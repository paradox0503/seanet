# coding = utf-8

import numbers
from typing import Union, List

import torch
from torch import Tensor, Size
from torch.nn import Module, init
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.functional import normalize


# cite: https://github.com/lancopku/AdaNorms
# cite: NeurIPS19 Understanding and Improving Layer Normalization
'''
这段代码定义了一个名为 AdaNorm 的 PyTorch 模块，用于自定义的归一化处理。它继承自 torch.nn.Module 类，以下是代码中关键部分的详细解释：

模块的初始化：

__init__ 方法用于初始化模块。
normalized_shape 参数决定了归一化的维度，它可以是整数、列表或 torch.Size 对象。
k 和 scale 参数是影响归一化计算的系数，默认值分别为 0.1 和 2.0。
eps 是用于防止除零错误的小常数，默认值是 1e-5。
elementwise_affine 是一个布尔值，表示是否为每个通道学习权重和偏移量。
参数重置：

reset_parameters 方法将 weight 初始化为全 1，将 bias 初始化为全 0。
向前传播：

forward 方法实现了模块的主要功能。
首先计算 input 的均值和标准差。
然后计算 graNorm，它是通过 k 系数缩放后的标准化梯度。
input_norm 是根据 graNorm 对输入进行归一化，然后乘以 scale。
最终，返回 scale 乘以 input_norm 的结果。
其他信息：

extra_repr 方法返回用于打印模块的额外信息。
其中包括 normalized_shape、eps、elementwise_affine、k 和 scale 等参数。
这个 AdaNorm 模块可以用于深度学习网络中的层归一化，其特殊的计算方式可能在某些应用中有独特的性能或训练优势。
'''
class AdaNorm(Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine', 'k', 'scale']
    
    normalized_shape: Union[int, List[int], torch.Size]
    eps: float
    elementwise_affine: bool
    k: float
    scale: float

    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], k: float = 1 / 10, scale: float = 2., eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(AdaNorm, self).__init__()

        self.k = k
        self.scale = scale
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        else:
            raise ValueError('Only last layer for AdaNorm currently')
        self.normalized_shape = tuple(normalized_shape)

        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)


    def forward(self, input: Tensor) -> Tensor:
        mean = input.mean(-1, keepdim=True)
        std = input.std(-1, keepdim=True)

        input = input - mean
        mean = input.mean(-1, keepdim=True)
        
        graNorm = (self.k * (input - mean) / (std + self.eps)).detach()
        input_norm = (input - input * graNorm) / (std + self.eps)
        
        return self.scale * input_norm


    def extra_repr(self) -> Tensor:
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}, k={k}, scale={scale}'.format(**self.__dict__)


'''
这段代码定义了一个函数 getSRIPTerm，用于计算模型参数的 Spectral Restricted Isometry Property (SRIPTerm)。这个函数的主要作用是在训练深度卷积神经网络时，通过对权重矩阵施加正交性约束来促进模型的稳定性和性能。

函数的关键部分解释如下：

通过迭代计算权重矩阵的近似谱值(sigma)。
使用正交约束，将谱值(sigma)的平方加总，作为 SRIPTerm。
返回计算得到的 SRIPTerm。
此函数的目的是根据给定模型的参数计算出 SRIPTerm，并且该术语在训练过程中作为正则化项被添加到损失函数中，以帮助保持输出特征/通道之间的正交性。'''
# regulize (preserve) orthogonality among output features/channels 
# under Spectral Restricted Isometry Property (of orthogonal matrix)
# extra hyper-parameters is added and should be searched: coefficient (weight) of SRIPTerm
# recommended by the authors: 1e-1(epoch 0) --> 1e-3(20) --> 1e-4(50) --> 1e-6(70) --> 0(120) of 200 epochs totally
# while at the same time changing coefficient of weight decay: 1e-8(0) --> 1e-4(20)
# cite: https://github.com/VITA-Group/Orthogonality-in-CNNs/blob/master/Imagenet/resnet/train_n.py
# cite: NeurIPS18 Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?
def getSRIPTerm(model: Module, device='cpu'):
    term = None

    for W in model.parameters():
        if W.ndimension() < 2:
            continue
        else:
            # for convolutional:
            # W.shape = [OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL_SIZE] 
            # rows = OUTPUT_CHANNELS, cols = INPUT_CHANNELS * KERNEL_SIZE

            # for linner:
            # W.shape = [OUTPUT_FEATURES, INTPUT_FEATURES]
            # rows = OUTPUT_FEATURES, cols = INTPUT_FEATURES

            cols = W[0].numel()
            rows = W.shape[0]

            w1 = W.view(-1, cols)
            wt = torch.transpose(w1, 0, 1)
            m  = torch.matmul(wt, w1)

            ident = Variable(torch.eye(cols,cols))
            ident = ident.to(device)

            w_tmp = (m - ident)
            height = w_tmp.size(0)

            # iterative computing approximate sigma
            u = normalize(w_tmp.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
            v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
            u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)

            sigma = torch.dot(u, torch.matmul(w_tmp, v))

            if term is None:
                term = (sigma) ** 2
            else:
                term = term + (sigma) ** 2
                
    return term

