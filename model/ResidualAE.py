# coding = utf-8
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from torch import nn, Tensor

from util.conf import Configuration
from model.commons import Squeeze, Reshape


class _OriginalResBlock(nn.Module):
    def __init__(self, conf: Configuration, in_channels, out_channels, dilation):
        super(_OriginalResBlock, self).__init__()

        dim_series = conf.getHP('dim_series')
        kernel_size = conf.getHP('size_kernel')
        padding = int(kernel_size / 2) * dilation
        activation_name = conf.getHP('activation_conv')
        bias = conf.getHP('layernorm_type') == 'none'

        self.__residual_link = nn.Sequential(conf.getWeightNorm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)),
                                             conf.getLayerNorm(dim_series), 
                                             conf.getActivation(activation_name),

                                             conf.getWeightNorm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)),
                                             conf.getLayerNorm(dim_series))
        
        if in_channels != out_channels:
            self.__identity_link = conf.getWeightNorm(nn.Conv1d(in_channels, out_channels, 1, bias=bias))
        else:
            self.__identity_link = nn.Identity()

        self.__after_addition = conf.getActivation(activation_name)
        
        
    def forward(self, input: Tensor) -> Tensor:
        residual = self.__residual_link(input)
        identity = self.__identity_link(input)

        return self.__after_addition(identity + residual)



class _PreActivatedResBlock(nn.Module):
    def __init__(self, conf: Configuration, in_channels, out_channels, dilation, first = False, last = False):
        super(_PreActivatedResBlock, self).__init__()

        dim_series = conf.getHP('dim_series')
        kernel_size = conf.getHP('size_kernel')
        padding = int(kernel_size / 2) * dilation
        activation_name = conf.getHP('activation_conv')
        bias = conf.getHP('layernorm_type') == 'none' or not conf.getHP('layernorm_elementwise_affine')

        if first:
            self.__first_block = conf.getWeightNorm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias))
            in_channels = out_channels
        else:
            self.__first_block = nn.Identity()

        self.__residual_link = nn.Sequential(conf.getLayerNorm(dim_series), 
                                             conf.getActivation(activation_name),
                                             conf.getWeightNorm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)),
                                      
                                             conf.getLayerNorm(dim_series),
                                             conf.getActivation(activation_name),
                                             conf.getWeightNorm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)))
        
        if in_channels != out_channels:
            self.__identity_link = conf.getWeightNorm(nn.Conv1d(in_channels, out_channels, 1, bias=bias))
        else:
            self.__identity_link = nn.Identity()

        if last:
            self.__after_addition = nn.Sequential(conf.getLayerNorm(dim_series), 
                                                   conf.getActivation(activation_name))
        else:
            self.__after_addition = nn.Identity()
        
        
    def forward(self, input: Tensor) -> Tensor:
        input = self.__first_block(input)

        residual = self.__residual_link(input)
        identity = self.__identity_link(input)

        return self.__after_addition(identity + residual)



class _ResNet(nn.Module):
    def __init__(self, conf: Configuration, to_encode: bool):
        super(_ResNet, self).__init__()

        num_resblock = conf.getHP('num_en_resblock') if to_encode else conf.getHP('num_de_resblock')

        if conf.getHP('dilation_type') == 'exponential':
            assert num_resblock > 1 and 2 ** (num_resblock + 1) <= conf.getHP('dim_series') + 1

        inner_channels = conf.getHP('num_en_channels') if to_encode else conf.getHP('num_de_channels')
        out_channels = conf.getHP('dim_en_latent') if to_encode else conf.getHP('dim_de_latent')

        if conf.getHP('resblock_pre_activation'):
            layers = [_PreActivatedResBlock(conf, 1, inner_channels, conf.getDilatoin(1, to_encode), first=True)]
            layers += [_PreActivatedResBlock(conf, inner_channels, inner_channels, conf.getDilatoin(depth, to_encode)) for depth in range(2, num_resblock)]
            layers += [_PreActivatedResBlock(conf, inner_channels, out_channels, conf.getDilatoin(num_resblock, to_encode), last=True)]
        else:
            layers = [_OriginalResBlock(conf, 1, inner_channels, conf.getDilatoin(1, to_encode))]
            layers += [_OriginalResBlock(conf, inner_channels, inner_channels, conf.getDilatoin(depth, to_encode)) for depth in range(2, num_resblock)]
            layers += [_OriginalResBlock(conf, inner_channels, out_channels, conf.getDilatoin(num_resblock, to_encode))]

        self.__model = nn.Sequential(*layers)

        
    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)


'''这段代码是用于创建一个Python的神经网络模型，该模型基于ResNet结构，用于对输入的图像进行编码。以下是代码的解释：

self.__model = nn.Sequential(_ResNet(conf, to_encode=True),：首先，创建一个神经网络模型，使用ResNet作为基本结构。
_ResNet()函数是用于创建ResNet模型的，它需要一个配置对象conf和一个布尔值to_encode作为参数。to_encode为True时，
表示当前正在编码阶段，为False时，表示当前在解码阶段。

nn.AdaptiveMaxPool1d(1),：接下来，使用nn.AdaptiveMaxPool1d()函数对输出进行池化操作，该函数的参数是一个整数1，
表示输出尺寸为1。这是因为ResNet模型最后是一个一维卷积层，需要将输出转换为长度为1的向量。

Squeeze(),：Squeeze()函数用于将输入的向量去掉batch dimension，
即将shape从(batch_size, 1, embedding_dim)变为(batch_size, embedding_dim)。
这是因为在某些情况下，模型可能需要忽略batch dimension。

nn.Linear(num_channels, dim_latent),：然后，使用nn.Linear()函数将上一个模块的输出与一个全连接层相连接。
该函数的参数包括输入特征维度num_channels和输出特征维度dim_latent。

conf.getActivation(conf.getHP('activation_linear')),：接下来，使用conf.getActivation()函数和
conf.getHP()函数来获取激活函数。这个函数接受一个配置对象conf和一个字符串activation_linear作为参数，并返回相应的激活函数。

nn.Linear(dim_latent, dim_embedding, bias=False),：然后，使用nn.Linear()函数将上一个模块的输出与一个全连接层相连接。
该函数的参数包括输入特征维度dim_latent和输出特征维度dim_embedding，以及一个布尔值bias，表示是否使用偏置。
这里bias设置为False，表示不使用偏置。

nn.LayerNorm(dim_embedding, elementwise_affine=False)：如果配置对象conf中encoder_normalize_embedding的值为True，
则使用nn.LayerNorm()函数对输出进行归一化操作。该函数的参数包括输入特征维度dim_embedding和布尔值elementwise_affine，
表示是否对每个元素进行独立的归一化。

if conf.getHP('encoder_normalize_embedding') else nn.Identity()：这个条件判断用于决定是否使用nn.LayerNorm()函数。
如果encoder_normalize_embedding的值为True，则使用LayerNorm；否则，使用Identity函数，表示不进行归一化。

self.__model.to(conf.getHP('device'))：最后，将模型移动到配置对象conf中指定的设备上（例如GPU）。



'''
class ResidualEncoder(nn.Module):
    def __init__(self, conf: Configuration):
        super(ResidualEncoder, self).__init__()

        dim_embedding = conf.getHP('dim_embedding')
        num_channels = conf.getHP('num_en_channels')
        dim_latent = conf.getHP('dim_en_latent')#256

        self.__model = nn.Sequential(_ResNet(conf, to_encode=True),
                                     nn.AdaptiveMaxPool1d(1),
                                     Squeeze(),

                                     nn.Linear(num_channels, dim_latent),
                                     conf.getActivation(conf.getHP('activation_linear')),

                                     nn.Linear(dim_latent, dim_embedding, bias=False),
                                     nn.LayerNorm(dim_embedding, elementwise_affine=False) if conf.getHP('encoder_normalize_embedding') else nn.Identity())

        self.__model.to(conf.getHP('device'))


    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)



class ResidualDecoder(nn.Module):
    def __init__(self, conf: Configuration):
        super(ResidualDecoder, self).__init__()

        dim_series = conf.getHP('dim_series')
        dim_embedding = conf.getHP('dim_embedding')
        num_channels = conf.getHP('num_de_channels')
        dim_latent = conf.getHP('dim_de_latent')

        self.__model = nn.Sequential(Reshape([-1, 1, dim_embedding]),
                                     nn.Linear(dim_embedding, dim_series),
                                     conf.getActivation(conf.getHP('activation_linear')),

                                     _ResNet(conf, to_encode=False),
                                     nn.AdaptiveMaxPool1d(1),
                                     Reshape([-1, 1, num_channels]),

                                     nn.Linear(num_channels, dim_latent),
                                     conf.getActivation(conf.getHP('activation_linear')),

                                     nn.Linear(dim_latent, dim_series, bias=False),
                                     nn.LayerNorm(dim_series, elementwise_affine=False) if conf.getHP('decoder_normalize_reconstruction') else nn.Identity())

        self.__model.to(conf.getHP('device'))


    def forward(self, input: Tensor) -> Tensor:
        return self.__model(input)



class SingleResidualDecoder(nn.Module):
    def __init__(self, conf: Configuration):
        super(SingleResidualDecoder, self).__init__()

        dim_series = conf.getHP('dim_series')
        assert dim_series == 256

        dim_embedding = conf.getHP('dim_embedding')
        assert dim_embedding == 16

        in_channels = 1
        assert in_channels == 1

        inner_channels = conf.getHP('num_de_channels')
        intermediate_channels = int(inner_channels / 2)
        kernel_size = conf.getHP('size_kernel')
        padding = int(kernel_size / 2)

        conv_activation_name = conf.getHP('activation_conv')
        linear_activation_name = conf.getHP('activation_linear')

        bias = conf.getHP('layernorm_type') == 'none'

        self.__reshape = Reshape([-1, in_channels, dim_embedding])

        if conf.getHP('resblock_pre_activation'):
            self.__residual_link = nn.Sequential(conf.getLayerNorm(16), 
                                                 conf.getActivation(conv_activation_name),
                                                 conf.getWeightNorm(nn.ConvTranspose1d(in_channels, intermediate_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)),
                                                 
                                                 conf.getLayerNorm(32), 
                                                 conf.getActivation(conv_activation_name),
                                                 conf.getWeightNorm(nn.ConvTranspose1d(intermediate_channels, inner_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)),
                                                 
                                                 conf.getLayerNorm(64), 
                                                 conf.getActivation(conv_activation_name),
                                                 conf.getWeightNorm(nn.ConvTranspose1d(inner_channels, intermediate_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)),
                                                 
                                                 conf.getLayerNorm(128), 
                                                 conf.getActivation(conv_activation_name),
                                                 conf.getWeightNorm(nn.ConvTranspose1d(intermediate_channels, in_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)))

            self.__after_addition = nn.Sequential(conf.getLayerNorm(256), 
                                                  conf.getActivation(conv_activation_name))
        else:
            self.__residual_link = nn.Sequential(conf.getWeightNorm(nn.ConvTranspose1d(in_channels, intermediate_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)),
                                                 conf.getLayerNorm(32), 
                                                 conf.getActivation(conv_activation_name),

                                                 conf.getWeightNorm(nn.ConvTranspose1d(intermediate_channels, inner_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)),
                                                 conf.getLayerNorm(64), 
                                                 conf.getActivation(conv_activation_name),

                                                 conf.getWeightNorm(nn.ConvTranspose1d(inner_channels, intermediate_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)),
                                                 conf.getLayerNorm(128), 
                                                 conf.getActivation(conv_activation_name),

                                                 conf.getWeightNorm(nn.ConvTranspose1d(intermediate_channels, in_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)),
                                                 conf.getLayerNorm(256))

            self.__after_addition = conf.getActivation(conv_activation_name)

        self.__identity_link = nn.Linear(dim_embedding, dim_series)

        self.__linear = nn.Sequential(nn.Linear(dim_series, dim_series),
                                      conf.getActivation(linear_activation_name),

                                      nn.Linear(dim_series, dim_series, bias=False))

        device = conf.getHP('device')

        self.__reshape.to(device)
        self.__residual_link.to(device)
        self.__identity_link.to(device)
        self.__after_addition.to(device)
        self.__linear.to(device)


    def forward(self, input: Tensor) -> Tensor:
        input = self.__reshape(input)
        
        residual = self.__residual_link(input)
        identity = self.__identity_link(input)
        output = self.__after_addition(identity + residual)

        return self.__linear(output)
