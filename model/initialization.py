# coding = utf-8

from __future__ import print_function

import numpy as np
import torch
import torch.nn.init
import torch.nn as nn

'''这段代码实现了一个名为LSUVinit的函数，该函数用于初始化神经网络模型，
使其权重具有正交性（Orthonormal）。在训练过程中，神经网络的权重可能会导致梯度弥散或梯度爆炸，
通过正交初始化可以提高模型的可训练性。

LSUVinit函数的主要参数如下：

model：神经网络模型
data：用于计算模型激活值的输入数据
needed_std：目标标准差值
std_tol：标准差容差值
max_attempts：最大尝试次数
do_orthonorm：是否进行正交初始化
needed_mean：目标均值
cuda：是否使用GPU
verbose：是否显示初始化过程
实现原理如下：

遍历模型中的所有卷积层和全连接层，统计总层数。
遍历模型中的所有卷积层和全连接层，并为每个层添加一个钩子函数store_activations，用于存储层的输出激活值。
遍历模型中的所有卷积层和全连接层，并为每个层添加一个钩子函数add_current_hook，用于在层的输出激活值发生变化时更新钩子函数。
遍历模型中的所有卷积层和全连接层，并为每个层添加一个钩子函数apply_weights_correction，用于在层的权重需要修正时应用修正。
使用torch.no_grad()上下文，遍历模型中的所有层，计算层的输出激活值。
检查层的输出激活值的标准差是否接近所需标准差值needed_std，如果接近，则修正层的权重以使其更加正交。
记录每次修正的权重，如果需要进一步修正，则累积权重修正系数和偏置修正。
检查修正后的层的权重是否满足正交性，如果满足，则停止修正；否则，继续下一轮修正。
将修正后的权重应用到模型中。
注意：在使用LSUVinit函数初始化模型时，确保模型处于评估模式（model.eval()），因为初始化过程不涉及到训练过程。'''
gg = {}
gg['hook_position'] = 0
gg['total_fc_conv_layers'] = 0
gg['done_counter'] = -1
gg['hook'] = None
gg['act_dict'] = {}
gg['counter_to_apply_correction'] = 0
gg['correction_needed'] = False
gg['current_coef'] = 1.0


# Orthonorm init code is taked from Lasagne
# https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
def svd_orthonormal(w):
    shape = w.shape
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)#w;
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    print (shape, flat_shape)
    q = q.reshape(shape)
    return q.astype(np.float32)

def store_activations(self, input, output):
    gg['act_dict'] = output.data.cpu().numpy();
    #print('act shape = ', gg['act_dict'].shape)
    return


def add_current_hook(m):
    if gg['hook'] is not None:
        return
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        #print 'trying to hook to', m, gg['hook_position'], gg['done_counter']
        if gg['hook_position'] > gg['done_counter']:
            gg['hook'] = m.register_forward_hook(store_activations)
            #print ' hooking layer = ', gg['hook_position'], m
        else:
            #print m, 'already done, skipping'
            gg['hook_position'] += 1
    return

def count_conv_fc_layers(m):
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        gg['total_fc_conv_layers'] +=1
    return

def remove_hooks(hooks):
    for h in hooks:
        h.remove()
    return
def orthogonal_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if hasattr(m, 'weight'):
            w_ortho = svd_orthonormal(m.weight.data.cpu().numpy())
            m.weight.data = torch.from_numpy(w_ortho)
            try:
                nn.init.constant_(m.bias, 0)
            except:
                pass
        else:
            #nn.init.orthogonal(m.weight)
            w_ortho = svd_orthonormal(m.weight.data.cpu().numpy())
            #print w_ortho
            #m.weight.data.copy_(torch.from_numpy(w_ortho))
            m.weight.data = torch.from_numpy(w_ortho)
            try:
                nn.init.constant(m.bias, 0)
            except:
                pass
    return

def apply_weights_correction(m):
    if gg['hook'] is None:
        return
    if not gg['correction_needed']:
        return
    if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
        if gg['counter_to_apply_correction'] < gg['hook_position']:
            gg['counter_to_apply_correction'] += 1
        else:
            if hasattr(m, 'weight'):
                m.weight.data *= float(gg['current_coef'])
                gg['correction_needed'] = False
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias.data += float(gg['current_bias'])
            return
    return


# cite: ICLR16 All you need is a good init
# cite: https://github.com/ducha-aiki/LSUV-pytorch/tree/0c45eb2a9bd8978f13572c328f2f0d5d11939c99
def LSUVinit(model,data, needed_std = 1.0, std_tol = 0.1, max_attempts = 10, do_orthonorm = True,needed_mean = 0., cuda = False, verbose = True):
    cuda = data.is_cuda
    gg['total_fc_conv_layers']=0
    gg['done_counter']= 0
    gg['hook_position'] = 0
    gg['hook']  = None
    model.eval();
    if cuda:
        model = model.cuda()
        data = data.cuda()
    else:
        model = model.cpu()
        data = data.cpu()
    if verbose: print( 'Starting LSUV')
    model.apply(count_conv_fc_layers)
    if verbose: print ('Total layers to process:', gg['total_fc_conv_layers'])
    with torch.no_grad():
        if do_orthonorm:
            model.apply(orthogonal_weights_init)
            if verbose: print ('Orthonorm done')
        if cuda:
            model = model.cuda()
        for layer_idx in range(gg['total_fc_conv_layers']):
            if verbose: print (layer_idx)
            model.apply(add_current_hook)
            out = model(data)
            # print("gg['act_dict']_len", len(gg['act_dict']))
            # print("gg['act_dict']", (gg['act_dict']))
            current_std = gg['act_dict'].std()
            current_mean = gg['act_dict'].mean()
            # print("---------**-----------", current_std)
            # print("----------&&----------", current_mean)

            if verbose: print ('std at layer ',layer_idx, ' = ', current_std)
            #print  gg['act_dict'].shape
            attempts = 0
            while (np.abs(current_std - needed_std) > std_tol):
                gg['current_coef'] =  needed_std / (current_std  + 1e-8);
                gg['current_bias'] =  needed_mean - current_mean * gg['current_coef'];
                gg['correction_needed'] = True
                model.apply(apply_weights_correction)
                if cuda:
                    model = model.cuda()
                out = model(data)
                current_std = gg['act_dict'].std()
                current_mean = gg['act_dict'].mean()
                if verbose: print ('std at layer ',layer_idx, ' = ', current_std, 'mean = ', current_mean)
                attempts+=1
                if attempts > max_attempts:
                    if verbose: print ('Cannot converge in ', max_attempts, 'iterations')
                    break
            if gg['hook'] is not None:
                gg['hook'].remove()
            gg['done_counter']+=1
            gg['counter_to_apply_correction'] = 0
            gg['hook_position'] = 0
            gg['hook']  = None
            if verbose: print ('finish at layer',layer_idx )
        if verbose: print ('LSUV init done!')
        if not cuda:
            model = model.cpu()
    return model
