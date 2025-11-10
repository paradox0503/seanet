import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn as nn
import math
from torch import Tensor
from util.conf import Configuration
import numpy as np


class patch_mask(nn.Module):
    def __init__(self, conf: Configuration):
        super(patch_mask,self).__init__()
        self.patch_len=conf.getHP('patch_len')
        self.stride=conf.getHP('stride')
        self.masking_ratio=conf.getHP('masking_ratio')
        self.mask_average_len=conf.getHP('mask_average_len')
    def geom_noise_mask_single(L, lm, masking_ratio):
        keep_mask = np.ones(L, dtype=bool)
        p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
        p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
        p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
        state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
        # print("1!!!!!!!!!", state)
        for i in range(L):
            keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
            if np.random.rand() < p[state]:
                state = 1 - state
        return keep_mask
    def noise_mask(X, masking_ratio, lm=3):
        # 原版X: (seq_length, feat_dim) 现#x:[bs,n_vars,num_patch, patch_len]

        mask = np.ones([int(X.shape[0]), int(X.shape[1])], dtype=bool)
        for n in range(X.shape[0]):  # iterate over batch dimension
              # iterate over number of patches

            mask[n, :] = patch_mask.geom_noise_mask_single(X.shape[1], lm, masking_ratio)  # apply noise mask
        return mask

    def forward(self, x: Tensor)->Tensor:#x:[bs,n_vars,num_patch, d_model]

        mask = patch_mask.noise_mask(x, self.masking_ratio, self.mask_average_len)
        mask = torch.tensor(mask, dtype=torch.bool).to("cuda")
        return mask


# input :seq_len*batch_num*dim_series
class CreatPatch(nn.Module):
    def __init__(self, conf: Configuration):
        super(CreatPatch,self).__init__()

        self.patch_len=conf.getHP('patch_len')
        self.stride=conf.getHP('stride')
        seq_len=conf.getHP('dim_series')
        self.num_patch = (max(seq_len, self.patch_len)-self.patch_len) // self.stride + 1
        tgt_len = self.patch_len  + self.stride*(self.num_patch-1)
        self.s_begin = seq_len - tgt_len

    def forward(self, x: Tensor):
        """
        x: [bs x seq_len x n_vars]   x = x[:, self.s_begin：, :]
        当前：seq_len*batch_num*dim_series
        """
        x = x[:, :, self.s_begin:]
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # x: [bs x n_vars x num_patch  x patch_len]
        return x



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout).to("cuda")#神经元有0.1概率不被激活
        pe = torch.zeros(max_len, d_model).to("cuda")
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to("cuda")
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to("cuda")
        pe[:, 0::2] = torch.sin(position * div_term).to("cuda")
        pe[:, 1::2] = torch.cos(position * div_term).to("cuda")
        pe = pe.unsqueeze(0).transpose(0, 1).to("cuda")
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:#(bs*n_vars, num_patch, self.d_model) *8
        # num_patch = x.size(1)
        #x = x + self.pe[:num_patch, :].transpose(0, 1).unsqueeze(0)
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return PositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))



class TEM(nn.Module):
    def __init__(self, conf: Configuration):
        super(TEM,self).__init__()
        dim_series = conf.getHP('dim_series')
        dropout = conf.getHP('dropout')
        nhead = conf.getHP('nhead')
        self.d_model = conf.getHP('d_model')
        num_encoder_layers = conf.getHP('num_encoder_layers')
        dim_feedforward = conf.getHP('dim_feedforward')
        dim_embedding = conf.getHP('dim_embedding')
        self.patch_len=conf.getHP('patch_len')
        self.stride=conf.getHP('stride')
        seq_len=conf.getHP('dim_series')
        self.seq_len=conf.getHP('dim_series')
        self.dim_embedding=dim_embedding
        n_test=conf.getHP("first_dim")
        self.num_patch = (max(n_test, self.patch_len)-self.patch_len) // self.stride + 1
        # self.num_patch1 = (max(96, self.patch_len)-self.patch_len) // self.stride + 1
        # self.num_patch2 = (max(128, self.patch_len)-self.patch_len) // self.stride + 1


        self.creatpatch = CreatPatch(conf).to("cuda")
        self.pos_encoder = PositionalEncoding(self.d_model, dropout).to("cuda")
        self.linear0= nn.Linear(self.patch_len, self.d_model).to("cuda")
        self.linear1 = nn.Linear(self.d_model, dim_embedding).to("cuda")
        encoder_layers = TransformerEncoderLayer(self.d_model, nhead, dim_feedforward, dropout).to("cuda")

        # self.encoder =  nn.TransformerEncoder(encoder_layers, num_encoder_layers).to("cuda")
        self.encoder =  TransformerEncoder(encoder_layers, num_encoder_layers).to("cuda")
        self.fuc = nn.Parameter(torch.tensor(0.948, dtype=torch.float32, requires_grad = False))
        # self.c = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad = False))
        self.fc = nn.Parameter(torch.tensor(0.999, dtype=torch.float32, requires_grad=False))

        n=dim_embedding*self.num_patch
        # n1=dim_embedding*self.num_patch1
        # n2=dim_embedding*self.num_patch2
        self.linear2 = nn.Linear(n, dim_embedding).to("cuda")
        self.linear0_1 = nn.Linear(96, n_test).to("cuda")
        self.linear0_2 = nn.Linear(128, n_test).to("cuda")
        self.linear0_3 = nn.Linear(256, n_test).to("cuda")



        self.norm = nn.LayerNorm(dim_embedding, elementwise_affine=False).to("cuda")

        self.seqline_1 = nn.Linear(self.d_model*self.num_patch, 96).to("cuda")
        self.seqline_2 = nn.Linear(self.d_model*self.num_patch, 128).to("cuda")
        self.seqline_3 = nn.Linear(self.d_model*self.num_patch, 256).to("cuda")
        self.norm_10 = nn.LayerNorm(96, elementwise_affine=False).to("cuda")
        self.norm_20 = nn.LayerNorm(128, elementwise_affine=False).to("cuda")
        self.norm_30 = nn.LayerNorm(256, elementwise_affine=False).to("cuda")

        self.begin_ns = nn.LayerNorm(seq_len, elementwise_affine=False).to("cuda")
        # self.batchnorm = nn.BatchNorm1d(dim_embedding)
        # self.norm = nn.BatchNorm1d(dim_embedding).to("cuda")
        self._reset_parameters()

        self.patch_mask=patch_mask(conf).to("cuda")
        self.fucb=nn.Parameter(torch.tensor(0.08, dtype=torch.float32, requires_grad = False))

        with torch.no_grad():  # 禁用梯度计算
            self.linear0_1.weight.copy_(torch.eye( n_test,96))  # 用单位矩阵赋值
            self.linear0_1.bias.zero_()  # 偏置设置为0
            self.linear0_2.weight.copy_(torch.eye( n_test,128))  # 用单位矩阵赋值
            self.linear0_2.bias.zero_()  # 偏置设置为0
            self.linear0_3.weight.copy_(torch.eye(n_test,256))  # 用单位矩阵赋值
            self.linear0_3.bias.zero_()  # 偏置设置为0


    def _reset_parameters(self):#遍历模型的所有参数，对维度大于1的权重矩阵进行Xavier初始化，以提高模型的性能。
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def new_forward(self, x):
        # x=self.begin_ns(x)
        # mean = x.mean(dim=-1, keepdim=True)
        # std = x.std(dim=-1, keepdim=True)
        # x = (x - mean) / (std + 1e-5)
        _,_,seq_len = x.size()
        if seq_len==96:
            x=self.linear0_1(x)
        if seq_len==128:
            x=self.linear0_2(x)
        if seq_len==256:
            x=self.linear0_3(x)
        x=self.creatpatch(x)#[bs,n-vars,dim_series]->[bs,n_vars,num_patch, patch_len]
        bs, n_vars, num_patch, patch_len = x.size()

        # print("a", end="")
        # x=x*padding_masks.to("cuda")

        x=self.linear0(x)#[bs,n_vars,num_patch, patch_len]->[bs,n_vars,num_patch, d_model]
        x=torch.reshape(x, (bs*n_vars, num_patch, self.d_model))#[bs,n_vars,num_patch, d_model]->[bs*n_vars,num_patch, d_model]
        padding_masks=self.patch_mask(x)#mvts  # ,bs*n_vars,num_patch,self.d_model # bs*n_vars  num_patch
        padding_masks=padding_masks.permute(1, 0)
        # print("遮罩形状：", padding_masks.shape)

        x = self.pos_encoder(x*math.sqrt(self.d_model))#[bs*n_vars,num_patch, d_model]->[bs*n_vars,num_patch, d_model



        # padding_masks = torch.tensor(padding_masks, dtype=torch.bool)
        # key_padding_mask = key_padding_mask.transpose(0, 3)

        x=self.encoder(x)#[bs*n_vars,num_patch, d_model]->, src_key_padding_mask=~padding_masks
        x=torch.reshape(x, (bs, n_vars, num_patch, self.d_model))#[bs*n_vars,num_patch, d_model]->[bs,n_vars,num_patch, d_model]

        y=x.transpose(2,3)
        y=torch.reshape(y, (bs, n_vars, num_patch*self.d_model))
        if seq_len==96:
            y=self.seqline_1(y)
            y=self.norm_10(y)
        elif seq_len==128:
            y=self.seqline_2(y)
            y=self.norm_20(y)
        elif seq_len==256:
            y=self.seqline_3(y)
            y=self.norm_30(y)

        x=self.linear1(x)#[bs,n_vars,num_patch, d_model]->[bs,n_vars,num_patch, dim_embedding]
        x=x.transpose(2,3)#[bs,n_vars,num_patch, dim_embedding]->[bs,n_vars,dim_embedding,num_patch]
        x = torch.reshape(x, (bs, n_vars, self.dim_embedding*num_patch))
        x = self.linear2(x)  # [bs, n_vars, num_patch * dim_embedding] -> [bs, n_vars, dim_embedding]


        # x = x.view(bs, 16)  # 调整形状为 [batch_size, 16]
        # x = self.batchnorm(x)  # 使用BatchNorm层
        # x = x.view(bs, 1, 16)

        x=self.norm(x)

        # 生成分为16段的paa
        # ox: [bs, n_vars, seq_len]
        paa_segments = 16
        paa = []
        for i in range(paa_segments):
            start = int(i * seq_len / paa_segments)
            end = int((i + 1) * seq_len / paa_segments)
            segment = ox[:, :, start:end]
            paa.append(segment.mean(dim=2, keepdim=True))
        paa = torch.cat(paa, dim=2)  # [bs, n_vars, 16]
        # 如果x的shape为[bs, n_vars, dim_embedding]，则需调整paa形状
        # 这里假设dim_embedding==16，否则需进一步处理
        x = x + torch.sigmoid(self.fc) * (paa - x)
        # print("n", end='')
        # x=torch.zeros_like(x)
        return [x,y]

    def forward(self, x: Tensor) -> Tensor:
        _,_,seq_len = x.size()
        if seq_len==96:
            x=self.linear0_1(x)
        if seq_len==128:
            x=self.linear0_2(x)
        if seq_len==256:
            x=self.linear0_3(x)


        x=self.creatpatch(x)#[bs,n-vars,dim_series]->[bs,n_vars,num_patch, patch_len]
        bs, n_vars, num_patch, patch_len = x.size()

        # print("a", end="")
        # x=x*padding_masks.to("cuda")

        x=self.linear0(x)#[bs,n_vars,num_patch, patch_len]->[bs,n_vars,num_patch, d_model]
        x=torch.reshape(x, (bs*n_vars, num_patch, self.d_model))#[bs,n_vars,num_patch, d_model]->[bs*n_vars,num_patch, d_model]
        padding_masks=self.patch_mask(x)#mvts  # ,bs*n_vars,num_patch,self.d_model # bs*n_vars  num_patch
        padding_masks=padding_masks.permute(1, 0)
        # print("遮罩形状：", padding_masks.shape)

        x = self.pos_encoder(x*math.sqrt(self.d_model))#[bs*n_vars,num_patch, d_model]->[bs*n_vars,num_patch, d_model



        # padding_masks = torch.tensor(padding_masks, dtype=torch.bool)
        # key_padding_mask = key_padding_mask.transpose(0, 3)

        x=self.encoder(x, src_key_padding_mask=~padding_masks)#[bs*n_vars,num_patch, d_model]->, src_key_padding_mask=~padding_masks
        x=torch.reshape(x, (bs, n_vars, num_patch, self.d_model))#[bs*n_vars,num_patch, d_model]->[bs,n_vars,num_patch, d_model]

        y=x.transpose(2,3)
        y=torch.reshape(y, (bs, n_vars, num_patch*self.d_model))
        if seq_len==96:
            y=self.seqline_1(y)
            y=self.norm_10(y)
        elif seq_len==128:
            y=self.seqline_2(y)
            y=self.norm_20(y)
        elif seq_len==256:
            y=self.seqline_3(y)
            y=self.norm_30(y)

        x=self.linear1(x)#[bs,n_vars,num_patch, d_model]->[bs,n_vars,num_patch, dim_embedding]
        x=x.transpose(2,3)#[bs,n_vars,num_patch, dim_embedding]->[bs,n_vars,dim_embedding,num_patch]
        x = torch.reshape(x, (bs, n_vars, self.dim_embedding*num_patch))
        x = self.linear2(x)  # [bs, n_vars, num_patch * dim_embedding] -> [bs, n_vars, dim_embedding]

        # x = x.view(bs, 16)  # 调整形状为 [batch_size, 16]
        # x = self.batchnorm(x)  # 使用BatchNorm层
        # x = x.view(bs, 1, 16)

        x=self.norm(x)

        # 生成分为16段的paa
        # ox: [bs, n_vars, seq_len]
        paa_segments = 16
        paa = []
        for i in range(paa_segments):
            start = int(i * seq_len / paa_segments)
            end = int((i + 1) * seq_len / paa_segments)
            segment = ox[:, :, start:end]
            paa.append(segment.mean(dim=2, keepdim=True))
        paa = torch.cat(paa, dim=2)  # [bs, n_vars, 16]
        # 如果x的shape为[bs, n_vars, dim_embedding]，则需调整paa形状
        # 这里假设dim_embedding==16，否则需进一步处理
        x = x + torch.sigmoid(self.fc) * (paa - x)
                        # print('-------------------------------------------------------------------------------')
        # print(x.shape)     #torch.Size([2000, 1, 16])
        # exit()


        return [x,y]

    def old_forward(self, x: Tensor) -> Tensor:
        # x=self.begin_ns(x)
        # mean = x.mean(dim=-1, keepdim=True)
        # std = x.std(dim=-1, keepdim=True)
        # x = (x - mean) / (std + 1e-5)
        _,_,seq_len = x.size()
        if seq_len==96:
            x=self.linear0_1(x)
        if seq_len==128:
            x=self.linear0_2(x)
        if seq_len==256:
            x=self.linear0_3(x)

        # print("f", end='')
        x=self.creatpatch(x)#[bs,n-vars,dim_series]->[bs,n_vars,num_patch, patch_len]
        bs, n_vars, num_patch, patch_len = x.size()
        x=self.linear0(x)#[bs,n_vars,num_patch, patch_len]->[bs,n_vars,num_patch, d_model]
        x=torch.reshape(x, (bs*n_vars, num_patch, self.d_model))#[bs,n_vars,num_patch, d_model]->[bs*n_vars,num_patch, d_model]
        padding_masks=self.patch_mask(x)#mvts  # ,bs*n_vars,num_patch,self.d_model # bs*n_vars  num_patch
        padding_masks=padding_masks.permute(1, 0)
        x = self.pos_encoder(x*math.sqrt(self.d_model))#[bs*n_vars,num_patch, d_model]->[bs*n_vars,num_patch, d_model
        x=self.encoder(x, src_key_padding_mask=~padding_masks)#[bs*n_vars,num_patch, d_model]->, src_key_padding_mask=~padding_masks
        x=torch.reshape(x, (bs, n_vars, num_patch, self.d_model))#[bs*n_vars,num_patch, d_model]->[bs,n_vars,num_patch, d_model]

        y=x.transpose(2,3)
        y=torch.reshape(y, (bs, n_vars, num_patch*self.d_model))
        if seq_len==96:
            y=self.seqline_1(y)
            y=self.norm_10(y)
        elif seq_len==128:
            y=self.seqline_2(y)
            y=self.norm_20(y)
        elif seq_len==256:
            y=self.seqline_3(y)
            y=self.norm_30(y)

        x=self.linear1(x)#[bs,n_vars,num_patch, d_model]->[bs,n_vars,num_patch, dim_embedding]
        x=x.transpose(2,3)#[bs,n_vars,num_patch, dim_embedding]->[bs,n_vars,dim_embedding,num_patch]
        x = torch.reshape(x, (bs, n_vars, self.dim_embedding*num_patch))
        x = self.linear2(x)  # [bs, n_vars, num_patch * dim_embedding] -> [bs, n_vars, dim_embedding]
        x=self.norm(x)

        # 生成分为16段的paa
        # ox: [bs, n_vars, seq_len]
        paa_segments = 16
        paa = []
        for i in range(paa_segments):
            start = int(i * seq_len / paa_segments)
            end = int((i + 1) * seq_len / paa_segments)
            segment = ox[:, :, start:end]
            paa.append(segment.mean(dim=2, keepdim=True))
        paa = torch.cat(paa, dim=2)  # [bs, n_vars, 16]
        # 如果x的shape为[bs, n_vars, dim_embedding]，则需调整paa形状
        # 这里假设dim_embedding==16，否则需进一步处理
        x = x + torch.sigmoid(self.fc) * (paa - x)
        return [x,y]



class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src: Tensor,src_key_padding_mask= None) -> Tensor:
        for layer in self.layers:
            src = layer(src, src_key_padding_mask)  # , src_key_padding_mask
        return src



class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=64, nhead=8, dim_feedforward=2048, dropout=0.1,):
        super(TransformerEncoderLayer, self).__init__()
        # conf=Configuration()
        # self.d_model = conf.getHP('d_model')
        # self.n_heads = conf.getHP('nhead')
        assert not d_model%nhead, f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout).to("cuda")
        self.linear1 = nn.Linear(d_model, dim_feedforward).to("cuda")
        self.dropout = nn.Dropout(dropout).to("cuda")
        self.linear2 = nn.Linear(dim_feedforward, d_model).to("cuda")
        self.norm1 = nn.LayerNorm(d_model).to("cuda")
        self.norm2 = nn.LayerNorm(d_model).to("cuda")
        self.dropout1 = nn.Dropout(dropout).to("cuda")
        self.dropout2 = nn.Dropout(dropout).to("cuda")

    def forward(self, src: Tensor, src_key_padding_mask= None) -> Tensor:
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



class TransformerDecoderModel(nn.Module):
    def __init__(self, conf: Configuration):
        super(TransformerDecoderModel, self).__init__()
        d_model = conf.getHP('d_model')
        dropout = conf.getHP('dropout')
        nhead = conf.getHP('nhead')
        num_decoder_layers = conf.getHP('num_decoder_layers')
        dim_feedforward = conf.getHP('dim_feedforward')
        dim_embedding = conf.getHP('dim_embedding')
        seq_len=conf.getHP('dim_series')

        self.pos_decoder = PositionalEncoding(dim_embedding, dropout).to("cuda")
        self.linear0 = nn.Linear(dim_embedding, d_model).to("cuda")
        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout).to("cuda")
        self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers).to("cuda")
        self.linear = nn.Linear(d_model, seq_len).to("cuda")
        self.norm = nn.LayerNorm(seq_len, elementwise_affine=False).to("cuda")
        self.dim_embedding = dim_embedding
        self._reset_parameters()

    def _reset_parameters(self):#遍历模型的所有参数，对维度大于1的权重矩阵进行Xavier初始化，以提高模型的性能。
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, embed: Tensor) -> Tensor:
        # print("!!_________________________!!")
        # print("!!_____in_the_decoder______!!")
        # print("!!_________________________!!")
        # print("embed:", embed.shape)
        embed = self.pos_decoder(embed * math.sqrt(self.dim_embedding))
        # print("embed_pe:", embed.shape)
        embed = self.linear0(embed)
        # print("embed_linear:", embed.shape)
        memory=self.decoder(embed, embed)
        # print("embed_dec:", memory.shape)
        memory=self.linear(memory)
        # print("embed_linear:", memory.shape)
        output=self.norm(memory)
        # print("output:", output.shape)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)]).to("cuda")
        self.num_layers = num_layers

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout).to("cuda")
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout).to("cuda")
        self.linear1 = nn.Linear(d_model, dim_feedforward).to("cuda")
        self.dropout = nn.Dropout(dropout).to("cuda")
        self.linear2 = nn.Linear(dim_feedforward, d_model).to("cuda")
        self.norm1 = nn.LayerNorm(d_model).to("cuda")
        self.norm2 = nn.LayerNorm(d_model).to("cuda")
        self.norm3 = nn.LayerNorm(d_model).to("cuda")
        self.dropout1 = nn.Dropout(dropout).to("cuda")
        self.dropout2 = nn.Dropout(dropout).to("cuda")
        self.dropout3 = nn.Dropout(dropout).to("cuda")

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

