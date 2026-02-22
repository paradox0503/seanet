import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from torch.nn.utils import weight_norm
from torch import Tensor


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNetEmbedder(nn.Module):
    """
    TimesNet Embedder for data embedding using TimesNet approach.
    """

    def __init__(self, configs):
        super(TimesNetEmbedder, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

    def forward(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # De-Normalization from Non-stationary Transformer
        enc_out = enc_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(
                1, self.seq_len + self.pred_len, 1)))
        enc_out = enc_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(
                1, self.seq_len + self.pred_len, 1)))
        return enc_out
class Configs_tmsnt:
    pass

class TimesNetEncoder(nn.Module):
    """
    TimesNet Encoder for autoencoder, outputting global embedding.
    """

    def __init__(self, conf):
        super(TimesNetEncoder, self).__init__()
        # Map SEAnet Configuration to TimesNet parameters
        self.seq_len = conf.getHP('first_dim')
        self.pred_len = 0  # Set to 0 for encoder
        self.d_model = conf.getHP('dim_embedding')  # Use dim_embedding as d_model
        self.e_layers = conf.getHP('num_en_resblock')  # Number of layers
        self.enc_in = 1  # Assume single channel
        self.embed = 'fixed'
        self.freq = 'h'
        self.dropout = 0.1
        self.top_k = 2
        self.d_ff = self.d_model * 4
        self.num_kernels = 6
        self.dim_embedding = conf.getHP('dim_embedding')

        # Create a simple configs object for TimesNet components
        configs = Configs_tmsnt()
        configs.seq_len = self.seq_len
        configs.pred_len = self.pred_len
        configs.d_model = self.d_model
        configs.e_layers = self.e_layers
        configs.enc_in = self.enc_in
        configs.embed = self.embed
        configs.freq = self.freq
        configs.dropout = self.dropout
        configs.top_k = self.top_k
        configs.d_ff = self.d_ff
        configs.num_kernels = self.num_kernels
        n_test=conf.getHP("first_dim")
        self.configs = configs
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.linear0_1 = nn.Linear(96, n_test).to("cuda")
        self.linear0_2 = nn.Linear(128, n_test).to("cuda")
        self.linear0_3 = nn.Linear(256, n_test).to("cuda")
        self.seqline_1 = nn.Linear(96, 96).to("cuda")
        self.seqline_2 = nn.Linear(128, 128).to("cuda")
        self.seqline_3 = nn.Linear(256, 256).to("cuda")
        self.norm_10 = nn.LayerNorm(96, elementwise_affine=False).to("cuda")
        self.norm_20 = nn.LayerNorm(128, elementwise_affine=False).to("cuda")
        self.norm_30 = nn.LayerNorm(256, elementwise_affine=False).to("cuda")

        self.fuc = nn.Parameter(torch.tensor(0.948, dtype=torch.float32, requires_grad = False))
        self.fc = nn.Parameter(torch.tensor(0.999, dtype=torch.float32, requires_grad=False))
        self.fucb=nn.Parameter(torch.tensor(0.08, dtype=torch.float32, requires_grad = False))


        # Pooling to global embedding
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.squeeze = nn.Flatten(start_dim=1)
        # Linear to dim_embedding
        self.linear = nn.Linear(configs.d_model, self.dim_embedding, bias=False)
        self.norm = nn.LayerNorm(self.dim_embedding, elementwise_affine=False)
    def forward(self, input):
        return self.big(input)
    def new_forward(self, input):
        return self.big(input)
    def old_forward(self, input):
        return self.big(input)
    def big(self, input):
        _,_,seq_len = input.size()
        if seq_len==96:
            x=self.linear0_1(input)
        if seq_len==128:
            x=self.linear0_2(input)
        if seq_len==256:
            x=self.linear0_3(input)
        # Assume input is [B, c_in, seq_len], permute to [B, seq_len, c_in]
        x_enc = x.permute(0, 2, 1)
        x_mark_enc = None  # No temporal marks for now
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # Pooling
        enc_out = self.pool(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # [B, 1, d_model]
        enc_out = self.squeeze(enc_out)  # [B, d_model]
        enc_out = self.linear(enc_out)  # [B, dim_embedding]
        enc_out = self.norm(enc_out)
        return [enc_out,input]


class TimesNetDecoder(nn.Module):
    """
    TimesNet Decoder for autoencoder, reconstructing from global embedding.
    """

    def __init__(self, conf):
        super(TimesNetDecoder, self).__init__()
        # Map SEAnet Configuration to TimesNet parameters
        self.seq_len = conf.getHP('dim_series')
        self.pred_len = 0
        self.d_model = conf.getHP('dim_embedding')
        self.e_layers = conf.getHP('num_de_resblock')
        self.enc_in = 1
        self.embed = 'fixed'
        self.freq = 'h'
        self.dropout = 0.1
        self.top_k = 2
        self.d_ff = self.d_model * 4
        self.num_kernels = 6
        self.dim_embedding = conf.getHP('dim_embedding')
        self.c_out = 1  # Assume single channel output

        # Create a simple configs object
        class Configs:
            pass
        configs = Configs()
        configs.seq_len = self.seq_len
        configs.pred_len = self.pred_len
        configs.d_model = self.d_model
        configs.e_layers = self.e_layers
        configs.enc_in = self.enc_in
        configs.embed = self.embed
        configs.freq = self.freq
        configs.dropout = self.dropout
        configs.top_k = self.top_k
        configs.d_ff = self.d_ff
        configs.num_kernels = self.num_kernels

        self.configs = configs
        # Linear from dim_embedding to seq_len * d_model
        self.linear1 = nn.Linear(self.dim_embedding, self.seq_len * configs.d_model)
        # Reshape to [B, T, d_model]
        self.reshape = lambda x: x.view(x.size(0), self.seq_len, configs.d_model)
        # TimesNet blocks for reconstruction
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.layer_norm = nn.LayerNorm(configs.d_model)
        # Projection to output
        self.projection = nn.Linear(configs.d_model, self.c_out, bias=True)

    def forward(self, embedding):
        # Expand embedding to sequence
        out = self.linear1(embedding)  # [B, seq_len * d_model]
        out = self.reshape(out)  # [B, T, d_model]
        # TimesNet
        for i in range(len(self.model)):
            out = self.layer_norm(self.model[i](out))
        # Project to output
        out = self.projection(out)  # [B, T, c_out]
        return out
