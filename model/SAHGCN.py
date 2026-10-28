import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math


class WavelengthEncoder(nn.Module):
    '''
    Learn band-wise spectral position modulation from wavelength metadata.
    The output keeps the same number of spectral channels, so downstream
    CNN/HGCN layers can stay unchanged.
    '''

    def __init__(self, num_bands, wavelengths=None, fwhm=None, embed_dim=32, num_frequencies=4):
        super(WavelengthEncoder, self).__init__()
        self.num_bands = num_bands
        self.num_frequencies = num_frequencies

        if wavelengths is None:
            wavelengths = torch.linspace(0.0, 1.0, steps=num_bands)
        else:
            wavelengths = torch.as_tensor(wavelengths, dtype=torch.float32)
        if wavelengths.numel() != num_bands:
            raise ValueError("wavelengths length must match num_bands")

        if fwhm is None:
            fwhm = torch.zeros(num_bands, dtype=torch.float32)
        else:
            fwhm = torch.as_tensor(fwhm, dtype=torch.float32)
        if fwhm.numel() != num_bands:
            raise ValueError("fwhm length must match num_bands")

        wavelengths = wavelengths.reshape(num_bands)
        fwhm = fwhm.reshape(num_bands)
        wave_range = torch.clamp(wavelengths.max() - wavelengths.min(), min=1e-6)
        wavelengths = (wavelengths - wavelengths.min()) / wave_range
        fwhm = fwhm / wave_range

        self.register_buffer("wavelengths", wavelengths)
        self.register_buffer("fwhm", fwhm)
        self.register_buffer("frequencies", torch.pow(2.0, torch.arange(num_frequencies)) * math.pi)

        metadata_dim = 2 + 2 * num_frequencies
        self.encoder = nn.Sequential(
            nn.Linear(metadata_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.to_scale_bias = nn.Linear(embed_dim, 2)
        nn.init.zeros_(self.to_scale_bias.weight)
        nn.init.zeros_(self.to_scale_bias.bias)

    def _metadata_features(self):
        wave = self.wavelengths.unsqueeze(1)
        phase = wave * self.frequencies.unsqueeze(0)
        return torch.cat(
            [wave, torch.sin(phase), torch.cos(phase), self.fwhm.unsqueeze(1)],
            dim=1,
        )

    def forward(self, x):
        metadata = self._metadata_features().to(device=x.device, dtype=x.dtype)
        band_embedding = self.encoder(metadata)
        scale_bias = self.to_scale_bias(band_embedding)
        scale = torch.tanh(scale_bias[:, 0]).view(1, 1, self.num_bands)
        bias = scale_bias[:, 1].view(1, 1, self.num_bands)
        return x * (1.0 + scale) + bias


class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        # LeakyReLU是ReLU函数的一个变体，解决了ReLU函数存在的问题，α的默认往往是非常小的，比如0.01，这样就保证了Dead Neurons的问题。
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))  # 初始化为可训练的参数
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        G = G.to(torch.float32)
        if self.bias is not None:
            x = x + self.bias
        x = torch.sparse.mm(G.to_sparse(), x)
        return x
class HGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes,Q,A, dropout=0.1, momentum=0.1):
        super(HGCN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(input_dim, hidden_dim)
        self.hgc2 = HGNN_conv(hidden_dim, 64)
        # self.computeG = compute_G(W)
        self.batch_normalzation1 = nn.BatchNorm1d(input_dim)
        self.batch_normalzation2 = nn.BatchNorm1d(hidden_dim)
        self.batch_normalzation3 = nn.BatchNorm1d(64)
        self.A = A

        self.lin = nn.Linear(64, num_classes)
        self.lin2 = nn.Linear(input_dim, input_dim)
        self.Q=Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))
    def forward(self, x):
        (h, w, c) = x.shape
        x_flaten = x.reshape([h * w, -1])
        # x_flaten = self.batch_normalzation1(x_flaten)
        # x_flaten=self.lin2(x_flaten)
        # x_flaten = self.batch_normalzation1(x_flaten)
        supX = torch.mm(self.norm_col_Q.t(), x_flaten)
        supX = self.hgc1(supX, self.A)
        supX = self.batch_normalzation2(supX)
        supX = F.relu(supX)
        supX = F.dropout(supX, self.dropout)
        supX = self.hgc2(supX, self.A)
        Y = torch.matmul(self.Q, supX)
        return Y

class SAHGCN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, alph,
                 spectral_bands=None, wavelengths=None, fwhm=None, use_wavelength_encoder=True,
                 wavelength_embed_dim=32, wavelength_num_frequencies=4):
        super(SAHGCN, self).__init__()
        # 类别数,即网络最终输出通道数
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q
        self.alph = alph
        if spectral_bands is None:
            spectral_bands = changel - 2 if changel > 2 else changel
        self.spectral_bands = spectral_bands
        if use_wavelength_encoder and spectral_bands > 0:
            self.wavelength_encoder = WavelengthEncoder(
                spectral_bands,
                wavelengths=wavelengths,
                fwhm=fwhm,
                embed_dim=wavelength_embed_dim,
                num_frequencies=wavelength_num_frequencies,
            )
        else:
            self.wavelength_encoder = None
        layers_count = 2

        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 64, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(64), )
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(64, 64, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i < layers_count - 1:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(64 , 128, kernel_size=3))
            else:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))


        self.HGNN_Branch = HGCN(64,128, 64,self.Q,self.A,dropout=0.)
        # Softmax layer
        self.Softmax_linearHG = nn.Sequential(nn.Linear(64, self.class_count))  # 128
        self.Softmax_linearCN = nn.Sequential(nn.Linear(64, self.class_count))  # 128
        self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))  # 128

    def forward(self, x: torch.Tensor):

        (h, w, c) = x.shape
        if self.wavelength_encoder is not None:
            spectral_x = x[:, :, :self.spectral_bands]
            extra_x = x[:, :, self.spectral_bands:]
            spectral_x = self.wavelength_encoder(spectral_x)
            if extra_x.shape[-1] > 0:
                x = torch.cat((spectral_x, extra_x), dim=2)
            else:
                x = spectral_x

        # 先去除噪声
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))  # torch.unsqueeze(a，N)：就是在a中指定位置N加上一个维数为1的维度。
        noise = torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x = noise  # 直连

        hx = clean_x
        # CNN与GCN分两条支路
        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))  # spectral-spatial convolution
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])
        # GCN层 1 转化为超像素 x_flat 乘以 列归一化Q
        H = clean_x
        HGNN_result= self.HGNN_Branch(H)
        YHGNN = self.Softmax_linearHG( HGNN_result)
        YCNN = self.Softmax_linearCN( CNN_result)
        YHGNN=F.softmax(YHGNN,-1)
        YCNN=F.softmax(YCNN,-1)
        Y = self.alph * CNN_result + (1 - self.alph) * HGNN_result
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)

        return Y,YHGNN,YCNN

        #OnlyCNN
        # Y=CNN_result
        #OnlyHGCN
        # Y=HGNN_result
        
        # Y = self.Softmax_linear(Y)
        # Y = F.softmax(Y, -1)


        # return Y

