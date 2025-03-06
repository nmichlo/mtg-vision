from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from h5py.h5f import namedtuple

# Assuming AeBase is provided elsewhere with _init_weights and quick_test
from mtgvision.models.nn import AeBase


def Activation():
    return nn.Mish(inplace=True)


# Coordinate Attention Module for advanced attention
class CoordinateAttention(nn.Module):
    def __init__(self, channels):
        super(CoordinateAttention, self).__init__()
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        self.conv_w = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        pool_h = F.adaptive_avg_pool2d(x, (1, w))  # (B, C, 1, W)
        pool_w = F.adaptive_avg_pool2d(x, (h, 1))  # (B, C, H, 1)
        h_att = self.conv_h(pool_h)
        w_att = self.conv_w(pool_w)
        return x * self.sigmoid(h_att) * self.sigmoid(w_att)


# Depthwise Separable Convolution with dilation, GroupNorm, and SiLU
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, padding=padding, dilation=dilation,
            groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        num_groups = max(out_ch // 4, 1)  # Ensure at least 1 group
        self.bn = nn.GroupNorm(num_groups, out_ch)
        self.act = Activation()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# Inverted Residual Block with intra skips, GroupNorm, SiLU, and Coordinate Attention
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(max(hidden_dim // 4, 1), hidden_dim),
            Activation(),
            DepthwiseSeparableConv(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, dilation=1),
            CoordinateAttention(hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.GroupNorm(max(out_channels // 4, 1), out_channels),
        )
        self.use_residual = in_channels == out_channels

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x  # Intra skip connection
        return Activation()(out)


# **Downscale Block** with Dilated Convolutions
def _downscale_block(in_ch, out_ch, dilation=1, final_stride=2):
    return nn.Sequential(
        DepthwiseSeparableConv(
            in_ch, out_ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation
        ),
        DepthwiseSeparableConv(
            out_ch, out_ch, kernel_size=3, stride=final_stride, padding=1, dilation=1
        ),
        CoordinateAttention(out_ch),  # Advanced attention
    )


# **Upscale Block** with Upgraded Upsampling (bilinear + conv) and Intra Skips
def _upscale_block(in_ch, out_ch, upsample: bool = True, expand_ratio: int = 4):
    layers = [
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Upgraded upsampling
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        InvertedResidualBlock(out_ch, out_ch, expand_ratio=expand_ratio),  # Includes intra skips
        CoordinateAttention(out_ch),
    ]
    if upsample:
        return nn.Sequential(*layers)
    else:
        return nn.Sequential(*layers[1:])


_MISSING = object()


# Improved Autoencoder Class
class Ae2(AeBase):

    @classmethod
    def create_model_small(cls, x_size, y_size, stn: bool = False, multiscale: bool = False):
        return cls.create_model_verbose(x_size, y_size, model='old_s', stn=stn, multiscale=multiscale)

    @classmethod
    def create_model_medium(cls, x_size, y_size, stn: bool = False, multiscale: bool = False):
        return cls.create_model_verbose(x_size, y_size, model='old_m', stn=stn, multiscale=multiscale)

    @classmethod
    def create_model_heavy(cls, x_size, y_size, stn: bool = False, multiscale: bool = False):
        return cls.create_model_verbose(x_size, y_size, model='old_l', stn=stn, multiscale=multiscale)

    @classmethod
    def create_model_verbose(
        cls,
        x_size,
        y_size,
        *,
        # model configs
        model: str = None,
        model_enc: str = None,
        model_dec: str = None,
        model_stn: str = None,
        # enable features
        stn: bool = False,
        multiscale: bool = False,
        # overrides - enc
        enc_chs: Tuple[int, int, int, int, int] = _MISSING,
        enc_extra_ch: Optional[int] = _MISSING,
        enc_repr_ch: int = _MISSING,
        # overrides - dec
        dec_extra_ch: Optional[int] = _MISSING,
        dec_chs: Tuple[int, int, int, int, int] = _MISSING,
        dec_expand_ratio: int = _MISSING,
        # overrides - stn
        stn_chs: Tuple[int, int, int] = _MISSING,
        stn_groups: Tuple[int, int, int] = _MISSING,
        stn_out_size: Tuple[int, int] = _MISSING,
        stn_hidden: int = _MISSING,

    ):
        if model is not None:
            pass
        elif model_enc is not None and model_dec is not None and model_stn is not None:
            pass
        else:
            raise ValueError('Provide `model` OR `model_enc,model_dec,model_stn`')

        AE = namedtuple('AE', ['e', 'd', 's'])
        E = namedtuple('E', ['enc_chs', 'enc_extra_ch', 'enc_repr_ch'])
        D = namedtuple('D', ['dec_extra_ch', 'dec_chs', 'dec_expand_ratio'])
        S = namedtuple('S', ['stn', 'stn_chs', 'stn_groups', 'stn_out_size', 'stn_hidden'])
        # ENCODER DECODER
        stnL = S(stn=stn, stn_chs=(32, 64, 64),   stn_groups=(8, 16, 16),  stn_out_size=(24, 16), stn_hidden=128)
        items = {
            'old_s': AE(E(enc_chs=(16, 16, 24, 32, 64),    enc_extra_ch=64,  enc_repr_ch=32), D(dec_extra_ch=None, dec_chs=(16, 16, 24, 32, 64),  dec_expand_ratio=2), S(stn=stn, stn_chs=(16, 24, 32), stn_groups=(4, 6, 8),   stn_out_size=(12, 8),  stn_hidden=96)),
            'old_m': AE(E(enc_chs=(16, 16, 32, 64, 128),   enc_extra_ch=128, enc_repr_ch=32), D(dec_extra_ch=None, dec_chs=(16, 16, 32, 64, 128), dec_expand_ratio=2), S(stn=stn, stn_chs=(16, 32, 32), stn_groups=(4, 8, 8),   stn_out_size=(24, 16), stn_hidden=128)),
            'old_l': AE(E(enc_chs=(16, 32, 64, 128, 256),  enc_extra_ch=256, enc_repr_ch=32), D(dec_extra_ch=None, dec_chs=(16, 16, 32, 64, 128), dec_expand_ratio=4), stnL),
            # newer models
            '16x16x16x32x64':      AE(E(enc_chs=(16, 16, 16, 32, 64),      enc_extra_ch=None, enc_repr_ch=32),D(dec_extra_ch=None, dec_chs=(16, 16, 16, 32, 64),     dec_expand_ratio=2), stnL),
            '16x16x32x32x64':      AE(E(enc_chs=(16, 16, 32, 32, 64),      enc_extra_ch=None, enc_repr_ch=32),D(dec_extra_ch=None, dec_chs=(16, 16, 32, 32, 64),     dec_expand_ratio=2), stnL),
            '16x16x32x64x128':     AE(E(enc_chs=(16, 16, 32, 64, 128),     enc_extra_ch=None, enc_repr_ch=32), D(dec_extra_ch=None, dec_chs=(16, 16, 32, 64, 128),    dec_expand_ratio=2), stnL),
            '16x32x32x64x128':     AE(E(enc_chs=(16, 32, 32, 64, 128),     enc_extra_ch=None, enc_repr_ch=32), D(dec_extra_ch=None, dec_chs=(16, 32, 32, 64, 128),    dec_expand_ratio=2), stnL),
            '16x32x64x128x256':    AE(E(enc_chs=(16, 32, 64, 128, 256),    enc_extra_ch=None, enc_repr_ch=32), D(dec_extra_ch=None, dec_chs=(16, 32, 64, 128, 256),   dec_expand_ratio=2), stnL),
            #
            '32x32x32x64x128':     AE(E(enc_chs=(32, 32, 32, 64, 128),     enc_extra_ch=None, enc_repr_ch=32), D(dec_extra_ch=None, dec_chs=(32, 32, 32, 64, 128),     dec_expand_ratio=2), stnL),
            '32x32x64x64x128':     AE(E(enc_chs=(32, 32, 64, 64, 128),     enc_extra_ch=None, enc_repr_ch=32), D(dec_extra_ch=None, dec_chs=(32, 32, 64, 64, 128),     dec_expand_ratio=2), stnL),
            '32x64x64x128x128':     AE(E(enc_chs=(32, 64, 64, 128, 128),   enc_extra_ch=None, enc_repr_ch=32), D(dec_extra_ch=None, dec_chs=(32, 64, 64, 128, 128),    dec_expand_ratio=2), stnL),
            '32x32x64x128x256':    AE(E(enc_chs=(32, 32, 64, 128, 256),    enc_extra_ch=None, enc_repr_ch=32), D(dec_extra_ch=None, dec_chs=(32, 32, 64, 128, 256),    dec_expand_ratio=2), stnL),
            '32x64x128x256x512':   AE(E(enc_chs=(32, 64, 128, 256, 512),   enc_extra_ch=None, enc_repr_ch=32), D(dec_extra_ch=None, dec_chs=(32, 64, 128, 256, 512),   dec_expand_ratio=2), stnL),
            #
            '64x64x64x128x256':    AE(E(enc_chs=(64, 64, 64, 128, 256),    enc_extra_ch=None, enc_repr_ch=32), D(dec_extra_ch=None, dec_chs=(64, 64, 64, 128, 256),    dec_expand_ratio=2), stnL),
            '64x64x128x256x512':   AE(E(enc_chs=(64, 64, 128, 256, 512),   enc_extra_ch=None, enc_repr_ch=32), D(dec_extra_ch=None, dec_chs=(64, 64, 128, 256, 512),   dec_expand_ratio=2), stnL),
            '64x128x256x512x1024': AE(E(enc_chs=(64, 128, 256, 512, 1024), enc_extra_ch=None, enc_repr_ch=32), D(dec_extra_ch=None, dec_chs=(64, 128, 256, 512, 1024), dec_expand_ratio=2), stnL),
        }
        # get
        try:
            e = items[model_enc or model].e
            d = items[model_dec or model].d
            s = items[model_stn or model].s
        except KeyError as e:
            raise KeyError(f'Invalid model: {model_enc or model} {model_dec or model} {model_stn or model}, valid: {list(items.keys())}') from e

        return cls.create_model(
            x_size, y_size,
            # enc
            enc_chs=e.enc_chs if (enc_chs is _MISSING) else enc_chs,
            enc_extra_ch=e.enc_extra_ch if (enc_extra_ch is _MISSING) else enc_extra_ch,
            enc_repr_ch=e.enc_repr_ch if (enc_repr_ch is _MISSING) else enc_repr_ch,
            # dec
            dec_extra_ch=d.dec_extra_ch if (dec_extra_ch is _MISSING) else dec_extra_ch,
            dec_chs=d.dec_chs if (dec_chs is _MISSING) else dec_chs,
            dec_expand_ratio=d.dec_expand_ratio if (dec_expand_ratio is _MISSING) else dec_expand_ratio,
            # stn
            stn=stn,
            stn_chs=s.stn_chs if (stn_chs is _MISSING) else stn_chs,
            stn_groups=s.stn_groups if (stn_groups is _MISSING) else stn_groups,
            stn_out_size=s.stn_out_size if (stn_out_size is _MISSING) else stn_out_size,
            stn_hidden=s.stn_hidden if (stn_hidden is _MISSING) else stn_hidden,
            # multiscale
            multiscale=multiscale,
        )

    def __init__(
        self,
        # enc
        enc_chs: Tuple[int, int, int, int, int] = (16, 32, 64, 128, 256),
        enc_extra_ch: Optional[int] = 256,  # useful for helping encode info to bottleneck
        enc_repr_ch: int = 256,  # 6x4x<repr_ch> e.g. 6*4*32 = 768
        # dec
        dec_extra_ch: Optional[int] = None,  # useful for helping decode info from bottleneck
        dec_chs: Tuple[int, int, int, int, int] = (16, 16, 32, 64, 128),  # reversed
        dec_expand_ratio: int = 4,
        # stn
        stn: bool = True,
        stn_chs: Tuple[int, int, int] = (32, 64, 64),
        stn_groups: Tuple[int, int, int] = (8, 16, 16),
        stn_out_size: Tuple[int, int] = (24, 16),
        stn_hidden: int = 128,
        # multiscale
        multiscale: bool = False,
    ):
        super(Ae2, self).__init__()
        self.enc_chs = enc_chs
        self.enc_extra_ch = enc_extra_ch
        self.enc_repr_ch = enc_repr_ch

        self.dec_extra_ch = dec_extra_ch
        self.dec_chs = dec_chs
        self.dec_expand_ratio = dec_expand_ratio

        self.stn = stn
        self.stn_out_size = stn_out_size
        self.stn_chs = stn_chs
        self.stn_gr = stn_groups
        self.stn_hidden = stn_hidden

        self.multiscale = multiscale

        # **Spatial Transformer Network (STN)** with Coordinate Attention
        if self.stn:
            self.localization = nn.Sequential(
                nn.Conv2d(3, self.stn_chs[0], kernel_size=7, stride=2, padding=3, bias=False),
                nn.GroupNorm(self.stn_gr[0], self.stn_chs[0]),  # Advanced normalization
                Activation(),            # Newer activation
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(self.stn_chs[0], self.stn_chs[1], kernel_size=5, stride=2, padding=2, bias=False),
                nn.GroupNorm(self.stn_gr[1], self.stn_chs[1]),
                Activation(),
                CoordinateAttention(self.stn_chs[1]),  # Advanced attention in STN
                nn.Conv2d(self.stn_chs[1], self.stn_chs[2], kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(self.stn_gr[2], self.stn_chs[2]),
                Activation(),
            )
            self.fc_loc = nn.Sequential(
                nn.Linear(self.stn_chs[2] * self.stn_out_size[0] * self.stn_out_size[1], self.stn_hidden),
                Activation(),
                nn.Linear(self.stn_hidden, 6)
            )
            self.fc_loc[-1].weight.data.zero_()
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        else:
            self.localization = None
            self.fc_loc = None

        # Encoder with Dilated Convolutions in later stages
        # --> 192x128x3
        self.stem = _downscale_block(3, self.enc_chs[0], dilation=1)
        # --> 96x64x[0]
        self.enc1 = _downscale_block(self.enc_chs[0], self.enc_chs[1], dilation=1)
        # --> 48x32x[1]
        self.enc2 = _downscale_block(self.enc_chs[1], self.enc_chs[2], dilation=1)
        # --> 24x16x[2]
        self.enc3 = _downscale_block(self.enc_chs[2], self.enc_chs[3], dilation=2)
        # --> 12x8x[3]
        self.enc4 = _downscale_block(self.enc_chs[3], self.enc_chs[4], dilation=2)
        # --> 6x4x[4]
        if self.enc_extra_ch is not None:
            # * extra layer for non-linear encoding, no downscaling here.
            self.enc_extra = _downscale_block(enc_chs[4], self.enc_extra_ch, dilation=2, final_stride=1)
            # --> 6x4x[extra]
            self.bottleneck = nn.Conv2d(self.enc_extra_ch, self.enc_repr_ch, 1, bias=False)
        else:
            self.enc_extra = None
            self.bottleneck = nn.Conv2d(enc_chs[4], self.enc_repr_ch, 1, bias=False)
        # --> 6x4x[repr]

        # Decoder starting from 6x4x<repr_ch>
        # --> 6x4x[repr]
        if self.dec_extra_ch is not None:
            # * extra layer for non-linear encoding, no upscaling here.
            self.dec_extra = _downscale_block(self.enc_repr_ch, self.dec_extra_ch, dilation=2, final_stride=1)
            # --> 6x4x[extra]
            self.dec4 = _upscale_block(self.dec_extra_ch, dec_chs[4], expand_ratio=self.dec_expand_ratio)
        else:
            self.dec_extra = None
            self.dec4 = _upscale_block(self.enc_repr_ch, dec_chs[4], expand_ratio=self.dec_expand_ratio)
        # --> 12x8x[4]
        self.dec3 = _upscale_block(dec_chs[4], dec_chs[3], expand_ratio=self.dec_expand_ratio)
        # --> 24x16x[3]
        self.dec2 = _upscale_block(dec_chs[3], dec_chs[2], expand_ratio=self.dec_expand_ratio)
        # --> 48x32x[2]
        self.dec1 = _upscale_block(dec_chs[2], dec_chs[1], expand_ratio=self.dec_expand_ratio)
        # --> 96x64x[1]
        self.dec0 = _upscale_block(dec_chs[1], dec_chs[0], expand_ratio=self.dec_expand_ratio)
        # --> 192x128x[0]

        # **Multi-Scale Outputs ENCODER**
        self.final_0_env = nn.Conv2d(enc_chs[0], 3, 1, bias=False)
        self.final_1_env = nn.Conv2d(enc_chs[1], 3, 1, bias=False)
        self.final_2_env = nn.Conv2d(enc_chs[2], 3, 1, bias=False)
        self.final_3_env = nn.Conv2d(enc_chs[3], 3, 1, bias=False)
        self.final_4_env = nn.Conv2d(enc_chs[4], 3, 1, bias=False)

        # **Multi-Scale Outputs DECODER**
        self.final_4 = nn.Conv2d(dec_chs[4], 3, 1, bias=False)
        self.final_3 = nn.Conv2d(dec_chs[3], 3, 1, bias=False)
        self.final_2 = nn.Conv2d(dec_chs[2], 3, 1, bias=False)
        self.final_1 = nn.Conv2d(dec_chs[1], 3, 1, bias=False)
        self.final_0 = nn.Conv2d(dec_chs[0], 3, 1, bias=False)
        # --> 192x128x3

        self._init_weights()

    def _apply_stn(self, x):
        if not self.stn:
            return x
        xs = self.localization(x)
        xs = F.adaptive_avg_pool2d(xs, self.stn_out_size)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def _encode(self, x):
        x = self._apply_stn(x)
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        if self.enc_extra is not None:
            x5 = self.enc_extra(x4)
            z = self.bottleneck(x5)
        else:
            z = self.bottleneck(x4)
        if self.multiscale:
            return z, [
                self.final_4_env(x4),
                self.final_3_env(x3),
                self.final_2_env(x2),
                self.final_1_env(x1),
                self.final_0_env(x0),
            ]
        else:
            return z, []

    def _decode(self, z):
        if self.dec_extra is not None:
            z = self.dec_extra(z)
        x4 = self.dec4(z)
        x3 = self.dec3(x4)
        x2 = self.dec2(x3)
        x1 = self.dec1(x2)
        x0 = self.dec0(x1)
        if self.multiscale:
            return [
                self.final_0(x0),
                self.final_1(x1),
                self.final_2(x2),
                self.final_3(x3),
                self.final_4(x4),
            ]
        else:
            return [
                self.final_0(x0),
            ]


if __name__ == '__main__':
    Ae2.quick_test(model=Ae2.create_model_small(x_size=(16, 192, 128, 3), y_size=(16, 192, 128, 3), stn=False), compile=False)
    Ae2.quick_test(model=Ae2.create_model_medium(x_size=(16, 192, 128, 3), y_size=(16, 192, 128, 3), stn=False), compile=False)
    Ae2.quick_test(model=Ae2.create_model_heavy(x_size=(16, 192, 128, 3), y_size=(16, 192, 128, 3), stn=False), compile=False)
