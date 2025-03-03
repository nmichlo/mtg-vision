import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming AeBase is provided elsewhere with _init_weights and quick_test
from mtgvision.models.nn import AeBase

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
        self.act = nn.SiLU()

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
            nn.SiLU(),
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
        return nn.SiLU()(out)



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
def _upscale_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Upgraded upsampling
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        InvertedResidualBlock(out_ch, out_ch),  # Includes intra skips
        CoordinateAttention(out_ch),
    )

# Improved Autoencoder Class
class Ae2(AeBase):
    def __init__(self, stn: bool = True):
        super(Ae2, self).__init__()
        self.stn = stn

        # **Spatial Transformer Network (STN)** with Coordinate Attention
        if self.stn:
            import kornia.geometry as kg
            self.localization = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
                nn.GroupNorm(8, 32),  # Advanced normalization
                nn.SiLU(),            # Newer activation
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
                nn.GroupNorm(16, 64),
                nn.SiLU(),
                CoordinateAttention(64),  # Advanced attention in STN
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(16, 64),
                nn.SiLU(),
            )
            self.fc_loc = nn.Sequential(
                nn.Linear(64 * 24 * 16, 128),
                nn.SiLU(),
                nn.Linear(128, 6)
            )
            self.fc_loc[-1].weight.data.zero_()
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Encoder with Dilated Convolutions in later stages
        self.stem = _downscale_block(3, 16, dilation=1)
        self.enc1 = _downscale_block(16, 32, dilation=1)
        self.enc2 = _downscale_block(32, 64, dilation=1)
        self.enc3 = _downscale_block(64, 128, dilation=2)  # Dilated conv
        self.enc4 = _downscale_block(128, 256, dilation=2)  # Dilated conv
        self.enc5 = _downscale_block(256, 512, dilation=2, final_stride=1)  # No downsampling
        self.bottleneck = nn.Conv2d(512, 256, 1, bias=False)

        # Decoder starting from 6x4x256
        self.dec4 = _upscale_block(256, 128)  # 6x4x256 -> 12x8x128
        self.dec3 = _upscale_block(128, 64)   # 12x8x128 -> 24x16x64
        self.dec2 = _upscale_block(64, 32)    # 24x16x64 -> 48x32x32
        self.dec1 = _upscale_block(32, 16)    # 48x32x32 -> 96x64x16
        self.dec0 = _upscale_block(16, 16)    # 96x64x16 -> 192x128x16

        # **Multi-Scale Outputs**
        self.final_192 = nn.Conv2d(16, 3, 1, bias=False)  # 192x128x3
        self.final_96 = nn.Conv2d(16, 3, 1, bias=False)   # 96x64x3
        self.final_48 = nn.Conv2d(32, 3, 1, bias=False)   # 48x32x3

        self._init_weights()

    def _apply_stn(self, x):
        if not self.stn:
            return x
        import kornia.geometry as kg
        xs = self.localization(x)
        xs = F.adaptive_avg_pool2d(xs, (24, 16))
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def _encode(self, x):
        x = self._apply_stn(x)
        x = self.stem(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.bottleneck(x)
        return x

    def _decode(self, z, multiscale: bool = True):
        x4 = self.dec4(z)   # 6x4x256 -> 12x8x128
        x3 = self.dec3(x4)  # 12x8x128 -> 24x16x64
        x2 = self.dec2(x3)  # 24x16x64 -> 48x32x32
        x1 = self.dec1(x2)  # 48x32x32 -> 96x64x16
        x0 = self.dec0(x1)  # 96x64x16 -> 192x128x16
        out_192 = self.final_192(x0)
        if multiscale:
            out_96 = self.final_96(x1)
            out_48 = self.final_48(x2)
            return [out_192, out_96, out_48]
        return [out_192]


if __name__ == '__main__':
    Ae2.quick_test(stn=True)
