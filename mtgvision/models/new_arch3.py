import torch
import torch.nn as nn
import torch.nn.functional as F
from mtgvision.models.nn import AeBase

# Coordinate Attention Module (unchanged)
class CoordinateAttention(nn.Module):
    def __init__(self, channels):
        super(CoordinateAttention, self).__init__()
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1))
        self.conv_w = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        pool_h = F.adaptive_avg_pool2d(x, (1, w))
        pool_w = F.adaptive_avg_pool2d(x, (h, 1))
        h_att = self.conv_h(pool_h)
        w_att = self.conv_w(pool_w)
        return x * self.sigmoid(h_att) * self.sigmoid(w_att)

# Depthwise Separable Convolution (unchanged)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, padding=padding, dilation=dilation,
            groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        num_groups = max(out_ch // 4, 1)
        self.bn = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# Inverted Residual Block with reduced expand_ratio
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2):  # Changed from 4 to 2
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
            out = out + x
        return nn.SiLU()(out)

# Downscale Block (unchanged)
def _downscale_block(in_ch, out_ch, dilation=1, final_stride=2):
    return nn.Sequential(
        DepthwiseSeparableConv(
            in_ch, out_ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation
        ),
        DepthwiseSeparableConv(
            out_ch, out_ch, kernel_size=3, stride=final_stride, padding=1, dilation=1
        ),
        CoordinateAttention(out_ch),
    )

# Upscale Block (unchanged structure, but uses new InvertedResidualBlock)
def _upscale_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        InvertedResidualBlock(out_ch, out_ch),
        CoordinateAttention(out_ch),
    )

# Optimized Autoencoder Class
class Ae3(AeBase):
    def __init__(self, stn: bool = True):
        super(Ae3, self).__init__()
        self.stn = stn

        # Optimized STN
        if self.stn:
            import kornia.geometry as kg
            self.localization = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False),
                nn.GroupNorm(4, 16),
                nn.SiLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
                nn.GroupNorm(8, 32),
                nn.SiLU(),
                CoordinateAttention(32),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.SiLU(),
            )
            self.fc_loc = nn.Sequential(
                nn.Linear(32 * 4 * 4, 64),  # Reduced from 64*24*16
                nn.SiLU(),
                nn.Linear(64, 6)
            )
            self.fc_loc[-1].weight.data.zero_()
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Encoder with enc5 removed and reduced channels
        self.stem = _downscale_block(3, 8, dilation=1)
        self.enc1 = _downscale_block(8, 16, dilation=1)
        self.enc2 = _downscale_block(16, 32, dilation=1)
        self.enc3 = _downscale_block(32, 64, dilation=2)
        self.enc4 = _downscale_block(64, 128, dilation=2)
        self.bottleneck = nn.Conv2d(128, 64, 1, bias=False)  # Applied after enc4

        # Decoder with reduced channels
        self.dec4 = _upscale_block(64, 32)  # 6x4x64 -> 12x8x32
        self.dec3 = _upscale_block(32, 16)  # 12x8x32 -> 24x16x16
        self.dec2 = _upscale_block(16, 8)   # 24x16x16 -> 48x32x8
        self.dec1 = _upscale_block(8, 4)    # 48x32x8 -> 96x64x4
        self.dec0 = _upscale_block(4, 4)    # 96x64x4 -> 192x128x4

        # Multi-scale Outputs with adjusted channels
        self.final_192 = nn.Conv2d(4, 3, 1, bias=False)
        self.final_96 = nn.Conv2d(4, 3, 1, bias=False)
        self.final_48 = nn.Conv2d(8, 3, 1, bias=False)

        self._init_weights()

    def _apply_stn(self, x):
        if not self.stn:
            return x
        import kornia.geometry as kg
        xs = self.localization(x)
        xs = F.adaptive_avg_pool2d(xs, (4, 4))  # Reduced pooling size
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
        x = self.bottleneck(x)  # Representation from 6x4 layer
        return x

    def _decode(self, z, multiscale: bool = True):
        x4 = self.dec4(z)   # 6x4x64 -> 12x8x32
        x3 = self.dec3(x4)  # 12x8x32 -> 24x16x16
        x2 = self.dec2(x3)  # 24x16x16 -> 48x32x8
        x1 = self.dec1(x2)  # 48x32x8 -> 96x64x4
        x0 = self.dec0(x1)  # 96x64x4 -> 192x128x4
        out_192 = self.final_192(x0)
        if multiscale:
            out_96 = self.final_96(x1)
            out_48 = self.final_48(x2)
            return [out_192, out_96, out_48]
        return [out_192]


if __name__ == '__main__':
    Ae3.quick_test(stn=True)
