import torch
import torch.nn as nn
import torch.nn.functional as F

from mtgvision.models.nn import (
    AeBase, DepthwiseSeparableConv, EfficientChannelAttention, InvertedResidualBlock,
    SEBlock,
)


# class LightInception(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_1x1_ch: int = 48,  # 64
#         out_3x3_ch: int = 64,  # 128
#         out_ave_ch: int = 32,  # 32
#         out_ch: Optional[int] = None,
#     ):
#         super(LightInception, self).__init__()
#
#         # scale according to target output sizes
#         if out_ch is not None:
#             total_ch = out_1x1_ch + out_3x3_ch + out_ave_ch
#             out_1x1_ch = int(round(out_1x1_ch / total_ch * out_ch))
#             out_3x3_ch = int(round(out_3x3_ch / total_ch * out_ch))
#             out_ave_ch = int(round(out_ave_ch / total_ch * out_ch))
#
#         self.branch1 = nn.Conv2d(in_channels, out_1x1_ch, kernel_size=1, bias=False)
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(in_channels, int(out_3x3_ch/4*3), kernel_size=1, bias=False),
#             DepthwiseSeparableConv(int(out_3x3_ch/4*3), out_3x3_ch, kernel_size=3, padding=2, dilation=2)
#         )
#         self.branch_pool = nn.Sequential(
#             nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(in_channels, out_ave_ch, kernel_size=1, bias=False)
#         )
#         self.se = SEBlock(out_1x1_ch + out_3x3_ch + out_ave_ch)  # out_1x1_channels + 96 + 32 = 176 channels
#
#     def forward(self, x):
#         b1 = self.branch1(x)
#         b3 = self.branch3(x)
#         bp = self.branch_pool(x)
#         x = torch.cat([b1, b3, bp], dim=1)
#         return self.se(x)


class Ae1b(AeBase):
    def __init__(self, stn: bool = False):
        super(Ae1b, self).__init__()
        self.stn = stn

        if self.stn:
            import kornia.geometry as kg
            self.localization = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU()
            )
            self.fc_loc = nn.Sequential(
                nn.Linear(64 * 24 * 16, 128),
                nn.GELU(),
                nn.Linear(128, 6)
            )
            self.fc_loc[-1].weight.data.zero_()
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        def _downscale_block(in_ch, out_ch):
            return nn.Sequential(
                DepthwiseSeparableConv(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                DepthwiseSeparableConv(out_ch, out_ch, kernel_size=3, stride=2, padding=1),
                SEBlock(out_ch)
            )

        def _upscale_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch*4, 1, bias=False),  # Expand for PixelShuffle
                nn.PixelShuffle(2),  # (B, 32, 12, 8)
                InvertedResidualBlock(out_ch, out_ch),
                EfficientChannelAttention(out_ch),
            )

        # Encoder (unchanged)
        # (IN) 192x128x3=73728
        self.stem = _downscale_block(3, 16)
        # -> 96x64x16=98304
        self.enc1 = _downscale_block(16, 32)
        # -> 48x32x32=49152
        self.enc2 = _downscale_block(32, 64)
        # -> 24x16x64=24576
        self.enc3 = _downscale_block(64, 128)
        # -> 12x8x128=12288
        self.enc4 = _downscale_block(128, 256)
        # -> 6x4x256=6144
        self.enc5 = _downscale_block(256, 512)
        # -> 3x2x512=3072
        self.bottleneck = nn.Conv2d(512, 128, 1, bias=False)
        # -> 3x2x128=768 (OUT)

        # Decoder
        # (IN) 3x2x128=768
        self.dec5 = _upscale_block(128, 256)
        # -> 6x4x256=6144
        self.dec4 = _upscale_block(256, 128)
        # -> 12x8x128=12288
        self.dec3 = _upscale_block(128, 64)
        # -> 24x16x64=24576
        self.dec2 = _upscale_block(64, 32)
        # -> 48x32x32=49152
        self.dec1 = _upscale_block(32, 16)
        # -> 96x64x16=98304
        self.dec0 = _upscale_block(16, 16)
        # -> 192x128x16=393216
        self.final = nn.Conv2d(16, 3, kernel_size=1, bias=False)
        # -> 192x128x3=73728 (OUT)

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

    def _decode(self, z, *, multiscale: bool = True):
        x = self.dec5(z)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)
        return [x]


if __name__ == '__main__':
    Ae1b.quick_test(stn=True)
