import kornia.feature
import torch
import torch.nn as nn
import torch.nn.functional as F

from mtgvision.models.nn import AeBase, DepthwiseSeparableConv, SEBlock


class LightInception(nn.Module):
    def __init__(self, in_channels):
        super(LightInception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, 48, kernel_size=1, bias=False)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            DepthwiseSeparableConv(64, 96, kernel_size=3, padding=2, dilation=2)
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, kernel_size=1, bias=False)
        )
        self.se = SEBlock(176)  # 48 + 96 + 32 = 176 channels

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        bp = self.branch_pool(x)
        x = torch.cat([b1, b3, bp], dim=1)
        return self.se(x)


class InvertedResidualBlock(nn.Module):
    """Inverted residual block inspired by MobileNetV3 for efficiency."""
    def __init__(self, in_channels, out_channels, expand_ratio=4):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            DepthwiseSeparableConv(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            SEBlock(hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.use_residual = in_channels == out_channels

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return F.gelu(out)


class EfficientChannelAttention(nn.Module):
    """Lightweight channel attention from ECA-Net."""
    def __init__(self, channels, kernel_size=3):
        super(EfficientChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


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

        # Encoder (unchanged)
        self.stem = DepthwiseSeparableConv(3, 64, kernel_size=7, stride=2, padding=3)
        self.enc1 = LightInception(64)
        self.enc2 = DepthwiseSeparableConv(176, 128, kernel_size=3, stride=2, padding=1)
        self.enc3 = DepthwiseSeparableConv(128, 96, kernel_size=3, stride=2, padding=1)
        self.enc4 = DepthwiseSeparableConv(96, 64, kernel_size=3, stride=2, padding=1)

        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(64, 64, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            SEBlock(64)
        )

        # Decoder
        self.dec4 = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False),  # Expand for PixelShuffle
            nn.PixelShuffle(2),  # (B, 32, 12, 8)
            InvertedResidualBlock(32, 32)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False),
            nn.PixelShuffle(2),  # (B, 16, 24, 16)
            InvertedResidualBlock(16, 16),
            EfficientChannelAttention(16)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(16, 32, 1, bias=False),
            nn.PixelShuffle(2),  # (B, 8, 48, 32)
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.GELU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(8, 24, 1, bias=False),
            nn.PixelShuffle(2),  # (B, 6, 96, 64)
            nn.Conv2d(6, 6, 3, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.GELU()
        )
        self.dec0 = nn.Sequential(
            nn.Conv2d(6, 12, 1, bias=False),
            nn.PixelShuffle(2),  # (B, 3, 192, 128)
            nn.Conv2d(3, 3, 3, padding=1, bias=False),
        )

        # self.final = nn.Conv2d(3, 3, 1, bias=False)

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
        x = self.stem(x)  # (B, 64, 96, 64)
        x = self.enc1(x)  # (B, 176, 96, 64)
        x = self.enc2(x)  # (B, 128, 48, 32)
        x = self.enc3(x)  # (B, 96, 24, 16)
        x = self.enc4(x)  # (B, 64, 12, 8)
        x = self.bottleneck(x)  # (B, 64, 6, 4) WRONG
        assert x.numel() // x.size(0) <= 1536, "Bottleneck exceeds 1536 elements"
        return x

    def _decode(self, z, *, multiscale: bool = True):
        x = self.dec4(z)  # (B, 32, 12, 8)
        # outputs = [self.final(F.interpolate(x, size=(192, 128), mode='bilinear', align_corners=False))] if multiscale else []

        x = self.dec3(x)  # (B, 16, 24, 16)
        # outputs.append(self.final(F.interpolate(x, size=(192, 128), mode='bilinear', align_corners=False))) if multiscale else []

        x = self.dec2(x)  # (B, 8, 48, 32)
        # outputs.append(self.final(F.interpolate(x, size=(192, 128), mode='bilinear', align_corners=False))) if multiscale else []

        x = self.dec1(x)  # (B, 6, 96, 64)
        # outputs.append(self.final(F.interpolate(x, size=(192, 128), mode='bilinear', align_corners=False))) if multiscale else []

        x = self.dec0(x)  # (B, 3, 192, 128)

        # x = F.interpolate(x, size=(192, 128), mode='bilinear', align_corners=False)
        # x = self.final(x)  # (B, 3, 192, 128)

        # outputs.insert(0, x)  # Final output first
        return [x]


if __name__ == '__main__':
    Ae1b.quick_test(stn=True)
