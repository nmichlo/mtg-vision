import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from mtgvision.models.nn import AeBase, DepthwiseSeparableConv, SEBlock


# ========================================================================= #
# Helper Modules                                                            #
# ========================================================================= #


class LightInceptionAlt(nn.Module):
    """Efficient Inception block with multi-scale feature extraction."""
    def __init__(self, in_channels):
        super(LightInceptionAlt, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, 32, kernel_size=1, bias=False)
        self.branch3 = DepthwiseSeparableConv(in_channels, 48, kernel_size=3, padding=2, dilation=2)
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 16, kernel_size=1, bias=False)
        )
        self.se = SEBlock(96)  # 32 + 48 + 16 = 96 channels

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        bp = self.branch_pool(x)
        x = torch.cat([b1, b3, bp], dim=1)
        return self.se(x)


class STN(nn.Module):
    """Spatial Transformer Network for global alignment."""
    def __init__(self, in_channels):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 48 * 32, 128),  # For input (192, 128) -> (48, 32) after pooling
            nn.ReLU(True),
            nn.Linear(128, 6)  # 6 parameters for affine transformation
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.reshape(-1, 32 * 48 * 32)  # Use reshape for non-contiguous tensors
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


# ========================================================================= #
# Main Model                                                                #
# ========================================================================= #


class Ae4(AeBase):
    """Encoder-Decoder model for fast, accurate image reconstruction."""

    def __init__(self):
        super(Ae4, self).__init__()

        # STN for global alignment
        self.stn = STN(3)

        # Encoder
        self.stem = nn.Sequential(
            DepthwiseSeparableConv(3, 48, kernel_size=3, stride=2, padding=2, dilation=2),
            DepthwiseSeparableConv(48, 48, kernel_size=3, padding=1)
        )
        self.enc1 = LightInceptionAlt(48)
        self.enc2 = DepthwiseSeparableConv(96, 64, kernel_size=3, stride=2, padding=1)
        self.enc3 = DepthwiseSeparableConv(64, 48, kernel_size=3, stride=2, padding=1)
        self.enc4 = DepthwiseSeparableConv(48, 32, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(32, 32, kernel_size=3, stride=3, padding=0),
            SEBlock(32)
        )

        # Decoder
        self.dec4_main = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4_conv = DepthwiseSeparableConv(32, 48, kernel_size=3, padding=1)
        self.dec3_main = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3_conv = DepthwiseSeparableConv(48, 64, kernel_size=3, padding=1)
        self.dec2_main = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2_conv = DepthwiseSeparableConv(64, 32, kernel_size=3, padding=1)
        self.dec1_main = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1_conv = DepthwiseSeparableConv(32, 16, kernel_size=3, padding=1)
        self.final = nn.Conv2d(16, 3, kernel_size=1, bias=False)

        self._init_weights()
        self.encoded = None

    def _encode(self, x):
        # STN alignment
        x = self.stn(x)
        # Shape: (batch, 3, 192, 128)

        # Encoder
        x = self.stem(x)
        # Shape: (batch, 48, 96, 64)
        x = self.enc1(x)
        # Shape: (batch, 96, 96, 64)
        x = self.enc2(x)
        # Shape: (batch, 64, 48, 32)
        x = self.enc3(x)
        # Shape: (batch, 48, 24, 16)
        x = self.enc4(x)
        # Shape: (batch, 32, 12, 8)

        # Bottleneck
        x = self.bottleneck(x)
        # Shape: (batch, 32, 4, 4)
        self.encoded = x
        return x

    def _decode(self, z, *, multiscale: bool = True):
        # Decoder
        x = self.dec4_main(z)
        # Shape: (batch, 32, 8, 8)
        x = self.dec4_conv(x)
        # Shape: (batch, 48, 8, 8)
        x = self.dec3_main(x)
        # Shape: (batch, 48, 16, 16)
        x = self.dec3_conv(x)
        # Shape: (batch, 64, 16, 16)
        x = self.dec2_main(x)
        # Shape: (batch, 64, 32, 32)
        x = self.dec2_conv(x)
        # Shape: (batch, 32, 32, 32)
        x = self.dec1_main(x)
        # Shape: (batch, 32, 64, 64)
        x = self.dec1_conv(x)
        # Shape: (batch, 16, 64, 64)
        x = self.final(x)
        # Shape: (batch, 3, 64, 64)
        x = F.interpolate(x, size=(192, 128), mode='bilinear', align_corners=False)
        # Shape: (batch, 3, 192, 128)
        if multiscale:
            return [x]
        else:
            return [x]


# ========================================================================= #
# Model Creation and Testing                                                #
# ========================================================================= #


if __name__ == '__main__':
    Ae4.quick_test()
