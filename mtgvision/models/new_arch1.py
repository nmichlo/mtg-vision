import kornia.feature
import torch
import torch.nn as nn
import torch.nn.functional as F

from mtgvision.models.nn import AeBase, DepthwiseSeparableConv, SEBlock


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #

class LightInception(nn.Module):
    """Efficient Inception block with multi-scale feature extraction.

    Inspired by the Inception architecture (Szegedy et al., 2015, "Going Deeper with Convolutions"),
    this block uses parallel branches (1x1, 3x3, 5x5, pool) to capture features at different scales,
    improving reconstruction accuracy for misaligned inputs. Modified here with depthwise separable
    convolutions and reduced channel counts for speed, plus dilation for a larger receptive field
    to handle misalignment, as seen in DeepLab (Chen et al., 2017).

    Args:
        in_channels (int): Number of input channels.
    """

    def __init__(self, in_channels):
        super(LightInception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, 64, kernel_size=1, bias=False)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=1, bias=False),
            DepthwiseSeparableConv(96, 128, kernel_size=3, padding=2, dilation=2)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=1, bias=False),
            DepthwiseSeparableConv(24, 64, kernel_size=5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 48, kernel_size=1, bias=False)
        )
        self.se = SEBlock(304)  # 64 + 128 + 64 + 48 = 304 channels

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)
        x = torch.cat([b1, b3, b5, bp], dim=1)
        return self.se(x)


class Ae1(AeBase):
    """Encoder-Decoder model for image reconstruction with modern efficiency techniques.

    Designed to reconstruct images from misaligned inputs with high accuracy and speed.
    The encoder compresses the input through multiple downsampling stages, capturing rich
    features with Inception and depthwise separable convolutions. The bottleneck reduces
    dimensionality to ≤512 elements, and the decoder upsamples back to the input resolution
    with residual refinements. Incorporates techniques from U-Net (Ronneberger et al., 2015),
    EfficientNet (Tan & Le, 2019), and DeepLab for a balance of performance and efficiency.

    Args:
        x_size (tuple): Input size in NCHW format (e.g., (1, 3, 192, 128)).
        y_size (tuple): Output size in NCHW format (e.g., (1, 3, 192, 128)).
    """

    def __init__(self):
        super(Ae1, self).__init__()

        def _dec_block(in_channels, out_channels):
            dec_main = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)
            )
            dec_residual = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
            return dec_main, dec_residual

        # Encoder: Progressive downsampling for compact representation
        self.stem = DepthwiseSeparableConv(3, 64, kernel_size=7, stride=2, padding=3)
        self.enc1 = LightInception(64)
        self.enc2 = DepthwiseSeparableConv(304, 128, kernel_size=3, stride=2, padding=1)
        self.enc3 = DepthwiseSeparableConv(128, 96, kernel_size=3, stride=2, padding=1)
        self.enc4 = DepthwiseSeparableConv(96, 64, kernel_size=3, stride=2, padding=1)

        # Bottleneck: Compact representation targeting ≤512 elements
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(64, 32, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=3),
            SEBlock(32)
        )

        # Decoder: Progressive upsampling with residual refinement
        self.dec4_main, self.dec4_residual = _dec_block(32, 64)
        self.dec3_main, self.dec3_residual = _dec_block(64, 96)
        self.dec2_main, self.dec2_residual = _dec_block(96, 64)
        self.dec1_main, self.dec1_residual = _dec_block(64, 32)
        self.dec0_main, self.dec0_residual = _dec_block(32, 32)
        self.final = nn.Conv2d(32, 3, kernel_size=1, bias=False)

        self._init_weights()

    def _encode(self, x):
        # Encoder
        x = self.stem(x)
        # Shape: (1, 64, 96, 64) - stride=2 halves H and W (192/2=96, 128/2=64)
        x = self.enc1(x)
        # Shape: (1, 304, 96, 64) - Inception outputs 64 + 128 + 64 + 48 = 304 channels
        x = self.enc2(x)
        # Shape: (1, 128, 48, 32) - stride=2 halves H and W (96/2=48, 64/2=32)
        x = self.enc3(x)
        # Shape: (1, 96, 24, 16) - stride=2 halves H and W (48/2=24, 32/2=16)
        x = self.enc4(x)
        # Shape: (1, 64, 12, 8) - stride=2 halves H and W (24/2=12, 16/2=8)

        # Bottleneck
        x = self.bottleneck(x)
        # Shape after DepthwiseSeparableConv: (1, 32, 12, 8) - stride=1 keeps spatial size, conv to 32 channels
        # Shape after AvgPool2d: (1, 32, 4, 4) - 3x3 pool with stride=3 reduces H and W ((12-3)/3+1=4, (8-3)/3+1=4)
        # Elements: 32 * 4 * 4 = 512
        self.encoded = x
        return x

    def _decode(self, z, *, multiscale: bool = True):
        # Decoder
        x = self.dec4_main(z)
        # Shape: (1, 64, 8, 8) - upsample x2 (4*2=8, 4*2=8), conv to 64 channels
        x = x + self.dec4_residual(x)  # Residual connection with matching channels
        # Shape: (1, 64, 8, 8) - residual preserves shape
        x = self.dec3_main(x)
        # Shape: (1, 96, 16, 16) - upsample x2 (8*2=16, 8*2=16), conv to 96 channels
        x = x + self.dec3_residual(x)  # Residual connection with matching channels
        # Shape: (1, 96, 16, 16) - residual preserves shape
        x = self.dec2_main(x)
        # Shape: (1, 64, 32, 32) - upsample x2 (16*2=32, 16*2=32), conv to 64 channels
        x = x + self.dec2_residual(x)  # Residual connection with matching channels
        # Shape: (1, 64, 32, 32) - residual preserves shape
        x = self.dec1_main(x)
        # Shape: (1, 32, 64, 64) - upsample x2 (32*2=64, 32*2=64), conv to 32 channels
        x = x + self.dec1_residual(x)  # Residual connection with matching channels
        # Shape: (1, 32, 64, 64) - residual preserves shape
        x = self.dec0_main(x)
        # Shape: (1, 16, 128, 128) - upsample x2 (64*2=128, 64*2=128), conv to 16 channels
        x = x + self.dec0_residual(x)  # Residual connection with matching channels
        # Shape: (1, 16, 128, 128) - residual preserves shape
        x = F.interpolate(x, size=(192, 128), mode='bilinear', align_corners=False)
        # Shape: (1, 16, 192, 128) - interpolate to match input resolution
        x = self.final(x)
        # Shape: (1, 3, 192, 128) - convert to 3-channel output matching input resolution
        return [x]


if __name__ == '__main__':
    Ae1.quick_test()
