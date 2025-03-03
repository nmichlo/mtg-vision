import torch
import torch.nn as nn
from mtgvision.models.nn import AeBase, DepthwiseSeparableConv, SEBlock

# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


class Ae1b(AeBase):
    """Encoder-Decoder model for image reconstruction with modern efficiency techniques.

    Designed to reconstruct images from misaligned inputs with high accuracy and speed.
    The encoder compresses the input through multiple downsampling stages, capturing rich
    features with inverted residuals (MobileNetV2-inspired) and depthwise separable convolutions.
    The bottleneck reduces dimensionality to ~384 elements with a 3x2 aspect ratio, and the decoder
    upsamples back to the input resolution using sub-pixel convolutions and residual refinements.
    Incorporates multi-scale outputs for faster training and techniques from EfficientNet and
    MobileNet for mobile efficiency.

    Args:
        x_size (tuple): Input size in NCHW format (e.g., (1, 3, 192, 128)).
        y_size (tuple): Output size in NCHW format (e.g., (1, 3, 192, 128)).
    """

    def __init__(self):
        super(Ae1b, self).__init__()

        def _inverted_residual(in_ch, exp_ch, out_ch, stride):
            """Inverted residual block inspired by MobileNetV2 for efficient feature extraction."""
            return nn.Sequential(
                nn.Conv2d(in_ch, exp_ch, 1, bias=False),  # Expansion
                nn.BatchNorm2d(exp_ch),
                nn.GELU(),
                DepthwiseSeparableConv(exp_ch, exp_ch, 3, stride=stride, padding=1),
                nn.Conv2d(exp_ch, out_ch, 1, bias=False),  # Projection
                nn.BatchNorm2d(out_ch),
                SEBlock(out_ch)  # Channel attention
            )

        def _dec_block(in_ch, out_ch):
            """Decoder block with sub-pixel convolution for high-quality upsampling."""
            dec_main = nn.Sequential(
                nn.PixelShuffle(2),  # Upsample by 2x, reduces channels by 4x
                DepthwiseSeparableConv(in_ch // 4, out_ch, kernel_size=3, padding=1)
            )
            dec_residual = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            )
            return dec_main, dec_residual

        # Encoder: Progressive downsampling for compact representation
        self.stem = DepthwiseSeparableConv(3, 64, kernel_size=7, stride=2, padding=3)
        """Initial layer with a large kernel to capture broad context from misaligned inputs."""
        self.enc1 = _inverted_residual(64, 384, 192, stride=1)
        """First encoder stage using inverted residual for multi-scale feature extraction."""
        self.enc2 = DepthwiseSeparableConv(192, 128, kernel_size=3, stride=2, padding=1)
        """Second encoder stage, downsamples and reduces channels for efficiency."""
        self.enc3 = DepthwiseSeparableConv(128, 96, kernel_size=3, stride=2, padding=1)
        """Third encoder stage, further compresses spatial dimensions."""
        self.enc4 = DepthwiseSeparableConv(96, 64, kernel_size=3, stride=2, padding=1)
        """Fourth encoder stage, added to reduce bottleneck size further."""

        # Bottleneck: Compact representation targeting ~384 elements with 3x2 aspect ratio
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(64, 64, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((3, 2)),  # Preserve 3:2 aspect ratio
            SEBlock(64)
        )
        """Bottleneck reduces to (1, 64, 3, 2) = 384 elements with attention for key features."""

        # Decoder: Progressive upsampling with residual refinement and multi-scale outputs
        self.dec5_main, self.dec5_residual = _dec_block(64, 96)
        """Fifth decoder stage, upsamples from 3x2 to 6x4 with residual refinement."""
        self.dec4_main, self.dec4_residual = _dec_block(96, 64)
        """Fourth decoder stage, upsamples to 12x8 with residual refinement."""
        self.dec3_main, self.dec3_residual = _dec_block(64, 96)
        """Third decoder stage, upsamples to 24x16 with residual refinement."""
        self.pred3 = nn.Conv2d(96, 3, kernel_size=1)
        """Prediction head for multi-scale loss at 24x16."""
        self.dec2_main, self.dec2_residual = _dec_block(96, 64)
        """Second decoder stage, upsamples to 48x32 with residual refinement."""
        self.pred2 = nn.Conv2d(64, 3, kernel_size=1)
        """Prediction head for multi-scale loss at 48x32."""
        self.dec1_main, self.dec1_residual = _dec_block(64, 32)
        """First decoder stage, upsamples to 96x64 with residual refinement."""
        self.pred1 = nn.Conv2d(32, 3, kernel_size=1)
        """Prediction head for multi-scale loss at 96x64."""
        self.dec0_main, self.dec0_residual = _dec_block(32, 32)
        """Zeroth decoder stage, upsamples to 192x128 with residual refinement."""
        self.final = nn.Conv2d(32, 3, kernel_size=1, bias=False)
        """Prediction head for multi-scale loss at 192x128."""
        """Final layer converts features to 3-channel output matching input resolution."""

        self._init_weights()

    def _encode(self, x):
        # Encoder
        x = self.stem(x)
        # Shape: (1, 64, 96, 64) - stride=2 halves H and W (192/2=96, 128/2=64)
        x = self.enc1(x)
        # Shape: (1, 192, 96, 64) - Inverted residual outputs 192 channels
        x = self.enc2(x)
        # Shape: (1, 128, 48, 32) - stride=2 halves H and W (96/2=48, 64/2=32)
        x = self.enc3(x)
        # Shape: (1, 96, 24, 16) - stride=2 halves H and W (48/2=24, 32/2=16)
        x = self.enc4(x)
        # Shape: (1, 64, 12, 8) - stride=2 halves H and W (24/2=12, 16/2=8)

        # Bottleneck
        x = self.bottleneck(x)
        # Shape after DepthwiseSeparableConv: (1, 64, 12, 8) - stride=1 keeps spatial size
        # Shape after AdaptiveAvgPool2d: (1, 64, 3, 2) - reduces to 3x2 aspect ratio
        # Elements: 64 * 3 * 2 = 384
        self.encoded = x
        return x

    def _decode(self, z, *, multiscale: bool = True):
        # Decoder with multi-scale outputs
        x = self.dec5_main(z)
        # Shape: (1, 96, 6, 4) - PixelShuffle x2 (3*2=6, 2*2=4), channels 64/4 -> 96
        x = x + self.dec5_residual(x)
        x = self.dec4_main(x)
        # Shape: (1, 64, 12, 8) - PixelShuffle x2 (6*2=12, 4*2=8), channels 96/4 -> 64
        x = x + self.dec4_residual(x)
        x = self.dec3_main(x)
        # Shape: (1, 96, 24, 16) - PixelShuffle x2 (12*2=24, 8*2=16), channels 64/4 -> 96
        x = x + self.dec3_residual(x)
        if multiscale:
            out3 = self.pred3(x)
        # Shape: (1, 3, 24, 16) - Multi-scale output
        x = self.dec2_main(x)
        # Shape: (1, 64, 48, 32) - PixelShuffle x2 (24*2=48, 16*2=32), channels 96/4 -> 64
        x = x + self.dec2_residual(x)
        if multiscale:
            out2 = self.pred2(x)
        # Shape: (1, 3, 48, 32) - Multi-scale output
        x = self.dec1_main(x)
        # Shape: (1, 32, 96, 64) - PixelShuffle x2 (48*2=96, 32*2=64), channels 64/4 -> 32
        x = x + self.dec1_residual(x)
        if multiscale:
            out1 = self.pred1(x)
        # Shape: (1, 3, 96, 64) - Multi-scale output
        x = self.dec0_main(x)
        # Shape: (1, 32, 192, 128) - PixelShuffle x2 (96*2=192, 64*2=128), channels 32/4 -> 32
        x = x + self.dec0_residual(x)
        # Shape: (1, 3, 192, 128) - Multi-scale output
        out_final = self.final(x)
        # Shape: (1, 3, 192, 128) - Final output
        if multiscale:
            return [out_final, out3, out2, out1]
        else:
            return [out_final]


if __name__ == '__main__':
    Ae1b.quick_test()
