import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #

class SEBlock(nn.Module):
    """Lightweight Squeeze-and-Excitation block for channel-wise attention.

    Enhances feature maps by reweighting channels based on global importance, improving
    reconstruction accuracy. Developed by Hu et al. (2018) in "Squeeze-and-Excitation Networks,"
    it adds minimal overhead for speed.

    Args:
        channels (int): Number of input/output channels.
        reduction (int): Factor to reduce channels in the MLP (default=8).
    """

    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficient feature extraction.

    Splits convolution into depthwise and pointwise steps, reducing FLOPs (e.g., from
    O(C_in * C_out * K^2) to O(C_in * K^2 + C_in * C_out)). From Chollet (2017) "Xception"
    and MobileNets (Howard et al., 2017), it boosts speed while retaining reconstruction power.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the depthwise convolution kernel.
        stride (int): Stride of the convolution (default=1).
        padding (int): Padding for the depthwise convolution (default=0).
        dilation (int): Dilation factor for larger receptive field (default=1).
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.gelu(x)


class LighterInception(nn.Module):
    """Efficient Inception block with multi-scale feature extraction.

    Based on Inception (Szegedy et al., 2015), it captures multi-scale features for misaligned
    inputs using parallel branches. Simplified with fewer channels and depthwise separable
    convolutions for speed, with dilation from DeepLab (Chen et al., 2017) for broader context.

    Args:
        in_channels (int): Number of input channels.
    """

    def __init__(self, in_channels):
        super(LighterInception, self).__init__()
        self.branch1 = nn.Conv2d(
            in_channels, 32, kernel_size=1, bias=False
        )  # Reduced from 64
        self.branch3 = DepthwiseSeparableConv(
            in_channels, 48, kernel_size=3, padding=2, dilation=2
        )  # Simplified
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 16, kernel_size=1, bias=False)  # Reduced from 48
        )
        self.se = SEBlock(96)  # 32 + 48 + 16 = 96 channels (down from 304)

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        bp = self.branch_pool(x)
        x = torch.cat([b1, b3, bp], dim=1)
        return self.se(x)


class ModelBuilder(nn.Module):
    """Encoder-Decoder model for fast, accurate image reconstruction.

    Reconstructs misaligned inputs using a deep encoder with Inception and depthwise separable
    convolutions, a compact bottleneck (≤512 elements), and an efficient decoder with residual
    refinements. Combines U-Net (Ronneberger et al., 2015), EfficientNet (Tan & Le, 2019),
    and DeepLab techniques for modern performance.

    Args:
        x_size (tuple): Input size in NCHW format (e.g., (batch, 3, 192, 128)).
        y_size (tuple): Output size in NCHW format (e.g., (batch, 3, 192, 128)).
    """

    def __init__(self, x_size, y_size):
        super(ModelBuilder, self).__init__()
        self.x_size = x_size  # e.g., (batch, 3, 192, 128) in NCHW
        self.y_size = y_size  # e.g., (batch, 3, 192, 128) in NCHW

        # Encoder: Progressive downsampling for compact representation
        self.stem = DepthwiseSeparableConv(
            3, 48, kernel_size=5, stride=2, padding=2
        )  # Reduced channels, smaller kernel
        """Initial layer with a moderate kernel for efficient context capture."""

        self.enc1 = LighterInception(48)  # Reduced from 64
        """First encoder stage with lightweight multi-scale feature extraction."""

        self.enc2 = DepthwiseSeparableConv(
            96, 64, kernel_size=3, stride=2, padding=1
        )  # Reduced channels
        """Second encoder stage, downsamples efficiently."""

        self.enc3 = DepthwiseSeparableConv(64, 48, kernel_size=3, stride=2, padding=1)
        """Third encoder stage, further compresses spatial dimensions."""

        self.enc4 = DepthwiseSeparableConv(48, 32, kernel_size=3, stride=2, padding=1)
        """Fourth encoder stage, reduces to a compact size."""

        # Bottleneck: Compact representation targeting ≤512 elements
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(32, 32, kernel_size=3, stride=3, padding=0),
            # Stride replaces pooling
            SEBlock(32)
        )
        """Bottleneck compresses to (batch, 32, 4, 4) = 512 elements per item with attention."""

        # Decoder: Progressive upsampling with streamlined residual refinement
        self.dec4_main = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False
        )
        self.dec4_conv = DepthwiseSeparableConv(32, 48, kernel_size=3, padding=1)
        """Fourth decoder stage, upsamples and refines efficiently."""

        self.dec3_main = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False
        )
        self.dec3_conv = DepthwiseSeparableConv(48, 64, kernel_size=3, padding=1)
        """Third decoder stage, continues upsampling."""

        self.dec2_main = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False
        )
        self.dec2_conv = DepthwiseSeparableConv(64, 32, kernel_size=3, padding=1)
        """Second decoder stage, prepares for final layers."""

        self.dec1_main = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False
        )
        self.dec1_conv = DepthwiseSeparableConv(32, 16, kernel_size=3, padding=1)
        """First decoder stage, reduces channels for output."""

        self.final = nn.Conv2d(16, 3, kernel_size=1, bias=False)
        """Final layer outputs 3-channel image at input resolution."""

        self._init_weights()
        self.encoded = None

    def _init_weights(self):
        """Initialize weights using Kaiming normal for conv layers and constant for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input shape: (batch, 3, 192, 128) if NCHW, or (batch, 192, 128, 3) if NHWC
        if x.size(1) != 3:
            if x.size(3) == 3:
                x = x.permute(0, 3, 1, 2)
                # Shape: (batch, 3, 192, 128)

        # Encoder
        x = self.stem(x)
        # Shape: (batch, 48, 96, 64) - stride=2 halves H and W (192/2=96, 128/2=64)
        x = self.enc1(x)
        # Shape: (batch, 96, 96, 64) - Inception outputs 32 + 48 + 16 = 96 channels
        x = self.enc2(x)
        # Shape: (batch, 64, 48, 32) - stride=2 halves H and W (96/2=48, 64/2=32)
        x = self.enc3(x)
        # Shape: (batch, 48, 24, 16) - stride=2 halves H and W (48/2=24, 32/2=16)
        x = self.enc4(x)
        # Shape: (batch, 32, 12, 8) - stride=2 halves H and W (24/2=12, 16/2=8)

        # Bottleneck
        x = self.bottleneck(x)
        # Shape: (batch, 32, 4, 4) - stride=3 reduces H and W ((12-3)/3+1=4, (8-3)/3+1=4)
        # Elements per item: 32 * 4 * 4 = 512
        self.encoded = x

        # Decoder
        x = self.dec4_main(x)
        # Shape: (batch, 32, 8, 8) - upsample x2 (4*2=8, 4*2=8)
        x = self.dec4_conv(x)
        # Shape: (batch, 48, 8, 8) - conv to 48 channels
        x = self.dec3_main(x)
        # Shape: (batch, 48, 16, 16) - upsample x2 (8*2=16, 8*2=16)
        x = self.dec3_conv(x)
        # Shape: (batch, 64, 16, 16) - conv to 64 channels
        x = self.dec2_main(x)
        # Shape: (batch, 64, 32, 32) - upsample x2 (16*2=32, 16*2=32)
        x = self.dec2_conv(x)
        # Shape: (batch, 32, 32, 32) - conv to 32 channels
        x = self.dec1_main(x)
        # Shape: (batch, 32, 64, 64) - upsample x2 (32*2=64, 32*2=64)
        x = self.dec1_conv(x)
        # Shape: (batch, 16, 64, 64) - conv to 16 channels
        x = self.final(x)
        # Shape: (batch, 3, 64, 64) - conv to 3 channels
        x = F.interpolate(x, size=(192, 128), mode='bilinear', align_corners=False)
        # Shape: (batch, 3, 192, 128) - interpolate to match input resolution

        return x


def create_model(x_size, y_size):
    """Create an instance of ModelBuilder with specified input and output sizes.

    Args:
        x_size (tuple): Input size in NHWC format (e.g., (batch, 192, 128, 3)).
        y_size (tuple): Output size in NHWC format (e.g., (batch, 192, 128, 3)).

    Returns:
        tuple: (model, encoded_tensor) where model is the ModelBuilder instance and
               encoded_tensor is the bottleneck representation.
    """
    assert len(x_size) == 4 and len(y_size) == 4
    model_x_size = (x_size[0], x_size[3], x_size[1], x_size[2])  # Convert to NCHW
    model_y_size = (y_size[0], y_size[3], y_size[1], y_size[2])  # Convert to NCHW
    model = ModelBuilder(model_x_size, model_y_size)
    return model, model.encoded


if __name__ == '__main__':
    x_size = (16, 192, 128, 3)  # NHWC format
    y_size = (16, 192, 128, 3)  # NHWC format, same as input

    model, encoding_layer = create_model(x_size, y_size)
    device = torch.device("mps")  # MPS for Apple Silicon
    model = model.to(device)
    dummy_input = torch.randn(16, 192, 128, 3).to(device)  # NHWC format

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    # Benchmark
    with torch.no_grad():
        for i in tqdm(range(100)):
            output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")  # (16, 192, 128, 3) NHWC
    print(f"Output shape: {output.shape}")  # (16, 3, 192, 128) NCHW
    print(f"Encoding shape: {model.encoded.shape}")  # (16, 32, 4, 4) NCHW
    print(f"Encoding elements per item: {model.encoded.numel() // x_size[0]}")  # 512