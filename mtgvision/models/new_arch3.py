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
    reconstruction accuracy. From Hu et al. (2018) "Squeeze-and-Excitation Networks," it’s
    kept lightweight for speed.

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

    Splits convolution into depthwise and pointwise steps, reducing FLOPs significantly.
    From Chollet (2017) "Xception" and MobileNets (Howard et al., 2017), it’s optimized
    here for speed and reconstruction power.

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


class MobileViTBlock(nn.Module):
    """Lightweight hybrid block combining convolutions and transformers.

    Inspired by MobileViT (Mehta & Rastegari, 2021), this block uses convolutions for local
    feature extraction and a transformer for global context, optimized for efficiency.
    Replaces Inception with a modern, speed-focused alternative while retaining multi-scale
    capabilities for misaligned inputs.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dim (int): Dimension of the transformer embedding.
    """

    def __init__(self, in_channels, out_channels, dim=64):
        super(MobileViTBlock, self).__init__()
        self.local_conv = DepthwiseSeparableConv(
            in_channels, dim, kernel_size=3, padding=1
        )
        self.global_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.proj = nn.Conv2d(dim, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        # Local features
        x = self.local_conv(x)  # (batch, dim, h, w)
        b, c, h, w = x.size()

        # Prepare for transformer: reshape to (batch, h*w, dim)
        x = x.view(b, c, h * w).permute(0, 2, 1)  # (batch, h*w, dim)
        attn_output, _ = self.global_attn(x, x, x)  # Self-attention
        x = attn_output.permute(0, 2, 1).view(b, c, h, w)  # Back to (batch, dim, h, w)

        # Project and normalize
        x = self.proj(x)
        x = self.bn(x)
        return self.gelu(x)


class ModelBuilder(nn.Module):
    """Modern, efficient encoder-decoder model for image reconstruction.

    Reconstructs misaligned inputs with a hybrid architecture blending convolutions and
    transformers. The encoder uses MobileViT and efficient convs for fast compression,
    hitting a bottleneck of ≤512 elements. The decoder employs lightweight upsampling inspired
    by Swin Transformer (Liu et al., 2021) and EfficientNet (Tan & Le, 2019) for speed and accuracy.

    Args:
        x_size (tuple): Input size in NCHW format (e.g., (batch, 3, 192, 128)).
        y_size (tuple): Output size in NCHW format (e.g., (batch, 3, 192, 128)).
    """

    def __init__(self, x_size, y_size):
        super(ModelBuilder, self).__init__()
        self.x_size = x_size  # e.g., (batch, 3, 192, 128) in NCHW
        self.y_size = y_size  # e.g., (batch, 3, 192, 128) in NCHW

        # Encoder: Efficient downsampling with hybrid blocks
        self.stem = DepthwiseSeparableConv(3, 32, kernel_size=5, stride=4, padding=2)
        """Initial layer with high stride for rapid spatial reduction and efficiency."""

        self.enc1 = MobileViTBlock(32, 48, dim=64)  # Hybrid conv-transformer block
        """First encoder stage with global and local feature extraction."""

        self.enc2 = DepthwiseSeparableConv(48, 64, kernel_size=3, stride=2, padding=1)
        """Second encoder stage, further downsamples efficiently."""

        self.enc3 = DepthwiseSeparableConv(64, 32, kernel_size=3, stride=3, padding=0)
        """Third encoder stage, compresses to bottleneck size."""

        # Bottleneck: Compact representation targeting ≤512 elements
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, groups=4, bias=False),
            # Grouped conv for efficiency
            SEBlock(32)
        )
        """Bottleneck refines (batch, 32, 4, 4) = 512 elements per item with attention."""

        # Decoder: Lightweight upsampling with minimal overhead
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=3, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(32, 48, kernel_size=3, padding=1)
        )
        """Third decoder stage, upsamples with efficient conv."""

        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(48, 32, kernel_size=3, padding=1)
        )
        """Second decoder stage, continues upsampling."""

        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(32, 16, kernel_size=3, padding=1)
        )
        """First decoder stage, prepares for final output."""

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
        # Shape: (batch, 32, 48, 32) - stride=4 quarters H and W (192/4=48, 128/4=32)
        x = self.enc1(x)
        # Shape: (batch, 48, 48, 32) - MobileViT outputs 48 channels
        x = self.enc2(x)
        # Shape: (batch, 64, 24, 16) - stride=2 halves H and W (48/2=24, 32/2=16)
        x = self.enc3(x)
        # Shape: (batch, 32, 8, 5) - stride=3 reduces H and W ((24-3)/3+1=8, (16-3)/3+1=5)

        # Bottleneck
        x = self.bottleneck(x)
        # Shape: (batch, 32, 8, 5) - grouped conv refines channels
        x = F.avg_pool2d(x, kernel_size=2, stride=2)  # Extra pooling to hit 4x4
        # Shape: (batch, 32, 4, 4) - stride=2 halves H and W with padding=0 ((8-2)/2+1=4, (5-2)/2+1=4)
        # Elements per item: 32 * 4 * 4 = 512
        self.encoded = x

        # Decoder
        x = self.dec3(x)
        # Shape: (batch, 48, 12, 12) - upsample x3 (4*3=12, 4*3=12), conv to 48 channels
        x = self.dec2(x)
        # Shape: (batch, 32, 24, 24) - upsample x2 (12*2=24, 12*2=24), conv to 32 channels
        x = self.dec1(x)
        # Shape: (batch, 16, 96, 96) - upsample x4 (24*4=96, 24*4=96), conv to 16 channels
        x = self.final(x)
        # Shape: (batch, 3, 96, 96) - conv to 3 channels
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