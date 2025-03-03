import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from mtgvision.models.new_arch1 import LightInception
from mtgvision.models.nn import DepthwiseSeparableConv, SEBlock


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


class ModelBuilder(nn.Module):
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

    OUTPUT_SIZES = [
        (192, 128),
        (96, 64),
        (48, 32),
        (24, 16),
    ]

    def __init__(self):
        super(ModelBuilder, self).__init__()

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
        """Initial layer with a large kernel to capture broad context from misaligned inputs."""
        self.enc1 = LightInception(64)
        """First encoder stage using Inception for multi-scale feature extraction."""
        self.enc2 = DepthwiseSeparableConv(304, 128, kernel_size=3, stride=2, padding=1)
        """Second encoder stage, downsamples and reduces channels for efficiency."""
        self.enc3 = DepthwiseSeparableConv(128, 96, kernel_size=3, stride=2, padding=1)
        """Third encoder stage, further compresses spatial dimensions."""
        self.enc4 = DepthwiseSeparableConv(96, 64, kernel_size=3, stride=2, padding=1)
        """Fourth encoder stage, added to reduce bottleneck size further."""

        # Bottleneck: Compact representation targeting ≤512 elements
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(64, 32, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=3),
            SEBlock(32)
        )
        """Bottleneck reduces to (1, 32, 4, 4) = 512 elements with attention for key features."""

        # Decoder: Progressive upsampling with residual refinement
        self.dec4_main, self.dec4_residual = _dec_block(32, 64)
        """Fourth decoder stage, upsamples and refines with a separate residual path."""
        self.dec3_main, self.dec3_residual = _dec_block(64, 96)
        """Third decoder stage, continues upsampling with a separate residual path."""
        self.dec2_main, self.dec2_residual = _dec_block(96, 64)
        """Second decoder stage, further upsamples with a separate residual path."""
        self.dec1_main, self.dec1_residual = _dec_block(64, 32)
        """Third decoder stage, continues upsampling with a separate residual path."""
        # weird size, should decrease to 16
        self.dec0_main, self.dec0_residual = _dec_block(32, 32)
        """First decoder stage, prepares for final output with a separate residual path."""
        self.final = nn.Conv2d(32, 3, kernel_size=1, bias=False)
        """Final layer converts features to 3-channel output matching input resolution."""

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

    def encode(self, x):
        # Input shape: (1, 3, 192, 128) if NCHW, or (1, 192, 128, 3) if NHWC
        if x.size(1) != 3:
            if x.size(3) == 3:
                x = x.permute(0, 3, 1, 2)
                # Shape: (1, 3, 192, 128)

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

    def decode(self, z, *, multiscale: bool = True):
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

        if multiscale:
            return [x]
        else:
            return [x]

    def forward(self, x, *, multiscale: bool = True):
        z = self.encode(x)
        out = self.decode(z, multiscale=multiscale)
        return z, out




def create_model(x_size, y_size):
    """Create an instance of ModelBuilder with specified input and output sizes.

    Args:
        x_size (tuple): Input size in NHWC format (e.g., (1, 192, 128, 3)).
        y_size (tuple): Output size in NHWC format (e.g., (1, 192, 128, 3)).

    Returns:
        tuple: (model, encoded_tensor) where model is the ModelBuilder instance and
               encoded_tensor is the bottleneck representation.
    """
    assert len(x_size) == 4 and len(y_size) == 4
    assert x_size[1:] == (192, 128, 3) and y_size[1:] == (192, 128, 3)
    model = ModelBuilder()
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

    print(f"Input shape: {dummy_input.shape}")  # (1, 192, 128, 3) NHWC
    print(f"Output shape: {output.shape}")  # (1, 3, 192, 128) NCHW
    print(f"Encoding shape: {model.encoded.shape}")  # (1, 32, 4, 4) NCHW
    print(f"Encoding elements: {model.encoded.numel()}")  # 512
