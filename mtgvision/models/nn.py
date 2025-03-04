import warnings
from typing import final, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AeBase(nn.Module):

    encoded: torch.Tensor = None
    multiscale: bool = False

    def _encode(self, x) -> torch.Tensor:
        raise NotImplementedError

    def _decode(self, z) -> List[torch.Tensor]:
        raise NotImplementedError

    def _init_weights(self):
        """Initialize weights for convolutional and batch norm layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @final
    def decode(self, z) -> List[torch.Tensor]:
        # should output more tensors if multiscale
        # * first is always the full scale
        # * second is half the scale
        # * third is quarter the scale, etc.
        return self._decode(z)

    @final
    def encode(self, x) -> torch.Tensor:
        # Input shape: (1, 3, 192, 128) if NCHW, or (1, 192, 128, 3) if NHWC
        # if x.size(1) != 3:
        #     if x.size(3) == 3:
        #         x = x.permute(0, 3, 1, 2)
                # Shape: (1, 3, 192, 128)
        z = self._encode(x)
        self.encoded = z
        return z

    @final
    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        z = self.encode(x)
        multiout = self.decode(z)
        return z, multiout

    @classmethod
    def create_model(cls, x_size, y_size, **kwargs) -> 'AeBase':
        assert len(x_size) == 4 and len(y_size) == 4
        assert x_size[1:] == (192, 128, 3) and y_size[1:] == (192, 128, 3)
        model = cls(**kwargs)
        return model

    @classmethod
    def quick_test(cls, batch_size: int = 16, n: int = 100, model=None, compile: bool = False, **model_kwargs):
        from tqdm import tqdm

        # Define input and output sizes in NHWC format
        x_size = (batch_size, 192, 128, 3)
        y_size = (batch_size, 192, 128, 3)

        # Create model and move to MPS device
        if model is None:
            model = cls.create_model(x_size, y_size, **model_kwargs)
        else:
            if model_kwargs:
                warnings.warn("Ignoring model_kwargs when model is provided.")

        # details
        num_params = model.num_params()
        print(model)
        print(f"params: {num_params} ({num_params/1_000_000:.3f}M)")

        device = torch.device("mps")
        model = model.to(device)

        # compile
        if compile:
            model = torch.compile(model)

        # Create dummy input
        dummy_input = torch.randn(batch_size, 192, 128, 3).to(device)

        # Warm-up runs
        with torch.no_grad():
            for _ in range(10):
                model(dummy_input)

        # Benchmark
        with torch.no_grad():
            for i in tqdm(range(n)):
                z, (output, *_) = model(dummy_input)

        # Print shapes and bottleneck size
        print(f"Input shape: {dummy_input.shape}")  # (16, 192, 128, 3) NHWC
        print(f"Output shape: {output.shape}")  # (16, 3, 192, 128) NCHW
        print(f"Encoding shape: {z.shape}")  # (16, 32, 4, 4) NCHW
        print(f"Encoding elements per item: {z.numel() // x_size[0]}")

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())



class SEBlock(nn.Module):
    """Lightweight Squeeze-and-Excitation block for channel-wise attention.

    This block enhances feature maps by reweighting channels based on their global importance,
    improving reconstruction accuracy by focusing on relevant features. Developed by Hu et al.
    (2018) in "Squeeze-and-Excitation Networks," it uses global average pooling and a small
    MLP to recalibrate channel responses, adding minimal computational overhead (thus preserving speed).

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

    Splits a standard convolution into a depthwise (per-channel) and pointwise (1x1) step,
    reducing parameters and FLOPs significantly (e.g., from O(C_in * C_out * K^2) to
    O(C_in * K^2 + C_in * C_out)). Introduced by Chollet (2017) in "Xception" and popularized
    in MobileNets (Howard et al., 2017), it’s used here to boost speed while maintaining
    expressive power for reconstruction.

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


class CoordinateAttention(nn.Module):
    """Coordinate Attention mechanism for spatial and channel-wise modulation."""
    def __init__(self, channels):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Pool along height
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Pool along width
        self.conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        h_pool = self.pool_h(x).permute(0, 1, 3, 2)  # Pool height, adjust dims
        w_pool = self.pool_w(x)                      # Pool width
        y = torch.cat([h_pool, w_pool], dim=2)       # Concatenate along spatial dim
        y = self.conv(y)                             # Apply 1x1 conv
        h_att, w_att = torch.split(y, [h, w], dim=2) # Split into height and width attention
        h_att = h_att.permute(0, 1, 3, 2)           # Adjust dims back
        return x * self.sigmoid(h_att) * self.sigmoid(w_att)  # Modulate input
