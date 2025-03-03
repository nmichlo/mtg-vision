from typing import final, List, Tuple

import torch
import torch.nn as nn


class AeBase(nn.Module):

    encoded: torch.Tensor = None

    def _encode(self, x) -> torch.Tensor:
        raise NotImplementedError

    def _decode(self, z, *, multiscale: bool = True) -> List[torch.Tensor]:
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
    def decode(self, z, *, multiscale: bool = True) -> List[torch.Tensor]:
        return self._decode(z, multiscale=multiscale)

    @final
    def encode(self, x) -> torch.Tensor:
        # Input shape: (1, 3, 192, 128) if NCHW, or (1, 192, 128, 3) if NHWC
        if x.size(1) != 3:
            if x.size(3) == 3:
                x = x.permute(0, 3, 1, 2)
                # Shape: (1, 3, 192, 128)
        z = self._encode(x)
        self.encoded = z
        return z

    @final
    def forward(self, x, *, multiscale: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        z = self.encode(x)
        multiout = self.decode(z, multiscale=multiscale)
        return z, multiout

    @classmethod
    def create_model(cls, x_size, y_size) -> 'AeBase':
        assert len(x_size) == 4 and len(y_size) == 4
        assert x_size[1:] == (192, 128, 3) and y_size[1:] == (192, 128, 3)
        model = cls()
        return model

    @classmethod
    def quick_test(cls, batch_size: int = 16, n: int = 100):
        from tqdm import tqdm

        # Define input and output sizes in NHWC format
        x_size = (batch_size, 192, 128, 3)
        y_size = (batch_size, 192, 128, 3)

        # Create model and move to MPS device
        model, encoding_layer = cls.create_model(x_size, y_size)
        device = torch.device("mps")
        model = model.to(device)

        # Create dummy input
        dummy_input = torch.randn(batch_size, 192, 128, 3).to(device)

        # Warm-up runs
        with torch.no_grad():
            for _ in range(10):
                model(dummy_input)

        # Benchmark
        with torch.no_grad():
            for i in tqdm(range(n)):
                output = model(dummy_input)

        # Print shapes and bottleneck size
        print(f"Input shape: {dummy_input.shape}")  # (16, 192, 128, 3) NHWC
        print(f"Output shape: {output.shape}")  # (16, 3, 192, 128) NCHW
        print(f"Encoding shape: {model.encoded.shape}")  # (16, 32, 4, 4) NCHW
        print(f"Encoding elements per item: {model.encoded.numel() // x_size[0]}")


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
