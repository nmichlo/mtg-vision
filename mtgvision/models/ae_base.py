import warnings
from typing import final, List, Tuple

import torch
import torch.nn as nn


class AeBase(nn.Module):
    encoded: torch.Tensor = None
    multiscale: bool = False

    def _encode(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError

    def _decode(self, z) -> List[torch.Tensor]:
        raise NotImplementedError

    def _init_weights(self):
        """Initialize weights for convolutional and batch norm layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @final
    def decode(self, z, **kwargs) -> List[torch.Tensor]:
        # should output more tensors if multiscale
        # * first is always the full scale
        # * second is half the scale
        # * third is quarter the scale, etc.
        return self._decode(z)

    @final
    def encode(self, x, **kwargs) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Input shape: (1, 3, 192, 128) if NCHW, or (1, 192, 128, 3) if NHWC
        # if x.size(1) != 3:
        #     if x.size(3) == 3:
        #         x = x.permute(0, 3, 1, 2)
        # Shape: (1, 3, 192, 128)
        z, multi = self._encode(x)
        self.encoded = z
        return z, multi

    @final
    def forward(self, x, **kwargs) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        z, multi = self.encode(x)
        multiout = self.decode(z)
        return z, multiout + multi

    @classmethod
    def create_model(cls, x_size, y_size, **kwargs) -> "AeBase":
        assert len(x_size) == 4 and len(y_size) == 4
        assert x_size[1:] == (192, 128, 3) and y_size[1:] == (192, 128, 3)
        model = cls(**kwargs)
        return model

    @classmethod
    def quick_test(
        cls,
        batch_size: int = 16,
        n: int = 100,
        model=None,
        compile: bool = False,
        **model_kwargs,
    ):
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
        print(f"params: {num_params} ({num_params / 1_000_000:.3f}M)")

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
