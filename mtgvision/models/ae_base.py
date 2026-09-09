from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import final

import torch
import torch.nn as nn


class AeBase(nn.Module):
    encoded: torch.Tensor | None = None
    multiscale: bool = False

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError

    def _decode(self, z: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError

    def _init_weights(self, m: nn.Module | None = None) -> None:
        """Initialize weights for convolutional and batch norm layers.

        If `m` is given (e.g. when used as `self.apply(self._init_weights)`),
        only that module is initialized. Otherwise, every submodule is.
        """
        for mod in self.modules() if m is None else [m]:
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

    @final
    def decode(self, z: torch.Tensor, **kwargs: object) -> list[torch.Tensor]:
        # should output more tensors if multiscale
        # * first is always the full scale
        # * second is half the scale
        # * third is quarter the scale, etc.
        return self._decode(z)

    @final
    def encode(self, x: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Input shape: (1, 3, 192, 128) if NCHW, or (1, 192, 128, 3) if NHWC
        # if x.size(1) != 3:
        #     if x.size(3) == 3:
        #         x = x.permute(0, 3, 1, 2)
        # Shape: (1, 3, 192, 128)
        z, multi = self._encode(x)
        self.encoded = z
        return z, multi

    @final
    def forward(self, x: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, list[torch.Tensor]]:
        z, multi = self.encode(x)
        multiout = self.decode(z)
        return z, multiout + multi

    @classmethod
    def create_model(
        cls,
        x_size: tuple[int, int, int, int],
        y_size: tuple[int, int, int, int],
        **kwargs: object,
    ) -> AeBase:
        assert len(x_size) == 4 and len(y_size) == 4
        assert x_size[1:] == (192, 128, 3) and y_size[1:] == (192, 128, 3)
        model = cls(**kwargs)
        return model

    @classmethod
    def quick_test(
        cls,
        batch_size: int = 16,
        n: int = 100,
        model: AeBase | None = None,
        compile: bool = False,
        **model_kwargs: object,
    ) -> None:
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
        runner: Callable[[torch.Tensor], tuple[torch.Tensor, list[torch.Tensor]]] = model
        if compile:
            runner = torch.compile(model)

        # Create dummy input
        dummy_input = torch.randn(batch_size, 192, 128, 3).to(device)

        # Warm-up runs
        with torch.no_grad():
            for _ in range(10):
                runner(dummy_input)

        # Benchmark
        with torch.no_grad():
            for i in tqdm(range(n)):
                z, (output, *_) = runner(dummy_input)

        # Print shapes and bottleneck size
        print(f"Input shape: {dummy_input.shape}")  # (16, 192, 128, 3) NHWC
        print(f"Output shape: {output.shape}")  # (16, 3, 192, 128) NCHW
        print(f"Encoding shape: {z.shape}")  # (16, 32, 4, 4) NCHW
        print(f"Encoding elements per item: {z.numel() // x_size[0]}")

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
