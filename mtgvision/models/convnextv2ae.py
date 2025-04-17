import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional, Union

import torch
from torch import nn
from tqdm import tqdm

from mtgvision.models.convnextv2 import Block, LayerNorm, trunc_normal_
from mtgvision.models.ae_base import AeBase


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def Act():
    return nn.Mish(inplace=True)


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


def Norm2d(*args, **kwargs):
    return LayerNorm(*args, **kwargs)


def ConvBlock(**kwargs):
    return Block(norm=Norm2d, act=Act, **kwargs)


class GlobalAveragePooling(nn.Module):
    def forward(self, x):
        # global average pooling, (N, C, H, W) -> (N, C, 1, 1)
        return x.mean([-2, -1])[:, :, None, None]


class Index(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x[self.shape]


class Print(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


class MLP(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, act=Act, act_out: bool = False
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, out_dim),
            act() if act_out else nn.Identity(),
        )

    def forward(self, x):
        return self.layers(x)


class _Branch(nn.Module):
    def __init__(self, branch_a: nn.Module, branch_b: nn.Module):
        super().__init__()
        self.branch_a = branch_a
        self.branch_b = branch_b

    def forward(self, x):
        x_a = self.branch_a(x)
        x_b = self.branch_b(x)
        return self._combine(x_a, x_b)


class BranchSum(_Branch):
    def _combine(self, x_a, x_b):
        return x_a + x_b


class BranchMul(_Branch):
    def _combine(self, x_a, x_b):
        return x_a * x_b  # element-wise multiplication


# ========================================================================= #
# Base Encoder & Decoder                                                    #
# ========================================================================= #


class _Base(nn.Module):
    def __init__(
        self,
        image_wh: tuple[int, int] = (192, 128),
        in_chans: int = 3,
        z_size: int = 1000,
        depths: tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
    ):
        super().__init__()
        self.image_wh = image_wh
        self.in_chans = in_chans
        self.z_size = z_size
        self.depths = depths
        self.dims = dims

        assert len(depths) == len(dims)
        iw, ih = self._get_internal_wh(image_wh)
        self.internal_wh = (iw, ih)
        self.internal_hw = (ih, iw)  # usually (6, 4)
        self.internal_num = iw * ih
        self.internal_chw = (dims[-1], ih, iw)
        self.internal_c = dims[-1]

        assert z_size % self.internal_num == 0

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        return 3, self.image_wh[1], self.image_wh[0]

    @property
    def internal_scale(self) -> int:
        return 4 * 2 * 2 * 2

    def _get_internal_wh(self, in_wh: tuple[int, int]) -> tuple[int, int]:
        assert len(in_wh) == 2
        assert in_wh[0] % self.internal_scale == 0
        assert in_wh[1] % self.internal_scale == 0
        iw = in_wh[0] // self.internal_scale
        ih = in_wh[1] // self.internal_scale
        return iw, ih

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        # else:
        #     print(f"skipping {type(m)}")


# ========================================================================= #
# Exporters                                                                 #
# ========================================================================= #


class Export:
    @staticmethod
    def to_coreml(
        obj: "ConvNeXtV2Encoder | ConvNeXtV2Decoder",
        in_name: str,
        out_name: str,
        **convert_kwargs,
    ):
        from coremltools import convert, TensorType

        model, i = obj.get_trace_items()
        m_jit = torch.jit.trace(func=model, example_inputs=i)
        c = convert(
            m_jit,
            inputs=[TensorType(name=in_name, shape=i.shape)],
            outputs=[TensorType(name=out_name)],
            **convert_kwargs,
        )
        return c  # coremltools.models.MLModel

    @staticmethod
    def to_onnx(
        obj: "ConvNeXtV2Encoder | ConvNeXtV2Decoder",
        **export_kwargs,
    ):
        import torch.onnx
        from torch.onnx import ONNXProgram

        model, i = obj.get_trace_items()
        program: ONNXProgram = torch.onnx.dynamo_export(model, i)
        return program

    @staticmethod
    def to_tflite(
        obj: "ConvNeXtV2Encoder | ConvNeXtV2Decoder",
    ):
        import ai_edge_torch
        from ai_edge_torch.model import TfLiteModel

        model, i = obj.get_trace_items()
        converted: TfLiteModel = ai_edge_torch.convert(model, (i,))
        return converted

    @staticmethod
    def to_keras(
        obj: "ConvNeXtV2Encoder | ConvNeXtV2Decoder",
    ):
        import torch
        import torch.nn.functional as F
        import tensorflow as tf
        import nobuco
        from nobuco import ChannelOrderingStrategy

        @nobuco.converter(
            F.mish,
            channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS,
        )
        def mish_converter(input: torch.Tensor, inplace: bool = False):
            return lambda input, inplace=False: tf.keras.activations.mish(input)

        print("Exporting to Keras")
        model, i = obj.get_trace_items()
        keras = nobuco.pytorch_to_keras(
            model,
            args=[i],
            kwargs=None,
        )
        return keras

    @staticmethod
    def export_tfjs(obj: "ConvNeXtV2Encoder | ConvNeXtV2Decoder", path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)  # Ensure output dir exists

        # 1. Convert to Keras
        keras_model = Export.to_keras(obj)
        if keras_model is None:
            raise RuntimeError("Keras conversion failed")

        # 2. Save Keras model temporarily
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_keras_path = Path(temp_dir) / "keras_model"
            print(f"Saving temporary Keras model to: {temp_keras_path}")
            keras_model.save(str(temp_keras_path))

            # 3. Convert saved Keras model to TFJS
            print(f"Converting Keras model to TFJS at: {path}")
            # Use subprocess to call the converter tool
            try:
                subprocess.run(
                    [
                        "tensorflowjs_converter",
                        "--input_format",
                        "tf_saved_model",  # Input is the saved Keras model
                        "--output_format",
                        "tfjs_graph_model",  # Or tfjs_layers_model
                        str(temp_keras_path),  # Path to saved Keras model
                        str(path),  # Output directory for TFJS files
                    ],
                    check=True,
                    capture_output=True,  # Capture output for better debugging
                    text=True,
                )
                print("TFJS conversion successful.")
            except subprocess.CalledProcessError as e:
                print("--- TFJS Conversion Failed ---")
                print("Command:", e.cmd)
                print("Return Code:", e.returncode)
                print("Stdout:", e.stdout)
                print("Stderr:", e.stderr)
                raise e
            except FileNotFoundError:
                print("ERROR: 'tensorflowjs_converter' command not found.")
                print(
                    "Ensure TensorFlow.js is installed ('pip install tensorflowjs') and the command is in your PATH."
                )
                raise


# ========================================================================= #
# Encoder                                                                   #
# ========================================================================= #


HeadHint = Literal["conv+linear", "conv+mlp", "conv+act+mlp", "pool+linear", "pool+mlp"]


class ConvNeXtV2Encoder(_Base):
    """ConvNeXt V2

    Args:
        in_chans: Number of input image channels. Default: 3
        z_size: Number of classes for classification head. Default: 1000
        depths: Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims: Feature dimension at each stage. Default: [96, 192, 384, 768]
    """

    def __init__(
        self,
        image_wh: tuple[int, int] = (224, 224),
        in_chans: int = 3,
        z_size: int = 1000,
        depths: tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
        head_type: HeadHint = "conv+mlp",
        scale_io: bool = True,
    ):
        super().__init__(
            image_wh=image_wh,
            in_chans=in_chans,
            z_size=z_size,
            depths=depths,
            dims=dims,
        )
        self.head_type = head_type
        self.scale_io = scale_io

        # **DOWNSAMPLE LAYERS**
        # + stem and 3 intermediate downsampling conv layers
        # + 4 feature resolution stages, each consisting of multiple residual blocks
        # --> Bx3x192x128
        self.block0 = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            Norm2d(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Sequential(*[ConvBlock(dim=dims[0]) for _ in range(depths[0])]),
        )
        # --> Bx[0]x48x32
        self.block1 = nn.Sequential(
            Norm2d(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
            nn.Sequential(*[ConvBlock(dim=dims[1]) for _ in range(depths[1])]),
        )
        # --> Bx[1]x24x16
        self.block2 = nn.Sequential(
            Norm2d(dims[1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2),
            nn.Sequential(*[ConvBlock(dim=dims[2]) for _ in range(depths[2])]),
        )
        # --> Bx[2]x12x8
        self.block3 = nn.Sequential(
            Norm2d(dims[2], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2),
            nn.Sequential(*[ConvBlock(dim=dims[3]) for _ in range(depths[3])]),
        )

        # **HEAD**
        # --> Bx[3]x6x4
        if head_type in ("conv+linear", "conv+mlp", "conv+act+mlp"):
            self.pool = nn.Sequential(
                nn.Conv2d(
                    dims[3], z_size // self.internal_num, kernel_size=1, stride=1
                ),
                # --> Bx<z_size//(6*4)>x6x4
                Act() if "+act" in head_type else nn.Identity(),
                Norm2d(
                    z_size // self.internal_num, eps=1e-6, data_format="channels_first"
                ),
                Reshape((-1, z_size)),
            )
            # --> Bx<z_size>
            if "+mlp" in head_type:
                self.head = MLP(z_size, z_size, z_size)
            else:
                self.head = nn.Linear(z_size, z_size)
        elif head_type in ("pool+linear", "pool+mlp"):
            self.pool = nn.Sequential(
                GlobalAveragePooling(),
                # --> Bx[3]x1x1
                Norm2d(dims[3], eps=1e-6, data_format="channels_first"),
                Reshape((-1, dims[3])),
                # --> Bx[3]
            )
            # --> Bx[3]
            if "+mlp" in head_type:
                self.head = MLP(dims[3], z_size, z_size)
            else:
                self.head = nn.Linear(dims[3], z_size)
        else:
            raise KeyError(f"head_type={head_type} not recognized")
        # --> Bx<z_size>

        # initialize weights
        self.apply(self._init_weights)

    def forward(self, x):
        if self.scale_io:
            x = (x * 2) - 1
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.head(x)
        x = x.reshape(x.size(0), self.z_size)
        return x

    def get_trace_items(self):
        i = torch.randn((1, *self.tensor_shape)).to("cpu")
        model = self.to("cpu").eval()
        return model, i

    def to_coreml(self, **convert_kwargs):
        return Export.to_coreml(self, "x", "z")


# ========================================================================= #
# Decoder                                                                   #
# ========================================================================= #


class ConvNeXtV2Decoder(_Base):
    """ConvNeXt V2

    Args:
        in_chans: Number of input image channels. Default: 3
        z_size: Number of classes for classification head. Default: 1000
        depths: Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims: Feature dimension at each stage. Default: [96, 192, 384, 768]
    """

    def __init__(
        self,
        image_wh: tuple[int, int] = (224, 224),
        in_chans: int = 3,
        z_size: int = 768,
        depths: tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
        head_type: HeadHint = "conv+mlp",
        scale_io: bool = True,
    ):
        super().__init__(
            image_wh=image_wh,
            in_chans=in_chans,
            z_size=z_size,
            depths=depths,
            dims=dims,
        )
        self.head_type = head_type
        self.scale_io = scale_io

        # **HEAD**
        # --> Bx<z_size>
        if head_type in ("conv+linear", "conv+mlp", "conv+act+mlp"):
            if "+mlp" in head_type:
                self.unhead = MLP(z_size, z_size, z_size)
            else:
                self.unhead = nn.Linear(z_size, z_size)
            # --> Bx<z_size>
            self.unpool = nn.Sequential(
                Reshape((-1, z_size // self.internal_num, *self.internal_hw)),
                # --> Bx<z_size//(6*4)>x6x4
                Norm2d(
                    z_size // self.internal_num, eps=1e-6, data_format="channels_first"
                ),
                Act() if "+act" in head_type else nn.Identity(),
                nn.ConvTranspose2d(
                    z_size // self.internal_num, dims[-1], kernel_size=1, stride=1
                ),
            )
        elif head_type in ("pool+linear", "pool+mlp"):
            if "+mlp" in head_type:
                self.unhead = MLP(z_size, z_size, dims[-1])
            else:
                self.unhead = nn.Linear(z_size, dims[-1])
            # --> Bx[3]
            self.unpool = nn.Sequential(
                Index((slice(None), slice(None), None, None)),  # arr[:, :, None, None]
                # --> Bx[3]x1x1
                Norm2d(
                    dims[-1], eps=1e-6, data_format="channels_first"
                ),  # extra, not in encoder
                # Act() if "+act" in head_type else nn.Identity(),
                nn.ConvTranspose2d(
                    dims[-1], dims[-1], kernel_size=self.internal_hw, stride=1
                ),
            )
        else:
            raise KeyError(f"head_type={head_type} not recognized")
        # --> Bx[3]x6x4

        # **UPSAMPLE LAYERS**
        # + stem and 3 intermediate upsampling conv layers
        # + 4 feature resolution stages, each consisting of multiple residual blocks
        # --> Bx[3]x6x4
        self.block3 = nn.Sequential(
            nn.Sequential(*[ConvBlock(dim=dims[-1]) for _ in range(depths[3])]),
            nn.ConvTranspose2d(dims[-1], dims[-2], kernel_size=2, stride=2),
            Norm2d(dims[-2], eps=1e-6, data_format="channels_first"),
        )
        # --> Bx[2]x12x8
        self.block2 = nn.Sequential(
            nn.Sequential(*[ConvBlock(dim=dims[-2]) for _ in range(depths[2])]),
            nn.ConvTranspose2d(dims[-2], dims[-3], kernel_size=2, stride=2),
            Norm2d(dims[-3], eps=1e-6, data_format="channels_first"),
        )
        # --> Bx[1]x24x16
        self.block1 = nn.Sequential(
            nn.Sequential(*[ConvBlock(dim=dims[-3]) for _ in range(depths[1])]),
            nn.ConvTranspose2d(dims[-3], dims[-4], kernel_size=2, stride=2),
            Norm2d(dims[-4], eps=1e-6, data_format="channels_first"),
        )
        # --> Bx[0]x48x32
        self.block0 = nn.Sequential(
            nn.Sequential(*[ConvBlock(dim=dims[-4]) for _ in range(depths[0])]),
            Norm2d(dims[-4], eps=1e-6, data_format="channels_first"),
            nn.ConvTranspose2d(dims[-4], in_chans, kernel_size=4, stride=4),
        )
        # --> Bx3x192x128

        # initialize weights
        self.apply(self._init_weights)

    def forward(self, x):
        assert x.ndim == 2
        x = self.unhead(x)
        x = self.unpool(x)
        x = self.block3(x)
        x = self.block2(x)
        x = self.block1(x)
        x = self.block0(x)
        if self.scale_io:
            x = (x + 1) / 2
        return x

    def get_trace_items(self):
        i = torch.randn((1, self.z_size)).to("cpu")
        model = self.to("cpu").eval()
        return model, i

    def to_coreml(self, **convert_kwargs):
        return Export.to_coreml(self, "z", "x_hat")


# ========================================================================= #
# AE                                                                        #
# ========================================================================= #


class ConvNeXtV2Ae(_Base, AeBase):
    encoder: Optional[ConvNeXtV2Encoder]
    decoder: Optional[ConvNeXtV2Decoder]

    def __init__(
        self,
        image_wh: tuple[int, int] = (224, 224),
        in_chans: int = 3,
        z_size: int = 768,
        depths: tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
        head_type: HeadHint = "conv+mlp",
        encoder_enabled: bool = True,
        decoder_enabled: bool = True,
        scale_io: bool = True,
    ):
        super().__init__(
            image_wh=image_wh,
            in_chans=in_chans,
            z_size=z_size,
            depths=depths,
            dims=dims,
        )

        if encoder_enabled:
            self.encoder = ConvNeXtV2Encoder(
                image_wh=image_wh,
                in_chans=in_chans,
                z_size=z_size,
                depths=depths,
                dims=dims,
                head_type=head_type,
                scale_io=scale_io,
            )
        else:
            self.encoder = None

        if decoder_enabled:
            self.decoder = ConvNeXtV2Decoder(
                image_wh=image_wh,
                in_chans=in_chans,
                z_size=z_size,
                depths=depths,
                dims=dims,
                scale_io=scale_io,
            )
        else:
            self.decoder = None

    def _encode(self, x) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.encoder is None:
            raise RuntimeError(f"encoder is not enabled on: {self.__class__.__name__}")
        return self.encoder(x), []

    def _decode(self, z) -> list[torch.Tensor]:
        if self.decoder is None:
            raise RuntimeError(f"decoder is not enabled on: {self.__class__.__name__}")
        return [self.decoder(z)]

    def save(
        self,
        base_path: Union[str, Path],
        fmt: Literal["coreml", "onnx", "tflite", "tfjs"],
    ) -> tuple[Optional[Path], Optional[Path]]:
        # get paths
        _exporters = {
            "coreml": (lambda m, p: m.to_coreml().save(p), "mlpackage"),
            "onnx": (lambda m, p: Export.to_onnx(m).save(p), "onnx"),
            "tflite": (lambda m, p: Export.to_tflite(m).export(p), "tflite"),
            "tfjs": (lambda m, p: Export.export_tfjs(m, p), "web_model"),
            # TODO: https://github.com/PINTO0309/onnx2tf
            # TODO: https://github.com/AlexanderLutsenko/nobuco
        }
        export_fn, suffix = _exporters[fmt]
        # export encoder
        encoder_path = None
        if self.encoder is not None:
            encoder_path = base_path.with_suffix(f".encoder.{suffix}")
            export_fn(self.encoder, str(encoder_path))
            print(f"Encoder exported to {encoder_path}")
        # export decoder
        decoder_path = None
        if self.decoder is not None:
            decoder_path = base_path.with_suffix(f".decoder.{suffix}")
            export_fn(self.decoder, str(decoder_path))
            print(f"Decoder exported to {decoder_path}")
        # done!
        return encoder_path, decoder_path


# ========================================================================= #
# Factory                                                                   #
# ========================================================================= #


def convnextv2_atto(**kwargs):
    model = ConvNeXtV2Ae(depths=(2, 2, 6, 2), dims=(40, 80, 160, 320), **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = ConvNeXtV2Ae(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384), **kwargs)
    return model


def convnextv2ae_pico(**kwargs):
    model = ConvNeXtV2Ae(depths=(2, 2, 6, 2), dims=(64, 128, 256, 512), **kwargs)
    return model


def convnextv2ae_nano(**kwargs):
    model = ConvNeXtV2Ae(depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), **kwargs)
    return model


def convnextv2ae_tiny(**kwargs):
    model = ConvNeXtV2Ae(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    return model


def convnextv2ae_tiny_9_128(**kwargs):
    model = ConvNeXtV2Ae(depths=(3, 3, 9, 3), dims=(128, 256, 384, 768), **kwargs)
    return model


def convnextv2ae_tiny_12_128(**kwargs):
    model = ConvNeXtV2Ae(depths=(3, 3, 12, 3), dims=(128, 256, 384, 768), **kwargs)
    return model


def convnextv2ae_base_9(**kwargs):
    model = ConvNeXtV2Ae(depths=(3, 3, 9, 3), dims=(128, 256, 512, 1024), **kwargs)
    return model


def convnextv2ae_base_12(**kwargs):
    model = ConvNeXtV2Ae(depths=(3, 3, 12, 3), dims=(128, 256, 512, 1024), **kwargs)
    return model


def convnextv2ae_base(**kwargs):
    model = ConvNeXtV2Ae(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024), **kwargs)
    return model


def convnextv2ae_large(**kwargs):
    model = ConvNeXtV2Ae(depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536), **kwargs)
    return model


def convnextv2ae_huge(**kwargs):
    model = ConvNeXtV2Ae(depths=(3, 3, 27, 3), dims=(352, 704, 1408, 2816), **kwargs)
    return model


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == "__main__":
    for make_fn in [
        # convnextv2_atto,  # ~52it/s
        # convnextv2_femto,  # ~52it/s
        # convnextv2ae_pico,  # ~52it/s
        # convnextv2ae_nano,  # ~47it/s starts slowing on laptop,
        convnextv2ae_tiny,  # ~33.5it/s
        convnextv2ae_tiny_9_128,
        convnextv2ae_tiny_12_128,
        convnextv2ae_base_9,  # ~34it/s
        convnextv2ae_base_12,  # ~34it/s
        convnextv2ae_base,  # ~21.53it/s
        # convnextv2ae_large,
        # convnextv2ae_huge,
    ]:
        for head_type in [
            # "conv+linear",
            "conv+mlp",
            # "conv+act+mlp",
            # "pool+linear",
            # "pool+mlp"
        ]:
            print(f"\n{make_fn.__name__}:{head_type}")

            ae = make_fn(head_type=head_type, image_wh=(128, 192), z_size=768)

            params_enc = sum(p.numel() for p in ae.encoder.parameters())
            params_dec = sum(p.numel() for p in ae.decoder.parameters())
            params_ae = sum(p.numel() for p in ae.parameters())

            print(f"params_ae: {params_ae} ({params_ae / 1_000_000:.3f}M)")
            print(f"params_enc: {params_enc} ({params_enc / 1_000_000:.3f}M)")
            print(f"params_dec: {params_dec} ({params_dec / 1_000_000:.3f}M)")

            x = torch.randn(1, 3, *ae.image_wh[::])
            z = torch.randn(1, ae.z_size)

            # to mps
            ae = ae.to(torch.device("mps"))
            x = x.to(torch.device("mps"))
            z = z.to(torch.device("mps"))

            # check
            # ae(x)
            # ae.encoder(x)
            # ae.decoder(z)

            def _repeat_secs(fn, sec=3):
                with tqdm() as pbar:
                    start_t = last_t = time.time()
                    while True:
                        t = time.time()
                        _delta, last_t = t - last_t, t
                        pbar.update()
                        if t - start_t > sec:
                            break
                        fn()

            _repeat_secs(lambda: ae(x))
            _repeat_secs(lambda: ae.encoder(x))
            _repeat_secs(lambda: ae.decoder(z))
