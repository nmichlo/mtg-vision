"""
Export a trained encoder to coreml or other formats.
"""

import argparse
import functools
from pathlib import Path
from typing import Literal, Sequence, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from mtgvision.encoder_train import (
    MtgVisionEncoder,
    RanMtgEncDecDataset,
    get_test_image_batches,
)
from mtgvision.util.image import img_float32

MODEL_DETAILS = {
    "head_type": "conv+linear",
    "path": "/Users/nathanmichlo/Desktop/active/mtg/data/gen/embeddings/encoder_nano_aivb8jvk/checkpoints/epoch=0-step=47500.ckpt",
}
MODEL_PATH = Path(MODEL_DETAILS["path"])


@functools.lru_cache(maxsize=1)
def _get_data():
    return get_test_image_batches(
        RanMtgEncDecDataset(default_batch_size=1),
        seed=42,
    )


def _debug(
    encoder_cls: "Type[_Encoder] | None",
    encoder_path: "Union[str, Path, None]",
    decoder_cls: "Type[_Decoder] | None",
    decoder_path: "Union[str, Path, None]",
    debug: bool = True,
):
    if not debug:
        return
    encoder = encoder_cls(encoder_path) if (encoder_cls and encoder_path) else None
    decoder = decoder_cls(decoder_path) if (decoder_cls and decoder_path) else None
    for img in _get_data()[:1]:
        plt.imshow(img["x"][0])
        plt.show()
        if encoder is not None:
            z = encoder.predict(img["x"][0])
            if decoder is not None:
                x_recon = decoder.predict(z)
                plt.imshow(x_recon)
                plt.show()


def _test_infer(
    enc_cls: "Type[_Encoder] | None", encoder_path: Path, test: bool = True
):
    if not test or not enc_cls:
        return
    encoder = enc_cls(encoder_path)
    encoder.ran_forward()
    for _ in tqdm(range(1000)):
        encoder.ran_forward()


def _export(
    path: Path,
    debug: bool = False,
    test: bool = False,
    # exports
    formats: Sequence[Literal["onnx", "tfjs", "tflite", "coreml"] | None] = (),
):
    # LOAD
    print("loading model from", path)
    model: MtgVisionEncoder = MtgVisionEncoder.load_from_checkpoint(path)
    # Optional loaders for testing
    _enc_dec = {
        "onnx": (OnnxEncoder, OnnxDecoder),
        "coreml": (CoreMlEncoder, CoreMlDecoder),
    }
    # EXPORT
    for fmt in set(formats):
        if fmt is None:
            continue
        enc_path, dec_path = model.model.save(base_path=path, fmt=fmt)
        enc_cls, dec_cls = _enc_dec.get(fmt, (None, None))
        _debug(enc_cls, enc_path, dec_cls, dec_path, debug=debug)
        _test_infer(enc_cls, enc_path, test=test)


class _Encoder:
    def _prepare_input(self, rgb_im: np.ndarray):
        rgb_im = img_float32(rgb_im)  # shape: [H, W, 3]
        assert rgb_im.ndim == 3, f"{rgb_im.shape}"
        assert rgb_im.shape[-1] == 3, f"{rgb_im.shape}"
        assert rgb_im.dtype == np.float32, f"{rgb_im.dtype}"
        rgb_im = rgb_im.transpose(2, 0, 1)[None, ...]
        return rgb_im  # [1, 3, H, W]

    def _prepare_output(self, z: np.ndarray):
        # [1, 768]
        assert z.ndim == 2
        assert z.shape[0] == 1
        return z[0]  # [768]

    def predict(self, rgb_im: np.ndarray):
        x = self._prepare_input(rgb_im)
        z = self._predict(x)
        z = self._prepare_output(z)
        return z

    def _predict(self, x: np.ndarray):
        raise NotImplementedError("Must be implemented in subclass.")

    @property
    def input_hwc(self) -> tuple[int, int, int]:
        raise NotImplementedError

    def ran_forward(self):
        return self.predict(np.random.rand(*self.input_hwc))


class _Decoder:
    def _prepare_input(self, z: np.ndarray):
        assert z.ndim == 1
        return z[None, ...]  # [1, 768]

    def _prepare_output(self, y: np.ndarray):
        # [1, 3, H, W]
        assert y.ndim == 4
        assert y.shape[0] == 1
        y = y[0].transpose(2, 3, 1)
        assert y.ndim == 3
        assert y.shape[-1] == 3
        return y

    def predict(self, z: np.ndarray):
        x = self._prepare_input(z)
        y = self._predict(x)
        y = self._prepare_output(y)
        return y

    def _predict(self, x: np.ndarray):
        raise NotImplementedError("Must be implemented in subclass.")


class OnnxEncoder(_Encoder):
    def __init__(self, model_path: Path = None):
        import onnxruntime as rt

        if model_path is None:
            model_path = MODEL_PATH.with_suffix(".encoder.onnx")
        self.model = rt.InferenceSession(str(model_path))
        [self.input_node] = self.model.get_inputs()

    def _predict(self, x: np.ndarray):
        return self.model.run(None, {self.input_node.name: x})[0]

    @property
    def input_hwc(self) -> tuple[int, int, int]:
        [x] = self.model.get_inputs()
        [_, c, h, w] = x.shape
        return h, w, c


class OnnxDecoder(_Decoder):
    def __init__(self, model_path: Path = None):
        import onnxruntime as rt

        if model_path is None:
            model_path = MODEL_PATH.with_suffix(".decoder.onnx")
        self.model = rt.InferenceSession(str(model_path))
        [self.input_node] = self.model.get_inputs()

    def _predict(self, x: np.ndarray):
        return self.model.run(None, {self.input_node.name: x})[0]


class CoreMlEncoder(_Encoder):
    def __init__(self, model_path: Path = None):
        import coremltools as ct

        if model_path is None:
            model_path = MODEL_PATH.with_suffix(".encoder.mlpackage")
        self.model = ct.models.MLModel(str(model_path))

    def _predict(self, x: np.ndarray):
        return self.model.predict({"x": x})["z"]

    @property
    def input_hwc(self) -> tuple[int, int, int]:
        [x] = self.model.input_description._fd_spec
        [_, c, h, w] = x.type.multiArrayType.shape
        return h, w, c


class CoreMlDecoder(_Decoder):
    def __init__(self, model_path: Path):
        import coremltools as ct

        if model_path is None:
            model_path = MODEL_PATH.with_suffix(".encoder.mlpackage")
        self.model = ct.models.MLModel(str(model_path))

    def _predict(self, z: np.ndarray):
        return self.model.predict({"z": z})["x_hat"]


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=MODEL_PATH, type=Path)
    parser.add_argument(
        "--export", action="append", choices=["onnx", "coreml", "tfjs", "tflite"]
    )
    parser.add_argument("--no-test", dest="test", action="store_false")
    parser.add_argument("--no-debug", dest="debug", action="store_false")
    args = parser.parse_args()

    if not args.export:
        args.export = ["tfjs"]

    print(f"Exporting {args.path} to all of: {args.export}")
    _export(
        args.path,
        debug=args.debug,
        test=args.test,
        formats=args.export,
    )


# Load the pytorch lightning checkpoint
# hardcoded export specifically for
if __name__ == "__main__":
    _cli()
