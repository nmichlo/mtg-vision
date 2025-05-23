"""
Export a trained encoder to coreml or other formats.
"""

import argparse
import functools
import warnings
from pathlib import Path

import coremltools as ct
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


def _export(path: Path, debug: bool = True):
    # LOAD
    print("loading model from", path)
    model: MtgVisionEncoder = MtgVisionEncoder.load_from_checkpoint(path)
    has_decoder = model.hparams.loss_recon is not None

    # DEBUG
    if debug:
        if has_decoder:
            for img in _get_data():
                plt.imshow(img["x"][0])
                plt.show()
                plt.imshow(model.forward_img(img["x"][0]))
                plt.show()
        else:
            warnings.warn("Model does not have decoder, skipping debug images.")

    # CONVERT
    coreml_encoder_path = path.with_suffix(".encoder.mlpackage")
    if model.model.encoder is not None:
        print("Exporting encoder to", coreml_encoder_path)
        encoder = model.model.encoder.to_coreml()
        encoder.save(coreml_encoder_path)

    decoder = None
    if has_decoder:
        coreml_decoder_path = path.with_suffix(".decoder.mlpackage")
        if model.model.decoder is not None:
            print("Encoder exported to", coreml_decoder_path)
            decoder = model.model.decoder.to_coreml()
            decoder.save(coreml_decoder_path)

    # DEBUG
    if debug:
        encoder = CoreMlEncoder(coreml_encoder_path)
        if has_decoder:
            decoder = CoreMlDecoder(coreml_decoder_path)
        for img in _get_data()[:1]:
            plt.imshow(img["x"][0])
            plt.show()
            z = encoder.predict(img["x"][0])
            if has_decoder:
                x_recon = decoder.predict(z)
                plt.imshow(x_recon)
                plt.show()


class CoreMlEncoder:
    def __init__(self, model_path: Path = None):
        if model_path is None:
            model_path = MODEL_PATH.with_suffix(".encoder.mlpackage")
        self.model = ct.models.MLModel(str(model_path))

    def predict(self, rgb_im: np.ndarray):
        rgb_im = img_float32(rgb_im)
        assert rgb_im.ndim == 3, f"{rgb_im.shape}"
        assert rgb_im.shape[-1] == 3, f"{rgb_im.shape}"
        assert rgb_im.dtype == np.float32, f"{rgb_im.dtype}"
        rgb_im = rgb_im.transpose(2, 0, 1)[None, ...]
        # print(img.dtype, img.shape)
        z = self.model.predict({"x": np.array(rgb_im)})["z"]
        assert z.ndim == 2
        assert z.shape[0] == 1
        return z[0]

    @property
    def input_hwc(self) -> tuple[int, int, int]:
        [x] = self.model.input_description._fd_spec
        [_, c, h, w] = x.type.multiArrayType.shape
        return h, w, c

    def ran_forward(self):
        return self.predict(np.random.rand(*self.input_hwc))


class CoreMlDecoder:
    def __init__(self, model_path: Path):
        if model_path is None:
            model_path = MODEL_PATH.with_suffix(".encoder.mlpackage")
        self.model = ct.models.MLModel(str(model_path))

    def predict(self, z: np.ndarray):
        assert z.ndim == 1
        y = self.model.predict({"z": z[None, ...]})["x_hat"]
        assert y.ndim == 4
        assert y.shape[0] == 1
        y = y[0].transpose(1, 2, 0)
        assert y.ndim == 3
        assert y.shape[-1] == 3
        return y


def _test_infer(path: Path):
    model = CoreMlEncoder(path.with_suffix(".encoder.mlpackage"))
    model.ran_forward()
    for _ in tqdm(range(1000)):
        model.ran_forward()


def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=MODEL_PATH, type=Path)
    parser.add_argument("--no-export", dest="export", action="store_false")
    parser.add_argument("--no-test", dest="test", action="store_false")
    args = parser.parse_args()

    if args.export:
        _export(args.path)
    if args.test:
        _test_infer(args.path)


# Load the pytorch lightning checkpoint
# hardcoded export specifically for
if __name__ == "__main__":
    _cli()
