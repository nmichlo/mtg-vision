import argparse
import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from mtgvision.train_encoder import (
    get_test_images,
    MtgVisionEncoder,
    RanMtgEncDecDataset,
)
import coremltools as ct

MODEL_PATH = Path(
    "/Users/nathanmichlo/Downloads/cnvnxt2ae-tiny9x128-ssim5l1-lr0.001-bs32__odcpoea5_265000.ckpt"
)


@functools.lru_cache(maxsize=1)
def _get_data():
    return get_test_images(
        RanMtgEncDecDataset(default_batch_size=1),
        seed=42,
    )


def _export(path: Path, debug: bool = True):
    # LOAD
    print("loading model from", path)
    model: MtgVisionEncoder = MtgVisionEncoder.load_from_checkpoint(path)

    # DEBUG
    if debug:
        for img in _get_data()[:1]:
            plt.imshow(img["x"])
            plt.show()
            plt.imshow(model.forward_img(img["x"]))
            plt.show()

    # CONVERT
    coreml_encoder_path = path.with_suffix(".encoder.mlpackage")
    if model.model.encoder is not None:
        print("Exporting encoder to", coreml_encoder_path)
        encoder = model.model.encoder.to_coreml()
        encoder.save(coreml_encoder_path)

    coreml_decoder_path = path.with_suffix(".decoder.mlpackage")
    if model.model.decoder is not None:
        print("Encoder exported to", coreml_decoder_path)
        decoder = model.model.decoder.to_coreml()
        decoder.save(coreml_decoder_path)

    # DEBUG
    if debug:
        encoder = CoreMlEncoder(coreml_encoder_path)
        decoder = CoreMlDecoder(coreml_decoder_path)
        for img in _get_data()[:1]:
            plt.imshow(img["x"])
            plt.show()
            plt.imshow(decoder.predict(encoder.predict(img["x"])))
            plt.show()


class CoreMlEncoder:
    def __init__(self, model_path: Path):
        self.model = ct.models.MLModel(str(model_path))

    def predict(self, img: np.ndarray):
        assert img.ndim == 3
        assert img.shape[-1] == 3
        img = img.transpose(2, 0, 1)[None, ...]
        z = self.model.predict({"x": img})["z"]
        assert z.ndim == 2
        assert z.shape[0] == 1
        return z[0]

    def ran_forward(self):
        [x] = self.model.input_description._fd_spec
        [_, c, h, w] = x.type.multiArrayType.shape
        return self.predict(np.random.rand(h, w, c))


class CoreMlDecoder:
    def __init__(self, model_path: Path):
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

    # if args.export:
    #     _export(args.path)
    if args.test:
        _test_infer(args.path)


# Load the pytorch lightning checkpoint
# hardcoded export specifically for
if __name__ == "__main__":
    _cli()
