"""
Embed images into qdrant and query them to validate models are working correctly.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from collections.abc import Hashable, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
from doorway.x import ProxyDownloader
from mtgdata.scryfall import ScryfallCardFace
from tqdm import tqdm

from mtgvision.encoder_datasets import SyntheticBgFgMtgImages
from mtgvision.encoder_export import MODEL_PATH, CoreMlEncoder
from mtgvision.encoder_train import RanMtgEncDecDataset
from mtgvision.qdrant_populate import VectorStoreQdrant
from mtgvision.util.image import imread_float, resize


def _cli(modes: tuple[str, ...] = ("virtual", "crop", "orig")) -> None:
    encoder = CoreMlEncoder(MODEL_PATH.with_suffix(".encoder.mlpackage"))

    dataset = RanMtgEncDecDataset(default_batch_size=1)

    proxy = ProxyDownloader()

    db = VectorStoreQdrant()

    # 2. check accuracy
    def _yield_virtual_points() -> Iterator[
        tuple[list[np.ndarray | None], list[list[float] | None], ScryfallCardFace]
    ]:
        for card in tqdm(dataset.mtg.card_iter(), total=len(dataset.mtg)):
            # get base image
            orig = imread_float(card.download(proxy=proxy))
            DS = SyntheticBgFgMtgImages

            def pred(x: np.ndarray) -> list[float]:
                return encoder.predict(x).tolist()

            # get modes
            im: list[np.ndarray | None] = [None, None, None]
            zs: list[list[float] | None] = [None, None, None]
            if "orig" in modes:
                im_orig = resize(orig, (192, 128))
                im[0] = im_orig
                zs[0] = pred(im_orig)
            if "crop" in modes:
                im_crop = DS.make_cropped(orig, size_hw=dataset.x_size_hw)
                im[1] = im_crop
                zs[1] = pred(im_crop)
            if "virtual" in modes:
                im_virtual = DS.make_virtual(
                    orig,
                    imread_float(dataset.ilsvrc.ran_path()),
                    size_hw=dataset.x_size_hw,
                )
                im[2] = im_virtual
                zs[2] = pred(im_virtual)
            yield im, zs, card

    N = 10000

    @dataclasses.dataclass
    class Stat:
        i = 0
        top_1_correct = 0
        top_5_correct = 0
        name: str = "N/A"
        _t: float = 0

        def update(self, targ: Hashable, nerby: Sequence[Hashable]) -> bool:
            self.i += 1
            correct = False
            if str(targ) == str(nerby[0]):
                self.top_1_correct += 1
                correct = True
            if str(targ) in {str(p) for p in nerby}:
                self.top_5_correct += 1
                correct = True
            return correct

        def print_correct(self) -> None:
            t = time.time()
            if t - self._t > 2:
                self._t = t
                print(
                    f"[{self.name}] top_1: {self.top_1_correct / (self.i + 1) * 100:.2f}%, "
                    f"top_5: {self.top_5_correct / (self.i + 1) * 100:.2f}%"
                )

    virtual = Stat()
    for i, ([imo, imc, imv], [o, c, v], card) in enumerate(
        itertools.islice(_yield_virtual_points(), N)
    ):
        # get matches
        v_match = True
        if o is not None:
            o_near = [p.id for p in db.query_nearby(o, k=5)]
            virtual.update(card.id, o_near)
        if c is not None:
            c_near = [p.id for p in db.query_nearby(c, k=5)]
            virtual.update(card.id, c_near)
        if v is not None:
            v_near = [p.id for p in db.query_nearby(v, k=5)]
            v_match = virtual.update(card.id, v_near)
        # done!
        if not v_match:
            print(card.id, v_near)
            assert imc is not None, (
                "'crop' must be in `modes` to render this debug plot"
            )
            plt.imshow(imc)
            plt.show()
            assert imv is not None
            plt.imshow(imv)
            plt.show()
            best_id = v_near[0]
            assert isinstance(best_id, str), f"expected a str card id, got {best_id!r}"
            plt.imshow(
                dataset.mtg.get_card_by_id(best_id).dl_and_open_im_resized(proxy=proxy)
            )
            plt.show()

    virtual.print_correct()


if __name__ == "__main__":
    _cli()
