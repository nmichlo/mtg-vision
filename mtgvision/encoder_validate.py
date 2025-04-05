"""
Embed images into qdrant and query them to validate models are working correctly.
"""

import dataclasses
import itertools
import time
from typing import Hashable, Sequence

import matplotlib.pyplot as plt
from tqdm import tqdm

from doorway.x import ProxyDownloader
from mtgvision.encoder_datasets import SyntheticBgFgMtgImages
from mtgvision.encoder_export import CoreMlEncoder, MODEL_PATH

from mtgvision.encoder_train import RanMtgEncDecDataset
from mtgvision.qdrant_populate import VectorStoreQdrant
from mtgvision.util.image import imread_float, resize


def _cli(modes: tuple[str, ...] = ("virtual", "crop", "orig")):
    encoder = CoreMlEncoder(MODEL_PATH.with_suffix(".encoder.mlpackage"))

    dataset = RanMtgEncDecDataset(default_batch_size=1)

    proxy = ProxyDownloader()

    db = VectorStoreQdrant()

    # 2. check accuracy
    def _yield_virtual_points():
        for card in tqdm(dataset.mtg.card_iter(), total=len(dataset.mtg)):
            # get base image
            orig = imread_float(card.download(proxy=proxy))
            DS = SyntheticBgFgMtgImages
            pred = lambda x: encoder.predict(x).tolist()
            # get modes
            im = [None, None, None]
            zs = [None, None, None]
            if "orig" in modes:
                im[0] = resize(orig, (192, 128))
                zs[0] = pred(im[0])
            if "crop" in modes:
                im[1] = DS.make_cropped(orig, size_hw=dataset.x_size_hw)
                zs[1] = pred(im[1])
            if "virtual" in modes:
                im[2] = DS.make_virtual(
                    orig,
                    imread_float(dataset.ilsvrc.ran_path()),
                    size_hw=dataset.x_size_hw,
                )
                zs[2] = pred(im[2])
            yield im, zs, card

    N = 10000

    @dataclasses.dataclass
    class Stat:
        i = 0
        top_1_correct = 0
        top_5_correct = 0
        name: str = "N/A"
        _t: float = 0

        def update(self, targ: Hashable, nerby: Sequence[Hashable]):
            self.i += 1
            correct = False
            if str(targ) == str(nerby[0]):
                self.top_1_correct += 1
                correct = True
            if str(targ) in {str(p) for p in nerby}:
                self.top_5_correct += 1
                correct = True
            return correct

        def print_correct():
            t = time.time()
            if t - self._t > 2:
                self._t = t
                print(
                    f"[{name}] top_1: {top_1_correct / (i + 1) * 100:.2f}%, top_5: {top_5_correct / (i + 1) * 100:.2f}%"
                )

    virtual = Stat()
    for i, ([imo, imc, imv], [o, c, v], card) in enumerate(
        itertools.islice(_yield_virtual_points(), N)
    ):
        # get matches
        o_match, c_match, v_match = True, True, True
        if o is not None:
            o_near = [p.id for p in db.query_nearby(o, k=5)]
            o_match = virtual.update(card.id, o_near)
        if c is not None:
            c_near = [p.id for p in db.query_nearby(c, k=5)]
            c_match = virtual.update(card.id, c_near)
        if v is not None:
            v_near = [p.id for p in db.query_nearby(v, k=5)]
            v_match = virtual.update(card.id, v_near)
        # done!
        if not v_match:
            print(card.id, v_near)
            plt.imshow(imc)
            plt.show()
            plt.imshow(imv)
            plt.show()
            plt.imshow(
                dataset.mtg.get_card_by_id(v_near[0]).dl_and_open_im_resized(
                    proxy=proxy
                )
            )
            plt.show()

    _print_correct()


if __name__ == "__main__":
    _cli()
