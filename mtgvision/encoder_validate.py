"""
Embed images into qdrant and query them to validate models are working correctly.
"""

import itertools
import time

import matplotlib.pyplot as plt
from tqdm import tqdm

from doorway.x import ProxyDownloader
from mtgvision.encoder_datasets import SyntheticBgFgMtgImages
from mtgvision.encoder_export import CoreMlEncoder, MODEL_PATH

from mtgvision.encoder_train import RanMtgEncDecDataset
from mtgvision.qdrant_populate import VectorStoreQdrant
from mtgvision.util.image import imread_float


def _cli(virtual: bool = False):
    encoder = CoreMlEncoder(MODEL_PATH.with_suffix(".encoder.mlpackage"))

    dataset = RanMtgEncDecDataset(default_batch_size=1)

    proxy = ProxyDownloader()

    db = VectorStoreQdrant()

    # 2. check accuracy
    def _yield_virtual_points():
        for card in tqdm(dataset.mtg.card_iter(), total=len(dataset.mtg)):
            card.download(proxy=proxy)
            orig = imread_float(card.img_path)
            if not virtual:
                x = SyntheticBgFgMtgImages.make_cropped(orig, size_hw=dataset.x_size_hw)
            else:
                bg = imread_float(dataset.ilsvrc.ran_path())
                x = SyntheticBgFgMtgImages.make_virtual(
                    orig, bg, size_hw=dataset.x_size_hw
                )
            z = encoder.predict(x).tolist()
            yield z, card

    N = 10000

    # actually query db
    i = 0
    top_1_correct = 0
    top_5_correct = 0

    def _print_correct():
        print(
            f"top_1: {top_1_correct / (i + 1) * 100:.2f}%, top_5: {top_5_correct / (i + 1) * 100:.2f}%"
        )

    t = time.time()
    for i, (z, card) in enumerate(itertools.islice(_yield_virtual_points(), N)):
        results = db.query_nearby(z, k=5)
        if str(card.id) == results[0].id:
            top_1_correct += 1
        if str(card.id) in {point.id for point in results}:
            top_5_correct += 1
        else:
            print(card)
            plt.imshow(imread_float(card.img_path))
            plt.show()
        if time.time() - t > 2:
            t = time.time()
            _print_correct()
    _print_correct()


if __name__ == "__main__":
    _cli()
