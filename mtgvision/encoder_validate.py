"""
Embed images into qdrant and query them to validate models are working correctly.
"""

import dataclasses
import itertools
import time
from typing import Any, Iterable
from tqdm import tqdm

from doorway.x import ProxyDownloader
from mtgvision.encoder_datasets import SyntheticBgFgMtgImages
from mtgvision.encoder_export import CoreMlEncoder, MODEL_PATH

from mtgvision.encoder_train import RanMtgEncDecDataset
from mtgvision.util.image import imread_float


def _dump_card(card):
    data = dataclasses.asdict(card)
    for k in ["oracle_id", "id"]:
        data[k] = str(data[k])
    for k in ["_img_type", "_bulk_type", "_sets_dir", "_proxy"]:
        data.pop(k, None)
    return data


@dataclasses.dataclass
class Point:
    id: str  # UUID
    vector: list[float]
    payload: dict[str, Any]


class VectorStoreBase:
    _VECTOR_SIZE: int = 768

    def save_points(self, iter_points: Iterable[Point]):
        raise NotImplementedError

    def query(self, vector: list[float], k: int) -> list[Point]:
        raise NotImplementedError

    def query_by_id(self, id_: str) -> Point:
        raise NotImplementedError


class VectorStoreQdrant(VectorStoreBase):
    _COLLECTION = "mtg"

    def __init__(self):
        import qdrant_client
        from qdrant_client.http.models import VectorParams, Distance

        self.client = qdrant_client.QdrantClient(location=":memory:")
        if self.client.collection_exists(self._COLLECTION):
            self.client.delete_collection(self._COLLECTION)
        self.client.create_collection(
            self._COLLECTION,
            vectors_config=VectorParams(
                size=self._VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

    def save_points(self, iter_points: Iterable[Point]):
        from qdrant_client.http.models import PointStruct

        self.client.upload_points(
            collection_name=self._COLLECTION,
            points=(
                PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload,
                )
                for point in iter_points
            ),
            batch_size=64,
        )

    def query(self, vector: list[float], k: int) -> list[Point]:
        from qdrant_client.http.models import QueryResponse

        results: QueryResponse = self.client.query_points(
            collection_name=self._COLLECTION,
            query=vector,
            limit=k,
        )
        return [
            Point(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
            )
            for point in results.points
        ]


def _cli():
    encoder = CoreMlEncoder(MODEL_PATH.with_suffix(".encoder.mlpackage"))

    dataset = RanMtgEncDecDataset(default_batch_size=1)

    proxy = ProxyDownloader()

    # db = VectorStoreQdrant()
    db = VectorStoreQdrant()

    # 1. create dataset of embeddings
    def _yield_gt_points():
        for card in tqdm(dataset.mtg.card_iter(), total=len(dataset.mtg)):
            card.download(proxy=proxy)
            x = SyntheticBgFgMtgImages.make_cropped(
                imread_float(card.img_path),
                size_hw=dataset.x_size_hw,
            )
            z = encoder.predict(x).tolist()
            yield z, card

    # 2. check accuracy
    def _yield_virtual_points():
        for card in tqdm(dataset.mtg.card_iter(), total=len(dataset.mtg)):
            card.download(proxy=proxy)
            x = SyntheticBgFgMtgImages.make_virtual(
                imread_float(card.img_path),
                imread_float(dataset.ilsvrc.ran_path()),
                size_hw=dataset.x_size_hw,
            )
            z = encoder.predict(x).tolist()
            yield z, card

    N = 10000

    # actually generate and query
    db.save_points(
        Point(id=str(card.id), vector=z, payload=_dump_card(card))
        for z, card in itertools.islice(_yield_gt_points(), N)
    )

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
        results = db.query(z, k=5)
        if str(card.id) == results[0].id:
            top_1_correct += 1
        if str(card.id) in {point.id for point in results}:
            top_5_correct += 1
        if time.time() - t > 2:
            t = time.time()
            _print_correct()
    _print_correct()


if __name__ == "__main__":
    _cli()
