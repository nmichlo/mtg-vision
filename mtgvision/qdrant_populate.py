from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
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

    def __init__(self, location: str = "localhost:6333"):
        import qdrant_client
        from qdrant_client.http.models import VectorParams, Distance

        self.client = qdrant_client.QdrantClient(location=location)
        if self.client.collection_exists(self._COLLECTION):
            self.client.delete_collection(self._COLLECTION)
        self.client.create_collection(
            self._COLLECTION,
            vectors_config=VectorParams(
                size=self._VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

    def drop_collection(self):
        self.client.delete_collection(self._COLLECTION)

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
    from mtgvision.encoder_export import CoreMlEncoder
    from mtgvision.encoder_export import MODEL_PATH
    from mtgvision.encoder_train import RanMtgEncDecDataset
    from doorway.x import ProxyDownloader
    from mtgvision.encoder_datasets import SyntheticBgFgMtgImages
    from tqdm import tqdm
    from mtgvision.util.image import imread_float

    encoder = CoreMlEncoder(MODEL_PATH.with_suffix(".encoder.mlpackage"))
    dataset = RanMtgEncDecDataset(default_batch_size=1)
    proxy = ProxyDownloader()
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

    # actually generate and query
    db.save_points(
        Point(id=str(card.id), vector=z, payload=None) for z, card in _yield_gt_points()
    )


if __name__ == "__main__":
    _cli()
