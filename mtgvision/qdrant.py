import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

import qdrant_client
from qdrant_client.http.models import ScoredPoint, VectorParams, Distance


@dataclass
class QdrantPoint:
    id: str  # UUID
    vector: list[float] | None = None
    payload: dict[str, Any] | None = None


class VectorStoreQdrant:
    _COLLECTION = "mtg"
    _VECTOR_SIZE: int = 768

    def __init__(self, location: str = "localhost:6333"):
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.client = qdrant_client.QdrantClient(
            location=location,
        )
        if not self.client.collection_exists(self._COLLECTION):
            self.client.create_collection(
                self._COLLECTION,
                vectors_config=VectorParams(
                    size=self._VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def drop_collection(self):
        self.client.delete_collection(self._COLLECTION)

    def scroll(
        self,
        *,
        batch_size: int = 1000,
        with_vectors: bool = False,
        with_payload: bool = True,
        offset: str | None = None,
    ):
        while True:
            [results, offset] = self.client.scroll(
                collection_name=self._COLLECTION,
                limit=batch_size,
                offset=offset,
                with_vectors=with_vectors,
                with_payload=with_payload,
            )
            if not results or offset is None:
                break
            for point in results:
                yield QdrantPoint(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload,
                )

    def retrieve(
        self,
        ids: Iterable[str],
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[QdrantPoint]:
        results = self.client.retrieve(
            collection_name=self._COLLECTION,
            ids=list(ids),
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return [
            QdrantPoint(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
            )
            for point in results
        ]

    def save_points(self, iter_points: Iterable[QdrantPoint]):
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

    def query_nearby(
        self,
        vector: list[float],
        k: int,
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: float = None,
    ) -> list[ScoredPoint]:
        from qdrant_client.http.models import QueryResponse

        results: QueryResponse = self.client.query_points(
            collection_name=self._COLLECTION,
            query=vector,
            limit=k,
            with_vectors=with_vectors,
            with_payload=with_payload,
            score_threshold=score_threshold,
        )
        return results.points

    def update_payload(
        self,
        id_: str,
        payload: dict[str, Any],
    ) -> QdrantPoint:
        self.client.overwrite_payload(
            collection_name=self._COLLECTION,
            points=[id_],
            payload=payload,
        )
        return QdrantPoint(
            id=id_,
            vector=None,
            payload=deepcopy(payload),
        )


if __name__ == "__main__":
    # ds = SyntheticBgFgMtgImages()

    db = VectorStoreQdrant()
    for i in [
        # "391a5fee-39e6-4192-93ab-134e7efe3990",
        # "391a5fee-39e6-4192-93ab-134e7efe3990",
        # "ce9ed217-8378-4a58-a00d-fa4e4cb27c9d",
        # "14de01ae-a52e-4530-9fe3-9888a8480fc8",
        # BAD DATA
        "000225fc-9bc3-4eb3-905e-02c19c873b0b",
        "007a6422-20b7-40d0-aed1-99eb7482556a",
        "00bbc009-ef6b-4f16-b737-086b7348e05e",
        "01498551-4c5d-42b8-9283-73244c680407",
        "01672157-7cf5-4bc2-90ba-080842625ea7",
    ]:
        # plt.imshow(ds.get_image_by_id(i))
        # plt.imshow(ds.make_cropped(ds.get_image_by_id(i)))
        # plt.show()

        [item] = db.retrieve([i], with_payload=True, with_vectors=True)

        for item in db.query_nearby(
            item.vector,
            k=3000,
            with_payload=False,
            with_vectors=False,
            score_threshold=0.1,
        ):
            print("-", item)
        break
