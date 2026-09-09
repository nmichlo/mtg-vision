from __future__ import annotations

import logging
from collections.abc import Iterable
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass

import qdrant_client
from qdrant_client.conversions.common_types import PointId
from qdrant_client.http.models import Distance
from qdrant_client.http.models import ScoredPoint
from qdrant_client.http.models import VectorParams
from typing_extensions import TypeIs

from mtgvision.util.json import Json


def _is_float_list(vector: object) -> TypeIs[list[float]]:
    return isinstance(vector, list) and all(isinstance(v, float) for v in vector)


def _as_flat_vector(vector: object) -> list[float] | None:
    if _is_float_list(vector):
        return vector
    return None


@dataclass
class QdrantPoint:
    id: str  # UUID
    vector: list[float] | None = None
    payload: dict[str, Json] | None = None


class VectorStoreQdrant:
    _COLLECTION = "mtg"
    _VECTOR_SIZE: int = 768

    def __init__(self, location: str = "localhost:6333") -> None:
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

    def drop_collection(self) -> None:
        self.client.delete_collection(self._COLLECTION)

    def count(self) -> int:
        return self.client.count(
            collection_name=self._COLLECTION,
        ).count

    def scroll_batches(
        self,
        *,
        batch_size: int = 1000,
        with_vectors: bool = False,
        with_payload: bool = True,
        offset: str | None = None,
    ) -> Iterator[list[QdrantPoint]]:
        cursor: PointId | None = offset
        while True:
            [results, cursor] = self.client.scroll(
                collection_name=self._COLLECTION,
                limit=batch_size,
                offset=cursor,
                with_vectors=with_vectors,
                with_payload=with_payload,
            )
            if not results or cursor is None:
                break
            yield [
                QdrantPoint(
                    id=str(point.id),
                    vector=_as_flat_vector(point.vector),
                    payload=point.payload,
                )
                for point in results
            ]

    def scroll(
        self,
        *,
        batch_size: int = 1000,
        with_vectors: bool = False,
        with_payload: bool = True,
        offset: str | None = None,
    ) -> Iterator[QdrantPoint]:
        for batch in self.scroll_batches(
            batch_size=batch_size,
            with_vectors=with_vectors,
            with_payload=with_payload,
            offset=offset,
        ):
            yield from batch

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
                id=str(point.id),
                vector=_as_flat_vector(point.vector),
                payload=point.payload,
            )
            for point in results
        ]

    def save_points(self, iter_points: Iterable[QdrantPoint]) -> None:
        from qdrant_client.http.models import PointStruct

        def _to_structs() -> Iterator[PointStruct]:
            for point in iter_points:
                if point.vector is None:
                    raise ValueError(f"point {point.id} has no vector to save")
                yield PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=point.payload,
                )

        self.client.upload_points(
            collection_name=self._COLLECTION,
            points=_to_structs(),
            batch_size=64,
        )

    def query_nearby(
        self,
        vector: list[float],
        k: int,
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: float | None = None,
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
        payload: dict[str, Json],
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

        [point] = db.retrieve([i], with_payload=True, with_vectors=True)
        assert point.vector is not None

        for item in db.query_nearby(
            point.vector,
            k=3000,
            with_payload=False,
            with_vectors=False,
            score_threshold=0.1,
        ):
            print("-", item)
        break
