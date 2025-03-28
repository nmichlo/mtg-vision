from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

import qdrant_client
from qdrant_client.http.models import VectorParams, Distance


@dataclass
class QdrantPoint:
    id: str  # UUID
    vector: list[float] | None = None
    payload: dict[str, Any] | None = None


class VectorStoreBase:
    _VECTOR_SIZE: int = 768

    def save_points(self, iter_points: Iterable[QdrantPoint]):
        raise NotImplementedError

    def query(self, vector: list[float], k: int) -> list[QdrantPoint]:
        raise NotImplementedError

    def query_by_id(self, id_: str) -> QdrantPoint:
        raise NotImplementedError


class VectorStoreQdrant(VectorStoreBase):
    _COLLECTION = "mtg"

    def __init__(self, location: str = "localhost:6333"):
        self.client = qdrant_client.QdrantClient(location=location)
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
    ) -> list[QdrantPoint]:
        from qdrant_client.http.models import QueryResponse

        results: QueryResponse = self.client.query_points(
            collection_name=self._COLLECTION,
            query=vector,
            limit=k,
            with_vectors=with_vectors,
            with_payload=with_payload,
        )
        return [
            QdrantPoint(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
            )
            for point in results.points
        ]

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
