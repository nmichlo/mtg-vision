import dataclasses
import itertools
import json
from typing import Any, Iterable
from uuid import UUID

import duckdb
from tqdm import tqdm

from doorway.x import ProxyDownloader
from mtgvision.datasets import SyntheticBgFgMtgImages
from mtgvision.export_encoder import CoreMlEncoder, MODEL_PATH

from mtgvision.train_encoder import RanMtgEncDecDataset
from mtgvision.util.image import imread_float


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


class VectorStoreDuckDB(VectorStoreBase):
    def __init__(self, db_path: str = "mtg.duckdb"):
        """
        Initialize the VectorStoreDuckDB with a DuckDB connection and set up the table and index.

        Args:
            db_path (str): Path to the DuckDB database file (default: 'mtg.duckdb').
        """
        # Connect to the DuckDB database (persistent storage)
        self.conn = duckdb.connect(db_path)

        # Install and load the VSS extension
        self.conn.execute("INSTALL vss;")
        self.conn.execute("LOAD vss;")

        # Enable experimental persistence for HNSW index
        self.conn.execute("SET hnsw_enable_experimental_persistence = true;")

        # Create the table to store vectors if it doesn't exist
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mtg_vectors (
                id UUID PRIMARY KEY,
                vector FLOAT[768],
                payload JSON
            )
        """)

        # Create an HNSW index on the vector column with cosine metric
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS mtg_hnsw_index
            ON mtg_vectors USING HNSW (vector)
            WITH (metric = 'cosine')
        """)

    def save_points(self, iter_points: Iterable[Point]):
        points = list(iter_points)
        if not points:
            return
        # Prepare data for insertion (convert UUID to string and payload to JSON)
        data = [
            (str(point.id), point.vector, json.dumps(point.payload)) for point in points
        ]
        # Insert or update points in the table
        self.conn.executemany(
            """
            INSERT INTO mtg_vectors (id, vector, payload)
            VALUES (?, ?, ?)
            ON CONFLICT (id) DO UPDATE
            SET vector = excluded.vector, payload = excluded.payload
        """,
            data,
        )

    def query(self, vector: list[float], k: int) -> list[Point]:
        results = self.conn.execute(
            """
            SELECT id, payload, vector
            FROM mtg_vectors
            ORDER BY array_cosine_distance(vector, ?)
            LIMIT ?
        """,
            [vector, k],
        ).fetchall()

        # Convert query results to Point objects
        return [
            Point(id=row[0], payload=json.loads(row[1]), vector=row[2])
            for row in results
        ]

    def query_by_id(self, id_: str) -> Point:
        row = self.conn.execute(
            """
            SELECT id, payload, vector
            FROM mtg_vectors
            WHERE id = ?
        """,
            [str(id_)],
        ).fetchone()

        if row is None:
            raise KeyError(f"No point with id {id_}")

        return Point(id=row[0], payload=json.loads(row[1]), vector=row[2])

    def clear(self):
        # Drop and recreate the table
        self.conn.execute("DROP TABLE IF EXISTS mtg_vectors")
        self.conn.execute("""
            CREATE TABLE mtg_vectors (
                id UUID PRIMARY KEY,
                vector FLOAT[768],
                payload JSON
            )
        """)

        # Recreate the HNSW index
        self.conn.execute("""
            CREATE INDEX mtg_hnsw_index
            ON mtg_vectors USING HNSW (vector)
            WITH (metric = 'cosine')
        """)

    def is_populated(self) -> bool:
        count = self.conn.execute("SELECT COUNT(*) FROM mtg_vectors").fetchone()[0]
        return count > 0


def _cli():
    encoder = CoreMlEncoder(MODEL_PATH.with_suffix(".encoder.mlpackage"))

    dataset = RanMtgEncDecDataset(default_batch_size=1)

    proxy = ProxyDownloader()

    # db = VectorStoreQdrant()
    db = VectorStoreDuckDB()

    # 1. create dataset of embeddings
    def _yield_gt_points():
        for card in tqdm(dataset.mtg.card_iter(), total=len(dataset.mtg)):
            if isinstance(card.id, UUID):
                card.__dict__["id"] = card.id.hex
                assert isinstance(card.id, str)
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
            if isinstance(card.id, UUID):
                card.__dict__["id"] = card.id.hex
                assert isinstance(card.id, str)
            card.download(proxy=proxy)
            x = SyntheticBgFgMtgImages.make_virtual(
                imread_float(card.img_path),
                imread_float(dataset.ilsvrc.ran_path()),
                size_hw=dataset.x_size_hw,
            )
            z = encoder.predict(x).tolist()
            yield z, card

    N = 100

    # actually generate and query
    db.save_points(
        Point(id=card.id, vector=z, payload=card.__dict__)
        for z, card in itertools.islice(_yield_gt_points(), N)
    )

    # actually query db
    top_1_correct = 0
    top_5_correct = 0
    with tqdm() as pbar:
        for i, (z, card) in enumerate(itertools.islice(_yield_virtual_points(), N)):
            print(card)
            results = db.query(z, k=5)
            print(results)
            if card.id == results[0].id:
                top_1_correct += 1
            if card.id in {point.id for point in results}:
                top_5_correct += 1
            pbar.set_postfix(
                {
                    "top_1": f"{top_1_correct / (i + 1) * 100:.2f}%",
                    "top_5": f"{top_5_correct / (i + 1) * 100:.2f}%",
                }
            )


if __name__ == "__main__":
    _cli()
