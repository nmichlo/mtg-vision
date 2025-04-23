import gzip
import itertools
import json
import os
import random
import uuid
from typing import Literal, Optional

import faiss
import numpy as np
from cachier import cachier
from usearch.index import Index, MetricKind, ScalarKind
from tqdm import tqdm

from mtgvision.qdrant import VectorStoreQdrant


def fetch_vectors_from_qdrant(
    max_vectors: Optional[int] = None, dtype=np.float32
) -> tuple[np.ndarray, list[str]]:
    if max_vectors is None:
        max_vectors = 2**63 - 1

    # get itr
    vstore = VectorStoreQdrant()
    itr = vstore.scroll(with_payload=False, with_vectors=True)
    itr = itertools.islice(itr, max_vectors)

    # collect everything
    ids = []
    vectors = []
    for point in tqdm(itr):
        ids.append(point.id)
        vectors.append(np.asarray(point.vector, dtype=dtype))

    # matrix
    vectors = np.stack(vectors)
    return vectors, ids


@cachier()
def fetch_vectors_from_qdrant_cached(
    max_vectors: Optional[int] = None,
    dtype=np.float32,
):
    return fetch_vectors_from_qdrant(max_vectors=max_vectors, dtype=dtype)


def print_vectors_info(vectors: np.ndarray, name: str):
    print(f"{name}: {vectors.shape}, {vectors.dtype}, {vectors.nbytes / 1024**2} MB")
    return vectors


def uuid_to_int(uuid_str: str) -> int:
    """
    Convert a UUID string to an integer.
    """
    return uuid.UUID(uuid_str).int % (2**64)


# 128, opq, 0.1, 81%
# 128, opq, 0.01, 94.7%
# 128, random, 0.1, 22.08%
# 128, random, 0.01, 93.2%
# 128, pca, 0.1, 88.8%%
# 128, pca, 0.01, 94.7%


def main(
    seed: int = 42,
    # load vectors
    max_vectors: Optional[int] = None,
    pre_norm_vectors: bool = True,
    # reduction
    reduce_dim: Optional[int] = 128,  # locked after training
    reduce_mode: Literal["pca", "opq", "random"] = "pca",  # locked after training
    # hnsw
    hnsw_metric: MetricKind = MetricKind.Cosine,  # locked after training
    hnsw_dtype: ScalarKind = ScalarKind.I8,  # locked after training
    hnsw_connectivity: int = 32,  # locked after training
    hnsw_expansion_add: int = 200,  # can change later
    hnsw_expansion_search: int = 200,  # can change later
    # validation only
    validate_perturb_scale: float = 0.1,
    validate_apply_mode: Literal["lib", "manual"] = "lib",
    cache: bool = True,
):
    random.seed(seed)
    np.random.seed(seed)

    # ============== CREATE METADATA ================== #

    print("Fetching vectors...")
    _fetch = fetch_vectors_from_qdrant_cached if cache else fetch_vectors_from_qdrant
    vectors, uuids = _fetch(max_vectors=max_vectors)
    N, D = vectors.shape

    if pre_norm_vectors:
        print("Normalizing vectors...")
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # ============== CREATE METADATA ================== #

    metadata = {
        # input data
        "in_dim": D,
        "in_norm": pre_norm_vectors,
        "in_num": N,
        "in_ids": uuids,
        # reduce dims
        "reduce_mode": reduce_mode,
        "reduce_transform": None,
        "reduce_dim": reduce_dim if reduce_dim else D,
        # hnsw index
        "hnsw_metric": hnsw_metric.name,
        "hnsw_dtype": hnsw_dtype.name,
        "hnsw_connectivity": hnsw_connectivity,
        "hnsw_expansion_add": hnsw_expansion_add,
        "hnsw_expansion_search": hnsw_expansion_search,
    }

    # ============== CREATE AND OPTIMIZE DATA ================== #

    # reduce op
    reduce = None

    # random order for training
    random_order = np.arange(N)
    np.random.shuffle(random_order)

    # actually reduce
    if reduce_dim and reduce_mode:
        # linear transforms
        if reduce_mode == "pca":
            reduce = faiss.PCAMatrix(d_in=D, d_out=reduce_dim, random_rotation=True)
        elif reduce_mode == "opq":
            reduce = faiss.OPQMatrix(D, 16, reduce_dim)  # M????
        elif reduce_mode == "random":
            reduce = faiss.RandomRotationMatrix(D, reduce_dim)
        else:
            raise ValueError(f"Unknown reduce mode: {reduce_mode}")

        # train!
        print(f"Training {reduce_mode}...")
        reduce.train(vectors[random_order].copy())

        # Extract the final transformation components computed by FAISS
        # - although PCAMat and eigenvectors are available, the PCAMatrix is a subclass
        #   of linear transformation, so when trained it modifies the A and b attributes
        #   instead of modifying the application pipeline.
        A_mat = faiss.vector_float_to_array(reduce.A).reshape(reduce_dim, D)
        b_vec = faiss.vector_float_to_array(reduce.b)
        metadata["reduce_transform"] = {"A": A_mat.tolist(), "b": b_vec.tolist()}

    # ============== CREATE AND EXPORT INDEX ================== #

    # create index
    index = Index(
        ndim=reduce_dim if reduce_dim else D,
        metric=hnsw_metric,
        dtype=hnsw_dtype,
        connectivity=hnsw_connectivity,  # Number of Graph connections per layer of HNSW. Original paper calls it "M". Can't be changed after construction.
        expansion_add=hnsw_expansion_add,  # Search depth when inserting new vectors. Original paper calls it "efConstruction". Can be changed afterwards.
        expansion_search=hnsw_expansion_search,  # Search depth when querying nearest neighbors. Original paper calls it "ef". Can be changed afterwards.
        multi=False,
    )

    # ============== PROCESS VECTORS ================== #

    def process_vector(
        v: np.ndarray, mode: Literal["lib", "manual"] = validate_apply_mode
    ) -> np.ndarray:
        if reduce is None:
            return v
        # reduce
        if mode == "lib":
            return reduce.apply(v[None, :])[0]
        elif mode == "manual":
            return (v.reshape(1, D) @ A_mat.T + b_vec).reshape(reduce_dim)
        elif mode == "manual_loop":
            output = np.zeros(reduce_dim, dtype=v.dtype)
            for i in range(reduce_dim):
                total = b_vec[i]
                for j in range(D):
                    total += A_mat[i, j] * v[j]
                output[i] = total
            return output
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # ============== CONSTRUCT ================== #

    # fill index with vectors
    print("Filling index...")
    with tqdm() as pbar:

        def progress(progress: int, total: int) -> bool:
            pbar.n = progress
            pbar.total = total
            return True

        index.add(
            keys=[uuid_to_int(uid) for uid in uuids],
            vectors=np.asarray([process_vector(v, mode="lib") for v in vectors]),
            progress=progress,
        )

    # ============== SAVE ================== #

    print("Saving index...")
    # save index and compress
    index.save("index.bin")
    with open("index.bin", "rb") as f_in:
        with gzip.open("index.bin.gz", "wb") as f_out:
            f_out.writelines(f_in)
    os.unlink("index.bin")
    # write metadata ids
    with gzip.open("index_meta.json.gz", "w") as fp:
        fp.write(json.dumps(metadata).encode("utf-8"))

    # ============== TEST INDEX ================== #

    def perturb_vector(vector: np.ndarray, scale=validate_perturb_scale):
        return vector + np.random.normal(0, scale, vector.shape)

    # for each vector, perturb it and check if it matches
    print("Validating index...")
    with tqdm(total=len(vectors)) as pbar:
        count, correct = 0, 0
        for uid, vector in zip(uuids, vectors):
            results = index.search(process_vector(perturb_vector(vector)), 1)
            count += 1
            correct += uuid_to_int(uid) in results.keys
            pbar.update()
            pbar.set_postfix_str(f"accuracy: {correct / count:.2%}")


if __name__ == "__main__":
    main()
