import gzip
import itertools
import json
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Literal, Optional

import faiss
import numpy as np
from cachier import cachier
from usearch.index import Index, ScalarKind
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


def resave_gz(path: str):
    with open(path, "rb") as f_in:
        with gzip.open(path + ".gz", "wb") as f_out:
            f_out.writelines(f_in)


def resave_parts(path: str, max_part_size_bytes: int = 3 * 1024 * 1024):
    """
    Split a file into parts.

    <input> --> <input>/part#

    A meta file is also saved listing all the parts, and the total size.
    """
    root = Path(f"{path}.parts")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    parts = []
    with open(path, "rb") as f_in:
        while True:
            part = f_in.read(max_part_size_bytes)
            if not part:
                break
            part_name = f"{root}/{len(parts)}.part"
            with open(part_name, "wb") as f_out:
                f_out.write(part)
            parts.append(Path(part_name).name)
    # save meta
    with open(f"{root}/meta.json", "w") as f_out:
        json.dump(
            {
                "total_size": os.path.getsize(path),
                "parts": parts,
            },
            f_out,
            indent=2,
            sort_keys=False,
        )


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
    reduce_quant: Optional[str] = "i8",  # locked after training
    # validation only
    validate_perturb_scale: float = 0.001,
    validate_apply_mode: Literal["lib", "manual", "manual_loop"] = "lib",
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
        "pipeline": [],
        "ids": uuids,
    }

    _pipeline_step_norm = {
        "type": "norm",
        "in_dim": D,
        "out_dim": D,
    }
    _pipeline_step_reduce = {
        "type": "linear_matrix",
        "in_dim": D,
        "out_dim": reduce_dim if reduce_dim else D,
        "params": None,
        "mode": reduce_mode,
    }
    _pipeline_step_quant = {
        "type": "scalar_quantizer",
        "in_dim": reduce_dim if reduce_dim else D,
        "out_dim": reduce_dim if reduce_dim else D,
        "params": None,
        "mode": reduce_quant,
    }

    # ============== CREATE PIPELINE ================== #

    # norm op
    norm = None
    if pre_norm_vectors:
        norm = faiss.normalize_L2
        metadata["pipeline"].append(_pipeline_step_norm)

    # reduce op
    reduce = None
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
        reduce.train(vectors)
        # Extract the final transformation components computed by FAISS
        # - although PCAMat and eigenvectors are available, the PCAMatrix is a subclass
        #   of linear transformation, so when trained it modifies the A and b attributes
        #   instead of modifying the application pipeline.
        A_mat = faiss.vector_float_to_array(reduce.A).reshape(reduce_dim, D)
        b_vec = faiss.vector_float_to_array(reduce.b)
        _pipeline_step_reduce["params"] = {
            "A": A_mat.tolist(),
            "b": b_vec.tolist(),
        }
        metadata["pipeline"].append(_pipeline_step_reduce)

    # quantize op
    quantizer = None
    if reduce_quant:
        # quantizers
        q_dims = reduce_dim if reduce_dim else D
        if reduce_quant == "i8":
            quantizer = faiss.ScalarQuantizer(q_dims, faiss.ScalarQuantizer.QT_8bit)
        else:
            raise ValueError(f"Unknown quantizer: {reduce_quant}")
        # train!
        x = vectors if reduce is None else reduce.apply(vectors)
        quantizer.train(x)
        # extract
        # - the trained quantizer operates over a min-max range of 0-1?
        #   this is configurable with RS_* settings??
        [vmin, vdiff] = faiss.vector_float_to_array(quantizer.trained).reshape(2, -1)
        _pipeline_step_quant["params"] = {
            "vmin": vmin.tolist(),
            "vdiff": vdiff.tolist(),
        }
        metadata["pipeline"].append(_pipeline_step_quant)

    # ============== PROCESS VECTORS ================== #

    def process_vector(
        v: np.ndarray,
        mode: Literal["lib", "manual", "manual_loop"] = validate_apply_mode,
    ) -> np.ndarray:
        for step in metadata["pipeline"]:
            # NORM
            if step["type"] == "norm":
                v = v / np.linalg.norm(v)
            # REDUCE
            elif step["type"] == "linear_matrix":
                if mode == "lib":
                    v = reduce.apply(v[None, :])[0]
                elif mode == "manual":
                    v = (v.reshape(1, D) @ A_mat.T + b_vec).reshape(reduce_dim)
                elif mode == "manual_loop":
                    output = np.zeros(reduce_dim, dtype=v.dtype)
                    for i in range(reduce_dim):
                        total = b_vec[i]
                        for j in range(D):
                            total += A_mat[i, j] * v[j]
                        output[i] = total
                    v = output
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            # QUANT
            elif step["type"] == "scalar_quantizer":
                vmin = step["params"]["vmin"]
                vdiff = step["params"]["vdiff"]
                if mode == "lib":
                    v = quantizer.compute_codes(v[None, :])[0]
                elif mode == "manual":
                    # encode
                    v = np.clip(((v - vmin) / vdiff) * 255, 0, 255).astype(np.uint8)
                    # decode
                    # v = vmin + ((v + 0.5) / 255) * vdiff
                elif mode == "manual_loop":
                    # https://github.com/facebookresearch/faiss/blob/d4fa401656fa413728f3c93bae4e34fb81803d54/faiss/impl/ScalarQuantizer.cpp#L381
                    out = np.zeros((len(v),), dtype=np.uint8)
                    for i in range(len(v)):
                        vd = vdiff[i]
                        vm = vmin[i]
                        xi = 0
                        if vd != 0:
                            xi = (v[i] - vm) / vd
                            if xi < 0:
                                xi = 0
                            if xi > 1.0:
                                xi = 1.0
                        out[i] = int(xi * 255)
                    v = out
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            # UNKNOWN
            else:
                raise ValueError(f"Unknown step type: {step['type']}")
        # done!
        return v

    # ============== VALIDATE QUANTIZATION ================== #

    # validate quantization
    for v in vectors[:10]:
        vlib = process_vector(v, mode="lib")
        vmanual = process_vector(v, mode="manual")
        vmanual_loop = process_vector(v, mode="manual_loop")
        error_lib = ((vlib - vlib) ** 2).sum() / (vlib**2).sum()
        error_manual = ((vlib - vmanual) ** 2).sum() / (vlib**2).sum()
        error_manual_loop = ((vlib - vmanual_loop) ** 2).sum() / (vlib**2).sum()
        print(
            f"lib error: {error_lib}, manual error: {error_manual}, manual loop error: {error_manual_loop}"
        )

    # ============== SAVE ================== #

    print("Saving metadata...")
    # save index and compress
    with open("gen/index_meta.json", "w") as fp:
        json.dump(metadata, fp, indent=2, sort_keys=False)
    # resave as gzip
    resave_gz("gen/index_meta.json")
    # split files into chunks
    resave_parts("gen/index_meta.json")

    # ============== TEST INDEX ================== #
    # -- THIS IS NOT EXPORTED
    # -- THIS IS JUST USED TO VALIDATE THE RESULTS

    hnsw_metric: str = "cosine"  # locked after training
    hnsw_connectivity: int = 16  # locked after training
    hnsw_expansion_add: int = 200  # can change later
    hnsw_expansion_search: int = 200  # can change later

    # create index
    index = Index(
        ndim=reduce_dim if reduce_dim else D,
        metric=hnsw_metric,
        dtype=ScalarKind.I8,
        connectivity=hnsw_connectivity,  # Number of Graph connections per layer of HNSW. Original paper calls it "M". Can't be changed after construction.
        expansion_add=hnsw_expansion_add,  # Search depth when inserting new vectors. Original paper calls it "efConstruction". Can be changed afterwards.
        expansion_search=hnsw_expansion_search,  # Search depth when querying nearest neighbors. Original paper calls it "ef". Can be changed afterwards.
        multi=False,
    )

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

    def perturb_vector(vector: np.ndarray, scale=validate_perturb_scale):
        noise = np.random.normal(0, scale, vector.shape)
        noise = noise / np.linalg.norm(noise)
        return vector + noise * scale

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
