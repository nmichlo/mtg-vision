import itertools
import subprocess

import faiss
import faiss.swigfaiss
import numpy as np
from tqdm import tqdm

# from doorway.x import ProxyDownloader
# from mtgvision.encoder_datasets import SyntheticBgFgMtgImages
# from mtgvision.encoder_export import CoreMlEncoder, MODEL_PATH
from mtgvision.qdrant import VectorStoreQdrant
import tensorflow as tf


def fetch_vectors_from_qdrant(max_vectors: int = 10_000, dtype=np.float32):
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


def print_vectors_info(vectors: np.ndarray, name: str):
    print(f"{name}: {vectors.shape}, {vectors.dtype}, {vectors.nbytes / 1024**2} MB")
    return vectors


def main(
    pq_num: int = 16,
    dim_reduction: int = 512,
    max_vectors: int = 1000,
    ef_construction: int = 200,
    ef_search: int = 100,
):
    # get shape of vectors
    _, D = fetch_vectors_from_qdrant(max_vectors=1)[0].shape

    # 1. CREATE INDEX
    index: faiss.IndexPreTransform = faiss.index_factory(
        D, f"L2norm,OPQ{pq_num}_{dim_reduction},HNSW32_PQ{pq_num}"
    )

    # 2. EXTRACT FROM INDEX
    hnsw_index: faiss.IndexHNSWPQ = faiss.downcast_index(index.index)
    assert isinstance(hnsw_index, faiss.IndexHNSWPQ)
    hnsw_index.efConstruction = ef_construction
    hnsw_index.efSearch = ef_search

    hnsw: faiss.HNSW = hnsw_index.hnsw
    assert isinstance(hnsw, faiss.HNSW)

    hnsw_pq_storage: faiss.IndexPQ = faiss.downcast_index(hnsw_index.storage)
    assert isinstance(hnsw_pq_storage, faiss.IndexPQ)

    pq: faiss.ProductQuantizer = hnsw_pq_storage.pq
    assert isinstance(pq, faiss.ProductQuantizer)

    assert index.chain.size() == 2
    _chn0 = index.chain.at(0)
    _chn1 = index.chain.at(1)

    chain_0_norm: faiss.NormalizationTransform = faiss.downcast_VectorTransform(_chn0)
    assert isinstance(chain_0_norm, faiss.NormalizationTransform)
    chain_1_opq: faiss.OPQMatrix = faiss.downcast_VectorTransform(_chn1)
    assert isinstance(chain_1_opq, faiss.OPQMatrix)

    # 3. GET VECTORS
    vectors, ids = fetch_vectors_from_qdrant()
    print_vectors_info(vectors, "ORIG_VECTORS")

    # 4. TRAINING INDEX
    print("Training...")
    index.train(vectors)
    print("Adding Vectors...")
    index.add(vectors)

    # --- EXPORTING ---

    # OPQ
    # * d_in: input dimensionality
    # * d_out: output dimensionality
    opq_mat = chain_1_opq.A.astype(np.float32)  # Transform matrix, size: (d_out, d_in)
    opq_bias = chain_1_opq.b.astype(np.float32)  # bias vector, size: (d_out)

    # PQ
    # * M: number of subquantizers
    # * dsub: dimensionality of each subvector
    # * ksub: number of centroids for each subquantizer
    pq_centroids = pq.centroids.astype(np.float32)  # (size: M, ksub, dsub)
    pq_codes = index.sa_encode(vectors).astype(np.int32)  # (N, pq_num)

    # HNSW
    # * levels: level of each vector (base level = 1), size: (ntotal,)
    # * offsets: offsets[i] is the offset in the neighbors array where vector i is stored size: (ntotal + 1,)
    # * neighbors: neighbors[offsets[i]:offsets[i+1]] is the list of neighbors of vector i for all levels. this is where all storage goes.
    # * entry_point: entry point in the search structure (one of the points with maximum level
    hnsw_levels = hnsw.levels.astype(np.int32)  # (N,)
    hnsw_offsets = hnsw.offsets.astype(np.int32)  # (N + 1,)
    hnsw_neighbors = hnsw.neighbors.astype(np.int32)  # (N, M)
    hnsw_entry_point = hnsw.entry_point.astype(np.int32)  # (1,)
    # TODO: WRONG
    # TODO: WRONG
    # TODO: WRONG

    # UUIDS
    uuids = np.array(ids, dtype=object)  # (N,)

    # --- MODEL ---
    # TODO: WRONG
    # TODO: WRONG
    # TODO: WRONG

    class IndexModule(tf.Module):
        def __init__(self):
            super().__init__()
            # OPQ: weight = A^T, bias = -mean
            self.opq_kernel = tf.Variable(
                opq_kernel, trainable=False, name="opq_kernel"
            )
            self.opq_bias = tf.Variable(opq_bias, trainable=False, name="opq_bias")
            # PQ
            self.pq_centroids = tf.Variable(
                pq_centroids, trainable=False, name="pq_centroids"
            )
            self.pq_codes = tf.Variable(pq_codes, trainable=False, name="pq_codes")
            # HNSW graph
            self.hnsw_links = tf.Variable(
                hnsw_links, trainable=False, name="hnsw_links"
            )
            # UUIDs
            self.uuids = tf.Variable(uuids, trainable=False, name="uuids")

        @tf.function(input_signature=[tf.TensorSpec([None, D], tf.float32)])
        def serve(self, q):
            # identity op so model has a signature
            return tf.identity(q)

    out_saved = "index.keras"
    out_tfjs = "index.savedmodel"

    print("Building and saving SavedModel…")
    model = IndexModule()
    tf.saved_model.save(
        model, str(out_saved), signatures={"serving_default": model.serve}
    )

    # --- CONVERTING ---

    print("Converting to TFJS…")
    max_file_size = 2 * 1024**2  # 2 MB
    subprocess.run(
        [
            "tensorflowjs_converter",
            "--input_format=tf_saved_model",
            "--split_weights_for_weight_shard=true",
            f"--weight_shard_size_bytes={max_file_size}",
            str(out_saved),
            str(out_tfjs),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
