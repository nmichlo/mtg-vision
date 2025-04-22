import itertools

import faiss
import numpy as np
from tqdm import tqdm

# from doorway.x import ProxyDownloader
# from mtgvision.encoder_datasets import SyntheticBgFgMtgImages
# from mtgvision.encoder_export import CoreMlEncoder, MODEL_PATH
from mtgvision.qdrant import VectorStoreQdrant


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


# def train_and_get(
#     vectors: np.ndarray,
#     quant: faiss.ScalarQuantizer = faiss.ScalarQuantizer.QT_4bit,
#     distance: Literal['cosine'] = 'cosine'
# ):
#     _, zsize = vectors.shape
#
#     if distance == 'cosine':
#         # we are using cosine similarity, so we need to normalize the vectors
#         # ...INPLACE...
#         faiss.normalize_L2(vectors)
#         metric = faiss.METRIC_INNER_PRODUCT
#     else:
#         raise ValueError(f'Unknown distance metric: {distance}')
#
#     # Scalar Quantization
#     quantizer = faiss.IndexScalarQuantizer(zsize, quant, metric)
#     quantizer.train(vectors)
#     qn_vectors = quantizer.sa_encode(vectors)
#
#     return quantizer, qn_vectors


def print_vectors_info(vectors: np.ndarray, name: str):
    print(f"{name}: {vectors.shape}, {vectors.dtype}, {vectors.nbytes / 1024**2} MB")
    return vectors


def main():
    vectors, ids = fetch_vectors_from_qdrant()
    print_vectors_info(vectors, "ORIG_VECTORS")

    z_size = vectors.shape[
        -1
    ]  # shape: (100000, 768), dtype float32 | ids: list[str] __len__=100000
    pq_num = 64
    dim_reduction = 512
    idx = faiss.index_factory(
        z_size, f"L2norm,OPQ{pq_num}_{dim_reduction},HNSW32_PQ{pq_num}"
    )
    idx.train(vectors)

    # TODO save in format that can be loaded in FE, maybe tfjs?


if __name__ == "__main__":
    main()
