import gzip
import itertools
import json
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Literal
from typing import TypedDict

import faiss
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from cachier import cachier
from tqdm import tqdm
from usearch.index import Index
from usearch.index import ScalarKind

from mtgvision.qdrant import VectorStoreQdrant
from mtgvision.util.json import Json

# a single vector as it flows through `process_vector`: float32 before
# quantization, uint8 after
type VecArray = npt.NDArray[np.float32] | npt.NDArray[np.uint8]


class ReducerNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        quant_bits: int = 8,
        mode: Literal["sigmoid", "linear"] = "linear",
    ) -> None:
        super().__init__()

        self.q_mode = mode
        self.q_min = 0
        self.q_max = 2**quant_bits - 1
        self.q_mid = 2 ** (quant_bits - 1) - 1

        if hidden_dim is None:
            hidden_dim = max(out_dim, in_dim // 2)
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # Learnable scale and offset for each dimension
        self.scale = nn.Parameter(torch.ones(out_dim))
        self.offset = nn.Parameter(torch.zeros(out_dim))
        with torch.no_grad():
            if self.q_mode == "sigmoid":
                self.scale.fill_(1)
                self.offset.fill_(0)
            else:
                self.scale.fill_(self.q_mid)
                self.offset.fill_(self.q_mid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        # forward
        if self.q_mode == "sigmoid":
            # Shift the output to good starting values
            x = x * 16
            # apply sigmoid to range 0-255
            x = torch.sigmoid(x) * (self.q_max - self.q_min)
        else:
            # Shift the output to good starting values
            x = x * self.q_mid * 8 + self.q_mid  # (works well)
            # clap to 0-255 range and allow gradients to flow through
            sub_greater = (x - self.q_max).detach() * (x > self.q_max).detach()
            sub_less = (x - self.q_min).detach() * (x < self.q_min).detach()
            x = x - sub_greater - sub_less
        # to int that allows grad flow
        x = x - (x % 1).detach()
        return x

    def get_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Convert quantized values back to normalized vectors"""
        # undo sigmoid
        if self.q_mode == "sigmoid":
            x = x / (self.q_max - self.q_min)
            x = torch.log(x / (1 - x + 1e-8) + 1e-8)  # logit function
        # Unquantize
        x = (x - self.offset) / self.scale
        # Normalize
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

    def get_quantization_params(self) -> dict[str, npt.NDArray[np.float32]]:
        """Get the learned scale and offset parameters for each dimension"""
        return {
            "scale": self.scale.detach().cpu().numpy(),
            "offset": self.offset.detach().cpu().numpy(),
        }


def train_reducer_net(
    vectors: npt.NDArray[np.float32],
    out_dim: int,
    epochs: int = 100,
    batch_size: int = 1024 * 4,
) -> ReducerNet:
    D = vectors.shape[1]
    device = torch.device("mps")
    print(f"Using device: {device}")

    # get model
    model = ReducerNet(D, out_dim).to(device)
    optimizer = optim.RAdam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    N, D = vectors.shape

    # ------------
    # BENCHMARK AGAINST
    # positional on purpose: faiss-cpu's runtime kwargs are (din, dout,
    # eigen_power_in, random_rotation_in) but its bundled .pyi stub declares
    # (d_in, d_out, eigen_power, random_rotation), so neither spelling of the
    # keywords satisfies both. Positional agrees with each.
    pca = faiss.PCAMatrix(D, out_dim, 0, True)
    pca.train(vectors)
    rrm = faiss.RandomRotationMatrix(D, out_dim)
    rrm.train(vectors)
    # ------------

    # ---- append data to vectors ----

    random_groups = [
        np.random.randn(N // 10, D).astype(np.float32),
        np.random.randn(N // 10, D).astype(np.float32),
    ]
    random_groups[1] /= np.linalg.norm(random_groups[1], axis=1, keepdims=True)
    vectors = np.concatenate((vectors, *random_groups), axis=0)

    # ------------
    # BENCHMARK AGAINST
    pca2 = faiss.PCAMatrix(D, out_dim, 0, True)
    pca2.train(vectors)
    rrm2 = faiss.RandomRotationMatrix(D, out_dim)
    rrm2.train(vectors)
    pca_A = faiss.vector_float_to_array(pca2.A).reshape(out_dim, D)
    pca_b = faiss.vector_float_to_array(pca2.b)
    # ------------

    norm = True

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(range(0, N, batch_size), desc=f"Epoch {epoch + 1}/{epochs}")
        # random order
        idxs = np.arange(N)
        np.random.shuffle(idxs)

        for batch_idx in pbar:
            optimizer.zero_grad()
            # make batch
            vecs = vectors[idxs[batch_idx : batch_idx + batch_size]]
            vecs = torch.from_numpy(vecs).to(device)

            # get original distances
            with torch.no_grad():
                orig_sim = vecs @ vecs.T
            # get new distances from compressed vectors
            quantized_vecs = model(vecs)  # output is soft-quantized 0-255
            # out of range loss
            if model.q_mode == "linear":
                oor_loss = torch.mean(
                    torch.abs((quantized_vecs <= model.q_min) * (quantized_vecs - model.q_mid))
                ) + torch.mean(torch.abs((quantized_vecs >= model.q_max) * (quantized_vecs - model.q_mid)))
            else:
                oor_loss = torch.full((), fill_value=0.0, device=device)
            # quant loss
            embedded_vecs = model.get_normalized(quantized_vecs)
            embedded_sim = embedded_vecs @ embedded_vecs.T
            # compute loss
            loss_sim = loss_fn(embedded_sim, orig_sim)
            loss = loss_sim + oor_loss

            # -------
            # benchmark
            if (epoch in (0, epochs - 1) or epoch % 10 == 0) and batch_idx == 0:
                print()
                # pca
                pca_vecs = pca.apply_py(vecs.cpu().numpy())
                pca_vecs = torch.from_numpy(pca_vecs).to(device)
                if norm:
                    pca_vecs = nn.functional.normalize(pca_vecs, p=2, dim=1)
                pca_sim = pca_vecs @ pca_vecs.T
                loss_pca_sim = loss_fn(pca_sim, orig_sim)
                print(f"PCA loss: {loss_pca_sim.item()}")
                # pca
                pca_vecs = pca2.apply_py(vecs.cpu().numpy())
                pca_vecs = torch.from_numpy(pca_vecs).to(device)
                if norm:
                    pca_vecs = nn.functional.normalize(pca_vecs, p=2, dim=1)
                pca_sim = pca_vecs @ pca_vecs.T
                loss_pca_sim = loss_fn(pca_sim, orig_sim)
                print(f"PCA loss (ran): {loss_pca_sim.item()}")
                # mock linear transform pca
                linear = nn.Linear(D, out_dim, bias=True).to(device)
                with torch.no_grad():
                    linear.weight.copy_(torch.from_numpy(pca_A).to(device))
                    linear.bias.copy_(torch.from_numpy(pca_b).to(device))
                linear_vecs = linear(vecs)
                if norm:
                    linear_vecs = nn.functional.normalize(linear_vecs, p=2, dim=1)
                linear_sim = linear_vecs @ linear_vecs.T
                loss_linear_sim = loss_fn(linear_sim, orig_sim)
                print(f"Linear transform PCA loss: {loss_linear_sim.item()}")
                # random rotation matrix
                rrm_vecs = rrm.apply_py(vecs.cpu().numpy())
                rrm_vecs = torch.from_numpy(rrm_vecs).to(device)
                if norm:
                    rrm_vecs = nn.functional.normalize(rrm_vecs, p=2, dim=1)
                rrm_sim = rrm_vecs @ rrm_vecs.T
                loss_rrm_sim = loss_fn(rrm_sim, orig_sim)
                print(f"Random rotation matrix loss: {loss_rrm_sim.item()}")
                # actual model
                print(f"Model loss: {loss.item()}")
                # Print quantization stats
                with torch.no_grad():
                    quantized = model(vecs)
                    print(f"Quantized range: [{quantized.min().item():.1f}, {quantized.max().item():.1f}]")
                    print(f"Scale range: [{model.scale.min().item():.3f}, {model.scale.max().item():.3f}]")
                    print(f"Offset range: [{model.offset.min().item():.3f}, {model.offset.max().item():.3f}]")
            # -------

            # optimize
            loss.backward()
            optimizer.step()
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "sim": loss_sim.item(),
                    "oor": oor_loss.item(),
                }
            )

    return model.eval()


def fetch_vectors_from_qdrant(
    max_vectors: int | None = None, dtype: type[np.float32] = np.float32
) -> tuple[npt.NDArray[np.float32], list[str]]:
    if max_vectors is None:
        max_vectors = 2**63 - 1

    # get itr
    vstore = VectorStoreQdrant()
    itr = vstore.scroll(with_payload=False, with_vectors=True)
    itr = itertools.islice(itr, max_vectors)

    # collect everything
    ids: list[str] = []
    vectors: list[npt.NDArray[np.float32]] = []
    for point in tqdm(itr):
        ids.append(point.id)
        vectors.append(np.asarray(point.vector, dtype=dtype))

    # matrix
    vectors_arr = np.stack(vectors)
    return vectors_arr, ids


@cachier()
def fetch_vectors_from_qdrant_cached(
    max_vectors: int | None = None,
    dtype: type[np.float32] = np.float32,
) -> tuple[npt.NDArray[np.float32], list[str]]:
    return fetch_vectors_from_qdrant(max_vectors=max_vectors, dtype=dtype)


def print_vectors_info(vectors: npt.NDArray[np.float32], name: str) -> npt.NDArray[np.float32]:
    print(f"{name}: {vectors.shape}, {vectors.dtype}, {vectors.nbytes / 1024**2} MB")
    return vectors


def uuid_to_int(uuid_str: str) -> int:
    """
    Convert a UUID string to an integer.
    """
    return uuid.UUID(uuid_str).int % (2**64)


def resave_gz(path: str) -> None:
    with open(path, "rb") as f_in:
        with gzip.open(path + ".gz", "wb") as f_out:
            f_out.writelines(f_in)


def resave_parts(path: str, max_part_size_bytes: int = 3 * 1024 * 1024) -> None:
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


# ============================================================================ #
# Metadata: describes the reduce/quantize pipeline so it can be replayed       #
# without the original faiss/torch objects (see `process_vector`)             #
# ============================================================================ #


class LinearTransformParams(TypedDict):
    A: list[list[float]]
    b: list[float]


class LinearTransformStep(TypedDict):
    type: Literal["LinearTransform"]
    in_dim: int
    out_dim: int
    in_dtype: str
    out_dtype: str
    params: LinearTransformParams


class NeuralNetReducerParams(TypedDict):
    state_dict: dict[str, Json]
    hidden_dim: int


class NeuralNetReducerStep(TypedDict):
    type: Literal["NeuralNetReducer"]
    in_dim: int
    out_dim: int
    in_dtype: str
    out_dtype: str
    params: NeuralNetReducerParams


type ChainStep = LinearTransformStep | NeuralNetReducerStep


class ScalarQuantizerParams(TypedDict):
    vmin: list[float]
    vdiff: list[float]


class QuantizeStep(TypedDict):
    type: Literal["ScalarQuantizer"]
    in_dim: int
    out_dim: int
    in_dtype: str
    out_dtype: str
    params: ScalarQuantizerParams
    mode: str


class Metadata(TypedDict):
    model: str | None
    chain: list[ChainStep]
    quantize: QuantizeStep | None
    ids: list[str]


def main(
    seed: int = 42,
    # load vectors
    max_vectors: int | None = None,
    # reduction
    reduce_dim: int | None = 128,  # locked after training
    reduce_mode: Literal["pca", "opq", "random", "nn"] = "nn",  # locked after training
    reduce_quant: str | None = "i8",  # locked after training
    # validation only
    validate_perturb_scale: float = 0.001,
    validate_apply_mode: Literal["lib", "manual", "manual_loop"] = "lib",
    cache: bool = True,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # ============== CREATE METADATA ================== #

    print("Fetching vectors...")
    _fetch = fetch_vectors_from_qdrant_cached if cache else fetch_vectors_from_qdrant
    vectors, uuids = _fetch(max_vectors=max_vectors)
    N, D = vectors.shape

    print("normalizing vectors...")
    faiss.normalize_L2(vectors)  # inplace

    # ============== CREATE METADATA ================== #

    metadata: Metadata = {
        "model": None,  # string
        "chain": [],
        "quantize": None,
        "ids": uuids,
    }

    # ============== CREATE PIPELINE ================== #

    # reduce op
    reduce: ReducerNet | faiss.PCAMatrix | faiss.OPQMatrix | faiss.RandomRotationMatrix | None = None
    # only set by the faiss reducers -- `reduce_mode="nn"` emits a
    # NeuralNetReducer chain step instead of a LinearTransform one
    A_mat: npt.NDArray[np.float32] | None = None
    b_vec: npt.NDArray[np.float32] | None = None
    if reduce_dim:
        # linear transforms
        if reduce_mode == "nn":
            print("Training reduction network...")
            reducer_model = train_reducer_net(vectors, out_dim=reduce_dim)
            reduce = reducer_model.cpu()
            # save state dict
            state_dict = reduce.state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.numpy().tolist()
            # save metadata
            metadata["chain"].append(
                {
                    "type": "NeuralNetReducer",
                    "in_dim": D,
                    "out_dim": reduce_dim,
                    "in_dtype": "float32",
                    "out_dtype": "float32",
                    "params": {
                        "state_dict": state_dict,
                        "hidden_dim": reduce.hidden_dim,
                    },
                }
            )
        elif reduce_mode == "pca":
            reduce = faiss.PCAMatrix(D, reduce_dim, 0, True)
        elif reduce_mode == "opq":
            reduce = faiss.OPQMatrix(D, 16, reduce_dim)  # M????
        elif reduce_mode == "random":
            reduce = faiss.RandomRotationMatrix(D, reduce_dim)
        else:
            raise ValueError(f"Unknown reduce mode: {reduce_mode}")
        # train! -- faiss transforms only. `reduce_mode="nn"` is already trained
        # by `train_reducer_net` above and has appended its own chain step.
        if not isinstance(reduce, ReducerNet):
            print(f"Training {reduce_mode}...")
            reduce.train(vectors)
            # Extract the final transformation components computed by FAISS
            # - although PCAMat and eigenvectors are available, the PCAMatrix is a subclass
            #   of linear transformation, so when trained it modifies the A and b attributes
            #   instead of modifying the application pipeline.
            A_mat = faiss.vector_float_to_array(reduce.A).reshape(reduce_dim, D)
            b_vec = faiss.vector_float_to_array(reduce.b)
            metadata["chain"].append(
                {
                    "type": "LinearTransform",
                    "in_dim": D,
                    "out_dim": reduce_dim if reduce_dim else D,
                    "in_dtype": "float32",
                    "out_dtype": "float32",
                    "params": {"A": A_mat.tolist(), "b": b_vec.tolist()},
                }
            )

    # quantize op
    quantizer: faiss.ScalarQuantizer | None = None
    if reduce_quant:
        # quantizers
        q_dims = reduce_dim if reduce_dim else D
        if reduce_quant == "i8":
            quantizer = faiss.ScalarQuantizer(q_dims, faiss.ScalarQuantizer.QT_8bit)
        else:
            raise ValueError(f"Unknown quantizer: {reduce_quant}")
        # train! -- the quantizer sees whatever the reducer emits
        x: npt.NDArray[np.float32]
        if reduce is None:
            x = vectors
        elif isinstance(reduce, ReducerNet):
            # ReducerNet already rounds to `quant_bits` integers; the scalar
            # quantizer only packs them into the 1-byte-per-dim layout that
            # `index_vecs.bin` and the usearch ScalarKind.I8 index require.
            with torch.no_grad():
                x = reduce(torch.from_numpy(vectors)).numpy()
        else:
            x = reduce.apply(vectors)
        quantizer.train(x)
        # extract
        # - the trained quantizer operates over a min-max range of 0-1?
        #   this is configurable with RS_* settings??
        [vmin, vdiff] = faiss.vector_float_to_array(quantizer.trained).reshape(2, -1)
        metadata["quantize"] = {
            "type": "ScalarQuantizer",
            "in_dim": reduce_dim if reduce_dim else D,
            "out_dim": reduce_dim if reduce_dim else D,
            "in_dtype": "float32",
            "out_dtype": "uint8",
            "params": {"vmin": vmin.tolist(), "vdiff": vdiff.tolist()},
            "mode": "QT_8bit",
        }

    # ============== PROCESS VECTORS ================== #

    def process_vector(
        v: npt.NDArray[np.float32],
        mode: Literal["lib", "manual", "manual_loop"] = validate_apply_mode,
        skip_quant: bool = False,
    ) -> VecArray:
        # REDUCE
        for step in metadata["chain"]:
            if step["type"] == "LinearTransform":
                assert reduce is not None and not isinstance(reduce, ReducerNet)
                assert reduce_dim is not None
                assert A_mat is not None and b_vec is not None
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
            elif step["type"] == "NeuralNetReducer":
                assert isinstance(reduce, ReducerNet)
                if mode == "lib":
                    with torch.no_grad():
                        v_tensor = torch.from_numpy(v[None, :].astype(np.float32))
                        v = reduce(v_tensor).numpy()[0]
                else:
                    raise ValueError(f"Mode {mode} not supported for NeuralNetReducer")
            else:
                raise ValueError(f"Unknown step type: {step['type']}")
        # QUANT
        result: VecArray = v
        if not skip_quant:
            step = metadata["quantize"]
            assert step is not None
            if step["type"] == "ScalarQuantizer":
                vmin = step["params"]["vmin"]
                vdiff = step["params"]["vdiff"]
                if mode == "lib":
                    assert quantizer is not None
                    result = quantizer.compute_codes(v[None, :])[0]
                elif mode == "manual":
                    # def _manual_encode(x):
                    #     x = (x - vmin) / vdiff
                    #     x = np.clip(x * 255, 0, 255).astype(np.uint8)
                    #     return x
                    # def _manual_decode(x):
                    #     x = (x + 0.5) / 255
                    #     x = vmin + x * vdiff
                    #     return x
                    # encode
                    result = np.clip(((v - vmin) / vdiff) * 255, 0, 255).astype(np.uint8)
                    # decode
                    # v = vmin + ((v + 0.5) / 255) * vdiff
                elif mode == "manual_loop":
                    # https://github.com/facebookresearch/faiss/blob/d4fa401656fa413728f3c93bae4e34fb81803d54/faiss/impl/ScalarQuantizer.cpp#L381
                    out = np.zeros((len(v),), dtype=np.uint8)
                    for i in range(len(v)):
                        vd = vdiff[i]
                        vm = vmin[i]
                        xi = 0.0
                        if vd != 0:
                            xi = (v[i] - vm) / vd
                            if xi < 0:
                                xi = 0
                            if xi > 1.0:
                                xi = 1.0
                        out[i] = int(xi * 255)
                    result = out
                else:
                    raise ValueError(f"Unknown mode: {mode}")
            else:
                raise ValueError(f"Unknown step type: {step['type']}")
        # done!
        return result

    # ============== VALIDATE QUANTIZATION ================== #

    # validate quantization
    for v in vectors[:10]:
        vlib = process_vector(v, mode="lib")
        vmanual = process_vector(v, mode="manual")
        vmanual_loop = process_vector(v, mode="manual_loop")
        error_lib = ((vlib - vlib) ** 2).sum() / (vlib**2).sum()
        error_manual = ((vlib - vmanual) ** 2).sum() / (vlib**2).sum()
        error_manual_loop = ((vlib - vmanual_loop) ** 2).sum() / (vlib**2).sum()
        print(f"lib error: {error_lib}, manual error: {error_manual}, manual loop error: {error_manual_loop}")

    # ============== SAVE ================== #

    print("Saving metadata...")

    # save metadata
    with open("gen/index_meta.json", "w") as fp:
        json.dump(metadata, fp, indent=2, sort_keys=False)
    # save vectors as binary values
    with open("gen/index_vecs.bin", "wb") as fp:
        for v in tqdm(vectors, "saving"):
            bytes_ = process_vector(v, mode="lib")
            bytes_ = bytes_.tobytes()
            assert len(bytes_) == (reduce_dim if reduce_dim else D)
            fp.write(bytes_)
        fp.flush()
    # print size of fp
    with open("gen/index_vecs.bin", "rb") as fp:
        len_ = len(fp.read())
        assert len_ == N * (reduce_dim if reduce_dim else D)
        print(len_, N, reduce_dim if reduce_dim else D)

    # resave
    resave_gz("gen/index_meta.json")
    resave_gz("gen/index_vecs.bin")
    resave_parts("gen/index_meta.json")
    resave_parts("gen/index_vecs.bin")

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
            vectors=np.asarray([process_vector(v, mode="lib", skip_quant=True) for v in vectors]),
            progress=progress,
        )

    def perturb_vector(
        vector: npt.NDArray[np.float32], scale: float = validate_perturb_scale
    ) -> npt.NDArray[np.float32]:
        noise = np.random.normal(0, scale, vector.shape).astype(np.float32)
        noise = noise / np.linalg.norm(noise)
        return vector + noise * scale

    # for each vector, perturb it and check if it matches
    print("Validating index...")
    with tqdm(total=len(vectors)) as pbar:
        count, correct = 0, 0
        for uid, vector in zip(uuids, vectors):
            results = index.search(process_vector(perturb_vector(vector), skip_quant=True), 1)
            count += 1
            correct += uuid_to_int(uid) in results.keys
            pbar.update()
            pbar.set_postfix_str(f"accuracy: {correct / count:.2%}")


if __name__ == "__main__":
    main()
