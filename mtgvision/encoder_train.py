"""
Train an encoder (and decoder because its pretty) in a contrastive learning style
to embed the same images with different distortions into the same space.

This is effectively facial recognition techniques.
"""

import argparse
import sys
import uuid
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, TypedDict
import matplotlib.pyplot as plt
import pydantic
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import random
import wandb
import kornia as K
import pytorch_lightning as pl
import pytorch_metric_learning.losses as mll
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
)

from mtgdata import ScryfallBulkType, ScryfallImageType
import mtgvision.models.convnextv2ae as cnv2ae
from mtgdata.scryfall import ScryfallCardFace
from mtgvision.encoder_datasets import IlsvrcImages, SyntheticBgFgMtgImages
from mtgvision.util.image import img_clip
from mtgvision.util.random import seed_all


Z_SIZE = 768


def _cnv2ae_fn(fn):
    return lambda hw, **k: fn(
        image_wh=hw[::-1],
        z_size=Z_SIZE,
        **k,
    )


_MODELS = {
    "cnvnxt2ae_atto": _cnv2ae_fn(cnv2ae.convnextv2_atto),
    "cnvnxt2ae_femto": _cnv2ae_fn(cnv2ae.convnextv2_femto),
    "cnvnxt2ae_pico": _cnv2ae_fn(cnv2ae.convnextv2ae_pico),
    "cnvnxt2ae_nano": _cnv2ae_fn(cnv2ae.convnextv2ae_nano),
    "cnvnxt2ae_tiny": _cnv2ae_fn(cnv2ae.convnextv2ae_tiny),
    # same number of layers as tiny. but more capacity
    "cnvnxt2ae_tiny_9_128": _cnv2ae_fn(cnv2ae.convnextv2ae_tiny_9_128),
    "cnvnxt2ae_tiny_12_128": _cnv2ae_fn(cnv2ae.convnextv2ae_tiny_12_128),
    # same number of layers as tiny. but more capacity
    "cnvnxt2ae_base_9": _cnv2ae_fn(cnv2ae.convnextv2ae_base_9),
    "cnvnxt2ae_base_12": _cnv2ae_fn(cnv2ae.convnextv2ae_base_12),
    "cnvnxt2ae_base": _cnv2ae_fn(cnv2ae.convnextv2ae_base),
    "cnvnxt2ae_large": _cnv2ae_fn(cnv2ae.convnextv2ae_large),
    "cnvnxt2ae_huge": _cnv2ae_fn(cnv2ae.convnextv2ae_huge),
}

# ========================================================================= #
# Basic Mtg Dataset                                                         #
# ========================================================================= #


class BatchHintNumpy(TypedDict, total=False):
    y: np.ndarray[np.float32]  # [:, H, W, C]
    x: np.ndarray[np.float32]  # [:, H, W, C]
    x_labels: np.ndarray[int]  # [:, 0] ids, [:, 1] names, [:, 2] sets
    x2: np.ndarray[np.float32]  # [:, H, W, C]
    x2_labels: np.ndarray[int]  # [:, 0] ids, [:, 1] names, [:, 2] sets


class BatchHintTensor(TypedDict, total=False):
    y: torch.Tensor  # [:, C, H, W]
    x: torch.Tensor  # [:, C, H, W]
    x_labels: torch.Tensor  # [:, 0] ids, [:, 1] names, [:, 2] sets
    x2: torch.Tensor  # [:, C, H, W]
    x2_labels: torch.Tensor  # [:, 0] ids, [:, 1] names, [:, 2] sets


class RanMtgEncDecDataset(IterableDataset):
    def __init__(
        self,
        default_batch_size: int,
        *,
        predownload: bool = False,
        paired: bool,  # for contrastive loss, two random aug of same cards
        targets: bool,
        x_size_hw: Tuple[int, int] = (192, 128),
        y_size_hw: Tuple[int, int] = (192, 128),
        half_upsidedown: bool,
        target_is_input_prob: float,
        similar_neg_prob: float,
        check_data: bool,
    ):
        assert default_batch_size > 0
        self.default_batch_size = default_batch_size
        self.paired = paired
        self.targets = targets
        self.x_size_hw = x_size_hw
        self.y_size_hw = y_size_hw
        self.mtg = SyntheticBgFgMtgImages(img_type="small", predownload=predownload)
        self.ilsvrc = IlsvrcImages()
        self.half_upsidedown = half_upsidedown
        self.target_is_input_prob = target_is_input_prob
        self.similar_neg_prob = similar_neg_prob
        self.check_data = check_data

    def __iter__(self):
        while True:
            yield self.random_tensor_batch()

    def image_batch_by_ids(
        self,
        ids: str | uuid.UUID | Sequence[str | uuid.UUID],
        *,
        force_target_input: bool = False,
        force_similar_neg: bool = False,
    ) -> BatchHintNumpy:
        if isinstance(ids, (str, uuid.UUID)):
            ids = [ids]
        t = 1.0 if force_target_input else (None if force_target_input is None else 0.0)
        n = 1.0 if force_similar_neg else (None if force_similar_neg is None else 0.0)
        imgs, lbls = self._make_image_batch(
            cards=[self.mtg.get_card_by_id(id_) for id_ in ids],
            bg_imgs=[self.ilsvrc.ran() for _ in ids],
            target_in_prob=t,
            similar_neg_prob=n,
        )
        return self._get_dict(imgs, lbls)

    def random_tensor_batch(self, n: int | None = None) -> BatchHintTensor:
        imgs, lbls = self._random_image_batch(n)
        return self._get_dict(imgs, lbls, K.image_to_tensor, torch.asarray)

    def random_image_batch(self, n: int | None = None) -> BatchHintNumpy:
        imgs, lbls = self._random_image_batch(n)
        return self._get_dict(imgs, lbls)

    def _random_image_batch(self, n: int | None = None):
        if n is None:
            n = self.default_batch_size
        # get random images
        return self._make_image_batch(
            cards=[self.mtg.ran_card() for _ in range(n)],
            bg_imgs=[self.ilsvrc.ran() for _ in range(n)],
        )

    @classmethod
    def _get_dict(
        cls,
        imgs: "dict[str, np.ndarray[np.float32]]",
        lbls: "dict[str, np.ndarray[int]]",
        apply_imgs=lambda v: v,
        apply_lbls=lambda v: v,
    ) -> BatchHintTensor | BatchHintNumpy:
        return {
            **{k: apply_imgs(v) for k, v in imgs.items()},
            **{k: apply_lbls(v) for k, v in lbls.items()},
        }

    def __make_y__(self, card_img: np.ndarray) -> np.ndarray:
        y = SyntheticBgFgMtgImages.make_cropped(card_img, size_hw=self.y_size_hw)
        return y

    def __make_x__(
        self, card_img: np.ndarray, bg: np.ndarray, target_is_input_prob: float
    ) -> np.ndarray:
        if random.random() < (target_is_input_prob or self.target_is_input_prob):
            x = SyntheticBgFgMtgImages.make_cropped(card_img, size_hw=self.x_size_hw)
        else:
            x = SyntheticBgFgMtgImages.make_virtual(
                card_img,
                bg,
                size_hw=self.x_size_hw,
                half_upsidedown=self.half_upsidedown,
            )
        return x

    def _make_image_batch(
        self,
        cards: list[ScryfallCardFace],
        bg_imgs: list[np.ndarray],
        *,
        target_in_prob: float = None,
        similar_neg_prob: float = None,
    ) -> "tuple[dict[str, np.ndarray[np.float32]], dict[str, np.ndarray[int]]]":
        assert len(cards) == len(bg_imgs), f"{len(cards)} != {len(bg_imgs)}"
        # A. generate random samples
        imgs, lbls = defaultdict(list), defaultdict(list)
        for i, (card, bg0) in enumerate(zip(cards, bg_imgs)):
            card_img = SyntheticBgFgMtgImages._load_card_image(card)
            # 1. make the target card, this is a slight crop of the actual original.
            if self.targets:
                imgs["y"].append(self.__make_y__(card_img))
            # 2. make the 1st input with card + random background and augment
            #    sometimes we swap this out with the target so that we learn everything
            #    correctly
            imgs["x"].append(self.__make_x__(card_img, bg0, target_in_prob))
            lbls["x_labels"].append(self.mtg.card_get_labels(card))
            # 3. make the contrastive learning example ... usually this is a positive
            #    example, but sometimes we randomly swap it out with a negative card
            #    that probably looks similar to the original card, but is not the same
            if self.paired:
                pair_card: "ScryfallCardFace" = card
                pair_card_im: "np.ndarray" = card_img
                # 3.a randomly swap out with nearby negative example
                if random.random() < (similar_neg_prob or self.similar_neg_prob):
                    _card = self.mtg.get_similar_card(card.id)
                    if _card is not None:
                        pair_card = _card
                        pair_card_im = SyntheticBgFgMtgImages._load_card_image(_card)
                # 3.b make 2nd input with SAME (if above not applied) card but different
                # augment and background for contrastive learning
                bg1 = random.choice(bg_imgs)
                imgs["x2"].append(self.__make_x__(pair_card_im, bg1, target_in_prob))
                lbls["x2_labels"].append(self.mtg.card_get_labels(pair_card))
        # B. stack
        imgs = {k: np.stack(v, axis=0) for k, v in imgs.items()}
        lbls = {k: np.stack(v, axis=0) for k, v in lbls.items()}
        return imgs, lbls

    def set_batch_size(self, batch_size):
        self.default_batch_size = batch_size

    @classmethod
    def from_hparams(cls, hparams: "Config"):
        return cls(
            default_batch_size=hparams.batch_size,
            predownload=hparams.force_download,
            paired=hparams.loss_contrastive is not None
            or hparams.loss_set_contrastive is not None,
            targets=hparams.loss_recon is not None,
            x_size_hw=hparams.x_size_hw,
            y_size_hw=hparams.y_size_hw,
            half_upsidedown=hparams.half_upsidedown,
            target_is_input_prob=hparams.target_is_input_prob,
            similar_neg_prob=hparams.similar_neg_prob,
            check_data=hparams.check_data,
        )


# ========================================================================= #
# Lightning Mtg Model & Optimisation                                        #
# ========================================================================= #


class MtgVisionEncoder(pl.LightningModule):
    hparams: "Config"
    model: "cnv2ae.ConvNeXtV2Ae"
    metric: "torch.nn.Module"
    metric_set: "torch.nn.Module"

    def __init__(self, config: dict):
        super().__init__()
        config = Config(**config).model_dump()  # add in missing defaults
        self.save_hyperparameters(config)

    def configure_model(self) -> None:
        model_fn = _MODELS[self.hparams.model_name]
        assert self.hparams.x_size_hw == self.hparams.y_size_hw, (
            f"different sizes: {self.hparams.x_size_hw} != {self.hparams.y_size_hw} are not yet supported."
        )
        model = model_fn(
            self.hparams.x_size_hw,
            encoder_enabled=True,
            decoder_enabled=self.hparams.loss_recon is not None,
            head_type=self.hparams.head_type,
        )
        self.model = model

        # set loss
        if self.hparams.loss_contrastive:
            self._metric = self._get_metric(self.hparams.loss_contrastive)
        if self.hparams.loss_set_contrastive:
            self._metric_set = self._get_metric(self.hparams.loss_set_contrastive)

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        self.configure_model()

    def _get_loss_recon(self):
        # generate
        if getattr(self, "_recon_loss", None) is not None:
            return self._recon_loss
        # make loss
        loss_fn = {
            "mse": F.mse_loss,
            "l1": F.l1_loss,
            "ssim5": K.losses.SSIMLoss(5),
            "ssim7": K.losses.SSIMLoss(7),
            "ssim9": K.losses.SSIMLoss(9),
            "ssim5+mse": lambda x, y: K.losses.ssim_loss(x, y, 5) * 0.5
            + F.mse_loss(x, y) * 0.5,
            "ssim5+l1": lambda x, y: K.losses.ssim_loss(x, y, 5) * 0.5
            + F.l1_loss(x, y) * 0.5,
            "ssim7+l1": lambda x, y: K.losses.ssim_loss(x, y, 7) * 0.5
            + F.l1_loss(x, y) * 0.5,
            "ms_ssim": K.losses.MS_SSIMLoss(),
        }[self.hparams.loss_recon]
        try:
            return loss_fn.to(self.device)
        except Exception:
            return loss_fn

    @classmethod
    def test_checkpoint(cls, path):
        model = MtgVisionEncoder.load_from_checkpoint(path, map_location="cpu")
        # Initialize model
        data_module = MtgDataModule(
            train_dataset=RanMtgEncDecDataset.from_hparams(model.hparams),
            num_workers=0,
            batch_size=1,
        )
        # Initial batch for visualization
        batch = data_module.train_dataset.random_tensor_batch(1)
        x = batch["x"].cpu()[0]
        y = batch["y"].cpu()[0]
        # feed forward
        _, ([outx], *_) = model(x[None])
        _, ([outy], *_) = model(y[None])
        # log images
        x = K.tensor_to_image(x)
        y = K.tensor_to_image(y)
        outx = K.tensor_to_image(outx)
        outy = K.tensor_to_image(outy)
        # show
        plt.imshow(x)
        plt.show()
        plt.imshow(outx)
        plt.show()
        plt.imshow(y)
        plt.show()
        plt.imshow(outy)
        plt.show()

    def forward(self, x):
        z, multi = self.model(x)
        return z, multi[0]

    def forward_img(self, img):
        assert img.ndim == 3
        assert img.shape[-1] == 3
        assert img.dtype == np.float32  # in range [0, 1]
        _, [y, *_] = self(K.image_to_tensor(img)[None, ...])
        return K.tensor_to_image(y)

    def encode(self, x):
        z, multi = self.model.encode(x)
        return z

    def decode(self, z):
        multi = self.model.decode(z)
        return multi[0]

    def _get_metric(self, name: str):
        # not good for qdrant search?
        if name == "ntxent":
            # needs very large batch sizes ~= 4096
            return mll.NTXentLoss(temperature=0.07)
        elif name == "triplet":
            # not circular
            return mll.TripletMarginLoss(margin=0.05)
        elif name == "triplet_smooth":
            # not circular, but probably better
            return mll.TripletMarginLoss(margin=0.05, smooth_loss=True)

        # Circular/normalized losses are better for qdrant cosine searches
        elif name == "arc_face":
            # This loss encourages embeddings of cards from the same set to be closer
            # than those from different sets, providing group-level guidance.
            return mll.ArcFaceLoss(
                num_classes=120000, embedding_size=Z_SIZE, margin=28.6, scale=64
            )
        elif name == "sub_center_arc_face":
            # This loss ensures that embeddings of distorted versions of
            # the same card are mapped closely together.
            return mll.SubCenterArcFaceLoss(
                num_classes=120000,
                embedding_size=Z_SIZE,
                margin=28.6,
                scale=64,
                sub_centers=3,
            )
        elif name == "sup_con":
            # seems to work well if not configured right?
            return mll.SupConLoss(temperature=0.1)
        elif name == "circle":
            # Face Recognition: m = 0.25, gamma = 256
            # Person Reidentification: m = 0.25, gamma = 128
            # Fine-grained Image Retrieval: m = 0.4, gamma = 80
            return mll.CircleLoss(m=0.25, gamma=256)
        else:
            raise KeyError(f"Unknown metric: {name}")

    def training_step(self, batch: BatchHintTensor, batch_idx: int):
        logs, loss = {}, 0

        # recon loss
        if not self.hparams.loss_recon:
            z = self.encode(batch["x"])
        else:
            z, y_recon = self.forward(batch["x"])
            y_recon = torch.clamp(y_recon, -0.25, 1.25)  # help with gradient explosion
            # recon loss
            recon_loss_fn = self._get_loss_recon()
            # input x size might have more elems than y, but y always corresponds to
            # starting elements in x, so slice array
            loss_recon = recon_loss_fn(y_recon[: len(batch["y"])], batch["y"])
            logs["loss_recon"] = loss_recon
            loss += loss_recon * self.hparams.scale_loss_recon

        # get z2
        z_all: "torch.Tensor | None" = None
        labels_all: "torch.Tensor | None" = None
        if self.hparams.loss_contrastive or self.hparams.loss_set_contrastive:
            _z2 = self.encode(batch["x2"])
            z_all = torch.cat([z, _z2], dim=0)
            # (N, 3) --> [:, 0] ids, [:, 1] names, [:, 2] sets
            labels_all = torch.cat([batch["x_labels"], batch["x2_labels"]], dim=0)

        # contrastive loss -- based on card IDs as labels
        if self.hparams.loss_contrastive:
            loss_cont = self._metric(z_all, labels_all[:, 0])
            logs["loss_metric"] = loss_cont
            loss += loss_cont * self.hparams.scale_loss_contrastive

        # contrastive set loss -- based on set codes as labels
        if self.hparams.loss_set_contrastive:
            loss_set_cont = self._metric_set(z_all, labels_all[:, 2])
            logs["loss_set_metric"] = loss_set_cont
            loss += loss_set_cont * self.hparams.scale_loss_set_contrastive

        # loss is required key
        logs["loss"] = loss
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True)

        # loss is required key
        return logs

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            opt = optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "radam":
            opt = optim.RAdam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "sgd":
            opt = optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                nesterov=True,
                momentum=0.9,
            )
        elif self.hparams.optimizer == "deepspeed_cpu_adam":
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            opt = DeepSpeedCPUAdam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        # reset load
        if self.hparams.skip_first_optimizer_load_state:
            _old_ = opt.load_state_dict

            def _skip_load_state_dict_(*args, **kwargs):
                print(
                    "Loading state dict... SKIPPED!, resetting load_state_dict to default."
                )
                opt.load_state_dict = _old_

            opt.load_state_dict = _skip_load_state_dict_

        # done!
        return opt


# ========================================================================= #
# Lightning Mtg Dataset                                                     #
# ========================================================================= #


class MtgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: RanMtgEncDecDataset,
        num_workers: int = 3,
        batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        if batch_size is not None:
            self.train_dataset.set_batch_size(batch_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,  # no auto collation
            shuffle=False,
            num_workers=self.num_workers,
        )


# ========================================================================= #
# Lightning Image Logging Callback                                          #
# ========================================================================= #


class ImageLoggingCallback(Callback):
    def __init__(self, vis_batches_np, log_every_n_steps=1000):
        self.vis_batches_np = vis_batches_np
        self.log_every_n_steps = log_every_n_steps
        self.last_steps = -(log_every_n_steps * 10)
        self._first_log = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_steps = trainer.global_step
        if current_steps - self.last_steps >= self.log_every_n_steps:
            self.log_images(pl_module)
            self.last_steps = current_steps

    @staticmethod
    def join_images_into_row(images: Sequence[np.ndarray], padding=5):
        const = 127
        if images[0].dtype in [np.float16, np.float32, np.float64]:
            const = 0.5
        images = [
            np.pad(
                image,
                [(padding, padding), (padding, padding), (0, 0)],
                mode="constant",
                constant_values=const,
            )
            for image in images
        ]
        return np.concatenate(images, axis=1)

    @staticmethod
    def _wandb_img(image: np.ndarray, caption: str):
        if image.dtype == np.uint8:
            pass
        elif image.dtype == np.float32:
            m, M = image.min(), image.max()
            if m < 0 or 1 < M:
                warnings.warn(
                    f"{caption} image must be in range [0, 1], but got [{m}, {M}], clipping"
                )
            image = img_clip(image)
        else:
            raise ValueError(f"{caption} image dtype unsupported: {image.dtype}")
        return wandb.Image(image, caption=caption)

    def log_images(self, model: MtgVisionEncoder):
        vis_batches_np: list[dict] = self.vis_batches_np
        logs = {}
        model.eval()

        def take_imgs(
            batches: list[dict],
            bkey: str,
            bidx: int,
        ) -> np.ndarray | None:
            if self._first_log and bkey in batches[0]:
                images = [batch[bkey][bidx] for batch in batches]
                images = np.stack(images, axis=0)
                return images
            return None

        def _log_images(images, logs_key: str, caption: str):
            if images is not None:
                image = self.join_images_into_row(images)
                logs[logs_key] = self._wandb_img(image, caption)

        def _forward_imgs(images: np.ndarray | None):
            if model.hparams.loss_recon and images is not None:
                inputs = torch.from_numpy(images).float().permute(0, 3, 1, 2)
                inputs = inputs.to(model.device)
                _, outputs = model(inputs)
                out_images = np.clip(K.tensor_to_image(outputs), 0, 1)
                return out_images
            return None

        with torch.no_grad():
            contrastive = model.hparams.loss_contrastive
            recon = model.hparams.loss_recon
            # INPUT/TARGETS
            y_np = take_imgs(vis_batches_np, "y", 0)
            x_np = take_imgs(vis_batches_np, "x", 0)
            x2_np = take_imgs(vis_batches_np, "x", -1) if contrastive else None
            # log images
            _log_images(y_np, "images_y", "Target")
            _log_images(x_np, "images_x", "Input")
            _log_images(x2_np, "images_x2", "Paired Inputs")
            # FORWARD
            if recon:
                out_y_np = _forward_imgs(y_np)
                out_x_np = _forward_imgs(x_np)
                out_x2_np = _forward_imgs(x2_np) if contrastive else None
                # log images
                _log_images(out_y_np, "images_out_y", "Target Output")
                _log_images(out_x_np, "images_out_x", "Input Output")
                _log_images(out_x2_np, "images_out_x2", "Paired Inputs Output")

        # stop logging after the first time
        self._first_log = False
        wandb.log(logs)


# ========================================================================= #
# Training Entrypoint                                                       #
# ========================================================================= #


def get_test_image_batches(
    train_dataset: RanMtgEncDecDataset,
    seed: int = None,
):
    if seed is not None:
        seed_all(seed)

    # Initial batch for visualization
    vis_batches = [
        # https://scryfall.com/card/e02/35/rancor
        train_dataset.image_batch_by_ids(
            "38e281ab-3437-4a2c-a668-9a148bc3eaf7",
            force_similar_neg=True,
            force_target_input=True,
        ),
        train_dataset.image_batch_by_ids(
            "38e281ab-3437-4a2c-a668-9a148bc3eaf7",
            force_similar_neg=False,
            force_target_input=False,
        ),
        # # https://scryfall.com/card/2x2/156/rancor
        # train_dataset.image_batch_by_ids("86d6b411-4a31-4bfc-8dd6-e19f553bb29b"),
        # https://scryfall.com/card/bro/238b/urza-planeswalker
        train_dataset.image_batch_by_ids("40a01679-3224-427e-bd1d-b797b0ab68b7"),
        # https://scryfall.com/card/ugl/70/blacker-lotus
        train_dataset.image_batch_by_ids("4a2e428c-dd25-484c-bbc8-2d6ce10ef42c"),
        # https://scryfall.com/card/zen/249/forest
        train_dataset.image_batch_by_ids("341b05e6-93bb-4071-b8c6-1644f56e026d"),
        # https://scryfall.com/card/mh3/132/ral-and-the-implicit-maze
        train_dataset.image_batch_by_ids("ebadb7dc-69a4-43c9-a2f8-d846b231c71c"),
    ]

    return vis_batches


def train(config: "Config"):
    seed_all(config.seed)

    # Initialize model
    data_module = MtgDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_dataset=RanMtgEncDecDataset.from_hparams(config),
    )

    # Initial batch for visualization
    vis_batches = get_test_image_batches(data_module.train_dataset, seed=config.seed)

    # Initialize wandb
    parts = [
        (config.prefix, config.prefix),
        (True, config.model_name),
        (True, config.head_type),
        (config.loss_recon, config.loss_recon),
        (config.loss_contrastive, config.loss_contrastive),
        (config.loss_set_contrastive, config.loss_set_contrastive),
        (config.learning_rate, f"lr={config.learning_rate}"),
        (config.batch_size, f"bs={config.batch_size}"),
    ]

    # choose device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # set seed again
    seed_all(config.seed)

    # Initialize model and compile
    model = MtgVisionEncoder(config.model_dump())
    if config.compile:
        # torch._dynamo.list_backends() # ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']
        model = torch.compile(
            model,
            **(
                {"backend": "aot_eager"}
                if device == "mps"
                else {"mode": "reduce-overhead"}
            ),
        )

    # logger
    wandb_logger = WandbLogger(
        name="_".join([v for k, v in parts if k]),
        project="mtgvision_encoder",
        config=config,
    )

    # Set up trainer with optimizations
    trainer = pl.Trainer(
        max_epochs=config.max_steps,
        logger=wandb_logger,
        callbacks=[
            ImageLoggingCallback(
                vis_batches, log_every_n_steps=config.log_every_n_steps
            ),
            ModelCheckpoint(
                monitor="loss",
                save_top_k=3,
                mode="min",
                every_n_train_steps=config.ckpt_every_n_steps,
            ),
            # StochasticWeightAveraging(swa_lrs=1e-2),
        ],
        accelerator=device,
        devices=1,
        precision=16 if device == "cuda" else 32,
        max_steps=config.max_steps,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        enable_checkpointing=True,
        default_root_dir=Path(__file__).parent / "lightning_logs",
        strategy="deepspeed_stage_2_offload"
        if config.optimizer == "deepspeed_cpu_adam"
        else "auto",
    )

    # allow model architecture to be changed
    model.strict_loading = False

    # Run training
    trainer.fit(
        model,
        data_module,
        ckpt_path=config.checkpoint,
    )

    # Save the final model checkpoint
    trainer.save_checkpoint("final_model.ckpt")

    # Convert to CoreML
    # | model.eval()
    # | example_input = torch.rand(1, 3, 192, 128).to(model.device)  # NCHW
    # | traced_model = torch.jit.trace(model, example_input)
    # | mlmodel = ct.convert(
    # |     traced_model,
    # |     inputs=[ct.ImageType(name="input", shape=example_input.shape)],
    # | )
    # | mlmodel.save("model.mlmodel")
    # | print("Model trained and converted to CoreML. Saved as 'model.mlmodel'.")


# ========================================================================= #
# CLI                                                                       #
# ========================================================================= #


def _cli():
    # generate parser
    _BOOLS = []
    parser = argparse.ArgumentParser()
    for name, field in Config().model_fields.items():
        if field.annotation is bool:
            _BOOLS.append(name)
            print("BOOL:", name, field.default)
            parser.add_argument(
                f"--{name}".replace("_", "-"),
                type=str,
                default="yes"
                if field.default
                else (None if field.default is None else "no"),
                help=str(field.description) + " (y/n/yes/no/true/false)",
            )
        else:
            parser.add_argument(
                f"--{name}".replace("_", "-"),
                type=_CONF_TYPE_OVERRIDES.get(name, field.annotation),
                default=field.default,
                help=field.description,
            )

    # parse args
    args = parser.parse_args()
    for name in _BOOLS:
        val = getattr(args, name)
        if val is not None:
            setattr(args, name, val.lower() in ("y", "yes", "true"))

    # get config
    print("ARGS:", args)
    config = Config(**vars(args))

    # update losses
    if config.loss_contrastive in ("none", "no") or config.scale_loss_contrastive <= 0:
        print("No contrastive loss.")
        config.loss_contrastive = None
    if config.loss_recon in ("none", "no") or config.scale_loss_recon <= 0:
        print("No reconstruction loss.")
        config.loss_recon = None

    # train
    print("CONFIG:", config.model_dump_json(indent=2))
    train(config)


_CONF_TYPE_OVERRIDES = {
    "checkpoint": str,
    "loss_recon": str,
    "loss_contrastive": str,
    "loss_set_contrastive": str,
    "prefix": str,
    "optimizer": str,
    "dec_skip_connections": str,
}


class Config(pydantic.BaseModel):
    seed: int = 42
    # dataset
    img_type: ScryfallImageType = ScryfallImageType.small
    bulk_type: ScryfallBulkType = ScryfallBulkType.default_cards
    force_download: bool = False
    half_upsidedown: bool = False
    target_is_input_prob: float = 0.05
    similar_neg_prob: float = 0.2
    check_data: bool = False  # ensure data is in the right range!
    # model
    model_name: str = "cnvnxt2ae_nano"
    head_type: str = "conv+linear"  # conv, conv+linear, pool+linear
    x_size_hw: tuple[int, int] = (192, 128)
    y_size_hw: tuple[int, int] = (192, 128)
    # optimisation
    optimizer: Literal["adam", "radam"] = "radam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-7  # hurts performance if < 1e-7, e.g. 1e-5 is really bad
    batch_size: int = (
        64  # effectively doubled with loss_contrastive / loss_set_contrastive
    )
    gradient_clip_val: float = 0.5
    accumulate_grad_batches: int = 1
    # loss
    loss_recon: Optional[str] = None  # 'ssim5+l1'
    # ntxent, triplet, triplet_smooth, arc_face, sub_center_arc_face, sup_con, circle
    loss_contrastive: Optional[str] = "circle"  # sub_center_arc_face
    loss_set_contrastive: Optional[str] = "circle"  # arc_face
    scale_loss_recon: float = 1
    scale_loss_contrastive: float = 1
    scale_loss_set_contrastive: float = 0.05
    # trainer
    compile: bool = False
    max_steps: int = 100001
    num_workers: int = 6
    # logging
    prefix: Optional[str] = None
    checkpoint: Optional[str] = None
    log_every_n_steps: int = 2500
    ckpt_every_n_steps: int = 2500
    # needed if model architecture changes or optimizer changes
    skip_first_optimizer_load_state: bool = False


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == "__main__":
    sys.argv.extend(
        [
            "--prefix=2_cnxt2",
            # "--checkpoint=/home/nmichlo/workspace/mtg/mtg-vision/mtgvision_encoder/6__dvea3b14/checkpoints/epoch=0-step=67500.ckpt",
            # "--checkpoint=/home/nmichlo/workspace/mtg/mtg-vision/mtgvision_encoder/6.2__o0yxl20m/checkpoints/epoch=0-step=125000.ckpt",
            # "--checkpoint=/home/nmichlo/workspace/mtg/mtg-vision/mtgvision_encoder/6.2__5u5qqmvz/checkpoints/epoch=0-step=217500.ckpt",
            # "--checkpoint=/home/nmichlo/workspace/mtg/mtg-vision/mtgvision_encoder/6.3__81rawzyz/checkpoints/epoch=0-step=365000.ckpt",
        ]
    )
    _cli()
