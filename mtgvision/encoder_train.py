import argparse
import sys
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Tuple, TypedDict
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
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
)

from mtgdata import ScryfallBulkType, ScryfallImageType
import mtgvision.models.convnextv2ae as cnv2ae
from mtgvision.encoder_datasets import IlsvrcImages, SyntheticBgFgMtgImages
from mtgvision.util.random import seed_all


def _cnv2ae_fn(fn):
    return lambda hw, **k: fn(
        image_wh=hw[::-1],
        z_size=768,
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
    x: np.ndarray
    y: np.ndarray
    x2: np.ndarray


class BatchHintTensor(TypedDict, total=False):
    x: torch.Tensor
    y: torch.Tensor
    x2: torch.Tensor


class RanMtgEncDecDataset(IterableDataset):
    def __init__(
        self,
        default_batch_size: int,
        predownload: bool = False,
        paired: bool = False,  # for contrastive loss, two random aug of same cards
        targets: bool = False,
        x_size_hw: Tuple[int, int] = (192, 128),
        y_size_hw: Tuple[int, int] = (192, 128),
        half_upsidedown: bool = False,
    ):
        assert default_batch_size >= 0
        self.default_batch_size = default_batch_size
        self.paired = paired
        self.targets = targets
        self.x_size_hw = x_size_hw
        self.y_size_hw = y_size_hw
        self.mtg = SyntheticBgFgMtgImages(img_type="small", predownload=predownload)
        self.ilsvrc = IlsvrcImages()
        self.half_upsidedown = half_upsidedown

    def __iter__(self):
        while True:
            if self.default_batch_size == 0:
                yield self.random_tensor()
            else:
                yield self.random_tensor_batch()

    def get_img_by_id(self, id_: str) -> BatchHintNumpy:
        x = self.mtg.get_image_by_id(id_)
        y = self.ilsvrc.ran()
        batch = self.make_image_batch([(x, y)])
        return {k: v[0] for k, v in batch.items()}

    def random_img(self) -> BatchHintNumpy:
        batch = self.random_image_batch(1)
        return {k: v[0] for k, v in batch.items()}

    def random_tensor(self) -> BatchHintTensor:
        batch = self.random_tensor_batch(1)  # already floats
        return {k: v[0] for k, v in batch.items()}

    def random_image_batch(self, n: Optional[int] = None) -> BatchHintNumpy:
        if n is None:
            n = self.default_batch_size
        # get random images
        img_pairs = [(self.mtg.ran(), self.ilsvrc.ran()) for _ in range(n)]
        return self.make_image_batch(img_pairs)

    def random_tensor_batch(self, n: Optional[int] = None) -> BatchHintTensor:
        batch = self.random_image_batch(n)
        return {k: K.image_to_tensor(v) for k, v in batch.items()}

    def make_image_batch(
        self, img_pairs: list[tuple[np.ndarray, np.ndarray]]
    ) -> BatchHintNumpy:
        # generate random samples
        xs0, xs1, ys = [], [], []
        for card, bg0 in img_pairs:
            _, bg1 = random.choice(img_pairs)
            if self.targets:
                ys.append(
                    SyntheticBgFgMtgImages.make_cropped(card, size_hw=self.y_size_hw)
                )
            xs0.append(
                SyntheticBgFgMtgImages.make_virtual(
                    card,
                    bg0,
                    size_hw=self.x_size_hw,
                    half_upsidedown=self.half_upsidedown,
                )
            )
            if self.paired:
                xs1.append(
                    SyntheticBgFgMtgImages.make_virtual(
                        card,
                        bg1,
                        size_hw=self.x_size_hw,
                        half_upsidedown=self.half_upsidedown,
                    )
                )
        # stack
        return {
            "x": np.stack(xs0, axis=0),
            **({"y": np.stack(ys, axis=0)} if self.targets else {}),
            **({"x2": np.stack(xs1, axis=0)} if self.paired else {}),
        }

    def set_batch_size(self, batch_size):
        self.default_batch_size = batch_size

    @classmethod
    def from_hparams(cls, hparams: "Config"):
        return cls(
            default_batch_size=hparams.batch_size,
            predownload=hparams.force_download,
            paired=hparams.loss_contrastive is not None,
            targets=hparams.loss_recon is not None,
            x_size_hw=hparams.x_size_hw,
            y_size_hw=hparams.y_size_hw,
            half_upsidedown=hparams.half_upsidedown,
        )


# ========================================================================= #
# Lightning Mtg Model & Optimisation                                        #
# ========================================================================= #


class MtgVisionEncoder(pl.LightningModule):
    hparams: "Config"
    model: "cnv2ae.ConvNeXtV2Ae"

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
        )
        self.model = model

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

    def training_step(self, batch, batch_idx):
        logs, loss = {}, 0
        # recon loss
        if self.hparams.loss_recon is None:
            if not self.hparams.loss_contrastive_batched:
                z = self.encode(batch["x"])
        else:
            z, y_recon = self.forward(batch["x"])
            y_recon = torch.clamp(y_recon, -0.25, 1.25)  # help with gradient explosion
            # recon loss
            recon_loss_fn = self._get_loss_recon()
            loss_recon = recon_loss_fn(y_recon, batch["y"])
            loss_recon *= self.hparams.scale_loss_recon
            loss += loss_recon
            logs["loss_recon"] = loss_recon

        # contrastive loss
        if self.hparams.loss_contrastive is not None:
            if not self.hparams.loss_contrastive_batched:
                z2 = self.encode(batch["x2"])
            else:
                _zs = self.encode(torch.concatenate([batch["x"], batch["x2"]], dim=0))
                z, z2 = _zs[: len(_zs) // 2], _zs[len(_zs) // 2 :]
            # shape (B, C, H, W) --> (B, C*H*W)
            z_flat = z.reshape(z.size(0), -1)
            z2_flat = z2.reshape(z2.size(0), -1)
            # normalize, may help with gradient explosion?
            z_flat = F.normalize(z_flat, dim=1)
            z2_flat = F.normalize(z2_flat, dim=1)
            # self-supervised loss
            assert self.hparams.loss_contrastive == "ntxent"
            loss_func = SelfSupervisedLoss(NTXentLoss(temperature=0.07), symmetric=True)
            loss_cont = loss_func(z_flat, z2_flat) * self.hparams.scale_loss_contrastive
            # scale
            loss += loss_cont
            logs["loss_contrastive"] = loss_cont

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
    def __init__(self, vis_batch, log_every_n_steps=1000):
        self.vis_batch = vis_batch
        self.log_every_n_steps = log_every_n_steps
        self.last_steps = -(log_every_n_steps * 10)
        self._first_log = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_steps = trainer.global_step
        if current_steps - self.last_steps >= self.log_every_n_steps:
            self.log_images(pl_module, self.vis_batch)
            self.last_steps = current_steps

    @staticmethod
    def join_images_into_row(images, padding=5):
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

    def log_images(self, model, vis_batch_np):
        print("Logging images...")
        if model.hparams.loss_recon is None:
            print("No reconstruction loss, skipping image logging.")
            return
        logs = {}
        model.eval()
        with torch.no_grad():
            x_np = np.stack([batch["x"] for batch in vis_batch_np], axis=0)
            y_np = np.stack([batch["y"] for batch in vis_batch_np], axis=0)
            x = torch.from_numpy(x_np).float().permute(0, 3, 1, 2).to(model.device)
            _, y = model(x)
            mout_np = []
            for out in [y]:
                mout_np.append(np.clip(K.tensor_to_image(out), 0, 1))
            # log images
            if self._first_log:
                logs["images_x"] = wandb.Image(
                    self.join_images_into_row(x_np), caption="Input"
                )
            if self._first_log:
                logs["images_y"] = wandb.Image(
                    self.join_images_into_row(y_np), caption="Target"
                )
            for i, out_np in enumerate(mout_np):
                name = "images_out" if i == 0 else f"images_out_{i + 1}"
                logs[name] = wandb.Image(
                    self.join_images_into_row(out_np), caption=f"Output {i}"
                )
        # stop logging after the first time
        self._first_log = False
        wandb.log(logs)


# ========================================================================= #
# Training Entrypoint                                                       #
# ========================================================================= #


def get_test_images(
    train_dataset: RanMtgEncDecDataset,
    seed: int = None,
):
    if seed is not None:
        seed_all(seed)

    # Initial batch for visualization
    vis_batch = [
        # https://scryfall.com/card/e02/35/rancor
        train_dataset.get_img_by_id("38e281ab-3437-4a2c-a668-9a148bc3eaf7"),
        # https://scryfall.com/card/2x2/156/rancor
        train_dataset.get_img_by_id("86d6b411-4a31-4bfc-8dd6-e19f553bb29b"),
        # https://scryfall.com/card/bro/238b/urza-planeswalker
        train_dataset.get_img_by_id("40a01679-3224-427e-bd1d-b797b0ab68b7"),
        # https://scryfall.com/card/ugl/70/blacker-lotus
        train_dataset.get_img_by_id("4a2e428c-dd25-484c-bbc8-2d6ce10ef42c"),
        # https://scryfall.com/card/zen/249/forest
        train_dataset.get_img_by_id("341b05e6-93bb-4071-b8c6-1644f56e026d"),
        # https://scryfall.com/card/mh3/132/ral-and-the-implicit-maze
        train_dataset.get_img_by_id("ebadb7dc-69a4-43c9-a2f8-d846b231c71c"),
    ]

    return vis_batch


def train(config: "Config"):
    seed_all(config.seed)

    # Initialize model
    data_module = MtgDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_dataset=RanMtgEncDecDataset.from_hparams(config),
    )

    # Initial batch for visualization
    vis_batch = get_test_images(data_module.train_dataset, seed=config.seed)

    # Initialize wandb
    parts = [
        (config.prefix, config.prefix),
        (True, config.model_name),
        (config.loss_recon, config.loss_recon),
        (config.loss_contrastive, config.loss_contrastive),
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
            ImageLoggingCallback(vis_batch, log_every_n_steps=config.log_every_n_steps),
            ModelCheckpoint(
                monitor="loss_recon",
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
    # model
    model_name: str = "cnvnxt2ae_base_12"
    x_size_hw: tuple[int, int] = (192, 128)
    y_size_hw: tuple[int, int] = (192, 128)
    # optimisation
    optimizer: Literal["adam", "radam"] = "adam"
    learning_rate: float = 5e-4
    weight_decay: float = 1e-9  # hurts performance if < 1e-7, e.g. 1e-5 is really bad
    batch_size: int = 12
    gradient_clip_val: float = 0.5
    accumulate_grad_batches: int = 4
    # loss
    loss_recon: Optional[str] = "ssim5+l1"  # 'ssim5+l1'
    loss_contrastive: Optional[str] = "ntxent"  # ntxent
    loss_contrastive_batched: bool = False
    scale_loss_recon: float = 1
    scale_loss_contrastive: float = 1
    # trainer
    compile: bool = False
    max_steps: int = 10_000_000
    num_workers: int = 6
    # logging
    prefix: Optional[str] = None
    checkpoint: Optional[str] = None
    log_every_n_steps: int = 5000
    ckpt_every_n_steps: int = 2500
    # needed if model architecture changes or optimizer changes
    skip_first_optimizer_load_state: bool = True


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == "__main__":
    sys.argv.extend(
        [
            "--prefix=cnxt2",
            # "--checkpoint=/home/nmichlo/workspace/mtg/mtg-vision/mtgvision_encoder/6__dvea3b14/checkpoints/epoch=0-step=67500.ckpt",
            # "--checkpoint=/home/nmichlo/workspace/mtg/mtg-vision/mtgvision_encoder/6.2__o0yxl20m/checkpoints/epoch=0-step=125000.ckpt",
            "--checkpoint=/home/nmichlo/workspace/mtg/mtg-vision/mtgvision_encoder/6.2__5u5qqmvz/checkpoints/epoch=0-step=217500.ckpt",
            # "--checkpoint=/home/nmichlo/workspace/mtg/mtg-vision/mtgvision_encoder/6.3__81rawzyz/checkpoints/epoch=0-step=365000.ckpt",
        ]
    )
    _cli()
