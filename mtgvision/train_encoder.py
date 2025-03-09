import argparse
import functools
import sys
from pathlib import Path
from typing import Any, IO, Literal, Mapping, Optional, Self, Tuple, Union

import matplotlib.pyplot as plt
import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import random
import wandb
import kornia as K
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
)


from mtgdata import ScryfallBulkType, ScryfallImageType
import mtgvision.models.convnextv2ae as cnv2ae
from mtgvision.models.new_arch2 import Ae2
from mtgvision.datasets import IlsvrcImages, MtgImages
from mtgvision.util.random import GLOBAL_RAN

_MODELS = {
    # Ae1.__name__.lower(): functools.partial(Ae1.create_model, stn=False),

    # Ae1b.__name__.lower(): functools.partial(Ae1b.create_model, stn=False),
    # Ae1b.__name__.lower() + '_stn': functools.partial(Ae1b.create_model, stn=True),

    Ae2.__name__.lower() + 'xl': functools.partial(Ae2.create_model_heavy, stn=False),
    Ae2.__name__.lower() + 'xl_stn': functools.partial(Ae2.create_model_heavy, stn=True),

    Ae2.__name__.lower() + 'l': functools.partial(Ae2.create_model_heavy, stn=False),
    Ae2.__name__.lower() + 'l_stn': functools.partial(Ae2.create_model_heavy, stn=True),

    Ae2.__name__.lower() + 'm': functools.partial(Ae2.create_model_medium, stn=False),
    Ae2.__name__.lower() + 'm_stn': functools.partial(Ae2.create_model_medium, stn=True),

    Ae2.__name__.lower() + 's': functools.partial(Ae2.create_model_small, stn=False),
    Ae2.__name__.lower() + 's_stn': functools.partial(Ae2.create_model_small, stn=True),

    "cnvnxt2ae_atto": lambda w, h, **k: cnv2ae.convnextv2_atto(image_wh=(w[2], w[1]), z_size=768),
    "cnvnxt2ae_femto": lambda w, h, **k: cnv2ae.convnextv2_femto(image_wh=(w[2], w[1]), z_size=768),
    "cnvnxt2ae_pico": lambda w, h, **k: cnv2ae.convnextv2ae_pico(image_wh=(w[2], w[1]), z_size=768),
    "cnvnxt2ae_nano": lambda w, h, **k: cnv2ae.convnextv2ae_nano(image_wh=(w[2], w[1]), z_size=768),
    "cnvnxt2ae_tiny": lambda w, h, **k: cnv2ae.convnextv2ae_tiny(image_wh=(w[2], w[1]), z_size=768),
    "cnvnxt2ae_base_9": lambda w, h, **k: cnv2ae.convnextv2ae_base_9(image_wh=(w[2], w[1]), z_size=768),  # same number of layers as tiny. but more capacity
    "cnvnxt2ae_base_12": lambda w, h, **k: cnv2ae.convnextv2ae_base_12(image_wh=(w[2], w[1]), z_size=768),
    "cnvnxt2ae_base": lambda w, h, **k: cnv2ae.convnextv2ae_base(image_wh=(w[2], w[1]), z_size=768),
    "cnvnxt2ae_large": lambda w, h, **k: cnv2ae.convnextv2ae_large(image_wh=(w[2], w[1]), z_size=768),
    "cnvnxt2ae_huge": lambda w, h, **k: cnv2ae.convnextv2ae_huge(image_wh=(w[2], w[1]), z_size=768),
}


class RanMtgEncDecDataset(IterableDataset):

    def __init__(
        self,
        default_batch_size: int,
        predownload: bool = False,
        paired: bool = False,  # generate second set of images
        size: Tuple[int, int] = (192, 128),
    ):
        assert default_batch_size >= 0
        self.default_batch_size = default_batch_size
        self.paired = paired
        self.size = size
        self.mtg = MtgImages(img_type='small', predownload=predownload)
        self.ilsvrc = IlsvrcImages()

    def __iter__(self):
        while True:
            if self.default_batch_size == 0:
                yield self.random_tensor()
            else:
                yield self.random_tensor_batch()

    def random_img(self):
        batch = self.random_image_batch(1)
        return {k: v[0] for k, v in batch.items()}

    def random_tensor(self):
        batch = self.random_tensor_batch(1)  # already floats
        return {k: v[0] for k, v in batch.items()}

    def random_image_batch(self, n: Optional[int] = None):
        if n is None:
            n = self.default_batch_size
        # get random images
        cards_bgs = [(self.mtg.ran(), self.ilsvrc.ran()) for _ in range(n)]
        # generate random samples
        xs0, xs1, ys = [], [], []
        for card, bg0 in cards_bgs:
            _, bg1 = random.choice(cards_bgs)
            ys.append(MtgImages.make_cropped(card, size=self.size))
            xs0.append(MtgImages.make_virtual(card, bg0, size=self.size))
            if self.paired:
                xs1.append(MtgImages.make_virtual(card, bg1, size=self.size))
        # stack
        return {
            "x": np.stack(xs0, axis=0),
            "y": np.stack(ys, axis=0),
            **({"x2": np.stack(xs1, axis=0)} if self.paired else {}),
        }

    def random_tensor_batch(self, n: Optional[int] = None):
        batch = self.random_image_batch(n)
        return {k: K.image_to_tensor(v) for k, v in batch.items()}


# Define the Lightning Module
class MtgVisionEncoder(pl.LightningModule):

    hparams: "Config"
    model: "nn.Module"

    def __init__(self, config: dict):
        super().__init__()
        config = Config(**config).model_dump()  # add in missing defaults
        self.save_hyperparameters(config)

    def configure_model(self) -> None:
        if self.hparams.model_name in _MODELS:
            model_fn = _MODELS[self.hparams.model_name]
            model = model_fn(self.hparams.x_size, self.hparams.y_size, multiscale=self.hparams.multiscale)
        else:
            name = self.hparams.model_name
            if name.startswith('ae2_'):
                name = name[4:]
            else:
                raise ValueError(f"Unknown model name: {name}")
            stn = False
            if name.endswith('_stn'):
                name = name[:-4]
                stn = True
            model = Ae2.create_model_verbose(
                self.hparams.x_size,
                self.hparams.y_size,
                model=name,
                multiscale=self.hparams.multiscale,
                stn=stn,
                dec_skip_connections=self.hparams.dec_skip_connections,
            )
        self.model = model

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        self.configure_model()

    @classmethod
    def _get_loss(cls, hparams):
        if hparams.loss == 'mse':
            return F.mse_loss, None
        elif hparams.loss == 'mse+edge':
            return F.mse_loss, K.filters.sobel
        elif hparams.loss == 'l1':
            return F.l1_loss, None
        elif hparams.loss == 'l1+edge':
            return F.l1_loss, K.filters.sobel
        # ssim
        elif hparams.loss == 'ssim5':
            return K.losses.SSIMLoss(5), None
        elif hparams.loss == 'ssim7':
            return K.losses.SSIMLoss(7), None
        elif hparams.loss == 'ssim9':
            return K.losses.SSIMLoss(9), None
        # ssim+mse
        elif hparams.loss == 'ssim5+mse':
            def loss(x, y):
                return K.losses.ssim_loss(x, y, 5) * 0.5 + F.mse_loss(x, y) * 0.5
            return loss, None
        else:
            raise ValueError(f"Unknown loss: {hparams.loss}")

    @classmethod
    def test_checkpoint(cls, path):
        model = MtgVisionEncoder.load_from_checkpoint(path, map_location="cpu")
        # Initialize model
        data_module = MtgDataModule(batch_size=1)
        # Initial batch for visualization
        batch = data_module.train_dataset.random_tensor_batch(1)
        x = batch['x'].cpu()[0]
        y = batch['y'].cpu()[0]
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
        if self.hparams.norm_io:
            x = x * 2 - 1
        z, multi_out = self.model(x)
        if self.hparams.norm_io:
            multi_out = [(out + 1) / 2 for out in multi_out]
        return z, multi_out

    def encode(self, x):
        if self.hparams.norm_io:
            x = x * 2 - 1
        z, multi = self.model.encode(x)
        return z, multi

    @classmethod
    def _recon_loss(
        cls,
        hparams,
        y_target,
        y_recon,
        y_recon_multi=None,
    ):
        loss_fn, loss_filter = cls._get_loss(hparams)
        # loss
        loss = 0
        # x - reconstruction loss
        loss_recon = loss_fn(y_recon, y_target)
        loss += loss_recon * hparams.scale_loss_recon
        # x - extra loss
        loss_recon_extra = 0
        if loss_filter is not None:
            loss_recon_extra = loss_fn(loss_filter(y_recon), loss_filter(y_target))
            loss += loss_recon_extra * hparams.scale_loss_recon_extra
        # x - multiscale loss
        loss_multiscale = 0
        if hparams.multiscale and y_recon_multi:
            for y_mid_recon in y_recon_multi:
                loss_multiscale += loss_fn(
                    F.interpolate(y_mid_recon, size=y_target.size()[2:], mode='bilinear', align_corners=False),
                    y_target
                )
            loss_multiscale /= len(y_recon_multi)
            loss += loss_multiscale * hparams.scale_loss_multiscale
        return loss, {
            "loss_recon": loss_recon,
            "loss_recon_edges": loss_recon_extra,
            "loss_multiscale_recon": loss_multiscale,
        }

    def training_step(self, batch, batch_idx):
        # recon loss
        z, y_recons = self.forward(batch['x'])
        loss, logs = self._recon_loss(
            hparams=self.hparams,
            y_target=batch['y'],
            y_recon=y_recons[0],
            y_recon_multi=y_recons[1:],
        )
        # paired loss
        if 'x2' in batch:
            z2, _ = self.encode(batch['x2'])
            loss_paired = F.mse_loss(z, z2) * self.hparams.scale_loss_paired
            loss += loss_paired
            logs["loss_paired"] = loss_paired

        # loss is required key
        logs["loss"] = loss
        logs["train_loss"] = loss

        # Cycle consistency losses
        # | loss_cyclic = 0
        # | if self.hparams.cyclic:
        # |     z2 = self.model.encode(out.detach())
        # |     loss_cyclic = loss_fn(z2, z)
        # |     loss += loss_cyclic * self.hparams.scale_loss_cyclic

        # Target consistency loss
        # | loss_target = 0
        # | if self.hparams.target_consistency:
        # |     z2 = self.model.encode(y)
        # |     loss_target = loss_fn(z2, z)
        # |     loss += loss_target * self.hparams.scale_loss_target

        # Cycle WITH target consistency loss
        # | loss_cycle_target = 0
        # | if self.hparams.cycle_with_target > 0:
        # |     assert self.hparams.cycle_with_target % 2 == 0
        # |     n = self.hparams.cycle_with_target // 2
        # |     # like cycle and target, but only take first n/2 elements
        # |     _z = torch.concatenate([z[:n], z[:n]], dim=0)
        # |     _x = torch.concatenate([x[:n], out[:n]], dim=0)
        # |     loss_cycle_target += loss_fn(self.model.encode(_x), _z)
        # |     loss += loss_cycle_target * self.hparams.scale_loss_cycle_target

        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True)
        # loss is required key
        return logs

    def configure_optimizers(self):
        if self.hparams.optimizer == 'adam':
            opt = optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'radam':
            opt = optim.RAdam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'sgd':
            opt = optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                nesterov=True,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")

        # reset load
        if self.hparams.skip_first_optimizer_load_state:
            _old_ = opt.load_state_dict

            def _skip_load_state_dict_(*args, **kwargs):
                print("Loading state dict... SKIPPED!, resetting load_state_dict to default.")
                opt.load_state_dict = _old_

            opt.load_state_dict = _skip_load_state_dict_

        # done!
        return opt



class MtgDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int,
        paired: bool = False,
        num_workers: int = 3,
        predownload: bool = False,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.train_dataset = RanMtgEncDecDataset(
            predownload=predownload,
            default_batch_size=batch_size,
            paired=paired,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,  # no auto collation
            shuffle=False,
            num_workers=self.num_workers,
        )


class ImageLoggingCallback(Callback):

    def __init__(self, vis_batch, log_every_n_steps=1000):
        self.vis_batch = vis_batch
        self.log_every_n_steps = log_every_n_steps
        self.last_steps = -(log_every_n_steps*10)
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
            np.pad(image, [(padding, padding), (padding, padding), (0, 0)], mode='constant', constant_values=const)
            for image in images
        ]
        return np.concatenate(images, axis=1)

    def log_images(self, model, vis_batch_np):
        print("Logging images...")
        logs = {}
        model.eval()
        with torch.no_grad():
            x_np = np.stack([batch['x'] for batch in vis_batch_np], axis=0)
            y_np = np.stack([batch['y'] for batch in vis_batch_np], axis=0)
            x = torch.from_numpy(x_np).float().permute(0, 3, 1, 2).to(model.device)
            _, multiscale = model(x)
            mout_np = []
            for out in multiscale:
                mout_np.append(np.clip(K.tensor_to_image(out), 0, 1))
            # log images
            if self._first_log:
                logs["images_x"] = wandb.Image(self.join_images_into_row(x_np), caption="Input")
            if self._first_log:
                logs["images_y"] = wandb.Image(self.join_images_into_row(y_np), caption="Target")
            for i, out_np in enumerate(mout_np):
                name = "images_out" if i == 0 else f"images_out_{i+1}"
                logs[name] = wandb.Image(self.join_images_into_row(out_np), caption=f"Output {i}")
        # stop logging after the first time
        self._first_log = False
        wandb.log(logs)


def train(config: "Config"):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    GLOBAL_RAN.reset(config.seed)

    # Initialize model
    data_module = MtgDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        predownload=config.force_download,
        paired=config.paired,
    )

    # Initial batch for visualization
    vis_batch = [
        data_module.train_dataset.random_img(),
        data_module.train_dataset.random_img(),
        data_module.train_dataset.random_img(),
        data_module.train_dataset.random_img(),
        data_module.train_dataset.random_img(),
    ]

    # Initialize wandb
    parts = [
        (config.prefix, config.prefix),
        (True, config.model_name),
        (config.paired, "pairs"),
        (config.norm_io, "norm"),
        (config.loss, config.loss),
        (config.learning_rate, f"lr={config.learning_rate}"),
        (config.batch_size, f"bs={config.batch_size}"),
        (config.multiscale, f"Lm={config.scale_loss_multiscale}"),
        # (config.cyclic, f"Lc={config.scale_loss_cyclic}"),
        # (config.target_consistency, f"Lt={config.scale_loss_target}"),
        # (config.cycle_with_target > 0, f"Lct{config.cycle_with_target}={config.scale_loss_cycle_target}"),
    ]

    # choose device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Initialize model and compile
    model = MtgVisionEncoder(config.model_dump())
    if config.compile:
        # torch._dynamo.list_backends() # ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']
        model = torch.compile(
            model,
            **({"backend": "aot_eager"} if device == 'mps' else {"mode": "reduce-overhead"}),
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
            ModelCheckpoint(monitor="train_loss", save_top_k=3, mode="min", every_n_train_steps=config.ckpt_every_n_steps),
            # StochasticWeightAveraging(swa_lrs=1e-2),
        ],
        accelerator=device,
        devices=1,
        precision="16-mixed" if device == "cuda" else 32,
        max_steps=config.max_steps,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        enable_checkpointing=True,
        default_root_dir=Path(__file__).parent / "lightning_logs",
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


def _main():
    parser = argparse.ArgumentParser()
    for name, field in Config().model_fields.items():
        if field.annotation is bool:
            if field.default:
                parser.add_argument(f"--no-{name}".replace("_", "-"), action="store_false", help=field.description, dest=name)
            else:
                parser.add_argument(f"--{name}".replace("_", "-"), action="store_true", help=field.description)
        else:
            parser.add_argument(
                f"--{name}".replace("_", "-"),
                type=_CONF_TYPE_OVERRIDES.get(name, field.annotation),
                default=field.default,
                help=field.description,
            )
    args = parser.parse_args()

    config = Config(**vars(args))
    print('CONFIG:', config)
    train(config)


_CONF_TYPE_OVERRIDES = {
    "checkpoint": str,
    "loss": str,
    "prefix": str,
    "optimizer": str,
    "dec_skip_connections": str,
}

class Config(pydantic.BaseModel):
    prefix: Optional[str] = None
    seed: int = 42
    # dataset
    x_size: tuple = (16, 192, 128, 3)
    y_size: tuple = (16, 192, 128, 3)
    img_type: ScryfallImageType = ScryfallImageType.small
    bulk_type: ScryfallBulkType = ScryfallBulkType.default_cards
    # model
    model_name: str = 'N/A'
    compile: bool = False
    checkpoint: Optional[str] = None
    dec_skip_connections: Optional[Literal['out', 'inner', 'inner_depthwise']] = None
    skip_first_optimizer_load_state: bool = False  # needed if model architecture changes or optimizer changes
    # optimizer
    max_steps: int = 1_000_000
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-7
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 0.5
    # losses
    optimizer: Literal['adam', 'radam'] = 'radam'
    loss: str = 'mse'
    multiscale: bool = True
    paired: bool = True
    # | cyclic: bool = False
    # | target_consistency: bool = False
    # | cycle_with_target: int = 0
    # loss scaling
    scale_loss_recon: float = 1
    scale_loss_recon_extra: float = 0
    scale_loss_multiscale: float = 0.5
    # | scale_loss_cyclic: float = 100
    # | scale_loss_target: float = 100
    # | scale_loss_cycle_target: float = 100
    scale_loss_paired: float = 100
    norm_io: bool = False,
    # dataset
    num_workers: int = 3
    force_download: bool = False
    # logging
    log_every_n_steps: int = 2500
    ckpt_every_n_steps: int = 2500


if __name__ == "__main__":

    # ae2_16x32x64x128x256
    # ae2_32x32x64x64x128
    # ae2_32x64x64x128x128
    # ae2_32x32x64x128x256
    # ae2_64x64x64x128x256

    # MID LAYERS DON'T SEEM TO INCREASE TIME MUCH
    # end layers are worst culprits?

    sys.argv.extend([
        "--prefix=cnxt2",
        "--model-name=cnvnxt2ae_tiny",
        "--num-workers=6",
        "--batch-size=24",
        "--learning-rate=0.001",
        "--checkpoint=mtgvision_encoder/3__psmlcp3p/checkpoints/epoch=0-step=67500.ckpt",
        "--accumulate-grad-batches=1",
        "--gradient-clip-val=1.0",
        "--scale-loss-recon=1.0",
        "--scale-loss-recon-extra=0.0",
        "--scale-loss-multiscale=0.0",
        "--scale-loss-paired=1.0",
        "--loss=ssim5",
        "--optimizer=radam",
        "--no-multiscale",
        # "--no-paired",
        "--skip-first-optimizer-load-state",
    ])
    _main()
