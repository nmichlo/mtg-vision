import argparse
import functools
import sys
from inspect import signature
from pathlib import Path
from typing import Literal, Optional

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
import coremltools as ct
import kornia as K
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    StochasticWeightAveraging,
)

from mtgdata import ScryfallBulkType, ScryfallImageType
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
}


@functools.lru_cache(maxsize=1)
def _load_image_ds(predownload: bool = False):
    orig = MtgImages(img_type='small', predownload=predownload)
    ilsvrc = IlsvrcImages()
    return orig, ilsvrc


# Custom Dataset for MTG Images (unchanged)
class RanMtgEncDecDataset(IterableDataset):
    def __init__(self, predownload: bool = False):
        self.orig, self.ilsvrc = _load_image_ds(predownload=predownload)

    def __iter__(self):
        while True:
            yield self.random_tensor()

    def random_img(self):
        o = self.orig.ran()
        l = self.ilsvrc.ran()
        x, y = MtgImages.make_virtual_pair(o, l, (192, 128), (192, 128), False)
        return x, y

    def random_tensor(self):
        x, y = self.random_img()
        # Convert from NHWC float16 numpy arrays to NCHW float32 PyTorch tensors
        x = torch.from_numpy(x).float().permute(2, 0, 1)
        y = torch.from_numpy(y).float().permute(2, 0, 1)
        return x, y


# Define the Lightning Module
class MtgVisionEncoder(pl.LightningModule):

    hparams: "Config"
    model: "nn.Module"

    def __init__(self, config: dict):
        super().__init__()
        config = Config(**config).model_dump()  # add in missing defaults
        self.save_hyperparameters(config)

    def configure_model(self) -> None:
        model_fn = _MODELS[self.hparams.model_name]
        model = model_fn(self.hparams.x_size, self.hparams.y_size, multiscale=self.hparams.multiscale)
        self.model = model

    def _get_loss(self):
        if self.hparams.loss == 'mse':
            return F.mse_loss, None
        elif self.hparams.loss == 'mse+edge':
            return F.mse_loss, K.filters.sobel
        else:
            raise ValueError(f"Unknown loss: {self.hparams.loss}")

    @classmethod
    def test_checkpoint(cls, path):
        model = MtgVisionEncoder.load_from_checkpoint(path, map_location="cpu")
        # Initialize model
        data_module = MtgDataModule(batch_size=1)
        # Initial batch for visualization
        x, y = data_module.train_dataset.random_tensor()
        x = x.cpu()
        y = y.cpu()
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
        z, multi_out = self.model(x)
        return z, multi_out

    def training_step(self, batch, batch_idx):
        # forward
        x, y = batch
        z, (out, *multi_out) = self(x)

        # loss
        loss = 0
        loss_fn, loss_filter = self._get_loss()

        # Reconstruction loss
        loss_recon = loss_fn(out, y)
        loss += loss_recon * self.hparams.scale_loss_recon

        # Extra loss
        loss_recon_extra = 0
        if loss_filter is not None:
            loss_recon_extra = loss_fn(loss_filter(out), loss_filter(y))
            loss += loss_recon_extra * self.hparams.scale_loss_recon_extra

        # Multiscale loss
        loss_multiscale = 0
        if self.hparams.multiscale and len(multi_out) > 0:
            for mout in multi_out:
                mout = F.interpolate(mout, size=y.size()[2:], mode='bilinear', align_corners=False)
                loss_multiscale += loss_fn(mout, y)
            loss_multiscale /= len(multi_out)
            loss += loss_multiscale * self.hparams.scale_loss_multiscale

        # Cycle consistency losses
        loss_cyclic = 0
        if self.hparams.cyclic:
            z2 = self.model.encode(out)
            loss_cyclic = loss_fn(z2, z)
            loss += loss_cyclic * self.hparams.scale_loss_cyclic

        # Target consistency loss
        loss_target = 0
        if self.hparams.target_consistency:
            z2 = self.model.encode(y)
            loss_target = loss_fn(z2, z)
            loss += loss_target * self.hparams.scale_loss_target

        # Cycle WITH target consistency loss
        loss_cycle_target = 0
        if self.hparams.cycle_with_target > 0:
            assert self.hparams.cycle_with_target % 2 == 0
            n = self.hparams.cycle_with_target // 2
            # like cycle and target, but only take first n/2 elements
            _z = torch.concatenate([z[:n], z[:n]], dim=0)
            _x = torch.concatenate([x[:n], out[:n]], dim=0)
            loss_cycle_target += loss_fn(self.model.encode(_x), _z)
            loss += loss_cycle_target * self.hparams.scale_loss_cycle_target

        logs = {
            "loss_recon": loss_recon,
            "loss_recon_extra": loss_recon_extra,
            "loss_multiscale": loss_multiscale,
            "loss_cyclic": loss_cyclic,
            "loss_target": loss_target,
            "loss_cycle_target": loss_cycle_target,
            "train_loss": loss,
        }
        self.log_dict(logs, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.RAdam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )


# Define the Data Module
class MtgDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, num_workers: int = 3, predownload: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = RanMtgEncDecDataset(predownload=predownload)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# Custom callback for periodic image logging
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

    # Helper function to join images into a row (unchanged)
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

    # Image logging function adapted for Lightning
    def log_images(self, model, vis_batch_np):
        print("Logging images...")
        logs = {}
        model.eval()
        with torch.no_grad():
            x_np = np.stack([x for x, _ in vis_batch_np], axis=0)
            y_np = np.stack([y for _, y in vis_batch_np], axis=0)
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



# Training function using PyTorch Lightning
def train(config: "Config"):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    GLOBAL_RAN.reset(config.seed)

    # Initialize model
    data_module = MtgDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        predownload=config.force_download,
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
        (config.loss, config.loss),
        (config.learning_rate, f"lr={config.learning_rate}"),
        (config.batch_size, f"bs={config.batch_size}"),
        (config.multiscale, f"Lm={config.scale_loss_multiscale}"),
        (config.cyclic, f"Lc={config.scale_loss_cyclic}"),
        (config.target_consistency, f"Lt={config.scale_loss_target}"),
        (config.cycle_with_target > 0, f"Lct{config.cycle_with_target}={config.scale_loss_cycle_target}"),
    ]
    wandb_logger = WandbLogger(
        name="_".join([v for k, v in parts if k]),
        project="mtgvision_encoder",
        config=config,
    )

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
        model = torch.compile(
            model,
            **({"backend": "aot_eager"} if device == 'mps' else {})
        )

    # Set up trainer with optimizations
    trainer = pl.Trainer(
        max_epochs=config.max_steps,
        logger=wandb_logger,
        callbacks=[
            ImageLoggingCallback(vis_batch, log_every_n_steps=2500),
            ModelCheckpoint(monitor="train_loss", save_top_k=3, mode="min", every_n_train_steps=2500),
            StochasticWeightAveraging(swa_lrs=1e-2),
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
    model_name: str = "ae2l"
    compile: bool = False
    checkpoint: Optional[str] = None
    # losses
    loss: Literal['mse', 'mse+edge'] = 'mse'
    multiscale: bool = True
    cyclic: bool = False
    target_consistency: bool = False
    cycle_with_target: int = 0
    # optimizer
    max_steps: int = 1_000_000
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-7
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 0.5
    # loss scaling
    scale_loss_recon: float = 1
    scale_loss_recon_extra: float = 5.0
    scale_loss_multiscale: float =  0.5
    scale_loss_cyclic: float = 100
    scale_loss_target: float = 100
    scale_loss_cycle_target: float = 100
    # inner-batch-sampling (load 1/2 data, but generate multiple for comparisons)
    # dataset
    num_workers: int = 3
    force_download: bool = False


if __name__ == "__main__":
    sys.argv.extend([
        "--prefix=test",
        "--model=ae2s_stn",
        # "--no-multiscale",
        "--cyclic",
        "--target-consistency",
        "--loss=mse+edge",
        "--learning-rate=0.001",
        "--batch-size=32",
        # "--accumulate-grad-batches=2",
        # "--checkpoint=mtgvision_encoder/nlsy52sj/checkpoints/epoch=0-step=152500.ckpt",
        "--num-workers=6",
    ])
    _main()
