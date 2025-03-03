import argparse
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import random
import wandb
import coremltools as ct
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback

from mtgvision.models.new_arch1 import Ae1
from mtgvision.datasets import IlsvrcImages, MtgImages
from mtgvision.util.random import GLOBAL_RAN

_MODELS = {
    Ae1.__name__.lower(): functools.partial(Ae1, stn=False),
    Ae1.__name__.lower() + '_stn': functools.partial(Ae1, stn=True),
}


@functools.lru_cache(maxsize=1)
def _load_image_ds():
    orig = MtgImages(img_type='small')
    ilsvrc = IlsvrcImages()
    return orig, ilsvrc


# Custom Dataset for MTG Images (unchanged)
class RanMtgEncDecDataset(IterableDataset):
    def __init__(self):
        self.orig, self.ilsvrc = _load_image_ds()

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
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = _MODELS[self.hparams.model]()
        print(self.model)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        z, multi_out = self.model(x, multiscale=self.hparams.multiscale)
        return z, multi_out

    def training_step(self, batch, batch_idx):
        x, y = batch
        z, (out, *multi_out) = self(x)

        loss = 0
        num = 0

        # recon loss
        loss_recon = self.criterion(out, y)
        loss += loss_recon
        num += 1

        # multiscale
        loss_multiscale = 0
        if self.hparams.multiscale and len(multi_out) > 0:
            for output in multi_out:
                output = F.interpolate(output, size=(192, 128), mode='bilinear', align_corners=False)
                loss_multiscale += self.criterion(output, y)
            loss_multiscale /= len(multi_out)
            loss += loss_multiscale
            num += 1

        # cyclic consistency loss
        loss_cyclic = 0
        if self.hparams.cyclic:
            z2 = self.model.encode(out)
            loss_cyclic = self.criterion(z2, z)
            loss += loss_cyclic
            num += 1

        # target consistency loss
        loss_target = 0
        if self.hparams.target_consistency:
            z2 = self.model.encode(y)
            loss_target = self.criterion(z2, z)
            loss += loss_target
            num += 1

        # done
        loss /= num
        logs = {
            "loss_recon": loss_recon,
            "loss_multiscale": loss_multiscale,
            "loss_cyclic": loss_cyclic,
            "loss_target": loss_target,
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

    def __init__(self, batch_size, num_workers=3):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = RanMtgEncDecDataset()

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
                mout_np.append(np.clip(out.cpu().permute(0, 2, 3, 1).numpy(), 0, 1))
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
def train(
    model_name: str,
    multiscale: bool,
    cyclic: bool,
    target_consistency: bool,
    seed: int,
    max_steps: int,
):
    random.seed(seed)
    torch.manual_seed(seed)
    GLOBAL_RAN.reset(seed)

    # Configuration
    config = {
        "seed": seed,
        "batch_size": 16,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "x_size": (16, 192, 128, 3),  # NHWC
        "y_size": (16, 192, 128, 3),  # NHWC
        "img_type": "small",
        "model": model_name,
        "multiscale": multiscale,
        "cyclic": cyclic,
        "target_consistency": target_consistency,
        "max_steps": max_steps,
    }

    # Initialize model
    data_module = MtgDataModule(batch_size=config["batch_size"])

    # Initial batch for visualization
    vis_batch = [
        data_module.train_dataset.random_img(),
        data_module.train_dataset.random_img(),
        data_module.train_dataset.random_img(),
        data_module.train_dataset.random_img(),
        data_module.train_dataset.random_img(),
    ]

    # Initialize wandb
    wandb_logger = WandbLogger(
        name=f"{config['model']}_lr{config['learning_rate']}_multi-{config['multiscale']}_cyc-{config['cyclic']}_targ-{config['target_consistency']}",
        project="mtgvision_encoder",
        config=config,
    )

    # Initialize model
    model = MtgVisionEncoder(config)

    # Set up trainer with optimizations
    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        logger=wandb_logger,
        callbacks=[ImageLoggingCallback(vis_batch)],
        accelerator="mps",
        devices=1,
        precision="mixed",
        max_steps=max_steps,
    )

    # Run training
    trainer.fit(model, data_module)

    # Save the final model checkpoint
    # trainer.save_checkpoint("final_model.ckpt")

    # Convert to CoreML
    model.eval()
    example_input = torch.rand(1, 3, 192, 128).to(model.device)  # NCHW
    traced_model = torch.jit.trace(model, example_input)
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="input", shape=example_input.shape)],
    )
    mlmodel.save("model.mlmodel")
    print("Model trained and converted to CoreML. Saved as 'model.mlmodel'.")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ae1", help=f"Model architecture to use, allowed: {list(_MODELS.keys())}")
    parser.add_argument("--multiscale", action="store_true", help="Use multiscale loss")
    parser.add_argument("--cyclic", action="store_true", help="Use cyclic consistency loss")
    parser.add_argument("--target-consistency", action="store_true", help="Use target consistency loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=100_000, help="Maximum training steps")
    args = parser.parse_args()

    train(
        model_name=args.model,
        multiscale=args.multiscale,
        cyclic=args.cyclic,
        target_consistency=args.target_consistency,
        seed=args.seed,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    _main()
