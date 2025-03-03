import functools
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import random
import wandb
import coremltools as ct
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback

# from mtgvision.models.new_arch3 import create_model
from mtgvision.models.new_arch1 import create_model


from mtgvision.datasets import IlsvrcImages, MtgImages
from mtgvision.util.random import GLOBAL_RAN


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
        self.model, _ = create_model(
            self.hparams.x_size,
            self.hparams.y_size,
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
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
    def __init__(self, vis_batch, log_every_n_seconds=120):
        self.vis_batch = vis_batch
        self.log_every_n_seconds = log_every_n_seconds
        self.last_log_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_every_n_seconds:
            self.log_images(pl_module, self.vis_batch)
            self.last_log_time = current_time

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
    @staticmethod
    def log_images(model, vis_batch_np):
        model.eval()
        with torch.no_grad():
            x_np = np.stack([x for x, _ in vis_batch_np], axis=0)
            y_np = np.stack([y for _, y in vis_batch_np], axis=0)
            x = torch.from_numpy(x_np).float().permute(0, 3, 1, 2).to(model.device)
            output = model(x)
            out_np = np.clip(output.cpu().permute(0, 2, 3, 1).numpy(), 0, 1)
            wandb.log({
                "images_x": wandb.Image(ImageLoggingCallback.join_images_into_row(x_np), caption="Input"),
                "images_y": wandb.Image(ImageLoggingCallback.join_images_into_row(y_np), caption="Target"),
                "images_out": wandb.Image(ImageLoggingCallback.join_images_into_row(out_np), caption="Output"),
            })


# Training function using PyTorch Lightning
def train(seed: int = 42):
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
        "model": str(create_model)
    }

    # Initialize wandb
    wandb.init(project="mtgvision_encoder", config=config)
    wandb_logger = WandbLogger()

    # Initialize model and data module
    model = MtgVisionEncoder(config)
    data_module = MtgDataModule(batch_size=config["batch_size"])

    # Initial batch for visualization
    vis_batch = [
        data_module.train_dataset.random_img(),
        data_module.train_dataset.random_img(),
        data_module.train_dataset.random_img(),
        data_module.train_dataset.random_img(),
        data_module.train_dataset.random_img(),
    ]

    # Set up trainer with optimizations
    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        logger=wandb_logger,
        callbacks=[ImageLoggingCallback(vis_batch)],
        accelerator="mps",
        devices=1,
        # precision="32",
        log_every_n_steps=1000,
        max_steps=100_000,
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


if __name__ == "__main__":
    train()