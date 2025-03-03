import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import wandb
import coremltools as ct
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from mtgvision.models.new_arch3 import create_model
from mtgvision.datasets import (
    IlsvrcImages, MtgImages,
)
from mtgvision.util.random import GLOBAL_RAN


# Custom Dataset for MTG Images
class RanMtgEncDecDataset(Dataset):

    def __init__(self, length=10000):
        self.orig = MtgImages(img_type='normal')
        self.ilsvrc = IlsvrcImages()
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        while True:
            yield self.random_tensor()

    def __getitem__(self, item):
        return self.random_tensor()

    def random_img(self):
        o = self.orig.ran()
        l = self.ilsvrc.ran()
        x, y = MtgImages.make_virtual_pair(o, l, (192, 128), (192, 128), True)
        return x, y

    def random_tensor(self):
        x, y = self.random_img()
        # Convert from NHWC float16 numpy arrays to NCHW float32 PyTorch tensors
        x = torch.from_numpy(x).float().permute(2, 0, 1)
        y = torch.from_numpy(y).float().permute(2, 0, 1)
        return x, y


def join_images_into_row(images, padding=5):
    const = 127
    if images[0].dtype in [np.float16, np.float32, np.float64]:
        const = 0.5
    images = [
        np.pad(image, [(padding, padding), (padding, padding), (0, 0)], mode='constant', constant_values=const)
        for image in images
    ]
    return np.concatenate(images, axis=1)


# Image Logging Function
def log_images(model, vis_batch_np, device, num_images=5):
    model.eval()
    with torch.no_grad():
        x_np = np.stack([x for x, _ in vis_batch_np], axis=0)
        y_np = np.stack([y for _, y in vis_batch_np], axis=0)
        x = torch.from_numpy(x_np).float().permute(0, 3, 1, 2).to(device)
        output = model(x)
        out_np = np.clip(output.cpu().permute(0, 2, 3, 1).numpy(), 0, 1)
        # get images
        wandb.log({
            "images_x": wandb.Image(join_images_into_row(x_np), caption="Input"),
            "images_y": wandb.Image(join_images_into_row(y_np), caption="Target"),
            "images_out": wandb.Image(join_images_into_row(out_np), caption="Output"),
        })


# Training Function
def train(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    GLOBAL_RAN.reset(seed)

    # make dataset
    train_dataset = RanMtgEncDecDataset()

    # initial batch
    vis_batch = [
        train_dataset.random_img(),
        train_dataset.random_img(),
        train_dataset.random_img(),
        train_dataset.random_img(),
        train_dataset.random_img(),
    ]

    # Configuration
    batch_size = 16
    config = {
        "seed": seed,
        "batch_size": batch_size,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "x_size": (batch_size, 192, 128, 3),  # NHWC
        "y_size": (batch_size, 192, 128, 3),  # NHWC
        "img_type": "small",
    }

    # Initialize wandb
    wandb.init(project="mtgvision_encoder", config=config)

    # Model Initialization
    model, _ = create_model(
        config["x_size"],
        config["y_size"],
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.RAdam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    # scaler = GradScaler()
    scaler = None

    # Dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
    )

    # Training Loop
    t0 = time.time()
    with tqdm(desc=f"training") as pbar:
        for epoch in range(config["num_epochs"]):
            pbar.set_postfix({"epoch": epoch})
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                if scaler is None:
                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                else:
                    with autocast(device_type=device.type):
                        output = model(x)
                        loss = criterion(output, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                train_loss += loss.item()
                pbar.update()
                pbar.set_postfix({"loss": loss.item()})
            train_loss /= len(train_loader)
            # Log metrics to wandb
            wandb.log({"train_loss": train_loss, "epoch": epoch})
            # Log images every 120 seconds
            if time.time() - t0 > 120:
                log_images(model, vis_batch, device)

    # Save the final model
    torch.save(model.state_dict(), "final_model.pth")

    # Convert to CoreML
    model.eval()
    example_input = torch.rand(1, 3, 192, 128).to(device)  # NCHW
    traced_model = torch.jit.trace(model, example_input)
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="input", shape=example_input.shape)],
    )
    mlmodel.save("model.mlmodel")
    print("Model trained and converted to CoreML. Saved as 'model.mlmodel'.")


if __name__ == "__main__":
    train()
