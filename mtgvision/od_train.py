import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

import mtgvision
from ultralytics import YOLO
from ultralytics import settings


def _main(
    data_dir: str | Path,
    weights: str | Path = None,
    size: str = "n",
    epochs: int = 100,
):
    settings.update({"wandb": True})

    # default root
    default_root = Path(mtgvision.__file__).parent.parent

    # get data
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = default_root / data_dir
    print(f"Using data from {data_dir}")

    # get contents
    img_dir = data_dir / "images"
    yaml_file = data_dir / "temp_mtg_obb.yaml"
    model_dir = data_dir / "models"

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create a new YOLO11n-OBB model from scratch
    with open(yaml_file, "w") as fp:
        data = {
            "path": ".",
            "train": str(img_dir / "train"),
            "val": str(img_dir / "val"),
            "names": {
                0: "card",
                1: "card_top",
                2: "card_bottom",
            },
        }
        yaml.safe_dump(data, fp)

    # Create the model & load weights if needed
    model_name = f"yolo11{size}-obb"
    model = YOLO(f"{model_name}.yaml")
    if weights is not None:
        weights = Path(weights)
        print(f"Loading weights from {weights}")
        if not weights.is_absolute():
            weights = default_root / weights
        model = model.load(weights)

    # Train the model on the DOTAv1 dataset
    results = model.train(data=yaml_file, epochs=epochs, imgsz=640, val=False)
    print(results)

    # save the model
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / f"{current_time}_{model_name}.pt"))


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to the weights file to use for training. E.g. for fine tuning.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="n",
        help="Size of the model to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train the model.",
    )
    args = parser.parse_args()

    _main(
        data_dir=args.data_dir,
        weights=args.weights,
        size=args.size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    # yolo_mtg_dataset_v2
    # | sys.argv.extend([
    # |     "--data", "data/yolo_mtg_dataset_v2",
    # |     "--epochs", "200",
    # | ])

    # yolo_mtg_dataset_v2_tune
    # sys.argv.extend([
    #     "--data", "data/yolo_mtg_dataset_v2_tune",
    #     "--weights", "data/yolo_mtg_dataset_v2/models/hnqxzc96_best.pt",
    #     "--epochs", "150",
    # ])

    # yolo_mtg_dataset_v2_tune_B
    # sys.argv.extend([
    #     "--data", "data/yolo_mtg_dataset_v2_tune_B",
    #     "--weights", "data/yolo_mtg_dataset_v2/models/hnqxzc96_best.pt",
    #     "--epochs", "200",
    # ])

    sys.argv.extend(
        [
            "--data",
            "data/yolo_mtg_dataset_v3",
            "--weights",
            "data/yolo_mtg_dataset/models/as9zo50r_best.pt",
            "--epochs",
            "100",
        ]
    )

    cli()
