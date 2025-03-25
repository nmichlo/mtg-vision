from datetime import datetime
from pathlib import Path

import yaml

import mtgvision
from ultralytics import YOLO
from ultralytics import settings


def main():
    settings.update({"wandb": True})

    data_dir = Path(mtgvision.__file__).parent.parent / "yolo_mtg_dataset_v2"
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

    # Create the model
    model_name = "yolo11n-obb"
    model = YOLO(f"{model_name}.yaml")

    # Train the model on the DOTAv1 dataset
    results = model.train(data=yaml_file, epochs=200, imgsz=640, val=False)
    print(results)

    # save the model
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / f"{current_time}_{model_name}.pt"))


if __name__ == "__main__":
    main()
