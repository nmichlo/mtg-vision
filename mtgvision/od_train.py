from pathlib import Path

import mtgvision
from ultralytics import YOLO


def main():
    # Create a new YOLO11n-OBB model from scratch
    model = YOLO("yolo11n-obb.yaml")

    # Train the model on the DOTAv1 dataset
    data = Path(mtgvision.__file__).parent.parent / "yolo_mtg_dataset" / "mtg_obb.yaml"
    results = model.train(data="DOTAv1.yaml", epochs=100, imgsz=640)
    print(results)


if __name__ == "__main__":
    main()
