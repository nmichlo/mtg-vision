from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_PATH_SEG = Path(
    "/Users/nathanmichlo/Desktop/active/mtg/mtg-vision/data/yolo_mtg_dataset_seg/models/9ss000i6_best_seg.pt"
)


@dataclass
class InstanceSeg:
    points: np.ndarray
    label: int
    conf: float
    xyxyxyxy: np.ndarray = None

    @property
    def scores(self) -> np.ndarray:
        return np.full_like(self.points[:, 0], self.conf)


class CardSegmenter:
    def __init__(self, model_path: str | Path = None):
        if model_path is None:
            model_path = MODEL_PATH_SEG.with_suffix(".mlpackage")
        from ultralytics import YOLO

        self.yolo = YOLO(model_path, task="segment", verbose=False)

    def __call__(self, frame: np.ndarray) -> list[InstanceSeg]:
        results = self.yolo([frame])[0]
        detections = []
        for points, conf in zip(results.masks.xy, results.boxes.conf):
            det = InstanceSeg(
                points=points,
                label=0,
                conf=conf,
            )
            detections.append(det)
        return detections


def main():
    from ultralytics import YOLO

    # Load your model
    model = YOLO(MODEL_PATH_SEG)
    path = model.export(format="onnx", nms=True)
    print(f"Model exported to {path}")
    assert path == str(MODEL_PATH_SEG.with_suffix(".onnx"))

    # Load your model
    model = YOLO(MODEL_PATH_SEG)
    path = model.export(format="coreml", nms=False)
    print(f"Model exported to {path}")
    assert path == str(MODEL_PATH_SEG.with_suffix(".mlpackage"))


if __name__ == "__main__":
    main()
