from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry.linestring import LineString
from shapely.geometry.polygon import Polygon

from mtgvision.util.cv2 import cv2_draw_arrow, cv2_draw_poly, cv2_draw_text

MODEL_PATH_SEG = Path(
    "/Users/nathanmichlo/Desktop/active/mtg/data/gen/"
    # "yolo_mtg_dataset_seg/models/9ss000i6_best_seg.pt"
    "yolo_mtg_dataset_seg/models/dk964hap_best_seg.pt"
)


@dataclass
class InstanceSeg:
    points: np.ndarray
    label: int
    conf: float

    # private
    _xyxyxyxy: np.ndarray = None
    _points_closed: np.ndarray = None
    _dir_vec: np.ndarray = None

    @property
    def scores(self) -> np.ndarray:
        return np.full_like(self.points[:, 0], self.conf)

    @property
    def center(self) -> np.ndarray:
        return np.mean(self.xyxyxyxy, axis=0)

    @property
    def points_closed(self) -> np.ndarray:
        self._orient()
        return self._points_closed

    @property
    def xyxyxyxy(self) -> np.ndarray:
        self._orient()
        return self._xyxyxyxy

    @property
    def dir_vec(self) -> np.ndarray:
        self._orient()
        return self._dir_vec

    def _orient(self, mode="u_shape") -> None:
        if self._xyxyxyxy is not None:
            return
        assert mode == "u_shape", "Only u_shape dataset mode is supported"
        # ===== ORIENT ===== #
        # each segmentation box is trained to be a U shape.
        # dilate polygon defined by points, similar to dilate operation in morphology
        # then erode it back to get the approximated box. This will close the U shape.
        # so we get an approximated rectangular bbox.
        orig_poly = Polygon(self.points)
        dilate = orig_poly.area**0.5 * 0.2
        closed_poly = orig_poly.buffer(dilate).buffer(-dilate)
        # MultiPoly to Poly
        if closed_poly.geom_type == "MultiPolygon":
            closed_poly = closed_poly.convex_hull
        # orient the box. subtract the one bbox from the other and get the centroids
        # because the bottom portion of the card is missing, and in the other we
        # close the U shape, the difference of the centroids will be a directional
        # vector pointing from the top to the bottom of the card.
        v = np.asarray(orig_poly.centroid.xy) - np.asarray(closed_poly.centroid.xy)
        v = v.flatten()
        v = v / np.linalg.norm(v)
        # ===== BBOX ===== #
        # get 4 corner points from closed polygon
        box_points = cv2.approxPolyN(self.points, 4, True)[0]
        box_poly = Polygon(box_points)
        # check if the centroid when extended by the direction vector passes through the edge
        idx = 0
        for i in range(1, 4):
            centroid = box_poly.centroid.coords[0]
            line = LineString([centroid, centroid + v * 10000000])
            exterior = box_poly.exterior.coords
            edge = LineString([exterior[i], exterior[(i + 1) % 4]])
            if line.intersects(edge):
                idx = i
                break
        # roll the points to get the top left corner of the card
        box_points = np.roll(box_points, -idx, axis=0)
        # ===== SAVE ===== #
        self._xyxyxyxy = box_points.astype(int)
        self._points_closed = np.asarray(closed_poly.exterior.coords[:-1]).astype(int)
        self._dir_vec = v

    def extract_dewarped(
        self,
        frame: np.ndarray,
        out_size_hw: tuple[int, int] = (192, 128),
        expand_ratio: float = 0.05,
    ) -> np.ndarray:
        h, w = out_size_hw
        dst_pts = np.asarray([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = (1 + expand_ratio) * dst_pts - (0.5 * expand_ratio) * np.asarray(
            [w, h]
        )
        M = cv2.getPerspectiveTransform(
            np.asarray(self.xyxyxyxy).astype(np.float32),
            np.asarray(dst_pts).astype(np.float32),
        )
        card_img = cv2.warpPerspective(frame, M, out_size_hw[::-1])
        return card_img

    def debug_draw_on(
        self,
        frame: np.ndarray,
        color: tuple[int, int, int] = (128, 128, 128),
        id: str = None,
    ):
        xyxyxyxy = self.xyxyxyxy
        # Calculate color based on detection score
        c = ((1 - self.conf) * 0 + self.conf * np.asarray(color)).astype(int).tolist()
        # Draw polygons
        cv2_draw_poly(frame, self.points, c=c, color_mod=0)
        cv2_draw_poly(frame, self.points_closed, c=c, color_mod=1)
        cv2_draw_poly(frame, self.xyxyxyxy, c=c)
        # Draw arrow from top to bottom of card
        line = LineString([xyxyxyxy[:2].mean(axis=0), xyxyxyxy[2:4].mean(axis=0)])
        point_a = line.interpolate(0.25, normalized=True)
        point_b = line.interpolate(0.5, normalized=True)
        cv2_draw_arrow(frame, [point_b.x, point_b.y], [point_a.x, point_a.y], c=c)
        # Draw direction vector
        p0 = self.xyxyxyxy.mean(axis=0)
        p1 = p0 + self.dir_vec * line.length * 0.25
        cv2_draw_arrow(frame, p0, p1, c=c, color_mod=0)
        # Draw card score
        center = self.xyxyxyxy.mean(axis=0).astype(int)
        text = (f"{id}: " if id else "") + f"{self.conf:.2f}"
        cv2_draw_text(frame, text, center, c=c)


class CardSegmenter:
    def __init__(self, model_path: str | Path = None):
        if model_path is None:
            model_path = MODEL_PATH_SEG.with_suffix(".mlpackage")
        from ultralytics import YOLO

        self.yolo = YOLO(model_path, task="segment", verbose=False)

    def __call__(self, rgb_im: np.ndarray) -> list[InstanceSeg]:
        results = self.yolo([rgb_im], verbose=False)[0]
        detections = []
        if results.masks and results.boxes:
            for points, conf in zip(results.masks.xy, results.boxes.conf):
                det = InstanceSeg(
                    points=np.asarray(points),
                    label=0,
                    conf=np.asarray(conf).tolist(),
                )
                detections.append(det)
        return detections


def main():
    from ultralytics import YOLO

    # Load your model
    model = YOLO(MODEL_PATH_SEG)

    # tensorflowjs
    path = model.export(format="tfjs", nms=True)
    print(f"Model exported to {path}")
    assert path == str(MODEL_PATH_SEG.with_suffix("_web_model"))

    # convert to onnx
    path = model.export(format="onnx", nms=True)  # don't think nms works here...
    print(f"Model exported to {path}")
    assert path == str(MODEL_PATH_SEG.with_suffix(".onnx"))

    # convert to coreml
    path = model.export(format="coreml", nms=True)  # don't think nms works here...
    print(f"Model exported to {path}")
    assert path == str(MODEL_PATH_SEG.with_suffix(".mlpackage"))


if __name__ == "__main__":
    main()
