import itertools
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from ultralytics import YOLO
from norfair import Detection

from mtgvision.encoder_export import CoreMlEncoder, MODEL_PATH
from mtgvision.util.image import imwait


class CardDetections:
    COLORS = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]

    def __init__(self, dets: list[Detection]):
        self.detections = dets

        # group detections
        self.groups: dict[int, list[Detection]] = defaultdict(list)
        for det in dets:
            self.groups[det.label].append(det)

        # for each card
        for card in self.groups[0]:
            poly = Polygon(card.points)
            # get overlapping tops and bots
            tops = [
                d
                for d in self.groups[1]
                if not getattr(d, "used", False) and Polygon(d.points).intersects(poly)
            ]
            bots = [
                d
                for d in self.groups[2]
                if not getattr(d, "used", False) and Polygon(d.points).intersects(poly)
            ]
            # get the best pair of top and bot that overlaps with the card
            best_iou, best_pair = 0, None
            for top, bot in itertools.product(tops, bots):
                top_poly = Polygon(top.points)
                bot_poly = Polygon(bot.points)
                merged_poly = top_poly.union(bot_poly)
                iou = poly.intersection(merged_poly).area / poly.union(merged_poly).area
                if iou > best_iou:
                    best_iou = iou
                    best_pair = (top, bot)
            # store the best pair
            if best_pair:
                card.top, card.bot = best_pair
                # mark used
                card.top.used = True
                card.bot.used = True
            else:
                card.top, card.bot = None, None

        # Orient each card based on the top half
        for card in self.groups[0]:
            # orient
            top_points = None
            if card.top:
                closest_side_idx = self._find_closest_side(card.points, card.top.points)
                top_points = np.roll(card.points, -closest_side_idx, axis=0)
            bot_points = None
            if card.bot:
                closest_side_idx = self._find_closest_side(card.points, card.bot.points)
                bot_points = np.roll(card.points, -closest_side_idx - 2, axis=0)
            # apply
            if top_points is None and bot_points is None:
                warnings.warn("No top or bottom found for card")
            elif top_points is None and bot_points is not None:
                warnings.warn("Top not found for card")
                card.points = bot_points
            elif top_points is not None and bot_points is None:
                warnings.warn("Bot not found for card")
                card.points = top_points
            else:
                if np.all(top_points == bot_points):
                    warnings.warn("Top and bottom do not match for card, using top")
                card.points = top_points

    @classmethod
    def _find_closest_side(cls, card_pts: np.ndarray, half_pts: np.ndarray) -> int:
        """
        Find the index of the card's side closest to the half's polygon.
        Returns the starting point index of that side.
        """
        half_poly = Polygon(half_pts)
        # ave distance from each side to the half
        best_idx = None
        best_dist = np.inf
        for i, (p0, p1) in enumerate(zip(card_pts, np.roll(card_pts, -1, axis=0))):
            dist = (half_poly.distance(Point(p0)) + half_poly.distance(Point(p1))) / 2
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def extract_warped_cards(
        self,
        frame: np.ndarray,
        out_size_hw: tuple[int, int] = (192, 128),
    ) -> list[tuple[Detection, np.ndarray]]:
        card_imgs = []
        for det in self.groups[0]:
            card_img = cv2.warpPerspective(
                frame,
                cv2.getPerspectiveTransform(
                    det.points.astype(np.float32),
                    np.asarray(
                        [
                            [0, 0],
                            [out_size_hw[1], 0],
                            [out_size_hw[1], out_size_hw[0]],
                            [0, out_size_hw[0]],
                        ]
                    ).astype(np.float32),
                ),
                out_size_hw[::-1],
            )
            card_imgs.append((det, card_img))
        return card_imgs

    def draw_all_on(
        self,
        frame: np.ndarray,
        idx: int = None,
        classes: Sequence = (0,),
        threshold: float = 0.7,
    ):
        for det in self.get_detections(classes=classes, threshold=threshold):
            self.draw_on(frame, det, color=self.COLORS[det.label], idx=idx)

    @classmethod
    def draw_on(
        cls,
        frame: np.ndarray,
        det,
        idx: int = None,
        color: tuple[int, int, int] = (255, 255, 255),
    ):
        # draw bounding box
        s = np.mean(det.scores)
        c = ((1 - s) * 0 + s * np.asarray(color)).astype(int).tolist()
        cv2.polylines(
            frame,
            [det.points.astype(int)],
            isClosed=True,
            color=c,
            thickness=1,
        )
        # draw card number
        if idx is not None:
            center = det.points.mean(axis=0).astype(int)
            center[1] += 10 * idx
            cv2.putText(
                frame,
                f"{idx}: {s:.2f}",
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                c,
                1,
            )
        # get line pointing from top to bottom and draw the arrow
        center_edge_1 = det.points[:2].mean(axis=0).astype(int)
        center_edge_2 = det.points[2:4].mean(axis=0).astype(int)
        line = LineString([center_edge_1, center_edge_2])
        point_25: Point = line.interpolate(0.25, normalized=True)
        point_75: Point = line.interpolate(0.75, normalized=True)
        cv2.arrowedLine(
            frame,
            (int(point_75.x), int(point_75.y)),
            (int(point_25.x), int(point_25.y)),
            color=c,
            thickness=1,
        )

    def get_detections(
        self, threshold: float = 0.7, classes: Sequence = (0,)
    ) -> list[Detection]:
        return [
            det
            for det in self.detections
            if det.label in classes and np.mean(det.scores) >= threshold
        ]


class CardDetector:
    def __init__(self, model_path: str | Path):
        self.yolo = YOLO(model_path)

    def __call__(self, frame: np.ndarray) -> CardDetections:
        results = self.yolo(frame)  # Run YOLO detection
        detections = []
        if results:
            for box in results[0].obb:
                for xyxyxyxy, cls, conf in zip(box.xyxyxyxy, box.cls, box.conf):
                    points = xyxyxyxy.cpu().numpy()
                    _scores = float(conf.cpu().numpy())
                    label = int(cls.cpu().numpy())
                    det = Detection(
                        points=points,
                        label=label,
                        scores=np.full_like(points[:, 0], _scores),
                    )
                    detections.append(det)
        return CardDetections(detections)


def main():
    # Init model

    root = Path("/Users/nathanmichlo/Desktop/active/mtg/mtg-vision/data")

    # This one is quite good, seems more robust to objects in the world that look similar to cards, like bright box or dark box.
    # probably need to add more augments / random erasing to fix this.
    detector_tune3 = CardDetector(root / "yolo_mtg_dataset_v3/models/1ipho2mn_best.pt")
    # These models are not idea, they don't really handle warping that well, and they
    # produce a lot of false positives. V3 tries to improve this slightly, but doesn't
    # always work
    # detector_tuneB = CardDetector(root / "yolo_mtg_dataset_v2_tune_B/models/861evuqv_best.pt")
    # detector_tune = CardDetector(root / "yolo_mtg_dataset_v2_tune/models/um2w5i7m_best.pt")
    # detector = CardDetector(root / "yolo_mtg_dataset_v2/models/hnqxzc96_best.pt")
    # detector_old = CardDetector(root / "yolo_mtg_dataset/models/as9zo50r_best.pt")

    encoder = CoreMlEncoder(MODEL_PATH.with_suffix(".encoder.mlpackage"))

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # e.g., 640x480
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # loop!
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        detectionss = [
            (detector_tune3(frame), (0, 0, 255)),
        ]
        for i, (detections, c) in enumerate(detectionss):
            detections.COLORS = [c, c, c]
            detections.draw_all_on(frame, i)

        # Display the frame and wait
        cv2.imshow("frame", frame)
        if imwait(delay=1, window_name="frame"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
