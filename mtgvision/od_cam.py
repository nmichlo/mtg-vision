import itertools
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from ultralytics import YOLO
# from norfair import Detection

from mtgvision.util.image import imwait


@dataclass
class Detection:
    points: np.ndarray
    label: int
    scores: np.ndarray
    top: "Detection" = None
    bot: "Detection" = None


class CardDetections:
    COLORS = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]

    def __init__(
        self,
        dets: list[Detection] = (),
        segs: list[Detection] = (),
    ):
        # shape 4,2
        self.detections = list(dets)
        self.detection_groups: dict[int, list[Detection]] = defaultdict(list)
        for det in dets:
            self.detection_groups[det.label].append(det)
        self._dets_orient(self.detection_groups)

        # shape M,2
        self.segments = list(segs)
        self._process_segments(self.segments)

    def _process_segments(self, segments: list[Detection]):
        # filter
        segs: list[Detection] = []
        for seg in segments:
            if seg.label != 0:
                warnings.warn("Segment label is not 0")
                continue
            segs.append(seg)

        # orient.
        for seg in segs:
            # ========== #
            # ORIENT
            # ========== #

            # each segmentation box is trained to be a U shape.
            # dilate polygon defined by points, similar to dilate operation in morphology
            # then erode it back to get the approximated box. This will close the U shape.
            # so we get an approximated rectangular bbox.
            orig_poly = Polygon(seg.points)
            dilate = orig_poly.area**0.5 * 0.2
            closed_poly = orig_poly.buffer(dilate).buffer(-dilate)
            # MultiPoly?????
            if not hasattr(closed_poly, "exterior"):
                continue
            # orient the box. subtract the one bbox from the other and get the centroids
            # because the bottom portion of the card is missing, and in the other we
            # close the U shape, the difference of the centroids will be a directional
            # vector pointing from the top to the bottom of the card.
            v = np.asarray(orig_poly.centroid.xy) - np.asarray(closed_poly.centroid.xy)
            v = v.flatten()
            v = v / np.linalg.norm(v)

            # ========== #
            # BBOX
            # ========== #

            # get 4 corner points from closed polygon
            box_points = cv2.approxPolyN(seg.points, 4, True)[0]
            box_poly = Polygon(box_points)
            # check if the centroid when extended by the direction vector passes through the edge
            idx = 0
            for i in range(1, 4):
                line = LineString(
                    [
                        box_poly.centroid.coords[0],
                        box_poly.centroid.coords[0] + v * 10000000,
                    ]
                )
                edge = LineString(
                    [box_poly.exterior.coords[i], box_poly.exterior.coords[(i + 1) % 4]]
                )
                if line.intersects(edge):
                    idx = i
                    break
            # roll the points to get the top left corner of the card
            box_points = np.roll(box_points, -idx, axis=0)

            # ========== #
            # SAVE
            # ========== #

            # create copied bounding box
            det = deepcopy(seg)
            det.points = box_points.astype(int)
            seg.points_closed = np.asarray(closed_poly.exterior.coords[:-1]).astype(int)
            seg.dir_vec = v

            self.detections.append(det)
            self.detection_groups[det.label].append(det)

    @classmethod
    def _dets_orient(cls, det_groups: dict[int, list[Detection]]):
        """
        Orient the detections based on the top half of the card.
        Do this in-place, modifying the input detections.
        """
        # for each card
        for card in det_groups[0]:
            poly = Polygon(card.points)
            # get overlapping tops and bots
            tops = [
                d
                for d in det_groups[1]
                if not getattr(d, "used", False) and Polygon(d.points).intersects(poly)
            ]
            bots = [
                d
                for d in det_groups[2]
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
        for card in det_groups[0]:
            # orient
            top_points = None
            if card.top:
                closest_side_idx = cls._dets_find_closest_side(
                    card.points, card.top.points
                )
                top_points = np.roll(card.points, -closest_side_idx, axis=0)
            bot_points = None
            if card.bot:
                closest_side_idx = cls._dets_find_closest_side(
                    card.points, card.bot.points
                )
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
    def _dets_find_closest_side(cls, card_pts: np.ndarray, half_pts: np.ndarray) -> int:
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
        for det in self.detection_groups[0]:
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

    def _lerp_color(self, color1, color2, t) -> tuple[int, int, int]:
        if isinstance(color1, int):
            color1 = (color1, color1, color1)
        if isinstance(color2, int):
            color2 = (color2, color2, color2)
        return tuple(int(c1 * (1 - t) + c2 * t) for c1, c2 in zip(color1, color2))

    def draw_all_on(
        self,
        frame: np.ndarray,
        idx: int = None,
        classes: Sequence = (0,),
        threshold: float = 0.7,
    ):
        for det in self.get_detections(classes=classes, threshold=threshold):
            c = self.COLORS[det.label]
            self.draw_on(frame, det, color=c, idx=idx)
        for det in self.get_segmentations(classes=classes, threshold=threshold):
            c = self._lerp_color(self.COLORS[det.label], 0, 0.5)
            self.draw_on(frame, det, color=c, idx=idx)

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
        # check has value
        if hasattr(det, "dir_vec"):
            p0 = det.points.mean(axis=0).astype(int)
            p1 = (
                det.points.mean(axis=0) + det.dir_vec * Polygon(det.points).length / 4
            ).astype(int)
            cv2.arrowedLine(
                frame,
                p0,
                p1,
                color=c,
                thickness=1,
            )
        if hasattr(det, "points_closed"):
            cv2.polylines(
                frame,
                [det.points_closed.astype(int)],
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

    def get_segmentations(self, threshold: float = 0.7, classes: Sequence = (0,)):
        return [
            det
            for det in self.segments
            if det.label in classes and np.mean(det.scores) >= threshold
        ]


class CardDetector:
    def __init__(self, model_path: str | Path):
        self.yolo = YOLO(model_path, verbose=False)

    def __call__(self, frame: np.ndarray) -> CardDetections:
        results = self.yolo([frame])[0]  # Run YOLO detection
        detections = []
        if results.obb:
            for box in results.obb:
                for xyxyxyxy, cls, conf in zip(box.xyxyxyxy, box.cls, box.conf):
                    points = xyxyxyxy.cpu().numpy()
                    conf = float(conf.cpu().numpy())
                    label = int(cls.cpu().numpy())
                    det = Detection(
                        points=points,
                        label=label,
                        scores=np.full_like(points[:, 0], conf),
                    )
                    detections.append(det)
        elif results.masks:
            for points, conf in zip(results.masks.xy, results.boxes.conf):
                det = Detection(
                    points=points,
                    label=0,
                    scores=np.full_like(points[:, 0], conf),
                )
                detections.append(det)
        return CardDetections(
            segs=detections,
        )


def main():
    # Init model

    root = Path("/Users/nathanmichlo/Desktop/active/mtg/mtg-vision/data")
    segmenter = CardDetector(root / "yolo_mtg_dataset_seg/models/9ss000i6_best_seg.pt")

    # This one is quite good, seems more robust to objects in the world that look similar to cards, like bright box or dark box.
    # probably need to add more augments / random erasing to fix this.
    # detector_tune3 = CardDetector(root / "yolo_mtg_dataset_v3/models/1ipho2mn_best.pt")

    # These models are not idea, they don't really handle warping that well, and they
    # produce a lot of false positives. V3 tries to improve this slightly, but doesn't
    # always work
    # detector_tuneB = CardDetector(root / "yolo_mtg_dataset_v2_tune_B/models/861evuqv_best.pt")
    # detector_tune = CardDetector(root / "yolo_mtg_dataset_v2_tune/models/um2w5i7m_best.pt")
    # detector = CardDetector(root / "yolo_mtg_dataset_v2/models/hnqxzc96_best.pt")
    # detector_old = CardDetector(root / "yolo_mtg_dataset/models/as9zo50r_best.pt")

    # encoder = CoreMlEncoder(MODEL_PATH.with_suffix(".encoder.mlpackage"))

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
            (segmenter(frame), (0, 0, 255)),
        ]
        for i, (detections, _) in enumerate(detectionss):
            for j, (det, img) in enumerate(detections.extract_warped_cards(frame)):
                cv2.imshow(f"card {i}:{j}", img)
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
