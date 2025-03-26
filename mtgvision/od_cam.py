import itertools
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from ultralytics import YOLO
from norfair import Detection

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

    def draw_on(self, frame: np.ndarray):
        for det in self.detections:
            if det.label not in (0, 1):
                continue
            # draw bounding box
            cv2.polylines(
                frame,
                [det.points.astype(int)],
                isClosed=True,
                color=self.COLORS[det.label],
                thickness=1,
            )
            if det.label not in (0,):
                continue
            for i, point in enumerate(det.points.astype(int)):
                cv2.putText(
                    frame,
                    str(i),
                    tuple(point),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )


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
                    det = Detection(points=points, label=label)
                    detections.append(det)
        return CardDetections(detections)


def main():
    # Init model
    detector_tuneB = CardDetector(
        "/Users/nathanmichlo/Desktop/active/mtg/mtg-vision/yolo_mtg_dataset_v2_tune_B/models/861evuqv_best.pt"
    )
    detector_tune = CardDetector(
        "/Users/nathanmichlo/Desktop/active/mtg/mtg-vision/yolo_mtg_dataset_v2_tune/models/um2w5i7m_best.pt"
    )
    detector = CardDetector(
        "/Users/nathanmichlo/Desktop/active/mtg/mtg-vision/yolo_mtg_dataset_v2/models/hnqxzc96_best.pt"
    )
    detector_old = CardDetector(
        "/Users/nathanmichlo/Desktop/active/mtg/mtg-vision/yolo_mtg_dataset/models/as9zo50r_best.pt"
    )
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

        # detect & draw
        # t = time.time()

        detectionss = [
            detector_tuneB(frame),
            detector_tune(frame),
            detector(frame),
            detector_old(frame),
        ]

        # detections_old = detector_old(frame)
        # for i, (det, img) in enumerate(detections.extract_warped_cards(frame)):
        #     cv2.imshow(f"card{i}", img)
        #     result = encoder.predict(img.astype(np.float32) / 255)

        for detections, c in zip(
            detectionss, [(0, 0, 255), (0, 128, 255), (0, 255, 255), (255, 0, 0)]
        ):
            detections.COLORS = [c, c, c]
            detections.draw_on(frame)

        # Display the frame and wait
        cv2.imshow("frame", frame)
        if imwait(delay=1, window_name="frame"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
