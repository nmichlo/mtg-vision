from collections import defaultdict

import cv2
import numpy as np
from shapely.geometry.polygon import Polygon
from ultralytics import YOLO
from norfair import Detection
from mtgvision.util.image import imwait


def get_best_iou(poly: Polygon, polygons: list[Polygon]) -> tuple[float, int]:
    best_iou = 0
    best_poly = None
    for i, p in enumerate(polygons):
        iou = poly.intersection(p).area / poly.union(p).area
        if iou > best_iou:
            best_iou = iou
            best_poly = i
    return best_iou, best_poly


class CardDetections:
    COLORS = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]

    def __init__(self, dets: list[Detection]):
        self.detections = dets

        # group detections
        self.groups: dict[int, list[Detection]] = defaultdict(list)
        for det in dets:
            self.groups[det.label].append(det)

    def extract_warped_cards(
        self, frame: np.ndarray, out_size_hw: tuple[int, int] = (192, 128)
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
                            [out_size_hw[0], 0],
                            [out_size_hw[0], out_size_hw[1]],
                            [0, out_size_hw[1]],
                        ]
                    ).astype(np.float32),
                ),
                out_size_hw,
            )
            card_imgs.append((det, card_img))
        return card_imgs

    def draw_on(self, frame: np.ndarray):
        for det in self.detections:
            if det.label != 0:
                continue
            # draw bounding box
            cv2.polylines(
                frame,
                [det.points.astype(int)],
                isClosed=True,
                color=self.COLORS[det.label],
                thickness=1,
            )


class CardDetector:
    def __init__(self):
        self.model = YOLO("/Users/nathanmichlo/Downloads/best.pt")

    def __call__(self, frame: np.ndarray) -> CardDetections:
        results = self.model(frame)  # Run YOLO detection
        detections = []
        if results:
            for box in results[0].obb:
                for xyxyxyxy, cls, conf in zip(box.xyxyxyxy, box.cls, box.conf):
                    points = xyxyxyxy.cpu().numpy()
                    scores = float(conf.cpu().numpy())
                    label = int(cls.cpu().numpy())
                    det = Detection(points=points, label=label)
                    detections.append(det)
        return CardDetections(detections)


def main():
    # Init model
    detector = CardDetector()

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
        detections = detector(frame)
        for i, (det, img) in enumerate(detections.extract_warped_cards(frame)):
            cv2.imshow(f"card{i}", img)
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
