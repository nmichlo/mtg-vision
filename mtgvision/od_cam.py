import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Sequence

import cv2
import numpy as np
from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from mtgvision.od_export import InstanceSeg, CardSegmenter, MODEL_PATH_SEG
from mtgvision.util.image import imwait


class CardDetections:
    COLORS = [(0, 255, 0), (0, 255, 255), (0, 0, 255)]

    def __init__(
        self,
        dets: list[InstanceSeg] = (),
        segs: list[InstanceSeg] = (),
    ):
        # shape 4,2
        self.detections = list(dets)
        self.detection_groups: dict[int, list[InstanceSeg]] = defaultdict(list)
        for det in dets:
            self.detection_groups[det.label].append(det)

        # shape M,2
        self.segments = list(segs)
        self._process_segments(self.segments)

    def _process_segments(self, segments: list[InstanceSeg]):
        # filter
        segs: list[InstanceSeg] = []
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

    def extract_warped_cards(
        self,
        frame: np.ndarray,
        out_size_hw: tuple[int, int] = (192, 128),
    ) -> list[tuple[InstanceSeg, np.ndarray]]:
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
    ) -> list[InstanceSeg]:
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


def main():
    # Init model

    segmenter = CardSegmenter(MODEL_PATH_SEG.with_suffix(".mlpackage"))

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
            (CardDetections(segmenter(frame)), (0, 0, 255)),
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
