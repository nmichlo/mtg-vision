"""
Card detection and segmentation utilities for MTG cards.

This module provides classes for detecting and segmenting Magic: The Gathering cards
in images or video frames. It includes:

- CardDetections: Processes segmentation results and manages detected cards
- CardDetector: Handles YOLO-based card detection
- CardSegmenter: Handles segmentation of cards in images

The module supports drawing detection results on frames and extracting card images
for further processing like encoding and identification.
"""

from functools import lru_cache

import cv2
import norfair
import numpy as np
import requests

from mtgvision.encoder_export import CoreMlEncoder
from mtgvision.od_export import CardSegmenter, MODEL_PATH_SEG
from mtgvision.qdrant import QdrantPoint, VectorStoreQdrant
from mtgvision.util.image import imwait


# ========================================================================= #
# CORE                                                                      #
# ========================================================================= #


# def embedding_distance(matched_not_init_trackers, unmatched_trackers):
#     snd_embedding = unmatched_trackers.last_detection.embedding
#     if snd_embedding is None:
#         for detection in reversed(unmatched_trackers.past_detections):
#             if detection.embedding is not None:
#                 snd_embedding = detection.embedding
#                 break
#         else:
#             return 1
#     for detection_fst in matched_not_init_trackers.past_detections:
#         if detection_fst.embedding is None:
#             continue
#         fst_embedding = detection_fst.embedding
#         cos_dist = np.dot(fst_embedding, snd_embedding) / (np.linalg.norm(fst_embedding) * np.linalg.norm(snd_embedding))
#         return 1 - cos_dist
#     return 1


def main():
    def reid_fn(d1, d2):
        # d1 and d2 are the embeddings of the two detections
        # we can use the euclidean distance as a metric
        return np.linalg.norm(d1 - d2)

    vstore = VectorStoreQdrant()
    encoder = CoreMlEncoder()
    segmenter = CardSegmenter(MODEL_PATH_SEG.with_suffix(".mlpackage"))
    tracker = norfair.Tracker(
        initialization_delay=1,
        distance_function="euclidean",
        hit_counter_max=10,
        distance_threshold=50,
        past_detections_length=5,
        # reid_distance_function=embedding_distance,
        # reid_distance_threshold=0.5,
        # reid_hit_counter_max=500,
    )

    @lru_cache
    def query_scryfall(id: str):
        response = requests.get(f"https://api.scryfall.com/cards/{id}")
        if response.status_code == 200:
            data = response.json()
            print(f"Got Card ID: {id}")
            return data
        else:
            print(f"Error: {response.status_code}")
            return None

    def get_nearby(z: np.ndarray) -> list[QdrantPoint]:
        results = vstore.query_nearby(z, 3, with_payload=True)
        for result in results:
            if not result.payload:
                result.payload = query_scryfall(result.id)
                if result.payload:
                    vstore.update_payload(result.id, result.payload)
        return results

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

        results = segmenter(frame)

        # generate detections and embed
        detections = []
        for seg in results:
            img = seg.extract_dewarped(frame)
            z = encoder.predict(img)
            near = get_nearby(z)
            det = norfair.Detection(
                points=seg.xyxyxyxy,
                label=seg.label,
                embedding=z,
                data={"set": seg, "near": near, "best": near[0] if near else None},
            )
            detections.append(det)

        # draw detections, must be after frame extractions
        for seg in results:
            seg.debug_draw_on(frame)

        # # track using norfair
        # tracked_objects = tracker.update(detections=detections)
        #
        # # draw tracked objects
        # for obj in tracked_objects:
        #     data = obj.last_detection.data if obj.last_detection else {}
        #     if data:
        #         seg: InstanceSeg = data["seg"]
        #         best: QdrantPoint | None = data["best"]
        #         id = "N/A" if not best else best.id
        #         cv2_draw_text(frame, id, seg.center, c=(0, 255, 0))

        # Display the frame and wait
        cv2.imshow("frame", frame)
        if imwait(delay=1, window_name="frame"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
