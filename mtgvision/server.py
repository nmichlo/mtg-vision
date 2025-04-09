# ========== Imports ==========
import dataclasses
import functools
import hashlib
import time
from typing import Hashable

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import cv2
import numpy as np
import base64
from norfair import Tracker, Detection
from norfair.distances import mean_euclidean
from qdrant_client.http.models import ScoredPoint

from mtgdata.scryfall import ScryfallCardFace
from mtgvision.encoder_datasets import SyntheticBgFgMtgImages
from mtgvision.encoder_export import CoreMlEncoder
from mtgvision.od_export import CardSegmenter, InstanceSeg
from mtgvision.qdrant import VectorStoreQdrant
from mtgvision.util.image import imshow


# ========== Global Context ==========


@functools.lru_cache()
def get_ctx():
    SEGMENTER = CardSegmenter()
    ENCODER = CoreMlEncoder()
    VECS = VectorStoreQdrant()
    DATA = SyntheticBgFgMtgImages()
    return SEGMENTER, ENCODER, VECS, DATA


@dataclasses.dataclass
class DetData:
    seg: InstanceSeg


@dataclasses.dataclass
class TrackedData:
    # Instance tracking
    id: int
    color: str
    # Update info
    last_update_time: float = dataclasses.field(default_factory=lambda: time.time())
    # Last Tracking info
    last_instance: InstanceSeg = None
    last_rgb_im: np.ndarray = None
    last_rgb_im_encoded: str = None
    # Embed and search
    avg_z: np.ndarray = None
    ave_nearby_points: list[ScoredPoint] = dataclasses.field(default_factory=list)
    ave_nearby_cards: list[ScryfallCardFace] = dataclasses.field(default_factory=list)

    def to_dict(self):
        matches = []
        for point, card in zip(self.ave_nearby_points, self.ave_nearby_cards):
            match = {
                "id": str(point.id),
                "score": point.score,
                "name": card.name,
                "set_name": card.set_name,
                "set_code": card.set_code,
                "img_uri": card.img_uri,
                "all_data": point.payload,  # fetched from scryfall
            }
            matches.append(match)

        return {
            "id": str(self.id),
            "points": self.last_instance.xyxyxyxy.tolist(),
            "color": self.color,
            "img": self.last_rgb_im_encoded,
            "score": self.last_instance.conf,
            "matches": matches,
        }


class TrackerCtx:
    segmenter: CardSegmenter
    encoder: CoreMlEncoder
    vecs: VectorStoreQdrant
    data: SyntheticBgFgMtgImages

    def __init__(
        self,
        update_wait_sec: float = 0.5,
        ewma_weight: float = 0.1,
    ):
        self.update_wait_sec = update_wait_sec
        self.ewma_weight = ewma_weight
        # create
        self.segmenter, self.encoder, self.vecs, self.data = get_ctx()
        self.tracker = Tracker(
            distance_function=mean_euclidean,
            distance_threshold=200,
            hit_counter_max=5,
            initialization_delay=2,
            past_detections_length=10,
        )
        # Dictionary to store persistent data per tracked object
        self.tracked_data = {}  # {id: {'z_avg': np.ndarray, 'last_query_time': float, 'nearby_points': list, 'nearby_cards': list}}

    def update(self, rgb_frame: np.ndarray) -> list[TrackedData]:
        # 0. Segment the frame (RGB)
        segments = self.segmenter(rgb_frame)

        # 1. Create Norfair detections with minimal initial data
        detections = []
        for seg in segments:
            detection = Detection(
                points=np.asarray(seg.xyxyxyxy),
                data=DetData(seg=seg),
            )
            detections.append(detection)

        # 2. Track objects
        tracked_objects = self.tracker.update(detections)

        # 3. Update tracked objects
        objs = []
        current_time = time.time()
        for obj in tracked_objects:
            # 3.A If we have detections, otherwise we are work on predicted positions
            #     of detections that are not yet removed
            if obj.last_detection not in detections:
                continue
            seg: InstanceSeg = obj.last_detection.data.seg

            # 3.B Get the object or create it
            trk: TrackedData = self.tracked_data.get(obj.id)
            if trk is None:
                trk = TrackedData(
                    id=obj.id,
                    color=get_color(obj.id),
                    last_update_time=current_time,
                    last_instance=seg,
                )
                self.tracked_data[obj.id] = trk

            # 3.C Update the tracked data
            # - extract the dewarped image
            trk.last_instance = seg
            trk.last_rgb_im = seg.extract_dewarped(rgb_frame)
            trk.last_rgb_im_encoded = encode_rgb_im(trk.last_rgb_im)
            # - if enough time has passed, do a full update instead
            #   of a partial update
            if current_time - trk.last_update_time > self.update_wait_sec:
                # * embed the image
                _z = self.encoder.predict(trk.last_rgb_im)
                if trk.avg_z is None:
                    trk.avg_z = _z
                trk.avg_z = self.ewma_weight * _z + (1 - self.ewma_weight) * trk.avg_z
                # * query the vector store
                trk.ave_nearby_points = self.vecs.query_nearby(
                    trk.avg_z, k=5, with_payload=False, with_vectors=False
                )
                trk.ave_nearby_cards = [
                    self.data.get_card_by_id(p.id) for p in trk.ave_nearby_points
                ]
                # * update the last update time
                trk.last_update_time = current_time

            # 3.D Update detection data with tracked object info
            objs.append(trk)

        return objs


# ========== Utility Functions ==========


def get_color(seed: Hashable):
    hash = hashlib.sha256(str(seed).encode())
    h = int(hash.hexdigest(), 16)
    r = (h >> 16) & 0xFF
    g = (h >> 8) & 0xFF
    b = h & 0xFF
    return f"#{r:02x}{g:02x}{b:02x}"


def encode_rgb_im(rgb_im):
    bgr_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", bgr_im, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return base64.b64encode(buffer).decode("utf-8")


# ========== FastAPI Setup ==========

app = FastAPI()

# ========== WebSocket Handler ==========


@app.websocket("/detect")
async def detect_websocket(websocket: WebSocket):
    ctx = TrackerCtx()

    await websocket.accept()
    while True:
        try:
            # 1. Receive binary image data from the browser
            data = await websocket.receive_bytes()  # RGB bytes

            # 2. Decode the JPEG image as RGB array (needed for models, must NOT be BGR)
            print(f"Received {len(data)} bytes")
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR_RGB)
            if frame is None:
                print("Failed to decode frame, skipping...")
                continue

            # 3. Process the frame
            objs = ctx.update(frame)
            for i, obj in enumerate(objs):
                print(
                    obj.id,
                    [(m["name"], m["set_code"]) for m in obj.to_dict()["matches"]],
                )
                if obj.last_rgb_im is not None:
                    imshow(obj.last_rgb_im, f"{i}")

            # 4. Send results
            response = {
                "detections": [obj.to_dict() for obj in objs],
            }
            await websocket.send_json(response)
        except Exception as e:
            print(f"WebSocket error: {e}")
            raise


# Serve static files from the 'www' directory
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent.parent / "www" / "dist", html=True),
    name="static",
)

# MAIN
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
