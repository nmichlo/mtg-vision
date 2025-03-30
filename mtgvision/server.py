# ========== Imports ==========
import dataclasses
import functools
import hashlib
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

# ========== Global Context ==========


@functools.lru_cache()
def get_ctx():
    SEGMENTER = CardSegmenter()
    ENCODER = CoreMlEncoder()
    VECS = VectorStoreQdrant()
    DATA = SyntheticBgFgMtgImages()
    return SEGMENTER, ENCODER, VECS, DATA


@dataclasses.dataclass
class ObjData:
    seg: InstanceSeg
    # instance
    id: int | None = None
    color: str | None = None
    # embed and search
    img: np.ndarray | None = None
    img_encoded: str | None = None
    z: np.ndarray | None = None
    # search
    nearby_points: list[ScoredPoint] | None = None
    nearby_cards: list[ScryfallCardFace] | None = None


class TrackerCtx:
    segmenter: CardSegmenter
    encoder: CoreMlEncoder
    vecs: VectorStoreQdrant
    data: SyntheticBgFgMtgImages

    def __init__(self):
        self.segmenter, self.encoder, self.vecs, self.data = get_ctx()
        self.tracker = Tracker(
            distance_function=mean_euclidean,
            distance_threshold=200,
            hit_counter_max=5,
            initialization_delay=2,
            past_detections_length=10,
        )

    def update(self, frame: np.ndarray) -> list[ObjData]:
        # segment
        segments = self.segmenter(frame)

        # create norfair detections
        detections = []
        for seg in segments:
            detection = Detection(
                points=np.asarray(seg.xyxyxyxy),
                data=ObjData(seg=seg),
            )
            detections.append(detection)

        # track
        tracked_objects = self.tracker.update(detections)

        # update tracked objects
        objects = []
        for tracked_object in tracked_objects:
            if tracked_object.last_detection not in detections:
                continue

            obj: ObjData = tracked_object.last_detection.data
            # 0. instance
            obj.id = tracked_object.id
            obj.color = get_color(obj.id)
            # 1. extract the dewarped image
            obj.im = obj.seg.extract_dewarped(frame)
            obj.img_encoded = encode_im(obj.im)
            # 2. embed the image
            obj.z = self.encoder.predict(obj.im)
            # 3. get the nearest matching cards
            obj.nearby_points = self.vecs.query_nearby(
                obj.z, k=5, with_payload=False, with_vectors=False
            )
            obj.nearby_cards = [
                self.data.get_card_by_id(p.id) for p in obj.nearby_points
            ]
            objects.append(obj)

        return objects


# ========== Utility Functions ==========


def get_color(seed: Hashable):
    # hash the seed using hashlib and convert to an RGB integer
    hash = hashlib.sha256(str(seed).encode())
    h = int(hash.hexdigest(), 16)
    r = (h >> 16) & 0xFF
    g = (h >> 8) & 0xFF
    b = h & 0xFF
    # convert to hex string
    # and format as a hex color code
    return f"#{r:02x}{g:02x}{b:02x}"


def encode_im(im):
    _, buffer = cv2.imencode(".jpg", im)
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
            # Receive binary image data from the browser
            data = await websocket.receive_bytes()

            # Decode the JPEG image
            print(f"Received {len(data)} bytes")
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                print("Failed to decode frame, skipping...")
                continue

            # Process the frame
            objects = ctx.update(frame)

            # Prepare the detection data
            det_data = []
            for obj in objects:
                det = {
                    "id": str(obj.id),
                    "points": obj.seg.xyxyxyxy.tolist(),
                    "color": obj.color,
                    "img": str(obj.img_encoded),
                    "score": float(obj.seg.conf),
                    "matches": [
                        {
                            "id": str(match.id),
                            "score": float(match.score),
                            "name": str(card.name),
                            "set_name": str(card.set_name),
                            "img_uri": str(card.img_uri),
                        }
                        for match, card in zip(obj.nearby_points, obj.nearby_cards)
                    ],
                }
                det_data.append(det)

            # DONE!
            await websocket.send_json({"detections": det_data})
        except Exception as e:
            print(f"WebSocket error: {e}")
            raise


# Serve static files from the 'www' directory
# this MUST be placed AFTER the `/detect` route above due to
# override order. Otherwise the above would not be reachable
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent.parent / "www", html=True),
    name="static",
)


# MAIN
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
