import functools

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import cv2
import numpy as np

from mtgvision.encoder_datasets import SyntheticBgFgMtgImages
from mtgvision.encoder_export import CoreMlEncoder
from mtgvision.od_export import CardSegmenter
from mtgvision.qdrant import VectorStoreQdrant


# Initialize the YOLO detector once at startup
@functools.lru_cache()
def get_ctx():
    SEGMENTER = CardSegmenter()
    ENCODER = CoreMlEncoder()
    VECS = VectorStoreQdrant()
    DATA = SyntheticBgFgMtgImages()
    return SEGMENTER, ENCODER, VECS, DATA


# Create the app
app = FastAPI()


# encode numpy array as jpg base64
def encode_im(im):
    pass


@app.websocket("/detect")
async def detect_websocket(websocket: WebSocket):
    """Handle WebSocket connections to receive frames and send detection metadata."""
    SEGMENTER, ENCODER, VECS, DATA = get_ctx()

    await websocket.accept()
    while True:
        try:
            # Receive binary image data from the browser
            data = await websocket.receive_bytes()

            # Decode the JPEG image
            print(f"Received {len(data)} bytes")
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR_RGB)
            if frame is None:
                print("Failed to decode frame, skipping...")
                continue

            # Run YOLO detection
            segments = SEGMENTER(frame)

            # extract cards & embed
            det_data = []
            for seg in segments:
                im = seg.extract_dewarped(frame)
                z = ENCODER.predict(im)
                [result] = VECS.query_nearby(z, k=1)
                card = DATA.get_card_by_id(result.id)
                det = {
                    "points": seg.xyxyxyxy.tolist(),
                    "score": float(seg.conf),
                    # "img": None
                    "match": {
                        "id": result.id,
                        "score": result.score,
                        # "img": None
                        "name": card.name,
                        "set_name": card.set_name,
                        "img_uri": card.img_uri,
                    },
                }
                det_data.append(det)

            # Send detection metadata back to the browser
            await websocket.send_json(
                {
                    "detections": det_data,
                }
            )
        except Exception as e:
            print(f"WebSocket error: {e}")
            raise


# Serve static files from the 'www' directory
# this must be placed AFTER the detect route
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent.parent / "www", html=True),
    name="static",
)


# MAIN
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
