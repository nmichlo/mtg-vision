from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import cv2
import numpy as np
from mtgvision.od_cam import CardDetector
from mtgvision.encoder_export import CoreMlEncoder, MODEL_PATH


# Initialize the YOLO detector once at startup
_DATA_ROOT = Path("/Users/nathanmichlo/Desktop/active/mtg/mtg-vision/data")
DETECTOR = CardDetector(_DATA_ROOT / "yolo_mtg_dataset_v3/models/1ipho2mn_best.pt")
ENCODER = CoreMlEncoder(MODEL_PATH.with_suffix(".encoder.mlpackage"))

# Create the app
app = FastAPI()


@app.websocket("/detect")
async def detect_websocket(websocket: WebSocket):
    """Handle WebSocket connections to receive frames and send detection metadata."""
    await websocket.accept()
    while True:
        try:
            # Receive binary image data from the browser
            data = await websocket.receive_bytes()

            # Decode the JPEG image
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Run YOLO detection
            detections = DETECTOR(frame)

            # extract cards
            dets = []
            for card in detections.detection_groups[0]:
                det = {
                    "points": det.points.tolist(),
                    "score": float(np.mean(det.scores)),
                }
                dets.append(det)

            # Send detection metadata back to the browser
            await websocket.send_json(
                {
                    "detections": det_data,
                }
            )
        except Exception as e:
            print(f"WebSocket error: {e}")
            break


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
