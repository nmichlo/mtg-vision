import cv2
from ultralytics import YOLO
from norfair import Detection
import threading
import queue
from mtgvision.util.image import imwait


def detection_thread(frame_queue, detection_queue, model):
    """Run YOLO detection in a separate thread."""
    while True:
        frame = frame_queue.get()  # Blocks until a frame is available
        results = model(frame)  # Run YOLO detection
        detections = []
        if results:
            for box in results[0].obb:
                for xyxyxyxy, cls, conf in zip(box.xyxyxyxy, box.cls, box.conf):
                    points = xyxyxyxy.cpu().numpy()
                    scores = float(conf.cpu().numpy())
                    label = int(cls.cpu().numpy())
                    det = Detection(points=points, label=label)
                    detections.append(det)
        try:
            detection_queue.put_nowait(detections)
        except queue.Full:
            pass  # Skip if queue is full, keeping only the latest detections


def main():
    # Load YOLO model
    model = YOLO("/Users/nathanmichlo/Downloads/best.pt")

    # Initialize norfair tracker with IoU distance
    # tracker = Tracker(
    #     distance_function=norfair.distances.mean_euclidean,
    #     distance_threshold=500,
    # )

    # Start detection thread
    # Queues for thread communication
    frame_queue = queue.Queue(maxsize=1)
    detection_queue = queue.Queue(maxsize=1)
    threading.Thread(
        target=detection_thread, args=(frame_queue, detection_queue, model), daemon=True
    ).start()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # e.g., 640x480
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Send the latest frame to the detection thread
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Skip if queue is full

        # Get the latest detections if available
        detections: list[Detection] = []
        try:
            detections = detection_queue.get_nowait()
            # tracked_objects = tracker.update(detections=detections)
        except queue.Empty:
            # tracked_objects = tracker.update()  # Predict without new detections
            pass

        # Draw tracked objects on the frame
        for obj in detections:
            points = obj.points
            # points = obj.estimate.astype(int)
            # Draw bounding box
            cv2.polylines(
                frame,
                [points.astype(int)],
                isClosed=True,
                color=[(0, 255, 0), (0, 255, 255), (0, 0, 255)][obj.label],
                thickness=2,
            )
            # Draw label and track ID if available
            # if obj.last_detection:
            #     label = str(obj.last_detection.label)
            #     cv2.putText(frame, label, (points[0, 0], points[0, 1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #     track_id = f"ID: {obj.id}"
            #     cv2.putText(frame, track_id, (points[0, 0], points[0, 1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("frame", frame)

        # Check for exit condition using imwait
        if imwait(delay=0, window_name="frame"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
