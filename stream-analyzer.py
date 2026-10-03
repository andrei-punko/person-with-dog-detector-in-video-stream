from ultralytics import YOLO
import argparse
import cv2
import math
import time
import numpy as np
from common import ScreenshotSaver, ThreadedVideoCapture, redact_url, setup_logging, source_label

LOG_FILE = "stream-analyzer.log"
logger = setup_logging(LOG_FILE)


def draw_bounding_box(frame, x1, y1, x2, y2, cls, conf):
    """Draw a bounding box and class label on the frame."""
    label = "person" if cls == 0 else "dog"
    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def log_time(res):
    """Log inference timing and effective FPS for the given result."""
    preprocess_speed = res.speed.get('preprocess', 0.0)
    inference_speed = res.speed.get('inference', 0.0)
    postprocess_speed = res.speed.get('postprocess', 0.0)
    total_speed = preprocess_speed + inference_speed + postprocess_speed
    fps_hardware = 1000 / total_speed if total_speed > 0 else 0.0
    logger.info(f"SPEED: Inference: {inference_speed:.1f}ms | Total: {total_speed:.1f}ms ({fps_hardware:.1f} FPS)")


# --- Settings ---

MODEL_PATH = "models/yolo26l.engine"

# Distance threshold in pixels between person and dog centres to trigger an event
DISTANCE_THRESHOLD = 100

# Per-class confidence thresholds (COCO class 0 = person, 16 = dog)
CONF_THRESHOLDS = {
    0: 0.2,   # person
    16: 0.02,  # dog
}

SCREENSHOTS_DIR = "screenshots"

# Reconnect settings: connection/read timeout and back-off delays (seconds)
STREAM_TIMEOUT_MS = 5000
RECONNECT_MIN_DELAY = 1
RECONNECT_MAX_DELAY = 30

# Model input size; must match the size used when exporting the .engine file
IMGSZ = 1280


# --- Entry point ---

parser = argparse.ArgumentParser(description="Detect persons and dogs in an RTSP stream.")
parser.add_argument("stream_url", help="RTSP or other stream URL")
parser.add_argument("--no-display", action="store_true", help="Disable the video window (for headless servers)")
args = parser.parse_args()

STREAM_URL = args.stream_url
NO_DISPLAY = args.no_display
STREAM_URL_SAFE = redact_url(STREAM_URL)
screenshots = ScreenshotSaver(SCREENSHOTS_DIR, source_label(STREAM_URL))
logger.info(f"Stream source: {STREAM_URL_SAFE}")

# Load and warm up the TensorRT model before opening the stream
model = YOLO(MODEL_PATH)
logger.info("Warming up TensorRT model...")
model.track(source=np.zeros((640, 640, 3), dtype=np.uint8), device='cuda:0', verbose=False)

cap = ThreadedVideoCapture(
    STREAM_URL, logger, STREAM_TIMEOUT_MS, RECONNECT_MIN_DELAY, RECONNECT_MAX_DELAY
).start()
logger.info("Analysis started. Press 'Q' in the video window to stop.")


# --- Main loop ---

try:
    while True:
        frame = cap.read(timeout=1.0)
        if frame is None:
            if not NO_DISPLAY and cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Analysis stopped by user.")
                break
            continue

        time_sec = time.time()

        results = model.track(
            source=frame,
            show=False,
            classes=[0, 16],
            conf=min(CONF_THRESHOLDS.values()),
            imgsz=IMGSZ,
            device='cuda:0',
            verbose=False,
            persist=True
        )

        result = results[0]
        # Uncomment to log per-frame inference timing:
        # log_time(result)
        boxes = result.boxes

        # --- Collect detections ---
        raw_persons = []
        raw_dogs = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                if conf < CONF_THRESHOLDS.get(cls, 0.25):
                    continue
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                if cls == 0:
                    raw_persons.append({"coords": (x1, y1, x2, y2), "conf": conf})
                elif cls == 16:
                    raw_dogs.append({"coords": (x1, y1, x2, y2), "conf": conf})

        # --- False-positive filter ---
        # Drop a dog detection whose centre falls inside a person box and whose
        # confidence is below 0.15; this catches the common "hood as dog" mistake.
        persons = []
        dogs = []

        for p in raw_persons:
            x1, y1, x2, y2 = p["coords"]
            persons.append(((x1 + x2) / 2, (y1 + y2) / 2))
            draw_bounding_box(frame, x1, y1, x2, y2, 0, p["conf"])

        for d in raw_dogs:
            dx1, dy1, dx2, dy2 = d["coords"]
            dcx = (dx1 + dx2) / 2
            dcy = (dy1 + dy2) / 2
            is_false_dog = any(
                px1 <= dcx <= px2 and py1 <= dcy <= py2 and d["conf"] < 0.15
                for p in raw_persons
                for px1, py1, px2, py2 in [p["coords"]]
            )
            if not is_false_dog:
                dogs.append((dcx, dcy))
                draw_bounding_box(frame, dx1, dy1, dx2, dy2, 16, d["conf"])

        # --- Distance check ---
        for i, (px, py) in enumerate(persons):
            for j, (dx, dy) in enumerate(dogs):
                distance = math.sqrt((px - dx) ** 2 + (py - dy) ** 2)
                if distance < DISTANCE_THRESHOLD:
                    logger.info(f"Person with dog: person#{i} <-> dog#{j}, distance={distance:.0f}px")
                    timestamp = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(time_sec))}_{int(time_sec * 1000) % 1000:03d}"
                    screenshots.save(frame, time_sec, timestamp, logger)

        if not NO_DISPLAY:
            # Scale down to fit a 1080p monitor if needed
            h, w = frame.shape[:2]
            scale = min(1920 / w, 1080 / h, 1.0)
            display_frame = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
            cv2.imshow("Stream", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Analysis stopped by user.")
                break

except KeyboardInterrupt:
    logger.info("Analysis interrupted by Ctrl+C.")
finally:
    cap.stop()
    if not NO_DISPLAY:
        cv2.destroyAllWindows()
