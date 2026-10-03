from ultralytics import YOLO
import cv2
import math
import sys
import time
import numpy as np
from common import ScreenshotSaver, redact_url, setup_logging, source_label

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


# --- Stream helpers ---

def open_stream(url):
    """Try to open the stream with explicit timeouts. Returns None on failure."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, STREAM_TIMEOUT_MS,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, STREAM_TIMEOUT_MS,
    ])
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def wait_or_quit(seconds):
    """Wait for seconds. Returns True if the user pressed 'q'."""
    end = time.time() + seconds
    while time.time() < end:
        if cv2.waitKey(100) & 0xFF == ord('q'):
            return True
    return False


def connect(url, url_safe):
    """Connect to the stream, retrying with exponential back-off. Returns None if user quit."""
    delay = RECONNECT_MIN_DELAY
    while True:
        cap = open_stream(url)
        if cap is not None:
            return cap
        logger.warning(f"Could not connect to {url_safe}, retrying in {delay}s")
        if wait_or_quit(delay):
            return None
        delay = min(delay * 2, RECONNECT_MAX_DELAY)


# --- Entry point ---

if len(sys.argv) < 2:
    print("Usage: python stream-analyzer.py <stream_url>")
    sys.exit(1)

STREAM_URL = sys.argv[1]
STREAM_URL_SAFE = redact_url(STREAM_URL)
screenshots = ScreenshotSaver(SCREENSHOTS_DIR, source_label(STREAM_URL))
logger.info(f"Stream source: {STREAM_URL_SAFE}")

# Load and warm up the TensorRT model before opening the stream
model = YOLO(MODEL_PATH)
logger.info("Warming up TensorRT model...")
model.track(source=np.zeros((640, 640, 3), dtype=np.uint8), device='cuda:0', verbose=False)

cap = connect(STREAM_URL, STREAM_URL_SAFE)
if cap is None:
    logger.info("Analysis stopped by user.")
    sys.exit(0)

fps = cap.get(cv2.CAP_PROP_FPS)
VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
logger.info(f"FPS: {fps}, Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")
logger.info("Analysis started. Press 'Q' in the video window to stop.")


# --- Main loop ---

while True:
    ret, frame = cap.read()
    if not ret:
        logger.warning("Stream lost or frame read error, reconnecting...")
        cap.release()
        if wait_or_quit(RECONNECT_MIN_DELAY):
            cap = None
        else:
            cap = connect(STREAM_URL, STREAM_URL_SAFE)
        if cap is None:
            logger.info("Analysis stopped by user.")
            break
        logger.info("Stream reconnected.")
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

    # --- Display (scale down to fit a 1080p monitor if needed) ---
    h, w = frame.shape[:2]
    scale = min(1920 / w, 1080 / h, 1.0)
    display_frame = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
    cv2.imshow("Stream", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("Analysis stopped by user.")
        break

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
