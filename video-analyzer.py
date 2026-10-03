from ultralytics import YOLO
import cv2
import math
import os
import sys
import numpy as np
from common import ScreenshotSaver, redact_url, setup_logging, source_label

LOG_FILE = "video-analyzer.log"
logger = setup_logging(LOG_FILE)

# --- Settings ---

MODEL_PATH = "models/yolo26l.engine"

# Distance threshold in pixels between person and dog centres to trigger an event
DISTANCE_THRESHOLD = 100

# Per-class confidence thresholds (COCO class 0 = person, 16 = dog)
CONF_THRESHOLDS = {
    0: 0.2,   # person
    16: 0.02,  # dog
}
MIN_CONF = min(CONF_THRESHOLDS.values())

# Maximum video duration to analyse (seconds)
MAX_DURATION_SEC = 3 * 60

SCREENSHOTS_DIR = "screenshots"

# Model input size; must match the size used when exporting the .engine file
IMGSZ = 1280


def draw_bounding_box(frame, x1, y1, x2, y2, cls, conf):
    """Draw a bounding box and class label on the frame."""
    label = "person" if cls == 0 else "dog"
    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# --- Entry point ---

if len(sys.argv) < 2:
    print("Usage: python video-analyzer.py <video_file>")
    sys.exit(1)

VIDEO_FILE = sys.argv[1]
logger.info(f"Analyzing file: {redact_url(VIDEO_FILE)}")

cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    logger.error(f"Could not open video file: {VIDEO_FILE}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

max_frames = min(int(fps * MAX_DURATION_SEC), total_frames)
logger.info(f"FPS: {fps}, total frames: {total_frames}, analysing: {max_frames} frames (~{MAX_DURATION_SEC}s)")

screenshots = ScreenshotSaver(SCREENSHOTS_DIR, source_label(VIDEO_FILE))
logger.info(f"Screenshots will be saved to: {SCREENSHOTS_DIR}/")

# Load and warm up the TensorRT model before processing the file
logger.info("Warming up TensorRT model...")
model = YOLO(MODEL_PATH)
model.track(source=np.zeros((640, 640, 3), dtype=np.uint8), device='cuda:0', verbose=False)

# Use stream=True so frames are yielded one by one without loading the whole video into memory
results = model.track(
    source=VIDEO_FILE,
    show=False,
    classes=[0, 16],
    conf=MIN_CONF,
    stream=True,
    imgsz=IMGSZ,
    device='cuda:0',
    verbose=False,
    persist=True
)

logger.info("Analysis started. Press 'Q' in the video window to stop.")

# --- Main loop ---

for frame_idx, result in enumerate(results):
    if frame_idx >= max_frames:
        logger.info(f"Limit reached: {MAX_DURATION_SEC}s — stopping.")
        break

    frame = result.orig_img.copy()
    boxes = result.boxes
    time_sec = frame_idx / fps

    persons = []
    dogs = []

    # --- Collect detections ---
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if conf < CONF_THRESHOLDS.get(cls, MIN_CONF):
                continue
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if cls == 0:
                persons.append((cx, cy))
            elif cls == 16:
                dogs.append((cx, cy))
                logger.info(f"  [frame {frame_idx}] Dog detected! conf={conf:.2f}, center=({cx:.0f}, {cy:.0f})")
            draw_bounding_box(frame, x1, y1, x2, y2, cls, conf)

    # --- Distance check ---
    for i, (px, py) in enumerate(persons):
        for j, (dx, dy) in enumerate(dogs):
            distance = math.sqrt((px - dx) ** 2 + (py - dy) ** 2)
            if distance < DISTANCE_THRESHOLD:
                logger.info(f"Person with dog: person#{i} <-> dog#{j}, distance={distance:.0f}px, time={time_sec:.1f}s")
                screenshots.save(frame, time_sec, f"{time_sec:.1f}s", logger)

    # --- Display (scale down to fit a 1080p monitor if needed) ---
    h, w = frame.shape[:2]
    scale = min(1920 / w, 1080 / h, 1.0)
    display_frame = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
    cv2.imshow("Video Analysis", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("Analysis interrupted by user.")
        break

cv2.destroyAllWindows()
logger.info("Analysis finished.")
