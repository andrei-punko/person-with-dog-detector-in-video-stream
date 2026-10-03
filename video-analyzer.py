from ultralytics import YOLO
import cv2
import math
import os
import sys

# Load pretrained YOLO model (knows classes: 0=person, 16=dog)
model = YOLO("models/yolo26l.pt")

# Distance threshold in real video pixels
DISTANCE_THRESHOLD = 100

# Video resolution is determined from the file itself

# Input frame size for the model (calculated from video resolution)

# Analysis time limit (seconds)
MAX_DURATION_SEC = 3*60

# Screenshots folder
SCREENSHOTS_DIR = "screenshots"

# Get file name from command line arguments
if len(sys.argv) < 2:
    print("Usage: python video-analyzer.py <video_file>")
    sys.exit(1)

VIDEO_FILE = sys.argv[1]
print(f"Analyzing file: {VIDEO_FILE}")

# Get FPS from video for frame count calculation
cap = cv2.VideoCapture(VIDEO_FILE)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Input frame size for the model (max 1920, multiple of 32)
IMGSZ = min(VIDEO_WIDTH, VIDEO_HEIGHT, 1920)
IMGSZ = (IMGSZ // 32) * 32
max_frames = min(int(fps * MAX_DURATION_SEC), total_frames)

print(f"FPS: {fps}, Total frames: {total_frames}, Analyzing: {max_frames} frames (~{MAX_DURATION_SEC} sec)")

# Process ~10 frames per second regardless of video FPS
frame_skip = max(1, int(fps / 10))
print(f"Frame skip: every {frame_skip} frames (~10 fps)")

# Create screenshots folder
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
print(f"Screenshots will be saved to: {SCREENSHOTS_DIR}/")

# Запуск анализа видеопотока с пониженным порогом уверенности
results = model.track(
    source=VIDEO_FILE,
    show=True,
    classes=[0, 16],
    conf=0.01,
    stream=True,
    imgsz=IMGSZ,
    device='cuda:0',
)

last_screenshot_time = -1.0

for frame_idx, result in enumerate(results):
    if frame_idx % frame_skip != 0:
        continue

    if frame_idx >= max_frames:
        print(f"Limit reached: {MAX_DURATION_SEC} sec — stopping.")
        break

    # 1. Get coordinates of all persons and dogs in the frame
    boxes = result.boxes
    persons = []
    dogs = []

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if cls == 0:
            persons.append((cx, cy))
        elif cls == 16:
            dogs.append((cx, cy))
            print(f"  [frame {frame_idx}] Dog detected! conf={conf:.2f}, center=({cx:.0f}, {cy:.0f})")

    # 2. Calculate distances between them (with conversion to real pixels)
    scale = VIDEO_WIDTH / IMGSZ
    for i, (px, py) in enumerate(persons):
        for j, (dx, dy) in enumerate(dogs):
            distance_model = math.sqrt((px - dx) ** 2 + (py - dy) ** 2)
            distance_real = distance_model * scale
            print(f"distance_real={distance_real}")

            # 3. If distance is less than N pixels -> log "Person with dog"
            if distance_real < DISTANCE_THRESHOLD:
                time_sec = frame_idx / fps
                print(f"Person with dog: person#{i} <-> dog#{j}, distance = {distance_real:.0f}px, time = {time_sec:.1f} sec")

                # Save screenshot (no more than once per second)
                if time_sec - last_screenshot_time >= 1.0:
                    video_name = os.path.splitext(os.path.basename(VIDEO_FILE))[0]
                    screenshot_path = os.path.join(SCREENSHOTS_DIR, f"{video_name}_{time_sec:.1f}s.jpg")
                    cv2.imwrite(screenshot_path, result.orig_img)
                    last_screenshot_time = time_sec
                    print(f"  Screenshot saved: {screenshot_path}")

cv2.destroyAllWindows()
