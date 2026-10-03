from ultralytics import YOLO
import cv2
import math
import os
import sys
import time

# Load pretrained YOLO model (knows classes: 0=person, 16=dog)
model = YOLO("models/yolo26l.pt")

# Distance threshold in real video pixels
DISTANCE_THRESHOLD = 100

# Screenshots folder
SCREENSHOTS_DIR = "screenshots"

# Get stream source from command line arguments
if len(sys.argv) < 2:
    print("Usage: python stream-analyzer.py <stream_url>")
    print("Examples:")
    print("  python stream-analyzer.py rtsp://login:password@192.168.1.80:554/stream1")
    print("  python stream-analyzer.py http://192.168.1.100:8080/video")
    print("  python stream-analyzer.py 0  # webcam")
    sys.exit(1)

STREAM_URL = sys.argv[1]
# If argument is a number, it's a camera index
if STREAM_URL.isdigit():
    STREAM_URL = int(STREAM_URL)

print(f"Stream source: {STREAM_URL}")

# Connect to stream
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"Error: could not connect to stream {STREAM_URL}")
    sys.exit(1)

# Get stream parameters
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0  # default for streams without FPS
VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if VIDEO_WIDTH <= 0 or VIDEO_HEIGHT <= 0:
    VIDEO_WIDTH = 1920
    VIDEO_HEIGHT = 1080

print(f"FPS: {fps}, Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")

# Process ~10 frames per second regardless of stream FPS
frame_skip = max(1, int(fps / 10))
print(f"Frame skip: every {frame_skip} frames (~10 fps)")

# Input frame size for the model (max 1920, multiple of 32)
IMGSZ = min(VIDEO_WIDTH, VIDEO_HEIGHT, 1920)
IMGSZ = (IMGSZ // 32) * 32

print(f"IMGSZ: {IMGSZ}")

# Create screenshots folder
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
print(f"Screenshots will be saved to: {SCREENSHOTS_DIR}/")
print("Analysis started. Stop with Ctrl+C")

last_screenshot_time = -1.0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or frame read error.")
        break

    if frame_idx % frame_skip != 0:
        frame_idx += 1
        continue

    # Current system time
    time_sec = time.time()

    # Run tracking on current frame
    results = model.track(
        source=frame,
        show=False,
        classes=[0, 16],
        conf=0.03,
        imgsz=IMGSZ,
        device='cuda:0',
        verbose=False,
    )

    result = results[0]
    num_boxes = len(result.boxes)
    timestamp = time.strftime('%H:%M:%S', time.localtime(time_sec))
    print(f"[{timestamp}] Frame {frame_idx}: {num_boxes} objects detected")

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

    # 2. Calculate distances between them (with conversion to real pixels)
    scale = VIDEO_WIDTH / IMGSZ
    for i, (px, py) in enumerate(persons):
        for j, (dx, dy) in enumerate(dogs):
            distance_model = math.sqrt((px - dx) ** 2 + (py - dy) ** 2)
            distance_real = distance_model * scale

            # 3. If distance is less than N pixels -> log "Person with dog"
            if distance_real < DISTANCE_THRESHOLD:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_sec))
                print(f"[{timestamp}] Person with dog: person#{i} <-> dog#{j}, distance = {distance_real:.0f}px")

                # Save screenshot (no more than once per second)
                if time_sec - last_screenshot_time >= 1.0:
                    stream_name = str(STREAM_URL).replace("/", "_").replace(":", "_").replace(".", "_")
                    screenshot_path = os.path.join(SCREENSHOTS_DIR, f"{stream_name}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(time_sec))}.jpg")
                    cv2.imwrite(screenshot_path, frame)
                    last_screenshot_time = time_sec
                    print(f"  Screenshot saved: {screenshot_path}")

    # Draw bounding boxes on the frame
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = "person" if cls == 0 else "dog"
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    frame_idx += 1

    # Scale window to fit FHD (1920x1080)
    max_width = 1920
    max_height = 1080
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cv2.imshow("Stream", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
