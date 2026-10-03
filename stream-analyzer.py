from ultralytics import YOLO
import cv2
import math
import os
import sys
import time

# Загружаем предобученную модель YOLO (она уже знает классы 0: person, 16: dog)
model = YOLO("models/yolo26l.pt")

# Пороговое расстояние в реальных пикселях видео
DISTANCE_THRESHOLD = 100

# Папка для скриншотов
SCREENSHOTS_DIR = "screenshots"

# Получаем источник потока из аргументов командной строки
if len(sys.argv) < 2:
    print("Использование: python stream-analyzer.py <stream_url>")
    print("Примеры:")
    print("  python stream-analyzer.py rtsp://login:password@192.168.1.80:554/stream1")
    print("  python stream-analyzer.py http://192.168.1.100:8080/video")
    print("  python stream-analyzer.py 0  # веб-камера")
    sys.exit(1)

STREAM_URL = sys.argv[1]
# Если аргумент — число, значит это индекс камеры
if STREAM_URL.isdigit():
    STREAM_URL = int(STREAM_URL)

print(f"Источник потока: {STREAM_URL}")

# Подключаемся к потоку
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"Ошибка: не удалось подключиться к потоку {STREAM_URL}")
    sys.exit(1)

# Получаем параметры потока
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0  # значение по умолчанию для потоков без FPS
VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if VIDEO_WIDTH <= 0 or VIDEO_HEIGHT <= 0:
    VIDEO_WIDTH = 1920
    VIDEO_HEIGHT = 1080

print(f"FPS: {fps}, Разрешение: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")

# Обрабатывать ~10 кадров в секунду независимо от FPS потока
frame_skip = max(1, int(fps / 10))
print(f"Frame skip: каждый {frame_skip}-й кадр (~10 кадров/сек)")

# Размер входного кадра для модели (не больше 1920, кратен 32)
IMGSZ = min(VIDEO_WIDTH, VIDEO_HEIGHT, 1920)
IMGSZ = (IMGSZ // 32) * 32

print(f"IMGSZ: {IMGSZ}")

# Создаём папку для скриншотов
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
print(f"Скриншоты будут сохранены в: {SCREENSHOTS_DIR}/")
print("Анализ запущен. Остановка — Ctrl+C")

last_screenshot_time = -1.0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Поток завершён или ошибка чтения кадра.")
        break

    if frame_idx % frame_skip != 0:
        frame_idx += 1
        continue

    # Текущее системное время
    time_sec = time.time()

    # Запускаем трекинг на текущем кадре
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

    # Масштабируем окно, чтобы влезло в FHD (1920x1080)
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
