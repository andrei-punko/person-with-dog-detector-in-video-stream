from ultralytics import YOLO
import cv2
import math
import os
import sys

# Загружаем предобученную модель YOLO (она уже знает классы 0: person, 16: dog)
model = YOLO("models/yolo26l.pt")

# Пороговое расстояние в реальных пикселях видео
DISTANCE_THRESHOLD = 100

# Разрешение видео определяется из самого файла

# Размер входного кадра для модели (вычисляется из разрешения видео)

# Ограничение анализа по времени (секунды)
MAX_DURATION_SEC = 3*60

# Папка для скриншотов
SCREENSHOTS_DIR = "screenshots"

# Получаем имя файла из аргументов командной строки
if len(sys.argv) < 2:
    print("Использование: python video-analyzer.py <video_file>")
    sys.exit(1)

VIDEO_FILE = sys.argv[1]
print(f"Анализируемый файл: {VIDEO_FILE}")

# Получаем FPS видео для подсчёта количества кадров
cap = cv2.VideoCapture(VIDEO_FILE)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Размер входного кадра для модели (не больше 1920, кратен 32)
IMGSZ = min(VIDEO_WIDTH, VIDEO_HEIGHT, 1920)
IMGSZ = (IMGSZ // 32) * 32
max_frames = min(int(fps * MAX_DURATION_SEC), total_frames)

print(f"FPS: {fps}, Всего кадров: {total_frames}, Анализируем: {max_frames} кадров (~{MAX_DURATION_SEC} сек)")

# Создаём папку для скриншотов
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
print(f"Скриншоты будут сохранены в: {SCREENSHOTS_DIR}/")

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
    if frame_idx % 2 != 0:
        continue

    if frame_idx >= max_frames:
        print(f"Достигнут лимит {MAX_DURATION_SEC} сек — остановка.")
        break

    # 1. Получаем координаты всех людей и собак в кадре
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
            print(f"  [кадр {frame_idx}] Собака обнаружена! conf={conf:.2f}, центр=({cx:.0f}, {cy:.0f})")

    # 2. Считаем расстояние между ними (с пересчётом в реальные пиксели)
    scale = VIDEO_WIDTH / IMGSZ
    for i, (px, py) in enumerate(persons):
        for j, (dx, dy) in enumerate(dogs):
            distance_model = math.sqrt((px - dx) ** 2 + (py - dy) ** 2)
            distance_real = distance_model * scale
            print(f"distance_real={distance_real}")

            # 3. Если расстояние меньше N пикселей -> фиксируем "Человек с собакой"
            if distance_real < DISTANCE_THRESHOLD:
                time_sec = frame_idx / fps
                print(f"Человек с собакой: person#{i} <-> dog#{j}, расстояние = {distance_real:.0f}px, время = {time_sec:.1f} сек")

                # Сохраняем скриншот (не чаще одного раза в секунду)
                if time_sec - last_screenshot_time >= 1.0:
                    video_name = os.path.splitext(os.path.basename(VIDEO_FILE))[0]
                    screenshot_path = os.path.join(SCREENSHOTS_DIR, f"{video_name}_{time_sec:.1f}s.jpg")
                    cv2.imwrite(screenshot_path, result.orig_img)
                    last_screenshot_time = time_sec
                    print(f"  Скриншот сохранён: {screenshot_path}")

cv2.destroyAllWindows()
