from ultralytics import YOLO
import cv2
import math
import os
import sys
import numpy as np
from common import ScreenshotSaver, redact_url, setup_logging, source_label

LOG_FILE = "video-analyzer.log"
logger = setup_logging(LOG_FILE)

# --- НАСТРОЙКИ ---
# Путь к оптимизированной TensorRT модели
MODEL_PATH = "models/yolo26l.engine"

# Distance threshold in real video pixels
DISTANCE_THRESHOLD = 100

# Ваши кастомные пороги уверенности
CONF_THRESHOLDS = {
    0: 0.2,   # person
    16: 0.02,  # dog
}
MIN_CONF_THRESHOLDS = min(CONF_THRESHOLDS.values())

# Ограничение по времени анализа файла (в секундах)
MAX_DURATION_SEC = 3 * 60

# Папка для скриншотов
SCREENSHOTS_DIR = "screenshots"

# Входной размер картинки для модели
IMGSZ = 1280


def draw_bounding_box(frame, x1, y1, x2, y2, cls, conf):
    """Рисует рамку и метку класса на кадре."""
    label = "person" if cls == 0 else "dog"
    color = (0, 255, 0) if cls == 0 else (0, 0, 255)

    # Отрисовка прямоугольника объекта
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Отрисовка текста
    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# --- ПОДГОТОВКА И ПРОВЕРКА АРГУМЕНТОВ ---
if len(sys.argv) < 2:
    print("Usage: python video-analyzer.py <video_file>")
    sys.exit(1)

VIDEO_FILE = sys.argv[1]
VIDEO_FILE_SAFE = redact_url(VIDEO_FILE)
logger.info(f"Analyzing file: {VIDEO_FILE_SAFE}")

# Получаем параметры видеофайла
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    logger.error(f"Could not open video file {VIDEO_FILE_SAFE}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

max_frames = min(int(fps * MAX_DURATION_SEC), total_frames)
logger.info(f"FPS: {fps}, Total frames: {total_frames}, Analyzing: {max_frames} frames (~{MAX_DURATION_SEC} sec)")

screenshots = ScreenshotSaver(SCREENSHOTS_DIR, source_label(VIDEO_FILE))
logger.info(f"Screenshots will be saved to: {SCREENSHOTS_DIR}/")

# --- ИНИЦИАЛИЗАЦИЯ И ПРОГРЕВ МОДЕЛИ ---
logger.info("Warming up TensorRT model...")
model = YOLO(MODEL_PATH)
model.track(source=np.zeros((640, 640, 3), dtype=np.uint8), device='cuda:0', verbose=False)

# Запуск анализа видеофайла с минимальным порогом детектора (0.01)
results = model.track(
    source=VIDEO_FILE,
    show=False,  # Отключаем дефолтное окно Ultralytics, рисуем сами через cv2.imshow
    classes=[0, 16],
    conf=MIN_CONF_THRESHOLDS,
    stream=True,
    imgsz=IMGSZ,
    device='cuda:0',
    verbose=False,
    persist=True
)

logger.info("Analysis started. Press 'Q' in the video window to stop.")

# --- ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ---
for frame_idx, result in enumerate(results):
    if frame_idx >= max_frames:
        logger.info(f"Limit reached: {MAX_DURATION_SEC} sec — stopping.")
        break

    # Берем оригинальный кадр для отрисовки рамок
    frame = result.orig_img.copy()
    boxes = result.boxes

    persons = []
    dogs = []

    # --- СБОР КООРДИНАТ И ОТРИСОВКА ---
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            # ИСПРАВЛЕНО: Убрали индексы, так как box — это одиночный объект в итераторе
            cls = int(box.cls)
            conf = float(box.conf)

            # Фильтрация по вашим кастомным порогам
            if conf < CONF_THRESHOLDS.get(cls, MIN_CONF_THRESHOLDS):
                continue

            # ИСПРАВЛЕНО: Исправлена распаковка двумерного списка xyxy
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if cls == 0:
                persons.append((cx, cy))
            elif cls == 16:
                dogs.append((cx, cy))
                logger.info(f"  [frame {frame_idx}] Dog detected! conf={conf:.2f}, center=({cx:.0f}, {cy:.0f})")

            # Вызов общего метода отрисовки
            draw_bounding_box(frame, x1, y1, x2, y2, cls, conf)

    # --- РАСЧЕТ РАССТОЯНИЙ ---
    # ИСПРАВЛЕНО: Убран коэффициент scale, так как координаты коробки уже соответствуют пикселям оригинального кадра
    for i, (px, py) in enumerate(persons):
        for j, (dx, dy) in enumerate(dogs):
            distance_real = math.sqrt((px - dx) ** 2 + (py - dy) ** 2)

            # Если расстояние меньше порога -> фиксируем событие
            if distance_real < DISTANCE_THRESHOLD:
                time_sec = frame_idx / fps
                logger.info(f"Person with dog: person#{i} <-> dog#{j}, distance = {distance_real:.0f}px, time = {time_sec:.1f} sec")

                # Сохраняем кадр уже с нарисованными рамками
                screenshots.save(frame, time_sec, f"{time_sec:.1f}s", logger)

    # --- ВИЗУАЛИЗАЦИЯ (ОКНО) ---
    # Масштабируем картинку, если видео в 2K, чтобы оно аккуратно влезало в FHD монитор
    max_width = 1920
    max_height = 1080
    h, w = frame.shape[:2]
    scale_window = min(max_width / w, max_height / h, 1.0)

    display_frame = frame
    if scale_window < 1.0:
        display_frame = cv2.resize(frame, (int(w * scale_window), int(h * scale_window)))

    cv2.imshow("Video Analysis", display_frame)

    # Кнопка 'q' закроет видео досрочно
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("Analysis interrupted by user.")
        break

cv2.destroyAllWindows()
logger.info("Analysis finished.")
