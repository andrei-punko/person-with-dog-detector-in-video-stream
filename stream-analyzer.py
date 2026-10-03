from ultralytics import YOLO
import cv2
import math
import os
import sys
import time
import logging
import re
import numpy as np
from urllib.parse import urlsplit

# Logging setup
LOG_FILE = "stream-analyzer.log"
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def draw_bounding_box(frame, x1, y1, x2, y2, cls, conf):
    """Рисует рамку и метку класса на кадре."""
    label = "person" if cls == 0 else "dog"
    color = (0, 255, 0) if cls == 0 else (0, 0, 255)

    # Отрисовка прямоугольника объекта
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Отрисовка текста с подложкой или просто поверх
    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def redact_url(url):
    """Возвращает URL без логина, пароля и query-параметров (безопасно для логов)."""
    parts = urlsplit(url)
    if parts.username is None and parts.password is None and not parts.query:
        return url
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    netloc = f"***@{host}" if (parts.username is not None or parts.password is not None) else parts.netloc
    return parts._replace(netloc=netloc, query="***" if parts.query else "").geturl()


def stream_label(url):
    """Короткое имя потока для файлов: хост, порт и путь, без учетных данных."""
    parts = urlsplit(url)
    raw = f"{parts.hostname or 'stream'}_{parts.port or ''}_{parts.path}"
    return re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")


def log_time(res):
    # Извлекаем время в миллисекундах из словаря скоростей модели
    preprocess_speed = res.speed.get('preprocess', 0.0)
    inference_speed = res.speed.get('inference', 0.0)
    postprocess_speed = res.speed.get('postprocess', 0.0)
    total_speed = preprocess_speed + inference_speed + postprocess_speed

    # Считаем реальный FPS обработки видеокартой
    fps_hardware = 1000 / total_speed if total_speed > 0 else 0.0

    logger.info(f"SPEED: Inference: {inference_speed:.1f}ms | Total: {total_speed:.1f}ms ({fps_hardware:.1f} FPS)")


# Load pretrained YOLO model
model = YOLO("models/yolo26l.engine")

# "Прогрев" TensorRT модели (warmup)
logger.info("Warming up TensorRT model...")
model.track(source=np.zeros((640, 640, 3), dtype=np.uint8), device='cuda:0', verbose=False)

# Distance threshold in real video pixels
DISTANCE_THRESHOLD = 100

# Пороги уверенности
CONF_THRESHOLDS = {
    0: 0.2,   # person
    16: 0.02,  # dog
}

SCREENSHOTS_DIR = "screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

if len(sys.argv) < 2:
    print("Usage: python stream-analyzer.py <stream_url>")
    sys.exit(1)

STREAM_URL = sys.argv[1]
STREAM_URL_SAFE = redact_url(STREAM_URL)
STREAM_LABEL = stream_label(STREAM_URL)
logger.info(f"Stream source: {STREAM_URL_SAFE}")

cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    logger.error(f"Could not connect to stream {STREAM_URL_SAFE}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

logger.info(f"FPS: {fps}, Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")

IMGSZ = 1280
logger.info("Analysis started. Press 'Q' in the video window to stop.")

last_screenshot_time = -1.0

while True:
    ret, frame = cap.read()
    if not ret:
        logger.info("Stream ended or frame read error.")
        break

    time_sec = time.time()

    # Запускаем трекинг
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
    # log processing speed (inference in ms)
    # log_time(result)
    boxes = result.boxes

    persons = []
    dogs = []

    # --- СБОР ДАННЫХ И ОТРИСОВКА ---
    raw_persons = []
    raw_dogs = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)

            # Проверка по вашему словарю порогов
            if conf < CONF_THRESHOLDS.get(cls, 0.25):
                continue

            x1, y1, x2, y2 = box.xyxy[0].int().tolist()

            # Сохраняем все детекции во временные списки для геометрического анализа
            if cls == 0:
                raw_persons.append({"coords": (x1, y1, x2, y2), "conf": conf})
            elif cls == 16:
                raw_dogs.append({"coords": (x1, y1, x2, y2), "conf": conf})

    # --- ГЕОМЕТРИЧЕСКИЙ ФИЛЬТР ЛОЖНЫХ НАЛОЖЕНИЙ ---
    persons = []
    dogs = []

    # 1. Сначала утверждаем всех валидных людей
    for p in raw_persons:
        x1, y1, x2, y2 = p["coords"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        persons.append((cx, cy))
        draw_bounding_box(frame, x1, y1, x2, y2, 0, p["conf"])

    # 2. Фильтруем собак, проверяя, не сидят ли они на голове у человека
    for d in raw_dogs:
        dx1, dy1, dx2, dy2 = d["coords"]
        dcx = (dx1 + dx2) / 2
        dcy = (dy1 + dy2) / 2

        is_false_dog = False

        for p in raw_persons:
            px1, py1, px2, py2 = p["coords"]

            # Проверяем, находится ли центр "собаки" внутри рамки человека
            # Или перекрывает ли рамка собаки верхнюю часть тела (голову/плечи)
            if (px1 <= dcx <= px2) and (py1 <= dcy <= py2):
                # Если собака внутри человека, но ее conf очень низкий — это 100% ошибка капюшона
                if d["conf"] < 0.15:
                    is_false_dog = True
                    break

        if not is_false_dog:
            dogs.append((dcx, dcy))
            draw_bounding_box(frame, dx1, dy1, dx2, dy2, 16, d["conf"])

    if len(persons) > 0 or len(dogs) > 0:
        logger.info(f"Detected: {len(persons)} persons, {len(dogs)} dogs")

    # --- РАСЧЕТ РАССТОЯНИЙ ---
    for i, (px, py) in enumerate(persons):
        for j, (dx, dy) in enumerate(dogs):
            distance_real = math.sqrt((px - dx) ** 2 + (py - dy) ** 2)

            if distance_real < DISTANCE_THRESHOLD:
                logger.info(f"Person with dog: person#{i} <-> dog#{j}, distance = {distance_real:.0f}px")

                # Сохранение скриншота (не чаще 3 раз в секунду)
                if time_sec - last_screenshot_time >= 0.33:
                    screenshot_path = os.path.join(SCREENSHOTS_DIR, f"{STREAM_LABEL}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(time_sec))}.jpg")
                    cv2.imwrite(screenshot_path, frame)
                    last_screenshot_time = time_sec
                    logger.info(f"  Screenshot saved: {screenshot_path}")

    # Изменение размера окна для вывода на экран (масштабируем под FHD)
    max_width = 1920
    max_height = 1080
    h, w = frame.shape[:2]
    scale_window = min(max_width / w, max_height / h, 1.0)

    display_frame = frame
    if scale_window < 1.0:
        display_frame = cv2.resize(frame, (int(w * scale_window), int(h * scale_window)))

    cv2.imshow("Stream", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("Analysis stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()
