from ultralytics import YOLO

# 1. Загружаем стандартную модель PyTorch (например, Medium-версию)
model = YOLO("yolo26m.pt")

# 2. Экспортируем её в формат TensorRT с включением FP16 (half=True)
# Параметр imgsz=640 или 1280 задает размер сетки, на которой модель будет обучаться/искать объекты
model.export(format="engine", device=0, half=True, imgsz=1280)
