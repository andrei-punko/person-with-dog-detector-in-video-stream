from ultralytics import YOLO
import os

# Path to the source PyTorch weights, relative to the project root
MODEL_PT = os.path.join(os.path.dirname(__file__), "..", "models", "yolo26l.pt")

model = YOLO(MODEL_PT)

# Export to TensorRT FP16 engine. imgsz must match the value used during inference (IMGSZ in the analyzer scripts).
model.export(format="engine", device=0, half=True, imgsz=1280)
