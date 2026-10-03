import os
import sys

from ultralytics import YOLO

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PROJECT_ROOT)

from pdd.config import load_config  # noqa: E402

# Path to the source PyTorch weights, relative to the project root
MODEL_PT = os.path.join(PROJECT_ROOT, "models", "yolo26l.pt")

model = YOLO(MODEL_PT)

# Export to TensorRT FP16 engine. imgsz comes from config.yaml so it always matches inference.
model.export(format="engine", device=0, half=True, imgsz=load_config()["model"]["imgsz"])
