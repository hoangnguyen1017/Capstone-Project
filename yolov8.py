from ultralytics import YOLO
import torch

# ===============================
# CONFIG
# ===============================
DATA = "dataset.yaml"
MODEL = "yolov8m-pose.pt"   # hoặc yolo11m-pose.pt
PROJECT = "runs"
NAME = "benchmark_frozen"
DEVICE = 0


# ===============================
# LOAD MODEL
# ===============================
model = YOLO(MODEL)

# ===============================
# FREEZE ALL PARAMETERS
# ===============================
for p in model.model.parameters():
    p.requires_grad = False

print("\n[INFO] All parameters frozen.\n")

# ===============================
# TRAIN (SHORT RUN FOR METRICS)
# ===============================
model.train(
    data=DATA,
    epochs=15,
    imgsz=640,
    batch=4,
    device=DEVICE,

    workers=0,   # ⭐ FIX LỖI WINDOWS

    project=PROJECT,
    name=NAME,

    val=True,
    save=False,
)


print("\n[INFO] Benchmark training completed.\n")
