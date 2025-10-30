import os
from ultralytics import YOLO

# ==== Config chung ====
DATA = "dataset.yaml"
PRETRAINED = "yolo11m-pose.pt"  # đổi sang model medium
PROJECT = "runs"
DEVICE = 0
IMAGE_SIZE = 640
BATCH = 4  # bạn có thể tăng batch nếu GPU đủ VRAM

# ==== Config Phase ====
PHASES = [
    {
        "name": "finetune_pose_yolo_phase1",
        "model": PRETRAINED,
        "epochs": 30,       # tăng từ 20 lên 30
        "freeze": "backbone"
    },
    {
        "name": "finetune_pose_yolo_phase2",
        "model": f"{PROJECT}/finetune_pose_yolo_phase1/weights/last.pt",
        "epochs": 100,      # tăng từ 80 lên 100
        "freeze": 0
    }
]

def run_phase(config):
    name = config["name"]
    model_path = config["model"]
    epochs = config["epochs"]
    freeze = config["freeze"]
    print(f"\n=== Running {name} ===")

    # Check nếu checkpoint đã tồn tại
    ckpt_last = os.path.join(PROJECT, name, "weights", "last.pt")

    if os.path.exists(ckpt_last):
        # Nếu đã train đủ epochs thì coi như xong
        results_dir = os.path.join(PROJECT, name, "results.csv")
        if os.path.exists(results_dir):
            with open(results_dir, "r") as f:
                lines = f.readlines()
                trained_epochs = len(lines) - 1  # trừ header
            if trained_epochs >= epochs:
                print(f"[INFO] {name} đã hoàn thành {trained_epochs}/{epochs} epochs.")
                return

        print(f"[INFO] Tiếp tục training {name} từ {ckpt_last}")
        model = YOLO(ckpt_last)
        model.train(
            data=DATA,
            epochs=epochs,
            imgsz=IMAGE_SIZE,
            batch=BATCH,
            device=DEVICE,
            project=PROJECT,
            name=name,
            resume=True
        )
    else:
        print(f"[INFO] Bắt đầu training mới {name}")
        model = YOLO(model_path)
        model.train(
            data=DATA,
            epochs=epochs,
            imgsz=IMAGE_SIZE,
            batch=BATCH,
            device=DEVICE,
            project=PROJECT,
            name=name,
            freeze=freeze
        )

if __name__ == "__main__":
    for phase in PHASES:
        run_phase(phase)
