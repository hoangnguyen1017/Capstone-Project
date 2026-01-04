import os
import torch
import torch.nn as nn
from ultralytics import YOLO

class PoseConvAdapter(nn.Module):
    def __init__(self, channels, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1, 1)
        )

    def forward(self, x):
        return x + self.conv(x)

def add_adapter_to_pose(model):
    print("\n=== ADDING CONV ADAPTER TO POSE HEAD ===\n")

    full_model = model.model.model
    pose_head = full_model[23]  # YOLO11m Pose Head

    if hasattr(pose_head, "cv4") and isinstance(pose_head.cv4, nn.ModuleList):
        last = pose_head.cv4[0]

        if isinstance(last, nn.Sequential):
            out_c = last[-1].out_channels
            last.add_module("adapter", PoseConvAdapter(out_c))
            print("[OK] Adapter added to pose_head.cv4[0]\n")
        else:
            print("[FAIL] cv4[0] không phải Sequential")
    else:
        print("[FAIL] Không thấy pose_head.cv4")

    return model

def freeze_pose_only(model):
    print("\n=== FREEZING EVERYTHING EXCEPT POSE HEAD + ADAPTER ===\n")

    full_model = model.model.model
    pose_head = full_model[23]
    for _, p in model.named_parameters():
        p.requires_grad = False

    for name, p in pose_head.named_parameters():
        p.requires_grad = True
        print("UNFREEZE:", name)

    print("\n[INFO] Freeze complete.\n")


def train_pose(resume=False):
    DATA = "dataset.yaml"
    MODEL = "yolo11m-pose.pt"
    PROJECT = "runs"
    NAME = "finetune_pose_stable"
    DEVICE = 0

    if resume:
        print("\n[INFO] Resume training...\n")
        model = YOLO(f"{PROJECT}/{NAME}/weights/last.pt")
    else:
        print("\n[INFO] Loading base model...\n")
        model = YOLO(MODEL)
        add_adapter_to_pose(model)
        freeze_pose_only(model)
    model.train(
        data=DATA,
        epochs=300,
        imgsz=640,
        batch=4,
        device=DEVICE,
        workers=4,
        project=PROJECT,
        name=NAME,

        lr0=0.0005,         
        optimizer="SGD",    
        momentum=0.937,

        cos_lr=True,
        patience=30,
        save_period=10,

        resume=resume,
    )

    print("\n[INFO] Training Completed\n")

if __name__ == "__main__":
    train_pose(resume=True)
