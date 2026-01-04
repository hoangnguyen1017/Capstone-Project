import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
import wandb
from collections import Counter
from Preprocess import NpySequenceDataset, collate_batch
from Model import STGCNThreeStream
import matplotlib.pyplot as plt
import seaborn as sns
from torch_optimizer import RAdam, Lookahead

# ===== Config =====
FALL_DIR = r"C:\Users\giang\Desktop\data\data_npy\Fall\labels"
NONFALL_DIR = r"C:\Users\giang\Desktop\data\data_npy\Non_Fall\labels"
BATCH_SIZE = 16
EPOCHS = 30
LR = 3e-4
T = 32
PATIENCE = 7
MAX_GRAD_NORM = 3
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()

# ===== Utilities =====
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total_params), int(trainable_params)

# ===== Checkpoint Utils =====
def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_f1, scaler=None):
    optimizer_state = optimizer.optimizer.state_dict() if hasattr(optimizer, 'optimizer') else optimizer.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler.state_dict(),
        "best_val_f1": best_val_f1
    }
    if scaler is not None:
        try:
            checkpoint["scaler_state"] = scaler.state_dict()
        except Exception:
            pass
    torch.save(checkpoint, path)
    print(f"Full checkpoint saved: {path}")

# ===== EarlyStopping =====
class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=5e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if score < self.best_score + self.min_delta:
            self.counter += 1
            return self.counter >= self.patience
        else:
            self.best_score = score
            self.counter = 0
        return False

# ===== Validation =====
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for joint, bone, motion, labels in loader:
            joint, bone, motion, labels = joint.to(DEVICE), bone.to(DEVICE), motion.to(DEVICE), labels.long().to(DEVICE)
            
            use_amp = DEVICE.type == "cuda"
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _ = model(joint, bone, motion)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, zero_division=0), all_labels, all_preds

# ===== Training =====
def train():
    set_seed()
    print(f"Using device: {DEVICE}")

    # ===================================================
    # 1️⃣ Build FULL dataset (KHÔNG augment – chỉ để lấy label)
    # ===================================================
    base_ds = NpySequenceDataset(
        FALL_DIR,
        NONFALL_DIR,
        seq_len=T,
        augment=False
    )

    sample_indices = np.arange(len(base_ds))
    sample_labels = np.array([label for _, label in base_ds.samples])

    # ===================================================
    # 2️⃣ Stratified split
    # ===================================================
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        sample_indices,
        sample_labels,
        test_size=0.3,
        stratify=sample_labels,
        random_state=42
    )

    val_idx, test_idx, val_labels, test_labels = train_test_split(
        temp_idx,
        temp_labels,
        test_size=1/3,
        stratify=temp_labels,
        random_state=42
    )

    print("Train:", np.bincount(train_labels))
    print("Val  :", np.bincount(val_labels))
    print("Test :", np.bincount(test_labels))

    # ===================================================
    # 3️⃣ Build SEPARATE datasets (augment đúng chỗ)
    # ===================================================
    def build_dataset_from_indices(indices, augment):
        ds = NpySequenceDataset(
            FALL_DIR,
            NONFALL_DIR,
            seq_len=T,
            augment=augment
        )
        ds.samples = [ds.samples[i] for i in indices]
        return ds

    train_ds = build_dataset_from_indices(train_idx, augment=True)
    val_ds   = build_dataset_from_indices(val_idx,   augment=False)
    test_ds  = build_dataset_from_indices(test_idx,  augment=False)
    # ===================================================
    #           ⭐ ADD WeightedRandomSampler ⭐
    # ===================================================
    train_labels_arr = np.array(train_labels)

    class_counts = np.bincount(train_labels_arr)
    class_weights = 1.0 / np.sqrt(class_counts)
    sample_weights = class_weights[train_labels_arr]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        collate_fn=collate_batch
    )
    # ===================================================

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # Model
    model = STGCNThreeStream(num_class=2, in_channels_each=3).to(DEVICE)

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params}, Trainable: {trainable_params}")

    # Phân nhóm params: decay / no_decay (giữ nguyên từ AdamW)
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "ln" in name.lower() or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    radam = RAdam(
        [
            {"params": decay_params, "weight_decay": 1e-4},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LR
    )

    # Wrap RAdam with Lookahead
    optimizer = Lookahead(radam, k=5, alpha=0.5)

    # ===== Scheduler: CosineAnnealingLR =====
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,  # Lookahead
        T_max=EPOCHS,
        eta_min=1e-5
    )

    criterion = nn.CrossEntropyLoss()

    early_stopper = EarlyStopping(patience=PATIENCE)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler()

    # WandB
    wandb.init(project="fall-detection-stgcn", config={
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "optimizer": "RAdam+Lookahead",
        "loss": "CrossEntropy + WeightedSampler",
        "scheduler": "CosineAnnealingLR",
    })
    wandb.watch(model, log="all", log_freq=200)

    best_val_f1 = 0.0
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_full_checkpoint.pth")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        all_labels, all_preds = [], []

        for joint, bone, motion, labels in train_loader:
            joint, bone, motion, labels = joint.to(DEVICE), bone.to(DEVICE), motion.to(DEVICE), labels.long().to(DEVICE)
            optimizer.zero_grad()
            use_amp = DEVICE.type == "cuda"
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _ = model(joint, bone, motion)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if MAX_GRAD_NORM:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        train_loss_avg = total_loss / max(len(train_loader), 1)
        train_f1 = f1_score(all_labels, all_preds, zero_division=0)
        train_acc = accuracy_score(all_labels, all_preds)
        train_fall_recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)

        # ---- Validation ----
        val_loss, val_acc, val_f1, val_labels_all, val_preds_all = validate(model, val_loader, criterion)

        fall_recall = recall_score(val_labels_all,val_preds_all,pos_label=1,zero_division=0)

        # Scheduler step AFTER validation (CosineAnnealingLR uses epoch step)
        scheduler.step()

        print(f"\nEpoch {epoch}/{EPOCHS} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
        print(classification_report(val_labels_all, val_preds_all, digits=4))
        print("Confusion Matrix:\n", confusion_matrix(val_labels_all, val_preds_all))

        # Save best model (include scaler state)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, best_val_f1, scaler=scaler)
            print(f"Saved BEST model at epoch {epoch} | Val F1: {best_val_f1:.4f}")

        # Early stopping
        if early_stopper.step(val_f1):
            print(f"Early stopping triggered at epoch {epoch}")
            break

        # WandB log
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "train_f1": train_f1,
            "train_acc": train_acc,
            "train_fall_recall": train_fall_recall,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "val_acc": val_acc,
            "fall_recall": fall_recall,
            "lr": optimizer.param_groups[0]['lr']
        })

    print(f"\nTraining complete! Best Val F1: {best_val_f1:.4f}")
    
    # ---- Final Test Evaluation ---- 
    print("\n=== Evaluating on Test Set ===") 
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    test_loss, test_acc, test_f1, test_labels_all, test_preds_all = validate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    print(classification_report(test_labels_all, test_preds_all, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(test_labels_all, test_preds_all))
    # ==== Plot Confusion Matrix ====
    cm = confusion_matrix(test_labels_all, test_preds_all)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Non-Fall", "Fall"],
                yticklabels=["Non-Fall", "Fall"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Fall Detection")
    plt.show()

    wandb.log({
        "test_loss": test_loss,
        "test_f1": test_f1,
        "test_acc": test_acc
    })

    wandb.finish()

if __name__ == "__main__":
    train()