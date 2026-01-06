import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    recall_score          
)
from collections import Counter
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import FallDataset
from model import ViTFallDetector

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", DEVICE)

wandb.init(
    project='ViT-Fall-Detection',
    config={
        'epochs': 30,
        'batch': 16,
        'lr': 2e-4,
        'seq_len': 32,
        'early_stop': 10
    }
)
cfg = wandb.config
fall_dir = r"D:\data_npy\Fall\labels"
nonfall_dir = r"D:\data_npy\Non_Fall\labels"
dataset = FallDataset(
    fall_dir,
    nonfall_dir,
    seq_len=cfg.seq_len,
    augment=True
)

N = len(dataset)
train_len = int(0.7 * N)
val_len   = int(0.2 * N)
test_len  = N - train_len - val_len

train_set, val_set, test_set = random_split(
    dataset, [train_len, val_len, test_len]
)

train_loader = DataLoader(train_set, cfg.batch, shuffle=True)
val_loader   = DataLoader(val_set, cfg.batch)
test_loader  = DataLoader(test_set, cfg.batch)

print(f"Dataset split | Train={train_len} | Val={val_len} | Test={test_len}")
cnt = Counter(dataset.labels)
total = cnt[0] + cnt[1]

class_weights = torch.tensor(
    [cnt[1] / total, cnt[0] / total],
    device=DEVICE
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)
sample_x, _ = dataset[0]
input_dim = sample_x.shape[-1]

model = ViTFallDetector(
    joints=14,
    channels=3,
    embed_dim=256,
    heads=4,
    classes=2,
    max_len=cfg.seq_len,
    dropout=0.2
).to(DEVICE)

optimizer = optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    weight_decay=1e-3
)

scheduler = ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5
)
def plot_confusion(cm, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='Blues',
        xticklabels=['Non-Fall', 'Fall'],
        yticklabels=['Non-Fall', 'Fall'],
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return fig
best_f1 = 0.0
patience_ctr = 0

for epoch in range(cfg.epochs):
    model.train()
    train_loss, train_preds, train_labels = 0, [], []

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()
        train_preds.extend(logits.argmax(1).cpu())
        train_labels.extend(y.cpu())

    train_loss /= len(train_loader)
    train_acc = accuracy_score(train_labels, train_preds)
    train_f1  = f1_score(train_labels, train_preds)
    model.eval()
    val_loss, val_preds, val_labels = 0, [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            val_loss += loss.item()
            val_preds.extend(logits.argmax(1).cpu())
            val_labels.extend(y.cpu())

    val_loss /= len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1  = f1_score(val_labels, val_preds)
    val_recall = recall_score(val_labels, val_preds)

    scheduler.step(val_f1)

    wandb.log({
        'epoch': epoch + 1,
        'train/loss': train_loss,
        'train/acc': train_acc,
        'train/f1': train_f1,
        'val/loss': val_loss,
        'val/acc': val_acc,
        'val/f1': val_f1,
        'val/recall': val_recall,
        'lr': optimizer.param_groups[0]['lr']
    })

    print(
        f"Epoch [{epoch+1:02d}] | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_acc:.3f} | "
        f"Val F1 {val_f1:.3f} | "
        f"Val Recall {val_recall:.3f}"
    )
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_ctr = 0
        torch.save(model.state_dict(), "best_vit_fall.pth")
    else:
        patience_ctr += 1

    if patience_ctr >= cfg.early_stop:
        print("Early stopping")
        break
model.load_state_dict(torch.load("best_vit_fall.pth"))
model.eval()

test_preds, test_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        logits = model(x.to(DEVICE))
        test_preds.extend(logits.argmax(1).cpu())
        test_labels.extend(y)

test_acc = accuracy_score(test_labels, test_preds)
test_f1  = f1_score(test_labels, test_preds)
test_recall = recall_score(test_labels, test_preds)

cm = confusion_matrix(test_labels, test_preds)
fig = plot_confusion(cm, "Test Confusion Matrix")

wandb.log({
    'test/acc': test_acc,
    'test/f1': test_f1,
    'test/recall': test_recall, 
    'test/confusion': wandb.Image(fig)
})

wandb.finish()

print(f"\nTEST Acc    : {test_acc:.4f}")
print(f"TEST F1     : {test_f1:.4f}")
print(f"TEST Recall : {test_recall:.4f}")
