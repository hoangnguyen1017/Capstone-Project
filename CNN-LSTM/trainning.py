import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from collections import Counter
import wandb
import time
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import FallDataset
from model import CNN_BiLSTM
def plot_confusion(cm, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Fall", "Fall"],
        yticklabels=["Non-Fall", "Fall"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    return fig

def train_model(
    fall_dir,
    nonfall_dir,
    device="cpu",
    epochs=30,
    batch_size=16,
    lr=1e-3
):
    assert device == "cpu", " Script này dành cho CPU"
    wandb.init(
        project="fall-detection-comparison",
        config={
            "model": "CNN_BiLSTM",
            "device": "cpu",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seq_len": 32,
            "split": "70/20/10"
        }
    )
    dataset = FallDataset(fall_dir, nonfall_dir)
    N = len(dataset)
    if N == 0:
        raise RuntimeError(" Dataset rỗng kiểm tra lại đường dẫn .npy")

    indices = torch.randperm(N)

    train_size = int(0.7 * N)
    val_size   = int(0.2 * N)

    train_idx = indices[:train_size]
    val_idx   = indices[train_size:train_size + val_size]
    test_idx  = indices[train_size + val_size:]

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Dataset split | Train={len(train_set)} | Val={len(val_set)} | Test={len(test_set)}")

    model = CNN_BiLSTM().to(device)

    labels = dataset.labels
    cnt = Counter(labels)

    weights = torch.tensor(
        [1.0 / cnt[0], 1.0 / cnt[1]],
        dtype=torch.float32
    )

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_true, train_pred = [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_true.extend(y.tolist())
            train_pred.extend(logits.argmax(1).tolist())

        train_acc = accuracy_score(train_true, train_pred)
        train_f1  = f1_score(train_true, train_pred)

        model.eval()
        val_loss = 0.0
        val_true, val_pred = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                start = time.time()
                logits = model(x)
                latency = (time.time() - start) * 1000 / x.size(0)

                if latency > 0:
                    fps = 1000.0 / latency
                else:
                    fps = 0.0

                loss = criterion(logits, y)
                val_loss += loss.item()

                val_true.extend(y.tolist())
                val_pred.extend(logits.argmax(1).tolist())

        val_acc    = accuracy_score(val_true, val_pred)
        val_f1     = f1_score(val_true, val_pred)
        val_recall = recall_score(val_true, val_pred)

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss / len(train_loader),
            "train/acc": train_acc,
            "train/f1": train_f1,

            "val/loss": val_loss / len(val_loader),
            "val/acc": val_acc,
            "val/f1": val_f1,
            "val/recall": val_recall,

            "latency_ms": latency,
            "fps": fps
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "cnn_bilstm_best_cpu.pth")

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Val Loss: {val_loss / len(val_loader):.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val Recall: {val_recall:.4f} | "
            f"FPS: {fps:.1f}"
        )

    print("\nRunning TEST set evaluation...")
    model.load_state_dict(torch.load("cnn_bilstm_best_cpu.pth", map_location="cpu"))
    model.eval()

    test_true, test_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            test_true.extend(y.tolist())
            test_pred.extend(logits.argmax(1).tolist())

    test_acc    = accuracy_score(test_true, test_pred)
    test_f1     = f1_score(test_true, test_pred)
    test_recall = recall_score(test_true, test_pred)
    cm_test     = confusion_matrix(test_true, test_pred)

    print("TEST Confusion Matrix\n", cm_test)
    print(
        f"\nTEST RESULT | "
        f"Acc: {test_acc:.4f} | "
        f"F1: {test_f1:.4f} | "
        f"Recall: {test_recall:.4f}"
    )

    wandb.finish()

if __name__ == "__main__":
    fall_dir = r"D:\data_npy\Fall\labels"
    nonfall_dir = r"D:\data_npy\Non_Fall\labels"

    train_model(fall_dir, nonfall_dir, device="cpu")
