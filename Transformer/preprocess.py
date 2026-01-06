import os
import numpy as np
import torch
from torch.utils.data import Dataset


def pad_or_crop(seq, target_len=32, center_crop=True):
    T = seq.shape[0]
    if T == target_len:
        return seq
    if T < target_len:
        return np.pad(seq, ((0, target_len - T),(0,0),(0,0)), mode="edge")
    start = (T - target_len) // 2 if center_crop else 0
    return seq[start:start + target_len]

def spatial_jitter(seq_xy, sigma=0.004):
    noise = np.random.normal(0, sigma, seq_xy.shape)
    mask = np.linalg.norm(seq_xy, axis=2, keepdims=True) > 0
    return seq_xy + noise * mask

def temporal_jitter_fine(seq_xy, sigma=0.004):
    noise = np.random.normal(0, sigma, seq_xy.shape)
    mask = np.linalg.norm(seq_xy, axis=2, keepdims=True) > 0
    return seq_xy + noise * mask

def motion_scale_safe(seq_xy, scale_range=(0.95,1.05)):
    delta = np.zeros_like(seq_xy)
    delta[1:] = seq_xy[1:] - seq_xy[:-1]
    delta *= np.random.uniform(*scale_range)
    out = seq_xy[0:1] + np.cumsum(delta, axis=0)
    return np.clip(out, 0.0, 1.0)

def conf_jitter(seq_conf, sigma=0.15, min_conf=0.05):
    noise = np.random.normal(1.0, sigma, seq_conf.shape)
    return np.clip(seq_conf * noise, min_conf, 1.0)

class FallDataset(Dataset):
    def __init__(
        self,
        fall_labels,
        nonfall_labels,
        seq_len=32,
        augment=False
    ):
        self.samples = []
        self.seq_len = seq_len
        self.augment = augment

        for folder, label in [(fall_labels, 1), (nonfall_labels, 0)]:
            for f in os.listdir(folder):
                if f.endswith(".npy"):
                    self.samples.append(
                        (os.path.join(folder, f), label)
                    )

        self.labels = [label for _, label in self.samples]

        print(
            f"✅ Loaded {len(self.samples)} samples | "
            f"Fall={sum(self.labels)} | "
            f"NonFall={len(self.labels)-sum(self.labels)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        seq_all = np.load(path).astype(np.float32)
        if seq_all.shape[2] == 2:
            conf = np.ones((seq_all.shape[0], seq_all.shape[1], 1), dtype=np.float32)
            seq_all = np.concatenate([seq_all, conf], axis=2)

        if seq_all.shape[1:] != (14, 3):
            raise ValueError(f"Wrong shape {path}: {seq_all.shape}")
        seq_all = pad_or_crop(seq_all, self.seq_len)

        seq_xy   = seq_all[..., :2]
        seq_conf = seq_all[..., 2:3]
        if seq_conf.max() > 1.5:
            seq_conf = seq_conf / seq_conf.max()
        if self.augment:
            if np.random.rand() < 0.3:
                seq_xy = spatial_jitter(seq_xy)

            if np.random.rand() < 0.2:
                seq_xy = temporal_jitter_fine(seq_xy)

            if np.random.rand() < 0.15:
                seq_xy = motion_scale_safe(seq_xy)

            if np.random.rand() < 0.15:
                seq_conf = conf_jitter(seq_conf)

        seq = np.concatenate([seq_xy, seq_conf], axis=2)
        return (
            torch.from_numpy(seq).float(),
            torch.tensor(label, dtype=torch.long)
        )
