import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FallDataset(Dataset):
    def __init__(self, fall_dir, nonfall_dir, seq_len=32, augment=False):
        self.paths = []
        self.labels = []
        self.seq_len = seq_len

        for f in os.listdir(fall_dir):
            if f.endswith(".npy"):
                self.paths.append(os.path.join(fall_dir, f))
                self.labels.append(1)

        for f in os.listdir(nonfall_dir):
            if f.endswith(".npy"):
                self.paths.append(os.path.join(nonfall_dir, f))
                self.labels.append(0)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])  
        label = self.labels[idx]

        T, J, C = data.shape
        center = data.mean(axis=1, keepdims=True)
        data = data - center
        scale = np.linalg.norm(data, axis=2).max(axis=1, keepdims=True)
        scale[scale < 1e-6] = 1.0
        data = data / scale[..., None]

        # ===== Pad / Cut =====
        if T > self.seq_len:
            data = data[:self.seq_len]
        elif T < self.seq_len:
            pad = np.zeros((self.seq_len - T, J, C))
            data = np.concatenate([data, pad], axis=0)

        x = data.reshape(self.seq_len, -1) 

        return torch.tensor(x, dtype=torch.float32), torch.tensor(label)
