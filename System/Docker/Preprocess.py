import os
import numpy as np
import torch
from torch.utils.data import Dataset

NOSE_ID = 0
NUM_JOINTS = 14

EDGES = [
    (0,1),(1,0),
    (1,2),(1,3),(2,1),(3,1),
    (2,4),(4,2),(4,6),(6,4),
    (3,5),(5,7),(5,3),(7,5),
    (2,8),(8,2),(3,9),(9,3),(8,9),(9,8),
    (8,10),(10,8),(10,12),(12,10),
    (9,11),(11,9),(11,13),(13,11),
]
def pad_or_crop(seq: np.ndarray, target_len: int = 32, center_crop=True):
    """Enforce fixed temporal length."""
    T = seq.shape[0]

    if T == target_len:
        return seq

    if T < target_len:
        return np.pad(
            seq,
            ((0, target_len - T), (0, 0), (0, 0)),
            mode="edge"
        )

    start = (T - target_len) // 2 if center_crop else 0
    return seq[start:start + target_len]


def compute_bone(seq_xy):
    """
    Bone feature with aggregation:
    Each joint may receive multiple incoming bone vectors.
    """
    bone = np.zeros_like(seq_xy)
    count = np.zeros((seq_xy.shape[1],), dtype=np.float32)

    for u, v in EDGES:
        bone[:, v] += seq_xy[:, v] - seq_xy[:, u]
        count[v] += 1.0

    count[count == 0] = 1.0
    bone = bone / count[None, :, None]
    return bone


def compute_motion(seq_xy):
    """First-order temporal difference."""
    motion = np.zeros_like(seq_xy)
    motion[1:] = seq_xy[1:] - seq_xy[:-1]
    return motion

def spatial_jitter(seq_xy, sigma=0.004):
    noise = np.random.normal(0, sigma, seq_xy.shape)
    mask = np.linalg.norm(seq_xy, axis=2, keepdims=True) > 0
    return seq_xy + noise * mask


def nose_jitter(seq_xy, sigma=0.02):
    noise = np.random.normal(0, sigma, seq_xy[:, NOSE_ID].shape)
    mask = np.linalg.norm(seq_xy[:, NOSE_ID], axis=1, keepdims=True) > 0
    seq_xy[:, NOSE_ID] += noise * mask
    return seq_xy


def nose_dropout(seq_xy, seq_conf):
    seq_xy[:, NOSE_ID] = 0.0
    seq_conf[:, NOSE_ID] = 0.0
    return seq_xy, seq_conf


def joint_dropout_whole_sequence(seq_xy, seq_conf, drop_prob=0.05):
    mask = np.random.rand(NUM_JOINTS) < drop_prob
    seq_xy[:, mask] = 0.0
    seq_conf[:, mask] = 0.0
    return seq_xy, seq_conf


def motion_scale_safe(seq_xy, scale_range=(0.95, 1.05)):
    """
    Scale motion magnitude while preserving trajectory continuity.
    """
    delta = np.zeros_like(seq_xy)
    delta[1:] = seq_xy[1:] - seq_xy[:-1]
    scale = np.random.uniform(*scale_range)
    delta *= scale

    out = np.zeros_like(seq_xy)
    out[0] = seq_xy[0]
    out[1:] = out[0] + np.cumsum(delta[1:], axis=0)
    return np.clip(out, 0.0, 1.0)


def temporal_jitter_fine(seq_xy, sigma=0.004):
    noise = np.random.normal(0, sigma, seq_xy.shape)
    mask = np.linalg.norm(seq_xy, axis=2, keepdims=True) > 0
    return seq_xy + noise * mask


def frame_dropout_xy_conf(seq_xy, seq_conf, drop_prob=0.03):
    keep = np.random.rand(seq_xy.shape[0]) > drop_prob
    seq_xy = pad_or_crop(seq_xy[keep], seq_xy.shape[0])
    seq_conf = pad_or_crop(seq_conf[keep], seq_conf.shape[0])
    return seq_xy, seq_conf

def conf_jitter(seq_conf, sigma=0.15, min_conf=0.05):
    noise = np.random.normal(1.0, sigma, seq_conf.shape)
    return np.clip(seq_conf * noise, min_conf, 1.0)

class NpySequenceDataset(Dataset):
    def __init__(self, fall_dir, nonfall_dir, seq_len=32, augment=True):
        self.seq_len = seq_len
        self.augment = augment

        fall = [(os.path.join(fall_dir, f), 1)
                for f in os.listdir(fall_dir) if f.endswith(".npy")]
        nonf = [(os.path.join(nonfall_dir, f), 0)
                for f in os.listdir(nonfall_dir) if f.endswith(".npy")]

        self.samples = fall + nonf
        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        seq = np.load(path).astype(np.float32)  
        if seq.shape[1:] != (NUM_JOINTS, 3):
            raise ValueError(f"Wrong shape {path}: {seq.shape}")
        seq = pad_or_crop(seq, self.seq_len)

        seq_xy = seq[..., :2]
        seq_conf = seq[..., 2:3]
        if seq_conf.max() > 1.5:
            seq_conf = seq_conf / seq_conf.max()

        if self.augment:
            if np.random.rand() < 0.3:
                seq_xy = spatial_jitter(seq_xy)

            if np.random.rand() < 0.3:
                seq_xy = nose_jitter(seq_xy)

            if np.random.rand() < 0.25:
                seq_xy, seq_conf = nose_dropout(seq_xy, seq_conf)

            if np.random.rand() < 0.05:
                seq_xy, seq_conf = joint_dropout_whole_sequence(seq_xy, seq_conf)

            if np.random.rand() < 0.03:
                seq_xy, seq_conf = frame_dropout_xy_conf(seq_xy, seq_conf)

            if np.random.rand() < 0.15:
                seq_xy = motion_scale_safe(seq_xy)

            if np.random.rand() < 0.2:
                seq_xy = temporal_jitter_fine(seq_xy)

            if np.random.rand() < 0.15:
                seq_conf = conf_jitter(seq_conf)

        joint = np.concatenate([seq_xy, seq_conf], axis=2)
        bone = np.concatenate([compute_bone(seq_xy), seq_conf], axis=2)
        motion = np.concatenate([compute_motion(seq_xy), seq_conf], axis=2)

        def to_tensor(x):
            return torch.from_numpy(x).permute(2, 0, 1).float()

        return {
            "joint": to_tensor(joint),
            "bone": to_tensor(bone),
            "motion": to_tensor(motion),
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_batch(batch):
    return (
        torch.stack([b["joint"] for b in batch]),
        torch.stack([b["bone"] for b in batch]),
        torch.stack([b["motion"] for b in batch]),
        torch.tensor([b["label"] for b in batch]),
    )
