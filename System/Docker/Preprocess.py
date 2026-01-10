import os
import numpy as np
import torch
from torch.utils.data import Dataset


NOSE_ID = 0
NUM_JOINTS = 14

SYMMETRIC_PAIRS = [
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
]

EDGES = [
    (0,1),
    (1,2),(1,3),
    (2,4),(4,6),
    (3,5),(5,7),
    (2,8),(3,9),(8,9),
    (8,10),(10,12),
    (9,11),(11,13)
]
def pad_or_crop(seq: np.ndarray, target_len: int = 32, center_crop=True):
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
    bone = np.zeros_like(seq_xy)
    T, V, _ = seq_xy.shape
    for u, v in EDGES:
        bone[:, v, :] = seq_xy[:, v, :] - seq_xy[:, u, :]
    return bone

def compute_motion(seq_xy):
    motion = np.zeros_like(seq_xy)
    motion[1:] = seq_xy[1:] - seq_xy[:-1]
    return motion

def rotate_skeleton_xy(seq_xy, angle_range=(-5, 5)):
    seq_rot = seq_xy.copy()
    angle_deg = np.random.uniform(*angle_range)
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]], dtype=np.float32)
    
    centroid = seq_rot.mean(axis=1, keepdims=True)
    seq_centered = seq_rot - centroid
    seq_rot = seq_centered @ R.T + centroid
    
    seq_rot = np.clip(seq_rot, 0.0, 1.0)
    return seq_rot

def spatial_jitter(seq_xy, sigma=0.008):
    noise = np.random.normal(0, sigma, seq_xy.shape)
    mask = np.linalg.norm(seq_xy, axis=2, keepdims=True) > 0
    seq_jitter = seq_xy + noise * mask
    return np.clip(seq_jitter, 0.0, 1.0)

def motion_scale_safe(seq_xy, scale_range=(0.95, 1.0)):

    delta = np.zeros_like(seq_xy)
    delta[1:] = seq_xy[1:] - seq_xy[:-1]

    scales = np.random.uniform(*scale_range, size=(delta.shape[0]-1, 1, 1))
    delta[1:] *= scales

    out = np.zeros_like(seq_xy)
    out[0] = seq_xy[0]
    out[1:] = out[0] + np.cumsum(delta[1:], axis=0)

    return np.clip(out, 0.0, 1.0)

def temporal_jitter_fine(seq_xy, sigma=0.008):
    noise = np.random.normal(0, sigma, seq_xy.shape)
    mask = np.linalg.norm(seq_xy, axis=2, keepdims=True) > 0
    return seq_xy + noise * mask

def conf_jitter(seq_conf, sigma=0.15, min_conf=0.05):
    noise = np.random.normal(1.0, sigma, seq_conf.shape)
    out = seq_conf.copy()
    mask = seq_conf > 0
    out[mask] = np.clip(seq_conf[mask] * noise[mask], min_conf, 1.0)
    return out

def back_facing_augment(seq_xy, seq_conf):
    seq_xy = seq_xy.copy()
    seq_conf = seq_conf.copy()
    for l, r in SYMMETRIC_PAIRS:
        seq_xy[:, [l, r], :] = seq_xy[:, [r, l], :]
        seq_conf[:, [l, r], :] = seq_conf[:, [r, l], :]
    return seq_xy, seq_conf

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

        if self.augment:
            if np.random.rand() < 0.3:
                seq_xy = rotate_skeleton_xy(seq_xy, angle_range=(-5,5))
            if np.random.rand() < 0.5:
                seq_xy, seq_conf = back_facing_augment(seq_xy, seq_conf)
            if np.random.rand() < 0.3:
                seq_xy = spatial_jitter(seq_xy)
            if np.random.rand() < 0.2:
                seq_xy = motion_scale_safe(seq_xy)
            if np.random.rand() < 0.3:
                seq_xy = temporal_jitter_fine(seq_xy)
            if np.random.rand() < 0.2:
                seq_conf = conf_jitter(seq_conf)

        joint = np.concatenate([seq_xy, seq_conf], axis=2)
        bone = np.concatenate([compute_bone(seq_xy), seq_conf], axis=2)
        motion = np.concatenate([compute_motion(seq_xy), seq_conf], axis=2)

        def to_tensor(x):
            return torch.from_numpy(x).permute(2,0,1).float()  

        return {
            "joint": to_tensor(joint), 
            "bone": to_tensor(bone),
            "motion": to_tensor(motion),
            "label": torch.tensor(label, dtype=torch.long)
        }

def collate_batch(batch):
    return (
        torch.stack([b["joint"] for b in batch]),
        torch.stack([b["bone"] for b in batch]),
        torch.stack([b["motion"] for b in batch]),
        torch.tensor([b["label"] for b in batch])
    )
