import os
import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

# ===== Skeleton edges (14-keypoint format) =====
EDGES = [
    (0,1),(1,2),(1,3),(2,4),(4,6),(3,5),(5,7),
    (2,8),(3,9),(8,9),(8,10),(10,12),(9,11),(11,13)
]

# ==========================================================
# 1️⃣ Sequence preprocessing utilities
# ==========================================================
def pad_or_crop(seq: np.ndarray, target_len: int = 32, center_crop=True):
    T = seq.shape[0]
    if T == target_len:
        return seq
    if T < target_len:
        pad_len = target_len - T
        return np.pad(seq, ((0, pad_len),(0,0),(0,0)), mode="edge")
    if center_crop:
        start = (T - target_len) // 2
        return seq[start:start + target_len]
    return seq[:target_len]

def compute_bone(seq_xy: np.ndarray):
    T, V, C = seq_xy.shape
    u = np.array([u for u,_ in EDGES])
    v = np.array([v for _,v in EDGES])
    bone = seq_xy[:,v] - seq_xy[:,u]
    bone_full = np.zeros((T,V,C), dtype=np.float32)
    bone_full[:,v] = bone
    return bone_full

def compute_motion(seq_xy: np.ndarray):
    motion = np.zeros_like(seq_xy)
    motion[1:] = seq_xy[1:] - seq_xy[:-1]
    return motion

def build_conf(seq_xy: np.ndarray):
    conf = (np.linalg.norm(seq_xy, axis=2) > 0).astype(np.float32)
    return conf[..., None]

def spatial_jitter(seq, sigma=0.01):
    noise = np.random.normal(0, sigma, seq.shape).astype(np.float32)
    mask = (np.linalg.norm(seq, axis=2, keepdims=True) > 0)
    return seq + noise * mask

def joint_dropout_whole_sequence(seq, drop_prob=0.1):
    T, V, C = seq.shape
    mask = np.random.rand(V) < drop_prob
    seq_aug = seq.copy()
    seq_aug[:, mask, :] = 0.0
    return seq_aug

def speed_change(seq, factor_range=(0.9,1.1)):
    T,V,C = seq.shape
    factor = np.random.uniform(*factor_range)
    new_T = max(int(T*factor),1)
    t_old = np.linspace(0,1,T)
    t_new = np.linspace(0,1,new_T)
    seq_new = np.zeros((new_T,V,C), dtype=np.float32)
    for v in range(V):
        for c in range(C):
            f = interp1d(t_old, seq[:,v,c], kind='linear', fill_value="extrapolate")
            seq_new[:,v,c] = f(t_new)
    return pad_or_crop(seq_new,T)

def motion_scale(seq, scale_range=(0.9,1.1)):
    T,V,C = seq.shape
    delta = np.zeros_like(seq)
    delta[1:] = seq[1:] - seq[:-1]
    scale = np.random.uniform(*scale_range)
    delta *= scale
    return seq[0:1] + np.cumsum(delta, axis=0)

def frame_dropout(seq, drop_prob=0.03):
    T,V,C = seq.shape
    mask = np.random.rand(T) > drop_prob
    seq_new = seq[mask]
    return pad_or_crop(seq_new, T)

def frame_duplicate(seq, dup_prob=0.03):
    T,V,C = seq.shape
    new_seq = []
    for i in range(T):
        new_seq.append(seq[i])
        if np.random.rand() < dup_prob:
            new_seq.append(seq[i])
    return pad_or_crop(np.array(new_seq), T)

def temporal_jitter_fine(seq, sigma=0.003):
    T,V,C = seq.shape
    noise = np.random.normal(0, sigma, size=(T,V,C))
    return seq + noise

# ==========================================================
# 2️⃣ Dataset class
# ==========================================================
class NpySequenceDataset(Dataset):
    def __init__(self, fall_dir, nonfall_dir, seq_len=32, shuffle=True, augment=True):
        self.seq_len = seq_len
        self.augment = augment

        fall_files = [(os.path.join(fall_dir,f),1) for f in os.listdir(fall_dir) if f.endswith(".npy")]
        nonfall_files = [(os.path.join(nonfall_dir,f),0) for f in os.listdir(nonfall_dir) if f.endswith(".npy")]
        self.samples = fall_files + nonfall_files
        if shuffle:
            np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path,label = self.samples[idx]
        seq = np.load(path).astype(np.float32)  # (T,V,2)
        if seq.ndim !=3 or seq.shape[1]!=14:
            raise ValueError(f"File lỗi shape: {path}, shape={seq.shape}")
        seq = pad_or_crop(seq, self.seq_len)

        # =============================
        # 🔥 DATA AUGMENTATION
        # =============================
        if self.augment:
            # ----- Spatial -----
            if np.random.rand() < 0.3: seq = spatial_jitter(seq, sigma=0.003)
            # ----- Joint dropout -----
            if np.random.rand() < 0.1: seq = joint_dropout_whole_sequence(seq, drop_prob=0.05)
            # ----- Motion augmentation nhẹ -----
            if np.random.rand() < 0.2: seq = speed_change(seq,(0.9,1.1))
            if np.random.rand() < 0.2: seq = motion_scale(seq,(0.9,1.1))
            if np.random.rand() < 0.05: seq = frame_dropout(seq,0.03)
            if np.random.rand() < 0.05: seq = frame_duplicate(seq,0.03)
            if np.random.rand() < 0.2: seq = temporal_jitter_fine(seq,0.003)

        # =============================
        # Build confidence
        # =============================
        conf = build_conf(seq)
        joint  = np.concatenate([seq, conf], axis=2)
        bone   = np.concatenate([compute_bone(seq), conf], axis=2)
        motion = np.concatenate([compute_motion(seq), conf], axis=2)

        def to_tensor(arr):
            return torch.from_numpy(arr).permute(2,0,1).float()

        return {
            "joint":  to_tensor(joint),
            "bone":   to_tensor(bone),
            "motion": to_tensor(motion),
            "label":  torch.tensor(label,dtype=torch.long)
        }

def collate_batch(batch):
    joint  = torch.stack([b["joint"]  for b in batch])
    bone   = torch.stack([b["bone"]   for b in batch])
    motion = torch.stack([b["motion"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch])
    return joint, bone, motion, labels
