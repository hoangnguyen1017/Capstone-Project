# model_server.py
import torch
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, jsonify
from Model import STGCNThreeStream
from collections import deque
import traceback

# =====================================================
# App & Device
# =====================================================

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Config (TỐI ƯU ĐỘ TRỄ)
# =====================================================

SEQ_LEN = 32
NUM_CLASS = 2
LABELS = ["non-fall", "fall"]

CONF_THRESHOLD = 0.2
FALL_PROB_THRESHOLD = 0.85   # nhạy hơn
FALL_DEBOUNCE = 1            # ⬅️ PHÁT HIỆN SỚM

# =====================================================
# Graph structure (MUST match preprocess.py)
# =====================================================
EDGES = [
    (0,1),(1,0),
    (1,2),(1,3),(2,1),(3,1),
    (2,4),(4,2),(4,6),(6,4),
    (3,5),(5,7),(5,3),(7,5),
    (2,8),(8,2),(3,9),(9,3),(8,9),(9,8),
    (8,10),(10,8),(10,12),(12,10),
    (9,11),(11,9),(11,13),(13,11),
]

# =====================================================
# Load model
# =====================================================

MODEL_PATH = r"C:\Users\ADMIN\Downloads\Do_an\web_final_bv\web_final\Docker\best_full_checkpoint.pth"

try:
    model = STGCNThreeStream(num_class=NUM_CLASS, in_channels_each=3)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(DEVICE).eval()
    print("✅ ST-GCN model loaded successfully.")
except Exception:
    print("❌ Model load failed")
    traceback.print_exc()

# =====================================================
# Helper functions
# =====================================================

def pad_or_crop(seq, target_len=SEQ_LEN, center_crop=True):
    T = seq.shape[0]
    if T == target_len:
        return seq
    if T < target_len:
        return np.pad(seq, ((0, target_len - T), (0, 0), (0, 0)), mode="edge")
    if center_crop:
        start = (T - target_len) // 2
    else:
        start = 0
    return seq[start:start + target_len]

def sanitize_joint(joint):
    joint = np.nan_to_num(joint, nan=0.0, posinf=0.0, neginf=0.0)
    
    # XY coordinates
    joint[..., :2] = np.clip(joint[..., :2], 0.0, 1.0)
    
    # Confidence
    conf = joint[..., 2:3]
    if conf.max() > 1.5:
        conf = conf / conf.max()
    conf = np.clip(conf, 0.0, 1.0)
    joint[..., 2:3] = conf
    
    return joint

def compute_bone(seq_xy, conf):
    """
    seq_xy: (T,V,2), conf: (T,V,1)
    """
    bone = np.zeros_like(seq_xy)
    T, V, _ = seq_xy.shape
    for u, v in EDGES:
        valid = (conf[:, u, 0] > 0) & (conf[:, v, 0] > 0)  # shape (T,)
        if valid.any():
            bone[valid, v, :] = seq_xy[valid, v, :] - seq_xy[valid, u, :]
    return bone

def compute_motion(seq_xy, conf):
    """
    seq_xy: (T,V,2), conf: (T,V,1)
    """
    motion = np.zeros_like(seq_xy)
    T, V, _ = seq_xy.shape
    valid = conf[:, :, 0] > 0  # shape (T,V)
    for t in range(1, T):
        valid_pair = valid[t] & valid[t-1]  # shape (V,)
        if valid_pair.any():
            motion[t, valid_pair, :] = seq_xy[t, valid_pair, :] - seq_xy[t-1, valid_pair, :]
    return motion

def to_tensor(arr):
    # (T,V,3) -> (1,3,T,V)
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()

# =====================================================
# API
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        joint = np.array(data["joint"], dtype=np.float32)  # (T,V,3)

        # ----------------------------
        # Validate input
        # ----------------------------
        if joint.ndim != 3 or joint.shape[1] != 14 or joint.shape[2] != 3:
            return jsonify({"success": False, "error": "Invalid joint shape"}), 400

        # ----------------------------
        # Sanitize & pad
        # ----------------------------
        joint = sanitize_joint(joint)
        joint = pad_or_crop(joint)

        joint_xy = joint[:, :, :2]
        conf     = joint[:, :, 2:3]

        # ----------------------------
        # Confidence gating
        # ----------------------------
        mean_conf = float(conf.mean())
        if mean_conf < CONF_THRESHOLD:
            return jsonify({
                "success": True,
                "prediction": "unknown",
                "confidence": 0.0,
                "mean_conf": mean_conf
            })

        # ----------------------------
        # Bone & Motion
        # ----------------------------
        bone_xy   = compute_bone(joint_xy, conf)
        motion_xy = compute_motion(joint_xy, conf)

        bone   = np.concatenate([bone_xy, conf], axis=2)
        motion = np.concatenate([motion_xy, conf], axis=2)

        # ----------------------------
        # To tensor
        # ----------------------------
        x_joint  = to_tensor(joint).to(DEVICE)
        x_bone   = to_tensor(bone).to(DEVICE)
        x_motion = to_tensor(motion).to(DEVICE)

        # ----------------------------
        # Inference
        # ----------------------------
        with torch.no_grad():
            logits, _ = model(x_joint, x_bone, x_motion)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # ----------------------------
        # Temporal smoothing (NHẸ)
        # ----------------------------
        pred_idx  = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])

        # ----------------------------
        # Fall detection (ƯU TIÊN NHANH)
        # ----------------------------
        fall_triggered = False
        if pred_idx == 1 and pred_conf > FALL_PROB_THRESHOLD:
            fall_triggered = True

        return jsonify({
            "success": True,
            "prediction": LABELS[pred_idx],
            "confidence": pred_conf,
            "fall_triggered": fall_triggered,
            "mean_conf": mean_conf,
            "probs": probs.tolist()
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "model": "STGCNThreeStream",
        "seq_len": SEQ_LEN
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000)
