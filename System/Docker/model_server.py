# model_server.py
import torch
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, jsonify
from Model import STGCNThreeStream
from collections import deque
import traceback

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Config (TỐI ƯU ĐỘ TRỄ)
# =====================================================

SEQ_LEN = 32                # ⬅️ GIẢM TỪ 32 → 16
NUM_CLASS = 2
LABELS = ["non-fall", "fall"]

CONF_THRESHOLD = 0.2
SMOOTH_WINDOW = 3            # ⬅️ GIẢM TỪ 5 → 3
FALL_PROB_THRESHOLD = 0.85   # nhạy hơn
FALL_DEBOUNCE = 1            # ⬅️ PHÁT HIỆN SỚM

# =====================================================
# Graph structure (MUST match preprocess.py)
# =====================================================

EDGES = [
    (0,1),(1,2),(1,3),(2,4),(4,6),(3,5),(5,7),
    (2,8),(3,9),(8,9),(8,10),(10,12),(9,11),(11,13)
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
# Runtime buffers
# =====================================================

PRED_QUEUE = deque(maxlen=SMOOTH_WINDOW)

# =====================================================
# Helper functions
# =====================================================

def pad_or_crop(seq, target_len=SEQ_LEN):
    T = seq.shape[0]
    if T < target_len:
        return np.pad(seq, ((0, target_len - T), (0, 0), (0, 0)), mode="edge")
    return seq[:target_len]

def sanitize_joint(joint):
    joint = np.nan_to_num(joint, nan=0.0, posinf=0.0, neginf=0.0)
    joint[..., :2] = np.clip(joint[..., :2], 0.0, 1.0)
    joint[..., 2:3] = np.clip(joint[..., 2:3], 0.0, 1.0)
    return joint

def compute_bone(seq_xy, conf):
    bone = np.zeros_like(seq_xy)
    for u, v in EDGES:
        valid = (conf[:, u, 0] > 0) & (conf[:, v, 0] > 0)
        bone[valid, v] = seq_xy[valid, v] - seq_xy[valid, u]
    return bone

def compute_motion(seq_xy, conf):
    motion = np.zeros_like(seq_xy)
    valid = conf[:, :, 0] > 0
    valid_pair = valid[1:] & valid[:-1]
    motion[1:][valid_pair] = seq_xy[1:][valid_pair] - seq_xy[:-1][valid_pair]
    return motion

def to_tensor(arr):
    # (T,V,C) → (1,C,T,V)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

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
        PRED_QUEUE.append(probs)
        avg_prob = np.mean(PRED_QUEUE, axis=0)

        pred_idx  = int(np.argmax(avg_prob))
        pred_conf = float(avg_prob[pred_idx])

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
            "probs": avg_prob.tolist()
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
