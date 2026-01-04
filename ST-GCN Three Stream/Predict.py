import os
from pathlib import Path
import cv2
import torch
import numpy as np
from collections import deque
from ultralytics import YOLO
import time
from Preprocess import compute_motion, compute_bone, motion_magnitude
from Model import STGCNThreeStream

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = r"C:\Users\giang\Desktop\ST-CGN\checkpoints\best_full_checkpoint.pth"
POSE_MODEL_PATH = r"D:\do_an\Model_ST-CGN\yolo11m-pose.pt"

NUM_FRAMES = 32
NUM_JOINTS = 14
ID_INACTIVE_FRAMES = 100
PROB_SMOOTH_LEN = 4
SMOOTH_LEN = 5
ALERT_THRESHOLD = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD MODELS
# -------------------------
print("🔹 Loading models...")
pose_model = YOLO(POSE_MODEL_PATH)
fall_model = STGCNThreeStream(num_class=2).to(DEVICE)
ckpt = torch.load(r"C:\Users\giang\Desktop\ST-CGN\checkpoints\best_full_checkpoint.pth", map_location=DEVICE)
fall_model.load_state_dict(ckpt["model_state"], strict=False)
fall_model.eval()
print("✅ Models loaded successfully on", DEVICE)

# -------------------------
# INIT
# -------------------------
buffers = {}
prev_pts = {}
prob_queue = {}
last_seen = {}
last_pred = {}
smooth_buffers = {}

SKELETON_EDGES = [
    (0,1),(1,2),(1,3),(2,4),(4,6),(3,5),(5,7),
    (2,8),(3,9),(8,9),(8,10),(10,12),(9,11),(11,13)
]

# -------------------------
# FUNCTIONS
# -------------------------
def extract_keypoints_pixels(result):
    if not hasattr(result, "keypoints") or result.keypoints is None:
        return None
    try:
        return result.keypoints.xy.cpu().numpy().astype(np.float32)
    except:
        return None

def coco17_to_14(coco_kps_xy):
    coco = coco_kps_xy.copy()
    if coco.shape[0] < 17:
        pad = np.zeros((17 - coco.shape[0], 2), dtype=np.float32)
        coco = np.vstack([coco, pad])

    left_sh, right_sh = coco[5], coco[6]
    center_sh = (left_sh + right_sh)/2.0

    mapped = np.stack([
        coco[0], center_sh, coco[5], coco[6], coco[7], coco[8],
        coco[9], coco[10], coco[11], coco[12], coco[13], coco[14],
        coco[15], coco[16]
    ], axis=0)
    return mapped.astype(np.float32)

def interpolate_missing(curr_pts, prev_pts):
    if prev_pts is None:
        return curr_pts
    interp = curr_pts.copy()
    for i in range(NUM_JOINTS):
        if np.all(curr_pts[i] == 0):
            interp[i] = prev_pts[i]
    return interp

def build_conf(pts):
    conf = np.ones((NUM_JOINTS,), dtype=np.float32)
    conf[np.all(pts == 0, axis=1)] = 0.0
    return conf

def smooth_pts(pid, pts, conf):
    if pid not in smooth_buffers:
        smooth_buffers[pid] = deque(maxlen=SMOOTH_LEN)
    smooth_buffers[pid].append((pts.copy(), conf.copy()))

    pts_stack = np.stack([p for p, _ in smooth_buffers[pid]], axis=0)
    conf_stack = np.stack([c for _, c in smooth_buffers[pid]], axis=0)

    denom = np.sum(conf_stack, axis=0)[:, None] + 1e-6
    weighted = np.sum(pts_stack * conf_stack[:, :, None], axis=0) / denom
    return weighted.astype(np.float32)

# -------------------------
# REAL-TIME VIDEO
# -------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam!")

print("🎥 Real-time fall detection started (press Q to stop)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ============================================
    # POSE TRACKING
    # ============================================
    results = pose_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    if not results or len(results) == 0:
        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    res = results[0]

    try:
        boxes_xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else None
        ids_raw = res.boxes.id.cpu().numpy() if hasattr(res.boxes, "id") else None
    except:
        boxes_xyxy, ids_raw = None, None

    kps_all = extract_keypoints_pixels(res)

    if ids_raw is None or kps_all is None:
        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    batch_joint, batch_bone, batch_motion = [], [], []
    batch_pids = []

    # ============================================
    # PROCESS EACH PERSON
    # ============================================
    for idx_det, pid_raw in enumerate(ids_raw):
        pid = int(pid_raw)
        last_seen[pid] = time.time()

        coco_kps = kps_all[idx_det]
        pts14 = coco17_to_14(coco_kps)
        pts14 = interpolate_missing(pts14, prev_pts.get(pid))
        conf = build_conf(pts14)
        pts14_sm = smooth_pts(pid, pts14, conf)
        prev_pts[pid] = pts14_sm.copy()

        if conf.sum() == 0:
            continue

        pts14_norm = pts14_sm / np.array([w, h], dtype=np.float32)
        kps14 = np.concatenate([pts14_norm, conf[:, None]], axis=1)

        # Draw skeleton
        for x, y in pts14_sm:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
        for a, b in SKELETON_EDGES:
            pa, pb = pts14_sm[a], pts14_sm[b]
            if pa.sum() != 0 and pb.sum() != 0:
                cv2.line(frame, (int(pa[0]), int(pa[1])),
                         (int(pb[0]), int(pb[1])), (0, 200, 200), 2)

        if pid not in buffers:
            buffers[pid] = deque(maxlen=NUM_FRAMES)
            prob_queue[pid] = deque(maxlen=PROB_SMOOTH_LEN)
            last_pred[pid] = ("NON-FALL", 0.0)

        buffers[pid].append(kps14)

        # Only infer if sequence full
        valid_frames = [b for b in buffers[pid] if b[:,2].sum() > NUM_JOINTS/2]
        if len(valid_frames) == NUM_FRAMES:
            seq = np.stack(valid_frames)

            joint_seq = seq.copy()
            bone_coords = compute_bone(seq[:, :, :2])
            bone_seq = np.concatenate([bone_coords, seq[:, :, 2:]], 2)
            motion_coords = compute_motion(seq[:, :, :2])        # (T, V, 2)  dx, dy
            motion_mag = motion_magnitude(seq[:, :, :2])         # (T, V, 1)  |v|

            motion_seq = np.concatenate([
                motion_coords,    # 2
                motion_mag,       # 1
                seq[:, :, 2:]     # conf (1)
            ], axis=2)            # ==> (T, V, 4)

            batch_joint.append(torch.tensor(joint_seq.transpose(2,0,1)))
            batch_bone.append(torch.tensor(bone_seq.transpose(2,0,1)))
            batch_motion.append(torch.tensor(motion_seq.transpose(2,0,1)))
            batch_pids.append(pid)

    # ============================================
    # BATCH INFERENCE
    # ============================================
    if batch_joint:
        joint_tensor = torch.stack(batch_joint).float().to(DEVICE)
        bone_tensor  = torch.stack(batch_bone).float().to(DEVICE)
        motion_tensor= torch.stack(batch_motion).float().to(DEVICE)

        with torch.no_grad():
            logits, _ = fall_model(joint_tensor, bone_tensor, motion_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        for i, pid in enumerate(batch_pids):
            pq = prob_queue[pid]
            pq.append(probs[i])
            avg_prob = np.mean(pq, axis=0)
            fall_prob = float(avg_prob[1])

            label = "FALL" if fall_prob >= ALERT_THRESHOLD else "NON-FALL"
            last_pred[pid] = (label, fall_prob)

            if label == "FALL":
                cv2.putText(frame, "⚠️ FALL DETECTED!", (50, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255), 4)

    # ============================================
    # DRAW BBOXES + LABELS
    # ============================================
    if boxes_xyxy is not None:
        for idx_det, pid_raw in enumerate(ids_raw):
            pid = int(pid_raw)
            if idx_det >= len(boxes_xyxy):
                continue

            x1,y1,x2,y2 = boxes_xyxy[idx_det].astype(int)
            label, prob = last_pred.get(pid, ("NON-FALL", 0.0))
            color = (0,0,255) if label=="FALL" else (0,255,0)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"ID {pid}: {label} {prob*100:.1f}%",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # ============================================
    # SHOW FRAME
    # ============================================
    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("🔚 Stopped.")
