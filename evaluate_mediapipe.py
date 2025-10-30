import os
import json
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
import mediapipe as mp
import cv2

# ==== Cấu hình ====
ANN_PATH = r"D:\Code\AIP\hrnet\HRNet-Human-Pose-Estimation\data\annotations\val_annotations.json"
IMG_DIR = r"D:\Code\AIP\dataset\data_txt\Fall\images"
PRED_JSON = "mediapipe_results.json"
ALPHA = 0.5  # Ngưỡng PCKh

# ==== Khởi tạo MediaPipe ====
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)

# ==== Load ground truth ====
coco = COCO(ANN_PATH)
gt_imgs = {img["id"]: img for img in coco.dataset["images"]}
gt_anns = coco.dataset["annotations"]

# ==== Ánh xạ Mediapipe (33 điểm) sang COCO (17 điểm) ====
mp_to_coco_idx = {
    0: 0,    # Nose
    1: 11,   # Left Shoulder
    2: 12,   # Right Shoulder
    3: 13,   # Left Elbow
    4: 14,   # Right Elbow
    5: 15,   # Left Wrist
    6: 16,   # Right Wrist
    7: 23,   # Left Hip
    8: 24,   # Right Hip
    9: 25,   # Left Knee
    10: 26,  # Right Knee
    11: 27,  # Left Ankle
    12: 28,  # Right Ankle
    13: 5,   # Left Eye
    14: 2,   # Left Ear
    15: 6,   # Right Eye
    16: 3    # Right Ear
}

# ==== Tạo file dự đoán ====
preds = []
for ann in tqdm(gt_anns, desc="Evaluating MediaPipe"):
    img_id = ann["image_id"]
    img_info = gt_imgs.get(img_id)
    if img_info is None:
        continue

    img_path = os.path.join(IMG_DIR, img_info["file_name"])
    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    h, w = image.shape[:2]
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        continue

    mp_kps = np.array([[lm.x * w, lm.y * h, 1] for lm in results.pose_landmarks.landmark])

    # === Ánh xạ sang 17 điểm COCO ===
    mapped_pred = []
    for coco_id, mp_id in mp_to_coco_idx.items():
        if mp_id < mp_kps.shape[0]:
            mapped_pred.append(mp_kps[mp_id])
        else:
            mapped_pred.append([0, 0, 0])
    pred_kps = np.array(mapped_pred).flatten().tolist()

    preds.append({
        "image_id": img_id,
        "category_id": 1,
        "keypoints": pred_kps,
        "score": 1.0
    })

# Lưu file dự đoán
with open(PRED_JSON, "w", encoding="utf-8") as f:
    json.dump(preds, f)
print(f"\nĐã lưu {len(preds)} kết quả vào {PRED_JSON}")

# ==== Tính PCKh và MKE ====
def compute_pckh_and_mke(gt_anns, preds):
    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    total_keypoints = 0
    correct_keypoints = 0
    mke_sum = 0.0

    for gt in tqdm(gt_anns, desc="Đánh giá MediaPipe"):
        img_id = gt["image_id"]
        if img_id not in preds_by_img:
            continue
        pred = max(preds_by_img[img_id], key=lambda x: x.get("score", 1.0))

        gt_kps = np.array(gt["keypoints"]).reshape(-1, 3)
        pred_kps = np.array(pred["keypoints"]).reshape(-1, 3)

        n_gt = gt_kps.shape[0]
        n_pred = pred_kps.shape[0]
        if n_gt != n_pred:
            m = min(n_gt, n_pred)
            gt_kps = gt_kps[:m]
            pred_kps = pred_kps[:m]

        valid_mask = gt_kps[:, 2] > 0
        if not np.any(valid_mask):
            continue

        head_size = np.linalg.norm(gt_kps[5, :2] - gt_kps[6, :2])
        if head_size < 1e-6:
            head_size = 1e-6

        dists = np.linalg.norm(gt_kps[:, :2] - pred_kps[:, :2], axis=1)
        dists_norm = dists / head_size

        correct_keypoints += np.sum((dists_norm < ALPHA) & valid_mask)
        total_keypoints += np.sum(valid_mask)
        mke_sum += np.sum(dists_norm[valid_mask])

    pckh = correct_keypoints / total_keypoints if total_keypoints > 0 else 0
    mke = mke_sum / total_keypoints if total_keypoints > 0 else 0
    return pckh, mke


# ==== Đánh giá ====
pckh, mke = compute_pckh_and_mke(gt_anns, preds)
print("\n=== Kết quả MediaPipe (chuẩn hóa) ===")
print(f"{'Model':<12} | {'PCKh@0.5':<10} | {'MKE':<10}")
print("-" * 36)
print(f"{'MediaPipe':<12} | {pckh*100:>8.2f}% | {mke:>8.4f}")
