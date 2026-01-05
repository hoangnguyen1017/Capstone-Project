import os
import shutil
import random

# ==== Đường dẫn gốc ====
SRC_ROOT = "./dataset/data_txt/Fall"
DST_ROOT = "./dataset/data_txt/Fall_split"

IMG_DIR = os.path.join(SRC_ROOT, "images")
LBL_DIR = os.path.join(SRC_ROOT, "labels")

# ==== Cấu hình tỉ lệ chia ====
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# ==== Lấy danh sách ảnh ====
images = [f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(images)

n = len(images)
n_train = int(n * train_ratio)
n_val = int(n * val_ratio)

splits = {
    "train": images[:n_train],
    "val": images[n_train:n_train + n_val],
    "test": images[n_train + n_val:]
}

# ==== Tạo thư mục đích ====
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DST_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DST_ROOT, "labels", split), exist_ok=True)

# ==== Copy ảnh và nhãn ====
for split, file_list in splits.items():
    for fname in file_list:
        name, _ = os.path.splitext(fname)
        src_img = os.path.join(IMG_DIR, fname)
        src_lbl = os.path.join(LBL_DIR, f"{name}.txt")

        dst_img = os.path.join(DST_ROOT, "images", split, fname)
        dst_lbl = os.path.join(DST_ROOT, "labels", split, f"{name}.txt")

        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

print("✅ Đã chia dữ liệu và sao chép sang thư mục mới:", DST_ROOT)
print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
