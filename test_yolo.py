from ultralytics import YOLO
import cv2
import os

# ===== Config =====
model_path = r"D:\Code\AIP\runs\finetune_pose_stable\weights\best.pt"
video_path = r"D:\Code\AIP\istockphoto-1286619966-640_adpp_is.mp4"
output_path = r"D:\Code\AIP\output_cam1.avi"

# Load model
model = YOLO(model_path)

# Mở video input
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Không mở được video:", video_path)
    exit()

# Lấy thông tin video gốc
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Khởi tạo VideoWriter để ghi output
fourcc = cv2.VideoWriter_fourcc(*"XVID")   # dùng "mp4v" nếu muốn .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# Xử lý từng frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán
    results = model(frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # Ghi ra file
    out.write(annotated_frame)

    # Hiển thị
    cv2.imshow("YOLO11 Pose Estimation", annotated_frame)

    # Nhấn Q để thoát
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# Giải phóng
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video output đã lưu tại:", os.path.abspath(output_path))