import json
import time
import smtplib
import ssl
from email.mime.text import MIMEText
from pymongo import MongoClient
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os

# =======================
#  CẤU HÌNH
# =======================
QUEUE_FILE = "fall_event_queue.jsonl"

MONGO_URI = "mongodb+srv://nguyenvinhkhang26042003:Khang26042003@do-an.nhzggvx.mongodb.net/?appName=Do-an"
DB_NAME = "test"          # <--- QUAN TRỌNG: DATABASE "test"
COLLECTION_NAME = "cameras"

GMAIL_USER = "khangnvse171562@fpt.edu.vn"
GMAIL_PASS = "prdgrgfuuhdnqcsu"

# =======================
# KẾT NỐI MONGODB
# =======================
client = MongoClient(MONGO_URI)
camera_col = client[DB_NAME][COLLECTION_NAME]

print("📡 Connected to MongoDB collection:", DB_NAME, COLLECTION_NAME)



# =======================
# GỬI EMAIL
# =======================
def send_email(to_email, subject, message, image_paths=None):
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = to_email

    msg.attach(MIMEText(message, "plain"))

    if image_paths:
        for path in image_paths:
            if not os.path.exists(path):
                continue
            with open(path, "rb") as f:
                img = MIMEImage(f.read())
                img.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=os.path.basename(path)
                )
                msg.attach(img)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)

# =======================
# TÌM CAMERA TRONG DB
# =======================
def find_camera_by_name(name):
    return camera_col.find_one({"camera_name": name})



# =======================
# XỬ LÝ EVENT FALL
# =======================
def handle_fall_event(event):
    timestamp = event.get("timestamp")
    frames = event.get("frames", [])
    camera_name = event.get("camera_name")
    location = event.get("location")

    # convert timestamp
    try:
        time_str = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(float(timestamp))
        )
    except:
        time_str = str(timestamp)

    cam = find_camera_by_name(camera_name)
    if not cam:
        print("⚠ Không tìm thấy camera trong database!")
        return

    responsible_email = cam.get("responsible_email")
    if not responsible_email:
        print("⚠ Camera không có responsible_email")
        return

    subject = f"[FALL ALERT] Phát hiện ngã tại {location}"
    message = (
        f"🚨 CẢNH BÁO NGÃ\n"
        f"\nCamera: {camera_name}"
        f"\nVị trí: {location}"
        f"\nThời gian: {time_str}"
        f"\n\nẢnh hiện trường được đính kèm."
    )

    send_email(responsible_email, subject, message, frames)

# =======================
# THEO DÕI QUEUE FILE
# =======================
def start_monitoring():
    print("🔔 ALERT SERVICE STARTED — Listening for fall events...")

    try:
        with open(QUEUE_FILE, "r", encoding="utf8") as f:
            f.seek(0, 2)  # Move to end of file

            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.2)
                    continue

                try:
                    event = json.loads(line.strip())
                    handle_fall_event(event)
                except json.JSONDecodeError:
                    print("⚠ Lỗi decode JSON, bỏ qua dòng.")

    except FileNotFoundError:
        print(f"⚠ Queue file '{QUEUE_FILE}' không tồn tại. Vui lòng tạo file trước.")
        open(QUEUE_FILE, "w").close()  # tạo file rỗng
        start_monitoring()


# =======================
# MAIN
# =======================
if __name__ == "__main__":
    start_monitoring()
