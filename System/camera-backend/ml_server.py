import asyncio
import cv2
import json
import base64
import numpy as np
from ultralytics import YOLO
import time
import websockets
import aiohttp
import torch
from collections import deque, defaultdict
import os
import signal

MODEL_SERVER_URL = "http://localhost:6000/predict"
CAM_FPS = 60
YOLO_MODEL_PATH = r"D:\nopbai\System\best912.pt"
WS_BIND = "127.0.0.1"

FALL_CONFIRM_HOLD = 5.0
FALL_RESET_HYSTERESIS = 2.0

KP_CONF_TH = 0.6
MIN_VALID_KP = 12

QUEUE_FILE = "fall_event_queue.jsonl"
FRAME_SAVE_DIR = "fall_frames"
os.makedirs(FRAME_SAVE_DIR, exist_ok=True)

connected_clients = dict() 
latest_results_by_track = defaultdict(lambda: {"label": "unknown", "confidence": 0.0, "ts": 0.0})
state_by_track = defaultdict(lambda: {
    "state": "normal",   
    "last_change": 0.0,
    "fall_ts": 0.0,
    "last_not_fall_ts": 0.0,
    "alert_logged": False
})
async def fetch_camera_metadata():
    url = "http://localhost:5000/api/cameras"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=5) as res:
                cams = await res.json()
                mapping = {}
                for cam in cams:
                    stream = cam.get("video_stream_url")
                    port = int(stream.split(":")[-1]) 
                    mapping[port] = cam
                return mapping
        except Exception as e:
            print("❌ Failed to fetch camera metadata:", e)
            return {}


def update_fall_state(track_id, st_label, st_conf, now_ts):
    s = state_by_track[track_id]
    state = s["state"]

    if st_label.lower() == "fall" and st_conf >= 0.6:
        if state != "fall_confirmed":
            s["state"] = "fall_confirmed"
            s["fall_ts"] = now_ts
            s["last_change"] = now_ts
            s["last_not_fall_ts"] = 0.0
            s["alert_logged"] = False
        return s["state"]

    if state == "fall_confirmed":
        if now_ts - s["fall_ts"] < FALL_CONFIRM_HOLD:
            return "fall_confirmed"

        if st_label.lower() != "fall":
            if s["last_not_fall_ts"] == 0.0:
                s["last_not_fall_ts"] = now_ts
            if now_ts - s["last_not_fall_ts"] >= FALL_RESET_HYSTERESIS:
                s["state"] = "normal"
                s["last_change"] = now_ts
                s["last_not_fall_ts"] = 0.0
                s["alert_logged"] = False
                return s["state"]
            else:
                return "fall_confirmed"
        else:
            s["last_not_fall_ts"] = 0.0
            return "fall_confirmed"

    s["state"] = "normal"
    s["last_change"] = now_ts
    s["last_not_fall_ts"] = 0.0
    s["alert_logged"] = False
    return "normal"

class TrackerManager:
    def __init__(self, min_frames=15, max_lost=20):
        self.min_frames = min_frames
        self.max_lost = max_lost
        self.temp_tracks = {}
        self.final_tracks = {}
        self.last_seen = {}
        self.frame_count = {}

    def update(self, raw_id):
        if raw_id not in self.frame_count:
            self.frame_count[raw_id] = 1
            self.last_seen[raw_id] = 0
        else:
            self.frame_count[raw_id] += 1
            self.last_seen[raw_id] = 0

        if self.frame_count[raw_id] < self.min_frames:
            return None

        if raw_id not in self.final_tracks:
            self.final_tracks[raw_id] = raw_id
        return self.final_tracks[raw_id]

    def mark_lost(self, seen_ids):
        lost_list = []
        for tid in list(self.frame_count.keys()):
            if tid not in seen_ids:
                self.last_seen[tid] += 1
            else:
                self.last_seen[tid] = 0

            if tid in self.final_tracks:
                if self.last_seen[tid] > self.max_lost:
                    lost_list.append(tid)
                    del self.final_tracks[tid]
                    del self.frame_count[tid]
                    del self.last_seen[tid]
            else:
                if self.last_seen[tid] > self.min_frames:
                    del self.frame_count[tid]
                    del self.last_seen[tid]
        return lost_list

def merge_tracks(raw_id, bbox, prev_boxes, final_tracks):
    if raw_id in final_tracks:
        return raw_id 

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    best_tid = None
    best_iou = 0.0
    for tid, old_bbox in prev_boxes.items():
        if tid not in final_tracks:
            continue
        cur_iou = iou(bbox, old_bbox)
        if cur_iou > best_iou and cur_iou > 0.4:
            best_iou = cur_iou
            best_tid = tid
    return best_tid

def push_event_to_queue(event: dict):
    try:
        with open(QUEUE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        print("Failed to write queue:", e)

async def stgcn_broadcast_loop(
    port,
    pose_windows_by_track,
    pose_lock,
    last_sent_ts,
    ws_port,
    recent_frames_by_track,
    camera_name,
    location,
    state_by_track,
    latest_results_by_track,
):
    SEND_COOLDOWN = 0.5
    PRUNE_TIMEOUT = 5.0

    session = aiohttp.ClientSession()

    try:
        while True:
            now_ts = time.time()
            async with pose_lock:
                for tid, window in list(pose_windows_by_track.items()):
                    if not window:
                        continue
                    if now_ts - window[-1]["ts"] > PRUNE_TIMEOUT:
                        pose_windows_by_track.pop(tid, None)
                        last_sent_ts.pop(tid, None)
                        latest_results_by_track.pop(tid, None)
                        state_by_track.pop(tid, None)

            async with pose_lock:
                tracks = list(pose_windows_by_track.items())

            for tid, window in tracks:
                current_len = len(window)
                pred_label = "unknown"
                pred_conf = 0.0
                if current_len not in (1, 16, 32):
                    pass
                else:
                    print(f"[ST-GCN] track={tid}, frames={current_len}")

                if current_len != 32:
                    continue

                window_end_ts = window[-1]["ts"]
                if window_end_ts - last_sent_ts.get(tid, 0) < SEND_COOLDOWN:
                    continue
                qualities = [x.get("quality", 0.0) for x in window]
                avg_quality = float(np.mean(qualities))
                bad_frames = sum(q < 0.5 for q in qualities)

                valid_kps = [
                    np.sum(np.array(x["kps"])[:, 2] > 0) for x in window
                ]
                avg_valid_kps = float(np.mean(valid_kps))

                reject_reasons = []
                if avg_valid_kps < MIN_VALID_KP:
                    reject_reasons.append("low_kp")
                if reject_reasons:
                    print(f"[ST-GCN] track {tid} -> UNKNOWN ({', '.join(reject_reasons)})")

                    latest_results_by_track[tid] = {
                        "label": "unknown",
                        "confidence": 0.0,
                        "ts": window_end_ts
                    }

                    payload = json.dumps({
                        "type": "stgcn_result",
                        "track_id": tid,
                        "window_end_ts": window_end_ts,
                        "result": {
                            "prediction": "unknown",
                            "confidence": 0.0,
                            "state": "normal"
                        }
                    })

                    for ws in connected_clients.get(ws_port, set()).copy():
                        try:
                            await ws.send(payload)
                        except:
                            connected_clients[ws_port].discard(ws)

                    last_sent_ts[tid] = window_end_ts
                    continue

                try:
                    kp_array = np.stack(
                        [np.array(x["kps"], dtype=np.float32) for x in window],
                        axis=0
                    )  

                    kp_array = np.nan_to_num(kp_array, nan=0.0)
                    kp_array[:, :, 2] = np.clip(kp_array[:, :, 2], 0.0, 1.0)
                    kp_list = kp_array.tolist()

                except Exception as e:
                    print(f"[ST-GCN] kp stack error (tid={tid}): {e}")
                    last_sent_ts[tid] = window_end_ts
                    continue
                try:
                    async with session.post(
                        MODEL_SERVER_URL,
                        json={
                            "track_id": tid,
                            "joint": kp_list,
                            "window_end_ts": window_end_ts
                        },
                        timeout=5
                    ) as res:
                        result = await res.json()
                except Exception as e:
                    print(f"[ST-GCN] request error: {e}")
                    result = {}

                pred_label = str(result.get("prediction", "unknown"))
                try:
                    pred_conf = float(result.get("confidence", 0.0))
                except:
                    pred_conf = 0.0
                final_state = update_fall_state(tid, pred_label, pred_conf, window_end_ts)
                print(f"[ST-GCN] track={tid}, frames={current_len}, confidence={pred_conf:.3f}")
                s = state_by_track.get(tid)
                if s is None:
                    continue

                if final_state == "fall_confirmed" and not s.get("alert_logged", False):

                    saved = []
                    try:
                        recent_bufs = recent_frames_by_track.get(tid, [])
                        for i, jpg_bytes in enumerate(list(recent_bufs)[-10:]):
                            fname = os.path.join(
                                FRAME_SAVE_DIR,
                                f"cam{port}_tid{tid}_{int(window_end_ts)}_{i}.jpg"
                            )
                            with open(fname, "wb") as f:
                                f.write(jpg_bytes)
                            saved.append(fname)
                    except Exception as e:
                        print("Frame save error:", e)

                    push_event_to_queue({
                        "track_id": tid,
                        "camera_index": port,
                        "camera_name": camera_name,
                        "location": location,
                        "timestamp": window_end_ts,
                        "frames": saved
                    })

                    alert_payload = json.dumps({
                        "type": "FALL_ALERT",
                        "camera_name": camera_name,
                        "location": location,
                        "timestamp": time.strftime(
                            "%Y-%m-%d %H:%M:%S",
                            time.localtime(window_end_ts)
                        )
                    })

                    for ws in connected_clients.get(ws_port, set()).copy():
                        try:
                            await ws.send(alert_payload)
                        except:
                            connected_clients[ws_port].discard(ws)

                    s["alert_logged"] = True

                if final_state == "normal":
                    s["alert_logged"] = False

                if final_state == "fall_confirmed":
                    pred_label = "fall"
                    pred_conf = max(pred_conf, 0.9)

                latest_results_by_track[tid] = {
                    "label": pred_label,
                    "confidence": pred_conf,
                    "ts": window_end_ts
                }

                payload = json.dumps({
                    "type": "stgcn_result",
                    "track_id": tid,
                    "window_end_ts": window_end_ts,
                    "result": {
                        "prediction": pred_label,
                        "confidence": pred_conf,
                        "state": final_state
                    }
                })

                for ws in connected_clients.get(ws_port, set()).copy():
                    try:
                        await ws.send(payload)
                    except:
                        connected_clients[ws_port].discard(ws)

                last_sent_ts[tid] = window_end_ts

            await asyncio.sleep(0.08)

    finally:
        await session.close()

class CameraPipeline:
    def __init__(self, cam_index, ws_port, camera_name="Camera", location="Unknown"):
        self.cam_index = cam_index
        self.ws_port = ws_port
        self.camera_name = camera_name
        self.location = location
        connected_clients.setdefault(ws_port, set())

        self.capture_queue = asyncio.Queue(maxsize=1)
        self.send_queue = asyncio.Queue(maxsize=15)
        self.pose_windows_by_track = defaultdict(lambda: deque(maxlen=32))
        self.pose_lock = asyncio.Lock()
        self.last_sent_ts = {}

        self.is_active = False
        self.cam_fps = CAM_FPS
        self.model = YOLO(YOLO_MODEL_PATH)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.prev_boxes = {}
        self.prev_keypoints = {}
        self.SMOOTH_ALPHA = 0.3
        self.tracker = TrackerManager(min_frames=10, max_lost=5)
        self.recent_frames_by_track = defaultdict(lambda: deque(maxlen=64))

    def kp_inside_bbox(self, kp_xy, bbox):
        """
        kp_xy: (14,2) normalized [0,1]
        bbox: [x1,y1,x2,y2] pixel
        """
        x1, y1, x2, y2 = bbox
        x1 /= self.W
        x2 /= self.W
        y1 /= self.H
        y2 /= self.H

        inside_x = (kp_xy[:, 0] >= x1) & (kp_xy[:, 0] <= x2)
        inside_y = (kp_xy[:, 1] >= y1) & (kp_xy[:, 1] <= y2)
        return inside_x & inside_y

    def smooth_bbox(self, track_id, bbox, alpha=0.2):
        if track_id not in self.prev_boxes:
            self.prev_boxes[track_id] = bbox
            return bbox

        prev = self.prev_boxes[track_id]
        sm = [
            prev[i] * (1 - alpha) + bbox[i] * alpha
            for i in range(4)
        ]

        self.prev_boxes[track_id] = sm
        return sm

    def smooth_keypoints(self, track_id, kps, kp_conf, conf_th=0.4):
        if track_id not in self.prev_keypoints:
            self.prev_keypoints[track_id] = kps
            return kps

        prev = self.prev_keypoints[track_id].copy()
        sm = np.zeros_like(kps)

        for i in range(len(kps)):
            if kp_conf[i] >= conf_th and not np.all(prev[i] == 0):
                sm[i] = prev[i] * (1 - self.SMOOTH_ALPHA) + kps[i] * self.SMOOTH_ALPHA
            else:
                sm[i] = kps[i]   

        self.prev_keypoints[track_id] = sm
        return sm

    async def capture_loop(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FPS, self.cam_fps)
        print(f"📷 CAPTURE[{self.cam_index}] START at WS port {self.ws_port}")
        while True:
            if not self.is_active or not connected_clients[self.ws_port]:
                await asyncio.sleep(0.01)
                continue
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.005)
                continue
            if self.capture_queue.full():
                try:
                    _ = self.capture_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await self.capture_queue.put(frame)
            await asyncio.sleep(1 / self.cam_fps)

    async def inference_loop(self):
        last_send = 0
        frame_count = 0
        fps_start = time.time()
        while True:
            if not self.is_active or not connected_clients[self.ws_port]:
                await asyncio.sleep(0.01)
                continue

            try:
                while self.capture_queue.qsize() > 1:
                    _ = self.capture_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

            frame = await self.capture_queue.get()

            now = time.time()
            frame_count += 1
            if now - fps_start >= 1.0:
                print(f"📊 CAM[{self.cam_index}] FPS ≈ {frame_count/(now-fps_start):.2f}")
                frame_count = 0
                fps_start = now

            H, W = frame.shape[:2]
            self.H = H
            self.W = W

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if not ok:
                await asyncio.sleep(0.001)
                continue
            jpg_bytes = buf.tobytes()
            b64 = base64.b64encode(jpg_bytes).decode()

            results = self.model.track(
                frame,
                device=self.device,
                persist=True,
                conf=0.25,
                iou=0.5
            )

            dets = []

            seen_ids = set()

            for r in results:
                for idx, box in enumerate(r.boxes):
                    if box.id is None:
                        continue

                    raw_id = int(box.id[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    final_id = merge_tracks(raw_id, [x1, y1, x2, y2], self.prev_boxes, self.tracker.final_tracks)
                    if final_id is None:
                        final_id = self.tracker.update(raw_id)
                    if final_id is None:
                        continue
                    track_id = final_id
                    if track_id in seen_ids:
                        continue
                    seen_ids.add(track_id)

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    sm_box = self.smooth_bbox(track_id, [x1, y1, x2, y2])
                    if hasattr(r, "keypoints") and idx < len(r.keypoints.data):
                        kp_xy = r.keypoints.data[idx].cpu().numpy()[:, :2]
                        kp_conf = r.keypoints.conf[idx].cpu().numpy()
                        kp_xy = kp_xy / np.array([W, H])
                        kp_xy = np.clip(kp_xy, 0.0, 1.0)

                        conf_mask = kp_conf >= KP_CONF_TH
                        bbox_mask = self.kp_inside_bbox(kp_xy, sm_box)
                        valid_mask = conf_mask & bbox_mask

                        num_valid = int(valid_mask.sum())
                        pose_quality = num_valid / 14.0

                        kp_xy_masked = kp_xy.copy()
                        kp_conf_masked = kp_conf.copy()
                        kp_xy_masked[~valid_mask] = 0.0
                        kp_conf_masked[~valid_mask] = 0.0

                        kp_xy_sm = self.smooth_keypoints(
                            track_id,
                            kp_xy_masked,
                            kp_conf_masked
                        )
                        kp_out = np.zeros((14, 3), dtype=np.float32)
                        kp_out[valid_mask, 0:2] = kp_xy_sm[valid_mask]
                        kp_out[valid_mask, 2] = kp_conf_masked[valid_mask]

                    else:
                        continue

                    lr = latest_results_by_track.get(track_id, {"label": "unknown", "confidence": 0.0, "ts": 0.0})
                    label, conf = lr["label"], lr["confidence"]
                    if now - lr["ts"] > 5.0:
                        label, conf = "unknown", 0.0
                        
                    try:
                        self.recent_frames_by_track[track_id].append(jpg_bytes)
                    except Exception:
                        pass

                    dets.append({"id": track_id, "bbox": sm_box, "keypoints": kp_out.tolist(), "label": label, "confidence": conf})

                    async with self.pose_lock:
                        window = self.pose_windows_by_track[track_id]
                        window.append({
                            "ts": now,
                            "kps": kp_out.tolist(),
                            "quality": pose_quality
                        })

            lost_ids = self.tracker.mark_lost(seen_ids)
            for tid in lost_ids:
                self.prev_keypoints.pop(tid, None)
                self.prev_boxes.pop(tid, None)
                self.pose_windows_by_track.pop(tid, None)
                self.recent_frames_by_track.pop(tid, None)

            hide_bbox = all(len(w) < 32 for w in self.pose_windows_by_track.values())
            send_dets = dets.copy()
            if hide_bbox:
                for d in send_dets:
                    d["label"] = "loading"
                    d["confidence"] = 0.0
            else:
                non_fall = [d for d in dets if d["label"].lower() != "fall"]
                fall = [d for d in dets if d["label"].lower() == "fall"]
                send_dets = non_fall + fall

            if now - last_send >= 1 / self.cam_fps:
                last_send = now
                payload = {"timestamp": now, "frame": b64, "detections": send_dets, "camera_index": self.cam_index}
                if self.send_queue.full():
                    try:
                        _ = self.send_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await self.send_queue.put(json.dumps(payload))

    async def websocket_send_loop(self):
        while True:
            payload = await self.send_queue.get()
            for ws in connected_clients[self.ws_port].copy():
                try:
                    await ws.send(payload)
                except:
                    connected_clients[self.ws_port].discard(ws)

    async def handle_ws_client(self, ws):
        connected_clients[self.ws_port].add(ws)
        print(f"🔗 Client connect CAM {self.cam_index} (port {self.ws_port})")
        try:
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except:
                    continue
                act = data.get("action")
                if act == "START":
                    self.is_active = True
                    print(f"🟢 CAM[{self.cam_index}] START")
                elif act == "STOP":
                    self.is_active = False
                    print(f"🔴 CAM[{self.cam_index}] STOP")
                elif act == "GET_LATEST":
                    tid = int(data.get("track_id", -1))
                    res = latest_results_by_track.get(tid, {"label": "unknown", "confidence": 0.0, "ts": 0.0})
                    try:
                        await ws.send(json.dumps({"type": "latest_result", "track_id": tid, "result": res}))
                    except:
                        pass
        finally:
            connected_clients[self.ws_port].discard(ws)
            print(f"❌ Client disconnect CAM {self.cam_index} (port {self.ws_port})")

    async def start_ws_server(self):
        server = await websockets.serve(self.handle_ws_client, WS_BIND, self.ws_port)
        print(f"🌐 WEBSOCKET CAM[{self.cam_index}] at port {self.ws_port}")
        await server.wait_closed()


def scan_cameras(max_test=5):
    cams = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"✅ Camera {i} FOUND")
                cams.append(i)
        cap.release()
    return cams

async def main():
    cameras = scan_cameras()
    if not cameras:
        print("❌ No camera found")
        return
    tasks = []
    camera_metadata = await fetch_camera_metadata()
    for idx, cam_index in enumerate(cameras):
        port = 8001 + idx
        cam_info = camera_metadata.get(port, {})

        pipeline = CameraPipeline(
            cam_index,
            port,
            camera_name=cam_info.get("camera_name", f"Camera_{port}"),
            location=cam_info.get("location", "Unknown")
        )

        tasks += [
            asyncio.create_task(pipeline.capture_loop()),
            asyncio.create_task(pipeline.inference_loop()),
            asyncio.create_task(
                stgcn_broadcast_loop(
                    port,
                    pipeline.pose_windows_by_track,
                    pipeline.pose_lock,
                    pipeline.last_sent_ts,
                    port, 
                    pipeline.recent_frames_by_track,
                    pipeline.camera_name,
                    pipeline.location,
                    state_by_track,             
                    latest_results_by_track,     
                )
            ),
            asyncio.create_task(pipeline.websocket_send_loop()),
            asyncio.create_task(pipeline.start_ws_server())
        ]
    stop_event = asyncio.Event()

    def _signal_handler(*_):
        print("🛑 Received stop signal, shutting down...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _signal_handler)
        except NotImplementedError:
            pass

    await stop_event.wait()
    print("Waiting tasks to finish (cancel)...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    print("Server stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 Server stopped manually")
