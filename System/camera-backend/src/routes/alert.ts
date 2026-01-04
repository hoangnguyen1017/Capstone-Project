import express from "express";
import Camera from "../models/Camera";
import Supervisor from "../models/Supervisor";
import { broadcastAlert } from "../index";

const router = express.Router();

router.post("/fall", async (req, res) => {
  try {
    const { camera_id } = req.body;
    if (!camera_id) return res.status(400).json({ message: "Missing camera_id" });

    const camera = await Camera.findById(camera_id).lean();
    if (!camera) return res.status(404).json({ message: "Camera not found" });

    const nowVN = new Date().toLocaleString("vi-VN", {
      timeZone: "Asia/Ho_Chi_Minh",
    });
    broadcastAlert({
      type: "FALL_ALERT",
      camera_id: camera_id,
      camera_name: camera.camera_name,
      location: camera.location,
      timestamp: nowVN,
    });

    return res.json({ message: "Alert broadcasted (Python handles email)" });
  } catch (err) {
    console.error("Alert route error:", err);
    res.status(500).json({ message: "Server error" });
  }
});

export default router;
