import express from "express";
import Camera from "../models/Camera";
import Supervisor from "../models/Supervisor";

const router = express.Router();
router.post("/", async (req, res) => {
  try {
    const { video_stream_url, camera_name, location } = req.body;

    if (!camera_name) {
      return res.status(400).json({ message: "Missing camera name" });
    }

    const newCamera = new Camera({ camera_name, video_stream_url, location });
    const savedCamera = await newCamera.save();

    res.status(201).json(savedCamera);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Failed to add camera" });
  }
});

router.get("/", async (req, res) => {
  try {
    const cameras = await Camera.find();
    res.json(cameras);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Error retrieving camera list" });
  }
});

router.delete("/:id", async (req, res) => {
  try {
    const cameraId = req.params.id;
    await Supervisor.deleteMany({ camera_id: cameraId });
    const deletedCamera = await Camera.findByIdAndDelete(cameraId);

    if (!deletedCamera) {
      return res.status(404).json({ message: "Camera does not exist" });
    }

    res.json({ message: "Camera deleted successfully", cameraId });
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Failed to delete camera" });
  }
});

router.put("/:id", async (req, res) => {
  try {
    const cameraId = req.params.id;
    console.log("Body received:", req.body);
    const updatedCamera = await Camera.findByIdAndUpdate(
      cameraId,
      { $set: req.body }, 
      { new: true, runValidators: true } 
    );

    if (!updatedCamera) {
      return res.status(404).json({ message: "Camera does not exist" });
    }

    res.json({ message: "Camera updated successfully", camera: updatedCamera });
  } catch (error: any) {
    console.error("Error updating camera:", error.message, error);
    res.status(500).json({ message: "Failed to update camera", error: error.message });
  }
});

export default router;
