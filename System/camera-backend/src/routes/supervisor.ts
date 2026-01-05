import express from "express";
import Supervisor from "../models/Supervisor";
import Camera from "../models/Camera";

const router = express.Router();

router.post("/", async (req, res) => {
  try {
    const { name, email, phone, camera_ids } = req.body;

    if (!name || !email || !camera_ids || camera_ids.length === 0) {
      return res.status(400).json({ message: "Missing required information" });
    }

    const newSupervisor = new Supervisor({
      name,
      email,
      phone,
      camera_ids,
    });

    const savedSupervisor = await newSupervisor.save();

    await Camera.updateMany(
      { _id: { $in: camera_ids } },
      {
        responsible_name: name,
        responsible_email: email,
        responsible_phone: phone,
      }
    );

    res.status(201).json(savedSupervisor);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: "Failed to add supervisor" });
  }
});

router.get("/", async (req, res) => {
  try {
    const supervisors = await Supervisor.find().populate(
      "camera_ids",
      "camera_name location"
    );
    res.json(supervisors);
  } catch (error) {
    res.status(500).json({ message: "Error fetching supervisor list" });
  }
});

router.delete("/:id", async (req, res) => {
  try {
    const supervisor = await Supervisor.findById(req.params.id);

    if (!supervisor) {
      return res.status(404).json({ message: "Supervisor not found" });
    }

    await Camera.updateMany(
      { _id: { $in: supervisor.camera_ids } },
      {
        responsible_name: "",
        responsible_email: "",
        responsible_phone: "",
      }
    );

    await Supervisor.findByIdAndDelete(req.params.id);

    res.json({ message: "Supervisor deleted successfully" });
  } catch (error) {
    res.status(500).json({ message: "Failed to delete supervisor" });
  }
});

router.put("/:id", async (req, res) => {
  try {
    const { name, email, phone, camera_ids } = req.body;
    const supervisor = await Supervisor.findById(req.params.id);

    if (!supervisor) {
      return res.status(404).json({ message: "Supervisor not found" });
    }

    await Camera.updateMany(
      { _id: { $in: supervisor.camera_ids } },
      {
        responsible_name: "",
        responsible_email: "",
        responsible_phone: "",
      }
    );

    supervisor.name = name;
    supervisor.email = email;
    supervisor.phone = phone;
    supervisor.camera_ids = camera_ids;
    await supervisor.save();

    await Camera.updateMany(
      { _id: { $in: camera_ids } },
      {
        responsible_name: name,
        responsible_email: email,
        responsible_phone: phone,
      }
    );

    res.json(supervisor);
  } catch (error) {
    res.status(500).json({ message: "Failed to update supervisor" });
  }
});

export default router;
