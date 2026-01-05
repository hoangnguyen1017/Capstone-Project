import dotenv from "dotenv";
dotenv.config();

import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import path from "path";
import { WebSocketServer, WebSocket } from "ws";  
import cameraRoutes from "./routes/camera";
import supervisorRoutes from "./routes/supervisor";
import alertRoutes from "./routes/alert";

const app = express();
const PORT = Number(process.env.PORT || 5000);

app.use(cors());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));
app.use("/uploads", express.static(path.join(__dirname, "../uploads")));
app.use("/api/cameras", cameraRoutes);
app.use("/api/supervisors", supervisorRoutes);
app.use("/api/alert", alertRoutes);

mongoose
  .connect(process.env.MONGO_URI || "mongodb://localhost:27017/camera_db")
  .then(() => console.log("MongoDB connected"))
  .catch((err) => console.log("MongoDB connect error:", err));

// HTTP server
const server = app.listen(PORT, () =>
  console.log(`Server running on port ${PORT}`)
);
const alertWSS = new WebSocketServer({ port: 5056 });

console.log("🔔 Alert WebSocket running at ws://localhost:5051");
export function broadcastAlert(data: any) {
  const json = JSON.stringify(data);

  alertWSS.clients.forEach((client: WebSocket) => {
    if (client.readyState === WebSocket.OPEN) {
      try {
        client.send(json);
      } catch (err) {
        console.warn("WS send error:", err);
      }
    }
  });
}
alertWSS.on("connection", (ws: WebSocket) => {
  console.log("🔔 Frontend connected, total:", alertWSS.clients.size);

  ws.on("close", () => {
    console.log("🔔 Frontend disconnected, total:", alertWSS.clients.size);
  });
});
