import { useState, useEffect } from "react";
import axios from "axios";
import React from "react";

interface CameraFormProps {
  onAdd: (camera: any) => void;
}

interface DetectedCamera {
  index: number;
  name: string;
  ws_url: string;
}

const CameraForm: React.FC<CameraFormProps> = ({ onAdd }) => {
  const [cameraName, setCameraName] = useState("");
  const [location, setLocation] = useState("");
  const [isActive, setIsActive] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [availableCams, setAvailableCams] = useState<DetectedCamera[]>([]);
  const [selectedCam, setSelectedCam] = useState<DetectedCamera | null>(null);

  useEffect(() => {
    const detectCams = async () => {
      const cams: DetectedCamera[] = [];
      for (let i = 0; i <= 2; i++) {
        const ws_url = `ws://localhost:${8001 + i}`;
        try {
          await new Promise<void>((resolve, reject) => {
            const ws = new WebSocket(ws_url);
            const timeout = setTimeout(() => {
              ws.close();
              reject("timeout");
            }, 500); 

            ws.onopen = () => {
              clearTimeout(timeout);
              ws.close();
              resolve();
            };
            ws.onerror = () => {
              clearTimeout(timeout);
              reject("error");
            };
          });

          cams.push({
            index: i,
            name: `Camera ${i}`,
            ws_url,
          });
        } catch {
          // Silent catch for failed connections
        }
      }

      setAvailableCams(cams);
      if (cams.length > 0) setSelectedCam(cams[0]);
    };

    detectCams();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!cameraName || !selectedCam) {
      alert("Please select a camera and enter a camera name.");
      return;
    }

    setIsLoading(true);
    try {
      const cameraData = {
        camera_name: cameraName,
        video_stream_url: selectedCam.ws_url,
        location,
        is_active: isActive,
      };

      const res = await axios.post(
        "http://localhost:5000/api/cameras",
        cameraData
      );
      onAdd(res.data);
      setCameraName("");
      setLocation("");
      setIsActive(true);
      alert("Camera added successfully!");
    } catch (err) {
      console.error("Error adding camera:", err);
      alert("Failed to add camera. Please check the server connection.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
      <h2 className="text-xl font-semibold text-center text-indigo-700 mb-6">
        📷 Camera Configuration
      </h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <input
            type="text"
            placeholder="Camera Name"
            value={cameraName}
            onChange={(e) => setCameraName(e.target.value)}
            required
            className="w-full border border-gray-300 rounded-lg p-2 focus:ring-2 focus:ring-indigo-400 outline-none transition"
          />

          <select
            value={selectedCam?.index ?? ""}
            onChange={(e) => {
              const cam = availableCams.find(c => c.index === Number(e.target.value));
              setSelectedCam(cam ?? null);
            }}
            required
            className="w-full border border-gray-300 rounded-lg p-2 focus:ring-2 focus:ring-indigo-400 outline-none transition"
          >
            {availableCams.length === 0 && <option>No cameras detected</option>}
            {availableCams.map(cam => (
              <option key={cam.index} value={cam.index}>
                {cam.name} ({cam.ws_url})
              </option>
            ))}
          </select>

          <input
            type="text"
            placeholder="Installation Location (e.g., Room 101)"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            className="w-full border border-gray-300 rounded-lg p-2 focus:ring-2 focus:ring-indigo-400 outline-none transition"
          />
        </div>

        <button
          type="submit"
          disabled={isLoading || !selectedCam}
          className="w-full bg-gradient-to-r from-indigo-600 to-blue-500 text-white py-2.5 rounded-lg font-semibold hover:opacity-90 transition-all duration-200 disabled:opacity-50 shadow-md"
        >
          {isLoading ? "Adding Camera..." : "➕ Add Camera"}
        </button>
      </form>
    </div>
  );
};

export default CameraForm;