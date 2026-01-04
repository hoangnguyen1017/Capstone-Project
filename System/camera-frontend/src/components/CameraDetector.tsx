import { useState, useRef, useEffect, useCallback } from "react";
import CameraCanvas from "./CameraCanvas";

interface Detection {
  id?: number;
  bbox?: [number, number, number, number];
  box?: [number, number, number, number];
  confidence: number;
  label: string;
  keypoints?: [number, number][];
  alerted?: boolean;
}

interface CameraDetectorProps {
  streamUrl: string;
  cameraId?: string;
  camIndex?: number;
  autoStart?: boolean;
  onFrame?: (frame: string) => void; 
}

const CameraDetector: React.FC<CameraDetectorProps> = ({
  streamUrl,
  cameraId,
  camIndex = 0,
  autoStart = false,
  onFrame,
}) => {
  const wsRef = useRef<WebSocket | null>(null);
  const lastFrameTime = useRef<number | null>(null);

  const [isActive, setIsActive] = useState(autoStart);
  const [status, setStatus] = useState("Idle");
  const [frame, setFrame] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [fps, setFps] = useState(0);
  const startWS = useCallback(() => {
    if (wsRef.current) wsRef.current.close();

    const ws = new WebSocket(streamUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("Connected");
      setIsActive(true);
      ws.send(
        JSON.stringify({
          action: "START",
          camera_id: cameraId,
          cam_index: camIndex,
        })
      );
    };

    ws.onmessage = (ev) => {
      const json = JSON.parse(ev.data);
      if (!json.frame) return;

      const frameData = `data:image/jpeg;base64,${json.frame}`;
      const dets = json.detections ?? [];

      setFrame(frameData);
      setDetections(dets);

      const now = performance.now();
      if (lastFrameTime.current) {
        setFps(1000 / (now - lastFrameTime.current));
      }
      lastFrameTime.current = now;
    };

    ws.onclose = () => {
      setStatus("Disconnected");
      setIsActive(false);
      setFrame(null);
      setDetections([]);
    };
  }, [streamUrl, cameraId, camIndex]);
  const toggle = () => {
    if (!isActive) {
      startWS();
    } else {
      wsRef.current?.send(
        JSON.stringify({ action: "STOP", camera_id: cameraId })
      );
      wsRef.current?.close();
    }
  };
  useEffect(() => {
    if (autoStart) startWS();
    return () => {
      wsRef.current?.close();
    };
  }, [autoStart, startWS]);

  return (
    <div className="flex flex-col w-full h-full">
      {/* Header */}
      <div className="flex justify-between mb-2 text-xs text-gray-600">
        <span>
          Camera {camIndex} – {status}
        </span>
        <button
          onClick={toggle}
          className={`px-3 py-1 rounded text-white ${
            isActive ? "bg-red-500" : "bg-green-600"
          }`}
        >
          {isActive ? "Turn off" : "Turn on"}
        </button>
      </div>

      {/* Canvas */}
      <CameraCanvas
        frame={frame}
        detections={detections}
        fps={fps}
        active={isActive}
        onRendered={(canvas) => {
          if (!onFrame) return;
          onFrame(canvas.toDataURL("image/jpeg", 0.8));
        }}
      />
    </div>
  );
};

export default CameraDetector;
