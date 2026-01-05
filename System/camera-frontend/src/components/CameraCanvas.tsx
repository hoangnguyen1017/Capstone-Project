import { useRef, useEffect, useCallback } from "react";

interface Detection {
  id?: number;
  bbox?: [number, number, number, number];
  box?: [number, number, number, number];
  confidence: number;
  label: string;
  keypoints?: [number, number][];
  alerted?: boolean;
}

interface CameraCanvasProps {
  frame: string | null;
  detections: Detection[];
  fps: number;
  active: boolean;
  onRendered?: (canvas: HTMLCanvasElement) => void;
}


const SKELETON_EDGES: [number, number][] = [
  [0, 1], [1, 2], [1, 3], [2, 4], [4, 6], [3, 5], [5, 7],
  [2, 8], [3, 9], [8, 9], [8, 10], [10, 12], [9, 11], [11, 13],
];

const SMOOTH_ALPHA = 0.35;
const CameraCanvas: React.FC<CameraCanvasProps> = ({
  frame,
  detections,
  fps,
  active,
  onRendered,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement>(new Image());
  const rafRef = useRef<number | null>(null);
  const boxMemory = useRef<Map<number, [number, number, number, number]>>(
    new Map()
  );

  const smoothBox = (
    id: number,
    box: [number, number, number, number]
  ): [number, number, number, number] => {
    const map = boxMemory.current;
    if (!map.has(id)) {
      map.set(id, box);
      return box;
    }

    const p = map.get(id)!;
    const s: [number, number, number, number] = [
      p[0] * (1 - SMOOTH_ALPHA) + box[0] * SMOOTH_ALPHA,
      p[1] * (1 - SMOOTH_ALPHA) + box[1] * SMOOTH_ALPHA,
      p[2] * (1 - SMOOTH_ALPHA) + box[2] * SMOOTH_ALPHA,
      p[3] * (1 - SMOOTH_ALPHA) + box[3] * SMOOTH_ALPHA,
    ];
    map.set(id, s);
    return s;
  };

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img.complete) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = img.width;
    canvas.height = img.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

 

    detections.forEach((d, i) => {
      const box = d.bbox ?? d.box;
      if (!box) return;

      const [x1, y1, x2, y2] = smoothBox(d.id ?? i, box);
      const color = d.label === "fall" ? "#ff4444" : "#10b981";

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const displayText = `ID:${d.id ?? i} ${d.label}`;
      ctx.fillStyle = color;  
      ctx.font = "12px sans-serif";
      ctx.fillText(displayText, x1, Math.max(10, y1 - 6));

      if (d.keypoints?.length) {
        const pts = d.keypoints.map(([x, y]) => {
          if (
            x == null || y == null ||
            x <= 0 || y <= 0 ||
            x > 1 || y > 1 ||
            Number.isNaN(x) || Number.isNaN(y)
          ) {
            return null;
          }
          return [x * canvas.width, y * canvas.height] as [number, number];
        });

        ctx.strokeStyle = "#ffdd00";
        ctx.lineWidth = 2;

        SKELETON_EDGES.forEach(([a, b]) => {
          const pa = pts[a];
          const pb = pts[b];
          if (!pa || !pb) return;

          ctx.beginPath();
          ctx.moveTo(pa[0], pa[1]);
          ctx.lineTo(pb[0], pb[1]);
          ctx.stroke();
        });

        pts.forEach((p) => {
          if (!p) return;
          ctx.fillStyle = "#ffdd00";
          ctx.beginPath();
          ctx.arc(p[0], p[1], 3, 0, Math.PI * 2);
          ctx.fill();
        });
      }
    }); 

    ctx.fillStyle = "#ff4444";
    ctx.font = "12px monospace";
    ctx.fillText(`FPS: ${fps.toFixed(1)}`, 10, 18);
    onRendered?.(canvas);

    rafRef.current = requestAnimationFrame(render);
  }, [detections, fps, onRendered]);
  useEffect(() => {
    if (!frame) return;

    imgRef.current.onload = () => {
      if (rafRef.current === null) {
        rafRef.current = requestAnimationFrame(render);
      }
    };

    imgRef.current.src = frame;

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [frame, render]);
  return (
    <div className="relative flex-1 bg-black rounded overflow-hidden">
      <canvas ref={canvasRef} className="w-full h-full" />
      {!active && (
        <div className="absolute inset-0 flex items-center justify-center text-white bg-black/60">
          Live Off
        </div>
      )}
    </div>
  );
};

export default CameraCanvas;
