import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import alertSound from "../assets/alert.mp3"
interface NavbarProps {
  onAddCameraClick?: () => void;
  onManageSupervisorClick: () => void;
}

interface Notif {
  id: number;
  text: string;
  timestamp?: string;
}

const Navbar: React.FC<NavbarProps> = ({
  onAddCameraClick,
  onManageSupervisorClick,
}) => {
  const [openNotif, setOpenNotif] = useState(false);
  const [notifications, setNotifications] = useState<Notif[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    audioRef.current = new Audio(alertSound);
    audioRef.current.loop = false;
    audioRef.current.load();

    const unlockAudio = () => {
      audioRef.current?.play().then(() => audioRef.current?.pause());
      window.removeEventListener("click", unlockAudio);
    };
    window.addEventListener("click", unlockAudio);
    const cameraPorts = [8001, 8002];
    const wsList: WebSocket[] = [];
    const handleWsMessage = (ev: MessageEvent) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === "FALL_ALERT") {
          const id = Date.now();
          const text = `⚠️ Fall detected at ${data.camera_name}${data.location ? " - " + data.location : ""}`;
          setNotifications((prev) => [
            { id, text, timestamp: data.timestamp },
            ...prev,
          ]);
          if (audioRef.current) {
            audioRef.current.currentTime = 0;
            audioRef.current.play().catch((e) => {
              console.warn("Audio play blocked:", e);
            });
          }
        }
      } catch (err) {
        console.warn("WS parse error", err);
      }
    };
    cameraPorts.forEach((port) => {
      const ws = new WebSocket(`ws://localhost:${port}`);
      wsList.push(ws);

      ws.onopen = () => console.log(`Connected to camera WS at port ${port}`);
      ws.onmessage = handleWsMessage;
      ws.onerror = (e) => console.warn(`WS error port ${port}:`, e);
      ws.onclose = () => console.log(`WS closed port ${port}`);
    });

    return () => {
      wsList.forEach((ws) => ws.close());
      audioRef.current?.pause();
      audioRef.current = null;
      window.removeEventListener("click", unlockAudio);
    };
  }, []);

  return (
    <nav className="relative w-full h-[70px] flex items-center justify-between px-4 sm:px-8 lg:px-10 py-4 bg-gradient-to-r from-slate-950/90 via-slate-900/70 to-slate-950/90 backdrop-blur-xl border-b border-white/10 shadow-[0_10px_35px_rgba(0,0,0,0.55)]">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => navigate("/")}
          className="flex h-11 w-11 items-center justify-center text-white hover:text-emerald-300 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-emerald-400/70 rounded-2xl border border-white/10 bg-white/5 shadow-inner shadow-white/5"
          aria-label="Go to home">
          <span className="text-2xl">🏠</span>
        </button>
        <div className="hidden sm:block">
          <p className="text-xs uppercase tracking-[0.3em] text-emerald-300/70">
            dashboard
          </p>
          <h1 className="text-xl font-semibold text-white">Camera Control</h1>
        </div>
      </div>

      <div className="flex gap-3 sm:gap-4 items-center">
        {/* Notification Icon */}
        <div className="relative">
          <button
            onClick={() => setOpenNotif(!openNotif)}
            className="relative w-12 h-12 rounded-2xl bg-white/5 border border-white/10 text-white text-2xl focus:outline-none focus:ring-2 focus:ring-blue-400/60 hover:text-blue-300 transition">
            <span role="img" aria-label="Notification">
              🔔
            </span>
            {/* Badge count */}
            <span className="absolute -top-2 -right-2 bg-red-600 text-white text-xs px-1.5 py-0.5 rounded-full">
              {notifications.length}
            </span>
          </button>

          {/* Notification dropdown */}
          {openNotif && (
            <div className="absolute right-0 mt-3 w-80 bg-slate-900/95 text-white rounded-2xl border border-white/10 shadow-[0_25px_60px_rgba(0,0,0,0.65)] p-4 z-50 backdrop-blur-lg">
              <h3 className="font-semibold text-white mb-2 flex items-center gap-2">
                <span className="text-lg">📡</span>
                Notifications
              </h3>
              <p className="text-xs text-gray-300/80 mb-3">
                Auto updated alerts
              </p>

              <ul className="space-y-3 max-h-60 overflow-y-auto">
                {notifications.length === 0 && (
                  <li className="p-3 text-center text-gray-300/70">No notifications</li>
                )}
                {notifications.map((n) => (
                  <li key={n.id} className="p-3 bg-white/5 border border-white/5 rounded-xl text-sm text-gray-100 hover:bg-white/10 transition flex flex-col gap-1">
                    <span>{n.text}</span>
                    {n.timestamp && (
                      <span className="text-xs text-gray-400/70">{n.timestamp}</span>
                    )}
                  </li>
                ))}
              </ul>

            </div>
          )}
        </div>

        {/* Supervisor Button */}
        <button
          onClick={onManageSupervisorClick}
          className="inline-flex items-center gap-2 bg-gradient-to-r from-sky-500 to-indigo-500 hover:from-sky-400 hover:to-indigo-400 px-5 py-2 rounded-full font-semibold text-white text-sm transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-sky-300/70 shadow-lg shadow-sky-500/30">
          <span className="text-lg">👤</span>
          Supervisor
        </button>
      </div>
    </nav>
  );
};

export default Navbar;
