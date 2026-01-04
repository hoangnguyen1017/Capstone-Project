import { useEffect, useState, useCallback } from "react";
import axios from "axios";
import CameraForm from "../components/CameraForm";
import SupervisorList from "../components/SupervisorList";
import CameraList from "../components/CameraList";
import CameraDetailModal from "../components/CameraDetailModal";
import Navbar from "../components/Navbar";
import "../components/Modal.css";
import BackgroundImg from "../assets/Hero.svg";
interface ICamera {
  _id: string;
  camera_name: string;
  location: string;
  responsible_name: string;
  alert_email: string;
  alert_phone: string;
  is_active: boolean;
  video_stream_url?: string;
}

const Dashboard: React.FC = () => {
  const [cameras, setCameras] = useState<ICamera[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<ICamera | null>(null);
  const [activeCameraId, setActiveCameraId] = useState<string | null>(null);
  const [cameraFrames, setCameraFrames] = useState<Record<string, string>>({});
  const [showCameraModal, setShowCameraModal] = useState(false);
  const [showSupervisorList, setShowSupervisorList] = useState(false);

  const BASE_URL = "http://localhost:5000";
  const fetchCameras = async (): Promise<ICamera[]> => {
    try {
      const res = await axios.get<ICamera[]>(`${BASE_URL}/api/cameras`);
      setCameras(res.data);
      return res.data;
    } catch (err) {
      console.error("Error fetching camera list:", err);
      return [];
    }
  };

  useEffect(() => {
    fetchCameras();
  }, []);
  const handleAddCamera = (camera: ICamera) => {
    setCameras([camera, ...cameras]);
    setShowCameraModal(false);
  };
  const handleSupervisorChange = async () => {
    const updatedCameras = await fetchCameras();
    if (activeCameraId) {
      const updatedCam = updatedCameras.find((c) => c._id === activeCameraId);
      if (updatedCam) setSelectedCamera(updatedCam);
    }
  };

  const handleDeleteCamera = async (id: string) => {
    try {
      await axios.delete(`${BASE_URL}/api/cameras/${id}`);
      setCameras((prev) => prev.filter((cam) => cam._id !== id));
      if (selectedCamera?._id === id) {
        setSelectedCamera(null);
        setActiveCameraId(null);
      }
      alert("Camera deleted successfully!");
    } catch (err) {
      console.error("Failed to delete camera:", err);
      alert("Failed to delete camera!");
    }
  };

  const handleSelectCamera = useCallback((camera: ICamera) => {
    setSelectedCamera(camera);
    setActiveCameraId(camera._id);
  }, []);

  const handleCloseModal = useCallback(() => {
    setSelectedCamera(null);
    setActiveCameraId(null);
  }, []);

  const handleUpdateCamera = (updatedCamera: ICamera) => {
    setCameras((prevCameras) =>
      prevCameras.map((cam) =>
        cam._id === updatedCamera._id ? { ...cam, ...updatedCamera } : cam
      )
    );
    setSelectedCamera(updatedCamera);
  };
  const handleFrameUpdate = (cameraId: string, frame: string) => {
    setCameraFrames((prev) => ({
      ...prev,
      [cameraId]: frame,
    }));
  };

  return (
    <div
      className="relative min-h-screen flex flex-col bg-cover bg-center bg-no-repeat"
      style={{ backgroundImage: `url(${BackgroundImg})` }}>
      {/* Overlay to darken background so content stands out */}
      <div className="absolute inset-0  from-black/60 via-black/40 to-black/70" />

      {/* Navbar */}
      <div className="relative z-20">
        <Navbar
          onAddCameraClick={() => setShowCameraModal(true)}
          onManageSupervisorClick={() => setShowSupervisorList(true)}
        />
      </div>

      {/* Main content */}
      <main className="relative z-10 flex-1 px-3 sm:px-6 lg:px-12 py-4 lg:py-6">
        <section className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1.1fr)] gap-4 lg:gap-6 min-h-[78vh] lg:min-h-[82vh]">
          {/* Left column: Camera Details */}
          <div className="relative rounded-2xl bg-white/5 border border-white/10 shadow-xl shadow-black/40 backdrop-blur-md overflow-hidden">
            <div className="absolute inset-x-0 top-0 h-10 from-white/10 to-transparent pointer-events-none" />
            <div className="h-full p-5 sm:p-6 lg:p-7 overflow-hidden">
              {selectedCamera ? (
                <CameraDetailModal
                  camera={selectedCamera}
                  frame={cameraFrames[selectedCamera._id]}
                  onClose={handleCloseModal}
                  onUpdate={(updatedCamera) =>
                    handleUpdateCamera(updatedCamera as ICamera)
                  }
                  onUpdateSupervisorList={handleSupervisorChange}
                />
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-gray-300/80">
                  <div className="mb-2 text-4xl">🎥</div>
                  <p className="text-sm sm:text-base font-medium">
                    Choose a camera from the list to view details.
                  </p>
                  <p className="mt-1 text-xs sm:text-sm text-gray-300/70">
                    Click &quot;Add Camera&quot; to register a new camera.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Right column: Camera List */}
          <div className="rounded-2xl bg-white/5 border border-white/10 shadow-xl shadow-black/40 backdrop-blur-md overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-5 sm:px-6 pt-5 pb-3 border-b border-white/5">
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-white">
                  Cameras list
                </h2>
                <p className="text-[11px] sm:text-xs text-gray-200/80">
                  Click on a camera to edit.
                </p>
              </div>
              <span className="inline-flex items-center gap-1 rounded-full border border-emerald-400/40 bg-emerald-500/10 text-emerald-200 text-xs sm:text-sm font-medium px-3 sm:px-4 py-1.5">
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                {cameras.length} mornitoring cameras
              </span>
            </div>

            <div className="flex-1 overflow-y-auto p-4 sm:p-5">
              <CameraList
                cameras={cameras}
                activeCameraId={activeCameraId}
                selectedCamera={selectedCamera}
                onSelectCamera={handleSelectCamera}
                onDeleteCamera={handleDeleteCamera}
                onAddCamera={() => setShowCameraModal(true)}
                onFrameUpdate={handleFrameUpdate} 
              />

            </div>
          </div>
        </section>
      </main>

      {/* Modal CameraForm */}
      {showCameraModal && (
        <div
          className="fixed inset-0 flex items-center justify-center pt-20 bg-black/60 backdrop-blur-md z-50"
          onClick={() => setShowCameraModal(false)}>
          <div
            className="bg-slate-900/95 rounded-2xl shadow-2xl shadow-black/70 border border-white/10 p-6 sm:p-7 max-w-md w-[90%] sm:w-full relative"
            onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setShowCameraModal(false)}
              className="absolute top-3 right-3 text-gray-300 hover:text-white text-lg font-bold transition">
              ✖
            </button>

            <h3 className=" text-white text-lg font-semibold mb-3">Add new camera</h3>
            <p className="text-xs text-gray-300/80 mb-4">
              Fill full the information
            </p>

            <CameraForm onAdd={handleAddCamera} />
          </div>
        </div>
      )}

      {/* Modal SupervisorList */}
      {showSupervisorList && (
        <div
          className="fixed inset-0 flex items-center justify-center pt-16 bg-black/60 backdrop-blur-md z-50"
          onClick={() => setShowSupervisorList(false)}>
          <div
            className="bg-slate-900/95 text-white rounded-2xl shadow-2xl shadow-black/70 border border-white/10 p-6 sm:p-7 max-w-5xl w-[94%] sm:w-[95%] relative overflow-y-auto max-h-[82vh]"
            onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setShowSupervisorList(false)}
              className="absolute top-3 right-3 text-gray-300 hover:text-white text-lg font-bold transition">
              ✖
            </button>

            <h3 className="text-lg font-semibold mb-3">
              supervisor management
            </h3>
            <p className="text-xs text-gray-300/80 mb-4">
              Add, edit, or remove supervisors who can monitor cameras
            </p>

            {/* SupervisorList quản lý inline form + card thêm */}
            <SupervisorList onChange={handleSupervisorChange} />
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
