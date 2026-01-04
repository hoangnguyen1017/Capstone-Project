import React from "react";
import CameraDetector from "./CameraDetector";
import { Plus, Trash2 } from "lucide-react";

export interface ICamera {
  _id: string;
  camera_name: string;
  location: string;
  responsible_name: string;
  alert_email: string;
  alert_phone: string;
  is_active: boolean;
  video_stream_url?: string;
}

interface CameraListProps {
  cameras: ICamera[];
  activeCameraId: string | null;
  selectedCamera: ICamera | null;
  onSelectCamera: (camera: ICamera) => void;
  onDeleteCamera: (id: string) => void;
  onAddCamera: () => void;
  onFrameUpdate: (cameraId: string, frame: string) => void;
}

const CameraItem: React.FC<{
  cam?: ICamera;
  onSelect?: (camera: ICamera) => void;
  isActive?: boolean;
  onDelete?: (id: string) => void;
  onAddCamera?: () => void;
  onFrameUpdate?: (cameraId: string, frame: string) => void;
}> = ({ cam, onSelect, onDelete, onAddCamera,onFrameUpdate, }) => {
  if (!cam) {
    return (
      <div className="border-2 border-dashed border-gray-300 rounded-xl shadow-sm hover:shadow-md transition-all flex items-center justify-center px-15 py-10 hover:border-green-400 group">
        <button
          onClick={onAddCamera}
          className="flex flex-col items-center justify-center text-gray-500 group-hover:text-green-600 transition-transform duration-300 hover:scale-105">
          <div className="w-16 h-16 rounded-full bg-white shadow-md flex items-center justify-center border border-gray-300 group-hover:border-green-500 group-hover:bg-green-50 transition-colors">
            <Plus className="w-8 h-8" />
          </div>
          <p className="mt-3 text-sm font-semibold">Add Camera</p>
        </button>
      </div>
    );
  }

  const isLocalStream = cam.video_stream_url?.startsWith("ws://");

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (
      onDelete &&
      window.confirm(`Are you sure you want to delete camera "${cam.camera_name}"?`)
    ) {
      onDelete(cam._id);
    }
  };

  return (
    <div className="relative border border-gray-200 rounded-2xl bg-white shadow-sm hover:shadow-xl transition-all duration-300 overflow-hidden group flex flex-col h-80">
      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
      <button
        onClick={handleDelete}
        className="absolute top-2 right-2 text-white bg-red-600 hover:bg-red-700 w-7 h-7 flex items-center justify-center rounded-full text-xs font-bold shadow-md opacity-0 group-hover:opacity-100 transition-opacity"
        title="Delete camera">
        <Trash2 size={14} />
      </button>

      <div className="px-3 pt-3">
        <h3 className="font-semibold text-gray-800 truncate text-base">
          {cam.camera_name}
        </h3>
        <p className="text-xs text-gray-500 flex items-center gap-1">
          📍 {cam.location}
        </p>
      </div>

      <div
        className="flex-1 bg-gray-100 rounded-md m-3 flex items-center justify-center overflow-hidden relative cursor-pointer"
        onClick={() => cam && onSelect?.(cam)} 
      >
        {isLocalStream && cam.video_stream_url ? (
          <CameraDetector
            streamUrl={cam.video_stream_url}
            onFrame={(frame: string) => {
              if (!onFrameUpdate || !cam) return;
              onFrameUpdate(cam._id, frame); 
            }}
          />
        ) : (
          <p className="text-gray-400 text-sm">No Live Stream</p>
        )}
      </div>

      <div className="px-3 pb-3 flex justify-between items-center">
        <p className="text-sm text-gray-600 truncate">
          👤 {cam.responsible_name}
        </p>
      </div>
    </div>
  );
};

const CameraList: React.FC<CameraListProps> = ({
  cameras,
  activeCameraId,
  selectedCamera,
  onSelectCamera,
  onDeleteCamera,
  onAddCamera,
  onFrameUpdate,
}) => {
  const displayCameras = [...cameras, undefined];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-6 p-6 rounded-xl">
      {displayCameras.map((cam, idx) => (
        <CameraItem
          key={cam?._id || `slot-${idx}`}
          cam={cam}
          onSelect={onSelectCamera}
          isActive={cam ? activeCameraId === cam._id && !selectedCamera : false}
          onDelete={onDeleteCamera}
          onAddCamera={onAddCamera}
          onFrameUpdate={onFrameUpdate}
        />
      ))}
    </div>
  );
};

export default CameraList;
