import React, { useEffect, useState } from "react";
import {
  Mail,
  MapPin,
  Phone,
  User,
  Activity,
  X,
  Edit2,
  Check,
  XCircle,
  Search,
} from "lucide-react";
import axios from "axios";
interface ICamera {
  _id: string;
  camera_name: string;
  location?: string;
  responsible_name?: string;
  responsible_email?: string;
  responsible_phone?: string;
  is_active: boolean;
  video_stream_url?: string;
  responsible_id?: string; 

interface ISupervisor {
  _id: string;
  name: string;
  email: string;
  phone?: string;
}

interface CameraDetailModalProps {
  camera: ICamera;
  frame?: string;
  onClose: () => void;
  onUpdate?: (updatedCamera: ICamera) => void;
  onUpdateSupervisorList: () => void;
}

const CameraDetailModal: React.FC<CameraDetailModalProps> = ({
  camera,
  frame,
  onClose,
  onUpdate,
  onUpdateSupervisorList,
}) => {
  const isLocalStream = camera.video_stream_url?.startsWith("ws://");

  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [supervisors, setSupervisors] = useState<ISupervisor[]>([]);
  const [selectedSupervisorId, setSelectedSupervisorId] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState("");
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const [formData, setFormData] = useState({
    camera_name: camera.camera_name,
    location: camera.location || "",
    responsible_name: camera.responsible_name || "",
    responsible_email: camera.responsible_email || "",
    responsible_phone: camera.responsible_phone || "",
    is_active: camera.is_active,
  });

  useEffect(() => {
    setFormData({
      camera_name: camera.camera_name,
      location: camera.location || "",
      responsible_name: camera.responsible_name || "",
      responsible_email: camera.responsible_email || "",
      responsible_phone: camera.responsible_phone || "",
      is_active: camera.is_active,
    });
    setSelectedSupervisorId("");
    setSearchQuery("");
    setIsEditing(false);
    setDropdownOpen(false);
  }, [camera]);

  useEffect(() => {
    if (isEditing) {
      axios
        .get<ISupervisor[]>("http://localhost:5000/api/supervisors")
        .then((res) => {
          setSupervisors(res.data);
          const found = res.data.find(
            (s) => s.email === camera.responsible_email
          );
          if (found) setSelectedSupervisorId(found._id);
        })
        .catch((err) => console.error("Error fetching supervisors:", err));
    }
  }, [isEditing, camera.responsible_email]);

  const handleSupervisorSelectFromList = (sup: ISupervisor) => {
    setSelectedSupervisorId(sup._id);
    setFormData((prev) => ({
      ...prev,
      responsible_name: sup.name,
      responsible_email: sup.email,
      responsible_phone: sup.phone || "",
    }));
    setDropdownOpen(false);
    setSearchQuery("");
  };

  const filteredSupervisors = supervisors.filter((s) =>
    s.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const target = e.target;
    const name = target.name;

    if (target instanceof HTMLInputElement && target.type === "checkbox") {
      setFormData((prev) => ({ ...prev, [name]: target.checked }));
    } else {
      setFormData((prev) => ({ ...prev, [name]: target.value }));
    }
  };

  const handleSave = async () => {
    setLoading(true);
    try {
      const res = await axios.put(
        `http://localhost:5000/api/cameras/${camera._id}`,
        formData
      );
      const updatedCamera = res.data.camera;

      setLoading(false);
      setIsEditing(false);

      if (onUpdate) onUpdate(updatedCamera);
      onUpdateSupervisorList();
    } catch (error: any) {
      console.error(error.response?.data || error.message);
      setLoading(false);
      alert("Update failed");
    }
  };

  const handleCancel = () => {
    setFormData({
      camera_name: camera.camera_name,
      location: camera.location || "",
      responsible_name: camera.responsible_name || "",
      responsible_email: camera.responsible_email || "",
      responsible_phone: camera.responsible_phone || "",
      is_active: camera.is_active,
    });

    const found = supervisors.find((s) => s.email === camera.responsible_email);
    setSelectedSupervisorId(found?._id || "");
    setSearchQuery("");
    setDropdownOpen(false);
    setIsEditing(false);
  };

  return (
    <div className="flex flex-col h-full from-white via-gray-50 to-gray-100 rounded-2xl shadow-2xl p-6 border border-gray-200 backdrop-blur-sm">
      {/* Header */}
      <div className="flex justify-between items-center border-b pb-3 mb-4">
        <h2 className="text-2xl font-bold from-indigo-600 to-purple-600 text-transparent bg-clip-text tracking-tight">
          Camera Details — {camera.camera_name || "N/A"}
        </h2>
        <div className="flex gap-2">
          {!isEditing ? (
            <button
              onClick={() => setIsEditing(true)}
              className="flex items-center gap-1 text-blue-600 hover:text-blue-800 font-medium transition">
              <Edit2 className="w-5 h-5" /> Edit
            </button>
          ) : (
            <div className="flex gap-2">
              <button
                onClick={handleSave}
                disabled={loading}
                className="flex items-center gap-1 bg-green-600 text-white px-4 py-2 rounded-xl hover:bg-green-700 transition font-medium shadow-md">
                <Check className="w-4 h-4" /> Save
              </button>
              <button
                onClick={handleCancel}
                className="flex items-center gap-1 bg-red-600 text-white px-4 py-2 rounded-xl hover:bg-red-700 transition font-medium shadow-md">
                <XCircle className="w-4 h-4" /> Cancel
              </button>
            </div>
          )}
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-red-500 transition-all hover:rotate-90 duration-300">
            <X className="w-7 h-7" />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex flex-col w-full h-screen bg-transparent gap-6 overflow-y-auto">
        {/* Video */}
        {/* Video (Canvas instead of CameraDetector) */}
        <div className="flex-1 rounded-2xl flex items-center justify-center overflow-hidden shadow-lg relative w-full h-full bg-black">
          {frame ? (
            <img
              src={frame}
              alt="camera-frame"
              className="
                w-full h-full
                object-contain
                transition-transform duration-200
                hover:scale-125
              "
              style={{
                imageRendering: "auto",   
              }}
            />
          ) : (
            <p className="text-gray-400 text-sm">No frame available</p>
          )}
        </div>
        {/* Detailed information */}
        <div className="bg-white/90 rounded-2xl p-6 shadow-lg border border-gray-100 backdrop-blur-md transition hover:shadow-xl">
          {isEditing ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <InputRow
                label="Camera Name"
                name="camera_name"
                value={formData.camera_name}
                onChange={handleChange}
              />
              <InputRow
                label="Location"
                name="location"
                value={formData.location}
                onChange={handleChange}
              />

              {/* Supervisor search card */}
              <div className="flex flex-col gap-1 relative">
                <label className="text-gray-800 font-semibold text-sm flex items-center gap-1">
                  Select Responsible Person{" "}
                  <Search className="w-4 h-4 text-gray-400" />
                </label>
                <input
                  type="text"
                  placeholder="Search responsible person..."
                  value={
                    searchQuery
                      ? searchQuery
                      : supervisors.find((s) => s._id === selectedSupervisorId)
                          ?.name || ""
                  }
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onFocus={() => setDropdownOpen(true)}
                  className="border border-gray-300 rounded-xl px-4 py-2 text-sm bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-400 shadow-sm transition"
                />
                {dropdownOpen && filteredSupervisors.length > 0 && (
                  <div className="absolute z-10 mt-1 w-full bg-white border border-gray-200 rounded-xl shadow-lg max-h-52 overflow-y-auto">
                    {filteredSupervisors.map((sup) => (
                      <div
                        key={sup._id}
                        className="p-2 hover:bg-indigo-50 cursor-pointer rounded-xl"
                        onClick={() => handleSupervisorSelectFromList(sup)}>
                        <p className="font-medium">{sup.name}</p>
                        <p className="text-xs text-gray-500">
                          {sup.email} {sup.phone && `— ${sup.phone}`}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <InputRow
                label="Responsible"
                name="responsible_name"
                value={formData.responsible_name}
                onChange={handleChange}
              />
              <InputRow
                label="Email"
                name="responsible_email"
                value={formData.responsible_email}
                onChange={handleChange}
              />
              <InputRow
                label="Phone"
                name="responsible_phone"
                value={formData.responsible_phone}
                onChange={handleChange}
              />

              <div className="flex flex-col gap-1">
                <label className="text-gray-800 font-semibold text-sm">
                  Status
                </label>
                <select
                  name="is_active"
                  value={formData.is_active ? "true" : "false"}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      is_active: e.target.value === "true",
                    }))
                  }
                  className="border border-gray-300 rounded-xl px-3 py-2 text-sm bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-400 shadow-sm transition">
                  <option value="true">Active ✅</option>
                  <option value="false">Non-active ❌</option>
                </select>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <DetailRow
                icon={<MapPin />}
                label="Location"
                value={formData.location || "N/A"}
              />
              <DetailRow
                icon={<Activity />}
                label="Status"
                value={formData.is_active ? "Active ✅" : "Inactive ❌"}
              />
              <DetailRow
                icon={<User />}
                label="Responsible"
                value={formData.responsible_name || "N/A"}
              />
              <DetailRow
                icon={<Mail />}
                label="Email"
                value={formData.responsible_email || "N/A"}
              />
              <DetailRow
                icon={<Phone />}
                label="Phone"
                value={formData.responsible_phone || "N/A"}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CameraDetailModal;

interface DetailRowProps {
  icon?: React.ReactNode;
  label: string;
  value: string;
}

const DetailRow: React.FC<DetailRowProps> = ({ icon, label, value }) => (
  <div className="flex items-center gap-3 py-2 text-sm">
    {icon && <span className="text-indigo-500">{icon}</span>}
    <span className="font-bold text-gray-700">{label}:</span>
    <span className="text-gray-900">{value}</span>
  </div>
);

interface InputRowProps {
  label: string;
  name: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

const InputRow: React.FC<InputRowProps> = ({
  label,
  name,
  value,
  onChange,
}) => (
  <div className="flex flex-col gap-1">
    <label className="text-gray-800 font-bold text-sm">{label}</label>
    <input
      type="text"
      name={name}
      value={value}
      onChange={onChange}
      className="
        border border-gray-300 rounded-xl px-4 py-2 text-sm
        bg-gray-50
        focus:outline-none focus:ring-2 focus:ring-indigo-400
        focus:border-indigo-400
        shadow-sm
        transition-all
        placeholder-gray-400
      "
      placeholder={`Enter ${label.toLowerCase()}`}
    />
  </div>
);
