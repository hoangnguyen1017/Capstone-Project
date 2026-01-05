import { useEffect, useState } from "react";
import axios from "axios";
import SupervisorForm from "./SupervisorForm";

interface Supervisor {
  _id: string;
  name: string;
  email: string;
  phone?: string;

  camera_ids?: {
    _id: string;
    camera_name: string;
    location?: string;
  }[];
}


interface SupervisorListProps {
  onChange: () => void; 
}

const SupervisorList: React.FC<SupervisorListProps> = ({ onChange }) => {
  const [supervisors, setSupervisors] = useState<Supervisor[]>([]);
  const [loading, setLoading] = useState(false);
  const [editingSupervisorId, setEditingSupervisorId] = useState<string | null>(
    null
  );
  const [showAddForm, setShowAddForm] = useState(false);

  const BASE_URL = "http://localhost:5000/api/supervisors";

  const fetchSupervisors = async () => {
    setLoading(true);
    try {
      const res = await axios.get(BASE_URL);
      setSupervisors(res.data);
    } catch (err) {
      console.error("Fail to load supervisors list", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSupervisors();
  }, []);

  const handleDelete = async (id: string) => {
    if (!window.confirm("You sure to delete this supervisor")) return;
    try {
      await axios.delete(`${BASE_URL}/${id}`);
      fetchSupervisors();
      onChange();
    } catch (err) {
      console.error("Fail to delete supervisor", err);
      alert("Cannot delete supervisor.");
    }
  };

  if (loading)
    return (
      <div className="flex justify-center items-center py-10">
        <p className="text-gray-500">Loading supervisors...</p>
      </div>
    );

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 auto-rows-fr p-6">
      {supervisors.map((s) => (
        <div
          key={s._id}
          className="bg-white shadow-lg rounded-2xl p-5 flex flex-col justify-between hover:shadow-xl transition">
          {editingSupervisorId === s._id ? (
            <SupervisorForm
              supervisor={{
                _id: s._id,
                name: s.name,
                email: s.email,
                phone: s.phone,
                camera_ids: s.camera_ids?.map((c) => c._id) || [],
              }}
              onAdd={() => {
                setEditingSupervisorId(null);
                fetchSupervisors();
                onChange();
              }}
            />

          ) : (
            <>
              <div className="flex-1 flex flex-col gap-1">
                <h3 className="text-xl font-semibold text-blue-700">
                  {s.name}
                </h3>
                <p className="text-gray-600 text-sm flex items-center gap-1">
                  📧 {s.email}
                </p>
                {s.phone && (
                  <p className="text-gray-500 text-sm flex items-center gap-1">
                    📞 {s.phone}
                  </p>
                )}
                {s.camera_ids && s.camera_ids.length > 0 && (
                  <div className="text-gray-500 text-sm flex flex-col gap-1 mt-2">
                    <span className="flex items-center gap-1">
                      🎥 Cameras:
                    </span>

                    <ul className="list-disc pl-5">
                      {s.camera_ids.map((cam) => (
                        <li key={cam._id}>
                          {cam.camera_name}
                          {cam.location ? ` - ${cam.location}` : ""}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

              </div>

              {/* Compact, modern buttons */}
              <div className="flex justify-end gap-2 mt-4">
                <button
                  onClick={() => setEditingSupervisorId(s._id)}
                  className="bg-yellow-400 hover:bg-yellow-500 text-white px-4 py-2 rounded-xl font-medium text-sm flex items-center justify-center gap-1 transition">
                  ✏️ Edit
                </button>
                <button
                  onClick={() => handleDelete(s._id)}
                  className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-xl font-medium text-sm flex items-center justify-center gap-1 transition">
                  🗑️ Delete
                </button>
              </div>
            </>
          )}
        </div>
      ))}

      {/* Add supervisor card always at the bottom */}
      {!editingSupervisorId && (
        <>
          {showAddForm ? (
            <div className="bg-white shadow-lg rounded-2xl p-5">
              <SupervisorForm
                onAdd={() => {
                  setShowAddForm(false);
                  fetchSupervisors();
                  onChange();
                }}
              />
            </div>
          ) : (
            <div
              onClick={() => setShowAddForm(true)}
              className="flex items-center justify-center bg-gray-100 border-2 border-dashed border-gray-300 rounded-2xl cursor-pointer hover:bg-gray-200 transition">
              <span className="text-5xl text-gray-400 font-bold">+</span>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default SupervisorList;
