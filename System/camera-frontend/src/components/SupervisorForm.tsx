import { useState, useEffect } from "react";
import axios from "axios";

interface SupervisorFormProps {
  onAdd: () => void; 
  supervisor?: {
    _id: string;
    name: string;
    email: string;
    phone?: string;
    camera_ids?: string[];

  };
}

const SupervisorForm: React.FC<SupervisorFormProps> = ({
  onAdd,
  supervisor,
}) => {
  const [name, setName] = useState(supervisor?.name || "");
  const [email, setEmail] = useState(supervisor?.email || "");
  const [phone, setPhone] = useState(supervisor?.phone || "");
  const [cameraIds, setCameraIds] = useState<string[]>(supervisor?.camera_ids || []);
  const [cameras, setCameras] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    axios
      .get("http://localhost:5000/api/cameras")
      .then((res) => setCameras(res.data))
      .catch((err) => console.error("Error fetching camera list:", err));
  }, []);
  const toggleCamera = (id: string) => {
    setCameraIds((prev) =>
      prev.includes(id)
        ? prev.filter((c) => c !== id)
        : [...prev, id]
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !email || !cameraIds)
      return alert("Please fill in all information");

    setIsLoading(true);
    try {
      if (supervisor?._id) {
        await axios.put(
          `http://localhost:5000/api/supervisors/${supervisor._id}`,
          {
            name,
            email,
            phone,
            camera_ids: cameraIds,
          }
        );
      } else {
        await axios.post("http://localhost:5000/api/supervisors", {
          name,
          email,
          phone,
          camera_ids: cameraIds,
        });
      }

      onAdd();
      if (!supervisor?._id) {
        setName("");
        setEmail("");
        setPhone("");
        setCameraIds([]);
      }
    } catch (err) {
      console.error("Error submitting supervisor:", err);
      alert("Operation failed.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <input
        type="text"
        placeholder="Supervisor Name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        className="w-full border border-gray-300 rounded-xl p-3 focus:ring-2 focus:ring-blue-400 outline-none text-gray-700 placeholder-gray-400 transition"
        required
      />

      <input
        type="email"
        placeholder="Email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        className="w-full border border-gray-300 rounded-xl p-3 focus:ring-2 focus:ring-blue-400 outline-none text-gray-700 placeholder-gray-400 transition"
        required
      />

      <input
        type="tel"
        placeholder="Phone Number"
        value={phone}
        onChange={(e) => setPhone(e.target.value)}
        className="w-full border border-gray-300 rounded-xl p-3 focus:ring-2 focus:ring-blue-400 outline-none text-gray-700 placeholder-gray-400 transition"
      />

      <div className="border border-gray-300 rounded-xl p-3">
        <p className="font-semibold mb-2 text-gray-700">
          Choose Cameras
        </p>

        <div className="flex flex-col gap-2 max-h-60 overflow-y-auto">
          {cameras.map((c) => (
            <label
              key={c._id}
              className="flex items-center gap-2 cursor-pointer"
            >
              <input
                type="checkbox"
                checked={cameraIds.includes(c._id)}
                onChange={() => toggleCamera(c._id)}
                className="accent-blue-600"
              />
              <span className="text-gray-700">
                {c.camera_name}
                {c.location ? ` - ${c.location}` : ""}
              </span>
            </label>
          ))}
        </div>
      </div>


      <button
        type="submit"
        disabled={isLoading}
        className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-xl font-semibold text-lg transition disabled:opacity-50">
        {isLoading ? "Saving..." : supervisor?._id ? "Update Supervisor" : "Add Supervisor"}
      </button>
    </form>
  );
};

export default SupervisorForm;
