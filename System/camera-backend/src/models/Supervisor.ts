import mongoose, { Schema, Document } from "mongoose";

export interface ISupervisor extends Document {
  name: string;
  email: string;
  phone?: string;
  camera_ids: mongoose.Types.ObjectId[];
}

const SupervisorSchema: Schema = new Schema(
  {
    name: { type: String, required: true },
    email: { type: String, required: true },
    phone: { type: String },

    camera_ids: [
      {
        type: Schema.Types.ObjectId,
        ref: "Camera",
        required: true,
      },
    ],
  },
  { timestamps: true }
);

export default mongoose.model<ISupervisor>(
  "Supervisor",
  SupervisorSchema
);
