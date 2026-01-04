import mongoose, { Schema, Document } from "mongoose";

export interface ICamera extends Document {
    camera_name: string;
    video_stream_url?: string; 
    location?: string;
    responsible_name?: string;  
    responsible_email?: string; 
    responsible_phone?: string; 
    is_active: boolean;
}

const CameraSchema: Schema = new Schema({
    camera_name: { type: String, required: true },
    video_stream_url: { type: String },
    location: { type: String },
    responsible_name: { type: String },
    responsible_email: { type: String },  
    responsible_phone: { type: String },  
    is_active: { type: Boolean, default: true },
});

export default mongoose.model<ICamera>("Camera", CameraSchema);
