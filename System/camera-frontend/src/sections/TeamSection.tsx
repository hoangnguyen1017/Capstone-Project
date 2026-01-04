import React from "react";
import member1 from "../assets/Hoang.png";
import member2 from "../assets/Thinh.jpg";
import member3 from "../assets/Nguyen.jpg";
import member4 from "../assets/Khang.jpg";

const teamMembers = [
  {
    name: "Nguyễn Nhật Hoàng",
    position: "Leader",
    bio: "Project Manager, Pipeline Coordinator, Data Preparation & Preprocessing",
    avatar: member1,
  },
  {
    name: "Nguyễn Đặng Phúc Thịnh",
    position: "Member",
    bio: "YOLOv11m-Pose Fine-tuning, Frontend Development, Data Preparation",
    avatar: member2,
  },
  {
    name: "Đồng Nguyễn Gia Nguyên",
    position: "Member",
    bio: "Three-Stream ST-GCN Architecture, Data Preparation",
    avatar: member3,
  },
  {
    name: "Nguyễn Vĩnh Khang",
    position: "Member",
    bio: "Baseline Model Builder, Backend Engine, Data Preparation",
    avatar: member4,
  },
];


const TeamSection: React.FC = () => {
  return (
    <section className="py-16 text-white lg:py-24">
      <div className="mx-auto max-w-6xl px-6 lg:px-12">
        <div className="text-center space-y-4">
          <p className="text-xs uppercase tracking-[0.4em] text-emerald-300/80">
            team
          </p>
          <h2 className="text-3xl font-bold sm:text-4xl">Meet the makers</h2>
          <p className="text-gray-300">
            A tight-knit engineering group focused on vision, temporal modeling,
            and production reliability.
          </p>
        </div>

        <div className="mt-12 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {teamMembers.map((member, index) => (
            <div
              key={index}
              className="rounded-3xl border border-white/10 bg-white/5 p-6 text-center backdrop-blur transition hover:-translate-y-1 hover:border-emerald-300/40">
              <div className="mx-auto mb-4 h-28 w-28 overflow-hidden rounded-full border border-white/20">
                <img
                  src={member.avatar}
                  alt={member.name}
                  className="h-full w-full object-cover"
                />
              </div>
              <h4 className="text-lg font-semibold">{member.name}</h4>
              <p className="text-sm text-emerald-200">{member.position}</p>
              <p className="mt-2 text-sm text-gray-300">{member.bio}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TeamSection;
