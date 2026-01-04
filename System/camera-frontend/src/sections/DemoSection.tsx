import React from "react";
import demoVideo from "../assets/demo.mp4";

const DemoSection: React.FC = () => {
  return (
    <section id="demo" className="py-16 text-white lg:py-24">
      <div className="mx-auto max-w-5xl px-6 lg:px-12">
        <div className="text-center space-y-4">
          <p className="text-xs uppercase tracking-[0.4em] text-emerald-300/80">
            live demo
          </p>
          <h2 className="text-3xl font-bold sm:text-4xl">
            Experience the detection flow
          </h2>
          <p className="text-base text-gray-300">
            Stream visualization, skeleton overlay, and alert pipeline in one
            place.
          </p>
        </div>

        <div className="mt-12 rounded-[32px] border border-white/10 bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900 p-1 shadow-[0_25px_80px_rgba(0,0,0,0.55)]">
          <div className="rounded-[28px] bg-black/80 p-4 sm:p-6">
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.6em] text-gray-400">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              live feed
            </div>
            <div className="mt-4 overflow-hidden rounded-2xl border border-white/5 bg-black">
              <video
                src={demoVideo}
                controls
                className="h-[260px] w-full object-cover sm:h-[360px]">
                Your browser does not support the video tag.
              </video>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DemoSection;
