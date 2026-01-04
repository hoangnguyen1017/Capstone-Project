import React from "react";
import Hero from "../assets/Hero.svg";

const HeroSection: React.FC = () => {
  return (
    <section className="relative isolate overflow-hidden bg-slate-950 text-white">
      <div className="absolute inset-0">
        <img
          src={Hero}
          alt="Hero background"
          className="h-full w-full object-cover opacity-60"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-slate-950 via-slate-950/80 to-slate-950" />
        <div className="absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-emerald-500/30 via-transparent to-transparent blur-3xl" />
      </div>

      <div className="relative z-10 mx-auto flex max-w-6xl flex-col gap-12 px-6 py-20 sm:py-24 lg:flex-row lg:items-center lg:gap-16 lg:px-12 lg:py-28">
        <div className="space-y-6">
          <p className="text-xs uppercase tracking-[0.5em] text-emerald-300/80">
            ai-powered safety
          </p>
          <h1 className="text-4xl font-bold leading-tight sm:text-5xl lg:text-6xl">
            Fall detection built for real care teams
          </h1>
          

          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <a
              href="/home"
              className="inline-flex items-center justify-center rounded-full bg-emerald-400 px-6 py-3 font-semibold text-slate-900 shadow-lg shadow-emerald-500/40 transition hover:bg-emerald-300">
              Open dashboard
            </a>
            <a
              href="#demo"
              className="inline-flex items-center justify-center rounded-full border border-white/20 px-6 py-3 font-semibold text-white transition hover:border-white hover:bg-white/10">
              Watch demo
            </a>
          </div>

          <div className="flex flex-wrap gap-4 border-t border-white/10 pt-6 text-sm text-gray-300">
            <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
              <p className="font-semibold text-white">Vision + motion fusion</p>
              <p>Bounding boxes, keypoints, and sequence context</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
              <p className="font-semibold text-white">Privacy & audit ready</p>
              <p>Access control, activity logs, anonymized snapshots</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
              <p className="font-semibold text-white">Alert workflows</p>
              <p>Supervisor routing and acknowledgement tracking</p>
            </div>
          </div>
        </div>

        <div className="w-full max-w-md self-center rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur">
          <div className="rounded-2xl bg-black/60 p-5">
            <p className="text-sm uppercase tracking-[0.4em] text-emerald-200">
              system snapshot
            </p>
            <h3 className="mt-4 text-2xl font-semibold">
              Multi-stage verification
            </h3>
            <p className="mt-3 text-gray-300">
            
            </p>
            <div className="mt-6 space-y-4 text-sm font-semibold">
              <div className="rounded-2xl border border-emerald-400/30 bg-emerald-400/10 p-3 text-emerald-100">
                Stage 1 · Visual detection
                <span className="block text-xs text-emerald-100/70">
                  YOLOv11m-Pose finds people, keypoints, and movement cues.
                </span>
              </div>
              <div className="rounded-2xl border border-sky-400/30 bg-sky-400/10 p-3 text-sky-100">
                Stage 2 · Motion reasoning
                <span className="block text-xs text-sky-100/70">
                  ST-GCN checks postural sequences to differentiate falls from non-falls.
                </span>
              </div>
              <div className="rounded-2xl border border-white/15 bg-white/5 p-3 text-gray-100">
                Stage 3 · Alert center
                <span className="block text-xs text-gray-200/80">
                  The dashboard displays events and audible alerts.
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
