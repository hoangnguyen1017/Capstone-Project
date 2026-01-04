import React from "react";

const features = [
  {
    title: "YOLOv11m-Pose detection",
    description:
      "Performs real-time human detection, multi-object tracking, and precise keypoint extraction for downstream action-recognition tasks.",
    accent: "⚡",
  },
  {
    title: "Skeleton-based action analysis",
    description:
      "The ST-GCN model leverages a three-stream architecture—joint, bone, and motion representations—to capture rich spatial–temporal dynamics. By integrating these complementary skeletal features, the system achieves robust discrimination between fall and non-fall actions.",
    accent: "🧠",
  },
  {
    title: "Multi-stage verification",
    description:
      "Spatial features are fused with temporal motion patterns to generate a reinforced, two-tier confidence score—ensuring only high-certainty fall events trigger alerts.",
    accent: "🛡️",
  },
  {
    title: "Alert & monitoring hub",
    description:
      "When a fall is detected, the system immediately triggers an audible alert and highlights the subject with a red bounding box on the live video stream, enabling caregivers or supervisors to quickly recognize and respond to the incident.",
    accent: "📟",
  },
];

const FeaturesSection: React.FC = () => {
  return (
    <section id="features" className="relative py-16 text-white lg:py-24">
      <div className="absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-white/10 via-transparent to-transparent blur-3xl" />
      <div className="relative mx-auto flex max-w-6xl flex-col gap-12 px-6 lg:px-12">
        <div className="text-center space-y-4">
          <p className="text-xs uppercase tracking-[0.4em] text-emerald-300/80">
            capabilities
          </p>
          <h2 className="text-3xl font-bold sm:text-4xl">
            Built for mission-critical care centers
          </h2>
          <p className="text-base text-gray-300 sm:text-lg">
            End-to-end tooling that spans perception, verification, and alerting
            so operators can focus on assisting residents faster.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="group relative rounded-3xl border border-white/10 bg-white/5 p-6 backdrop-blur-lg transition hover:-translate-y-1 hover:border-emerald-300/40">
              <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-emerald-400/10 text-2xl">
                {feature.accent}
              </div>
              <h4 className="text-lg font-semibold">{feature.title}</h4>
              <p className="mt-3 text-sm text-gray-300">{feature.description}</p>
              <div className="mt-5 h-px w-full bg-gradient-to-r from-transparent via-white/20 to-transparent" />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;
