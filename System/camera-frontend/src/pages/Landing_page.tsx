import { useEffect, useState } from "react";
import Header from "../sections/Header";
import HeroSection from "../sections/HeroSection";
import FeaturesSection from "../sections/FeaturesSection";
import DemoSection from "../sections/DemoSection";
import TeamSection from "../sections/TeamSection";
import Footer from "../sections/Footer";

import BackgroundImg from "../assets/Background.svg";

function LandingPage() {
  const [showScrollTop, setShowScrollTop] = useState(false);

  useEffect(() => {
    const handleScroll = () => setShowScrollTop(window.scrollY > 240);
    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleScrollToTop = () =>
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });

  return (
    <>
      {/* <Header /> */}

      <main className="bg-slate-950 text-white">
        {/* Hero có background RIÊNG */}
        <HeroSection />

        {/* Các section phía dưới dùng background CHUNG */}
        <div
          className="relative bg-cover bg-center bg-no-repeat"
          style={{ backgroundImage: `url(${BackgroundImg})` }}>
          <div className="absolute inset-0 bg-slate-950/85 backdrop-blur-[2px]" />
          <div className="relative">
            <FeaturesSection />
            <DemoSection />
            <TeamSection />
          </div>
        </div>
      </main>

      {showScrollTop && (
        <button
          type="button"
          onClick={handleScrollToTop}
          className="fixed bottom-6 right-6 z-50 inline-flex h-14 w-14 items-center justify-center rounded-full bg-emerald-400 text-slate-900 text-3xl font-bold leading-none shadow-xl shadow-emerald-500/40 transition hover:bg-emerald-300 focus:outline-none focus:ring-2 focus:ring-emerald-200">
          ↑
        </button>
      )}

      <Footer />
    </>
  );
}

export default LandingPage;
