import React from "react";
import Container from "../components/Container";
import { useNavigate } from "react-router-dom";
const Header: React.FC = () => {
  const navigate = useNavigate();
  return (
    <header className="fixed top-0 left-0 w-full z-50 bg-neutral-1000 shadow-sm">
      <Container>
        <div className="flex justify-between items-center">
          {/* Logo */}
          <div className="text-xl font-bold text-white">AIP491_EYKH-AI</div>

          {/* Call-to-action */}
          <button
            onClick={() => navigate("/home")}
            className="px-4 py-2 text-white rounded-lg">
            Get Started
          </button>
        </div>
      </Container>
    </header>
  );
};

export default Header;
