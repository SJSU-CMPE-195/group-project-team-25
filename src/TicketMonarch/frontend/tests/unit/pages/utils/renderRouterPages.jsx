import { render } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import Home from "@/pages/Home";
import SeatSelection from "@/pages/SeatSelection";
import Checkout from "@/pages/Checkout";
import Confirmation from "@/pages/Confirmation";

export function renderRouterPages({
  initialEntries = ["/"],
} = {}) {
  return render(
    <MemoryRouter 
      initialEntries={initialEntries}
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true,
      }}
    >
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/seats/:concertId" element={<SeatSelection />} />
        <Route path="/checkout" element={<Checkout />} />
        <Route path="/confirmation" element={<Confirmation />} />
      </Routes>
    </MemoryRouter>
  );
}