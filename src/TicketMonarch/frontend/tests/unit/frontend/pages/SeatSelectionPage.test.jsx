import { render, screen, within, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { vi, describe, test, expect } from "vitest";
import SeatSelection from "@/pages/SeatSelection";
import Home from "@/pages/Home";
import Checkout from "@/pages/Checkout";
import { concertsById } from "@/assets/concerts";

vi.mock("@/assets/concerts", () => ({
  concertsById: {
    1: {
      id: 1,
      name: "Chappell Roan",
      date: "Feb 14, 2026",
      eventName: "Midwest Princess Tour",
      venue: "Aragon Ballroom",
      city: "Chicago, IL",
      location: "Chicago, IL • Aragon Ballroom",
      image: "/chappellImg.jpg",
      price: 100,
    },
  },
  concerts: [
    {
      id: 1,
      name: 'Chappell Roan',
      date: 'Feb 14, 2026',
      eventName: 'Midwest Princess Tour',
      venue: 'Aragon Ballroom',
      city: 'Chicago, IL',
      location: 'Chicago, IL • Aragon Ballroom',
      image: "/chappellImg.jpg",
      price: 100,
    },
  ],
}));

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/seat-selection/1"]}>
      <Routes>
        <Route path="/seat-selection/:concertId" element={<SeatSelection />} />
        <Route path="/" element={<Home />} />
        <Route path="/checkout" element={<Checkout />} />
      </Routes>
    </MemoryRouter>
  );
}

describe("SeatSelection", () => {
  test("render page and check concert info", async () => {
    renderPage();

    expect(screen.getByText(/Ticket Monarch/i)).toBeInTheDocument();
    expect(screen.getByText(/← Back/i)).toBeInTheDocument();
    expect(screen.getByText(concertsById[1].name)).toBeInTheDocument();

    const eventInfoBar = screen.getByText(concertsById[1].name).closest(".ss-event-bar");
    expect(within(eventInfoBar).getByText(/\$60/i)).toBeInTheDocument();
    expect(screen.getByText(new RegExp(concertsById[1].date))).toBeInTheDocument();
    expect(screen.getByText(new RegExp(`${concertsById[1].venue}.*${concertsById[1].city}`))).toBeInTheDocument();
    expect(screen.getByText(new RegExp(concertsById[1].eventName))).toBeInTheDocument();
  })

  test("test back button", async () => {
    renderPage()
    const user = userEvent.setup()
    
    const backBtn = screen.getByRole("button", { name: "← Back"})
    await user.click(backBtn)
    expect(screen.getByText(/Ticket Monarch/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/search artists/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/sort/i)).toBeInTheDocument();
  })

  test("check section prices", async () => {
    renderPage();

    const section = screen.getByText("GA1").closest(".ss-section-cell");
    expect(within(section).getByText("$170")).toBeInTheDocument();

    const s105 = screen.getByText("105").closest(".ss-section-cell");
    expect(within(s105).getByText("$145")).toBeInTheDocument();

    const s203 = screen.getByText("203").closest(".ss-section-cell");
    expect(within(s203).getByText("$100")).toBeInTheDocument();

    const s301 = screen.getByText("301").closest(".ss-section-cell");
    expect(within(s301).getByText("$60")).toBeInTheDocument();
  })

  test("test seat section ticket counter", async () => {
    renderPage()

    const user = userEvent.setup()

    const section = screen.getByText("GA1").closest(".ss-section-cell")
    const plusBtn = within(section).getByRole("button", { name: "+"})
    const minusBtn = within(section).getByRole("button", { name: "−"})

    expect(within(section).getByText("0")).toBeInTheDocument()

    expect(minusBtn).toBeDisabled();
    await user.click(minusBtn)
    expect(within(section).getByText("0")).toBeInTheDocument()

    await user.click(plusBtn)
    expect(within(section).getByText("1")).toBeInTheDocument()
    await user.click(minusBtn)
    expect(within(section).getByText("0")).toBeInTheDocument()

    await user.dblClick(plusBtn)
    await user.dblClick(plusBtn)
    await user.dblClick(plusBtn)
    await user.dblClick(plusBtn)
    expect(plusBtn).toBeDisabled();
    expect(within(section).getByText("8")).toBeInTheDocument()
  })

  test("test checkout button", async () => {
    renderPage()
    const user = userEvent.setup()

    const section_g = screen.getByText("GA1").closest(".ss-section-cell")
    const plusBtn_g = within(section_g).getByRole("button", { name: "+"})
    await user.click(plusBtn_g)
    
    const checkoutBtn = screen.getByRole("button", { name: "Checkout — $170"})
    await user.click(checkoutBtn)
    expect(screen.getByText(/Payment Details/i)).toBeInTheDocument();
    expect(screen.getByText(/Purchase Details/i)).toBeInTheDocument();
  })

  test("test order summary add and subtracts", async () => {
    renderPage()

    const user = userEvent.setup()

    const section_g = screen.getByText("GA1").closest(".ss-section-cell")
    const plusBtn_g = within(section_g).getByRole("button", { name: "+"})
    const minusBtn_g = within(section_g).getByRole("button", { name: "−"})

    await user.dblClick(plusBtn_g)

    const checkout = document.querySelector(".ss-checkout-btn")

    const checkout_g = screen.getByText("Section GA1").closest(".ss-order-item")
    const plusBtnCheckout_g = within(checkout_g).getByRole("button", { name: "+"})
    const minusBtnCheckout_g = within(checkout_g).getByRole("button", { name: "−"})

    expect(within(checkout_g).getByText("$340")).toBeInTheDocument()
    expect(within(checkout).getByText("Checkout — $340")).toBeInTheDocument();

    await user.click(plusBtnCheckout_g)
    expect(within(checkout_g).getByText("$510")).toBeInTheDocument()
    expect(within(checkout).getByText("Checkout — $510")).toBeInTheDocument();

    await user.click(minusBtnCheckout_g)
    expect(within(checkout_g).getByText("$340")).toBeInTheDocument()
    expect(within(checkout).getByText("Checkout — $340")).toBeInTheDocument();

    await user.click(minusBtn_g)
    expect(within(checkout_g).getByText("$170")).toBeInTheDocument()
    expect(within(checkout).getByText("Checkout — $170")).toBeInTheDocument();
  })

  test("test order summary button disables", async () => {
    renderPage()

    const user = userEvent.setup()

    const section_g = screen.getByText("GA1").closest(".ss-section-cell")
    const plusBtn_g = within(section_g).getByRole("button", { name: "+"})
    const minusBtn_g = within(section_g).getByRole("button", { name: "−"})

    await user.dblClick(plusBtn_g)

    const checkout = document.querySelector(".ss-checkout-btn")

    await user.dblClick(minusBtn_g)

    await vi.waitFor(() => {
      expect(
        within(checkout).queryByRole(".ss-order-item", { name: "Section GA1" })
      ).not.toBeInTheDocument()
    })
  })

  test("test order summary multiple sections", async () => {
    renderPage()

    const user = userEvent.setup()

    const section_g = screen.getByText("GA1").closest(".ss-section-cell")
    const plusBtn_g = within(section_g).getByRole("button", { name: "+"})

    const section_200 = screen.getByText("205").closest(".ss-section-cell")
    const plusBtn_200 = within(section_200).getByRole("button", { name: "+"})

    await user.dblClick(plusBtn_g)
    await user.dblClick(plusBtn_200)
    await user.click(plusBtn_200)

    const checkout = document.querySelector(".ss-checkout-btn")

    const checkout_g = screen.getByText("Section GA1").closest(".ss-order-item")
    const checkout_200 = screen.getByText("Section 205").closest(".ss-order-item")

    expect(within(checkout_g).getByText("$340")).toBeInTheDocument()
    expect(within(checkout_200).getByText("$330")).toBeInTheDocument()
    expect(within(checkout).getByText("Checkout — $670")).toBeInTheDocument();
  })
})