import { beforeEach, vi, describe, test, expect } from "vitest";
import { screen, within, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { concertsById } from "@/assets/concerts";
import { renderRouterPages } from "./utils/renderRouterPages";

const {
  mockSubmitCheckout,
  mockRollingEvaluate,
  mockEvaluateSession,
  mockGetFlag,
  mockConfirmHumanSession,
  mockForceFlush,
  mockGetSessionId,
  mockResetSession,
} = vi.hoisted(() => ({
  mockSubmitCheckout: vi.fn(),
  mockRollingEvaluate: vi.fn(),
  mockEvaluateSession: vi.fn(),
  mockGetFlag: vi.fn(),
  mockConfirmHumanSession: vi.fn(),
  mockForceFlush: vi.fn(),
  mockGetSessionId: vi.fn(),
  mockResetSession: vi.fn(),
}));

vi.mock("@/services/api", () => ({
  submitCheckout: mockSubmitCheckout,
  rollingEvaluate: mockRollingEvaluate,
  evaluateSession: mockEvaluateSession,
  getFlag: mockGetFlag,
  confirmHumanSession: mockConfirmHumanSession,
}));

vi.mock("@/services/tracking", () => ({
  forceFlush: mockForceFlush,
  getSessionId: mockGetSessionId,
  resetSession: mockResetSession,
}));

vi.mock("@/components/ChallengeModal", () => ({
  default: ({ difficulty, onComplete }) => (
    <div>
      <p>Challenge Modal</p>
      <p>Difficulty: {difficulty}</p>
      <button onClick={() => onComplete(true)}>Pass Challenge</button>
      <button onClick={() => onComplete(false)}>Fail Challenge</button>
    </div>
  ),
}));

vi.mock("@/assets/concerts", async () => {
  const mod = await import("./fixtures/concerts");
  return {
    concerts: mod.mockConcerts,
    concertsById: mod.mockConcertsById,
  };
});

const mockBookingSelection = {
  concert: {
    id: 1,
    name: "Chappell Roan",
    date: "Feb 14, 2026",
    eventName: "Midwest Princess Tour",
    venue: "Aragon Ballroom",
    city: "Chicago, IL",
    location: "Chicago, IL • Aragon Ballroom",
    image: "/src/assets/images/chappell.jpg",
    price: 100
  },
  seats: [
    {
      id: "GA1-1",
      section: "GA1",
      row: "19",
      seat: "12",
      price: 170
    }
  ],
  selectedSection: "GA1",
  sectionPrice: 170,
  total: 170
};

let consoleLogSpy;

function renderPage() {
  return renderRouterPages({ initialEntries: ["/checkout"] })
}

async function fillValidCheckoutForm(user) {
  await user.type(screen.getByLabelText(/card number/i), "4242424242424242");
  await user.type(screen.getByLabelText(/mm\/yy/i), "1234");
  await user.type(screen.getByLabelText(/cvc/i), "123");
  await user.type(screen.getByLabelText(/name on card/i), "John Doe");
  await user.type(screen.getByLabelText(/^address/i), "123 Main Street");
  await user.type(screen.getByLabelText(/city/i), "Chicago");
  await user.type(screen.getByLabelText(/zip code/i), "60601");
  await user.selectOptions(screen.getByLabelText(/state/i), "Illinois");
}

describe("Checkout", () => {
  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, "log").mockImplementation(() => {});

    localStorage.clear();

    localStorage.setItem(
      "bookingSelection",
      JSON.stringify(mockBookingSelection)
    );

    mockSubmitCheckout.mockReset();
    mockRollingEvaluate.mockReset();
    mockEvaluateSession.mockReset();
    mockGetFlag.mockReset();
    mockForceFlush.mockReset();
    mockGetSessionId.mockReset();
    mockResetSession.mockReset();

    mockSubmitCheckout.mockResolvedValue({ success: true });
    mockRollingEvaluate.mockResolvedValue({
      success: true,
      deploy_honeypot: false,
      honeypot_triggered: false,
      events_processed: 0,
    });
    mockEvaluateSession.mockResolvedValue({
      success: true,
      decision: "allow",
      events_processed: 3,
      honeypot_triggered: false,
    });
    mockGetFlag.mockResolvedValue({
      success: true,
      data: { flag: "inactive" },
    });
    mockForceFlush.mockResolvedValue();
    mockGetSessionId.mockReturnValue("test-session-id");
  });

  afterEach(() => {
    consoleLogSpy.mockRestore();
  });

/* =========================
  Render and navigation
  ========================= */
  describe("render and navigation", () => {
    test("render page", async () => {
      renderPage();

      expect(screen.getByText(/Ticket Monarch/i)).toBeInTheDocument();
      expect(screen.getByText(/Payment Details/i)).toBeInTheDocument();
      expect(screen.getByText(/Purchase Details/i)).toBeInTheDocument();
    })

    test("render concert info", async () => {
      renderPage();

      expect(screen.getByText(/Chappell Roan/i)).toBeInTheDocument();

      const table = await screen.findByRole("table");

      const rows = within(table).getAllByRole("row");

      const sectionRow = rows[1];
      const cells = within(sectionRow).getAllByRole("cell");

      expect(cells[0]).toHaveTextContent("Section GA1");
      expect(cells[1]).toHaveTextContent("$170.00");
      expect(cells[2]).toHaveTextContent("1");
      expect(cells[3]).toHaveTextContent("$170.00");

      const totalRow = rows[rows.length - 1];
      expect(totalRow).toHaveTextContent("Total");
      expect(totalRow).toHaveTextContent("$170.00");
    })

    test("test back button", async () => {
      renderPage()
      const user = userEvent.setup()
      
      const backBtn = screen.getByRole("button", { name: "← Back"})
      await user.click(backBtn)

      expect(screen.getByText(/Ticket Monarch/i)).toBeInTheDocument();
      expect(screen.getByText(/← Back/i)).toBeInTheDocument();
      expect(screen.getByText(concertsById[1].name)).toBeInTheDocument();

      const eventInfoBar = screen.getByText(concertsById[1].name).closest(".ss-event-bar");
      expect(within(eventInfoBar).getByText(/\$60/i)).toBeInTheDocument();
      expect(screen.getByText(new RegExp(concertsById[1].date))).toBeInTheDocument();
      expect(screen.getByText(new RegExp(`${concertsById[1].venue}.*${concertsById[1].city}`))).toBeInTheDocument();
      expect(screen.getByText(new RegExp(concertsById[1].eventName))).toBeInTheDocument();
    })
    
    test("redirects to home if no bookingSelection exists", async () => {
      localStorage.clear()
      renderPage()

      expect(screen.getByText(/Ticket Monarch/i)).toBeInTheDocument();
      expect(screen.getByPlaceholderText(/search artists/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/sort/i)).toBeInTheDocument();
    })
  })

/* =========================
  Form validation
  ========================= */
  describe("form validation", () => {
    test("all required field errors are shown", async () => {
      renderPage()
      const user = userEvent.setup()

      const purchaseBtn = screen.getByRole("button", { name: /purchase/i })
      await user.click(purchaseBtn)

      expect(screen.getByText(/Card number is required/i)).toBeInTheDocument()
      expect(screen.getByText(/Expiry is required/i)).toBeInTheDocument()
      expect(screen.getByText(/CVV is required/i)).toBeInTheDocument()
      expect(screen.getByText(/Name is required/i)).toBeInTheDocument()
      expect(screen.getByText(/Address is required/i)).toBeInTheDocument()
      expect(screen.getByText(/City is required/i)).toBeInTheDocument()
      expect(screen.getByText(/Zip code is required/i)).toBeInTheDocument()
      expect(screen.getByText(/State is required/i)).toBeInTheDocument()
    })

    test("name cannot contain number", async () => {
      renderPage()
      const user = userEvent.setup()

      await user.type(screen.getByLabelText(/name on card/i), "1234")

      const purchaseBtn = screen.getByRole("button", { name: /purchase/i })
      await user.click(purchaseBtn)

      expect(screen.getByText(/Name cannot contain numbers/i)).toBeInTheDocument()
    })

    test("card number is 13-19 digits", async () => {
      renderPage()
      const user = userEvent.setup()

      await user.type(screen.getByLabelText(/card number/i), "1234")
      const purchaseBtn = screen.getByRole("button", { name: /purchase/i })
      await user.click(purchaseBtn)
      expect(screen.getByText(/Card number must be 13–19 digits/i)).toBeInTheDocument()

      await user.type(screen.getByLabelText(/card number/i), "123412341234123")
      await user.click(purchaseBtn)
      expect(screen.queryByText(/Card number must be 13–19 digits/i)).not.toBeInTheDocument()
    })

    test("expiery uses 4 digits", async () => {
      renderPage()
      const user = userEvent.setup()

      await user.type(screen.getByLabelText(/MM\/YY/i), "123")
      const purchaseBtn = screen.getByRole("button", { name: /purchase/i })
      await user.click(purchaseBtn)
      expect(screen.getByText(/Use MM\/YY format/i)).toBeInTheDocument()
    })

    test("CVV is 3-4 digits", async () => {
      renderPage()
      const user = userEvent.setup()

      await user.type(screen.getByLabelText(/CVC/i), "12")
      const purchaseBtn = screen.getByRole("button", { name: /purchase/i })
      await user.click(purchaseBtn)
      expect(screen.getByText(/CVV must be 3 or 4 digits/i)).toBeInTheDocument()

      await user.type(screen.getByLabelText(/CVC/i), "{backspace}{backspace}aaa")
      await user.click(purchaseBtn)
      expect(screen.getByText(/CVV must be 3 or 4 digits/i)).toBeInTheDocument()

      await user.type(screen.getByLabelText(/CVC/i), "{backspace}{backspace}{backspace}12345")
      await user.click(purchaseBtn)
      expect(screen.getByText(/CVV must be 3 or 4 digits/i)).toBeInTheDocument()
    })

    test("zip can only be 5 digits or 5+4 format", async () => {
      renderPage()
      const user = userEvent.setup()

      const input = screen.getByLabelText(/Zip Code/i)
      await user.type(input, "1234")
      const purchaseBtn = screen.getByRole("button", { name: /purchase/i })
      await user.click(purchaseBtn)
      expect(screen.getByText(/Enter a valid zip code/i)).toBeInTheDocument()

      await user.type(screen.getByLabelText(/Zip Code/i), "12345")
      await user.click(purchaseBtn)
      expect(screen.getByText(/Enter a valid zip code/i)).toBeInTheDocument()

      await user.type(screen.getByLabelText(/Zip Code/i), "{Control}a{/Control}{backspace}")
      await user.type(screen.getByLabelText(/Zip Code/i), "abcde")
      await user.click(purchaseBtn)
      expect(screen.getByText(/Enter a valid zip code/i)).toBeInTheDocument()

      await user.clear(input)
      await user.type(screen.getByLabelText(/Zip Code/i), "12345-1234")
      await user.click(purchaseBtn)
      expect(screen.queryByText(/Enter a valid zip code/i)).not.toBeInTheDocument()
    })

    test("city cannot contain numbers", async () => {
      renderPage()
      const user = userEvent.setup()

      await user.type(screen.getByLabelText(/city/i), "123")
      const purchaseBtn = screen.getByRole("button", { name: /purchase/i })
      await user.click(purchaseBtn)
      expect(screen.getByText(/City cannot contain numbers/i)).toBeInTheDocument()
    })

    test("user can select from state dropdown", async () => {
      renderPage()
      const user = userEvent.setup()

      const stateSelect = screen.getByLabelText(/state/i);
      await user.selectOptions(stateSelect, "California");

      expect(stateSelect).toHaveValue("California");
    })

    test("user can select from country dropdown", async () => {
      renderPage()
      const user = userEvent.setup()

      const countrySelect = screen.getByLabelText(/country/i);
      await user.selectOptions(countrySelect, "U.S.A.");

      expect(countrySelect).toHaveValue("U.S.A.");
    })
  })

/* =========================
   Checkout and Challenge flow
   ========================= */
  describe("checkout and challenge flow", () => {
    test("checkout works if not suspected", async () => {
      renderPage()
      const user = userEvent.setup()

      await fillValidCheckoutForm(user)
      await user.click(screen.getByRole("button", { name: /purchase/i }))

      await waitFor(() => {
        expect(mockForceFlush).toHaveBeenCalled();
        expect(mockGetFlag).toHaveBeenCalled();
        expect(mockEvaluateSession).toHaveBeenCalledWith("test-session-id");
        expect(mockSubmitCheckout).toHaveBeenCalled();
      });

      expect(await screen.findByText(/Ticket Monarch/i)).toBeInTheDocument();
      expect(await screen.findByText(/Congratulations/i)).toBeInTheDocument();
      expect(localStorage.getItem("bookingSelection")).toBeNull();
      expect(localStorage.getItem("orderDetails")).not.toBeNull();
    })

    test('force CAPTCHA flag "on" shows hard challenge and does not submit checkout', async () => {
      renderPage()
      const user = userEvent.setup()

      mockGetFlag.mockResolvedValue({
        success: true,
        data: { flag: "on" },
      })

      await fillValidCheckoutForm(user)
      await user.click(screen.getByRole("button", { name: /purchase/i }))

      expect(await screen.findByText("Challenge Modal")).toBeInTheDocument()
      expect(screen.getByText(/Difficulty: hard/i)).toBeInTheDocument()
      expect(mockSubmitCheckout).not.toHaveBeenCalled()
    })

    test('force CAPTCHA flag "off" bypasses challenge and submits checkout', async () => {
      renderPage()
      const user = userEvent.setup()

      mockGetFlag.mockResolvedValue({
        success: true,
        data: { flag: "off" },
      })

      await fillValidCheckoutForm(user)
      await user.click(screen.getByRole("button", { name: /purchase/i }))

      await waitFor(() => {
        expect(mockSubmitCheckout).toHaveBeenCalled()
      })

      expect(await screen.findByText(/Ticket Monarch/i)).toBeInTheDocument();
      expect(await screen.findByText(/Congratulations/i)).toBeInTheDocument();
    })

    test("shows verification unavailable message when evaluateSession fails", async () => {
      renderPage()
      const user = userEvent.setup()

      mockEvaluateSession.mockResolvedValue({
        success: false,
      })

      await fillValidCheckoutForm(user)
      await user.click(screen.getByRole("button", { name: /purchase/i }))

      expect(
        await screen.findByText(/Verification service unavailable\. Please try again\./i)
      ).toBeInTheDocument()

      expect(mockSubmitCheckout).not.toHaveBeenCalled()
    })

    test('decision "block" shows hard challenge', async () => {
      renderPage()
      const user = userEvent.setup()

      mockEvaluateSession.mockResolvedValue({
        success: true,
        decision: "block",
        events_processed: 3,
        honeypot_triggered: false,
      })

      await fillValidCheckoutForm(user)
      await user.click(screen.getByRole("button", { name: /purchase/i }))

      expect(await screen.findByText("Challenge Modal")).toBeInTheDocument()
      expect(screen.getByText(/Difficulty: hard/i)).toBeInTheDocument()
      expect(mockSubmitCheckout).not.toHaveBeenCalled()
    })

    test.each([
      ["easy_puzzle", "easy"],
      ["medium_puzzle", "medium"],
      ["hard_puzzle", "hard"],
    ])('decision "%s" shows %s challenge', async (decision, difficulty) => {
      renderPage()
      const user = userEvent.setup()

      mockEvaluateSession.mockResolvedValue({
        success: true,
        decision,
        events_processed: 3,
        honeypot_triggered: false,
      })

      await fillValidCheckoutForm(user)
      await user.click(screen.getByRole("button", { name: /purchase/i }))

      expect(await screen.findByText("Challenge Modal")).toBeInTheDocument()
      expect(screen.getByText(new RegExp(`Difficulty: ${difficulty}`, "i"))).toBeInTheDocument()
      expect(mockSubmitCheckout).not.toHaveBeenCalled()
    })

    test("unexpected decision shows fallback error", async () => {
      renderPage()
      const user = userEvent.setup()

      mockEvaluateSession.mockResolvedValue({
        success: true,
        decision: "weird_unknown_state",
        events_processed: 3,
        honeypot_triggered: false,
      })

      await fillValidCheckoutForm(user)
      await user.click(screen.getByRole("button", { name: /purchase/i }))

      expect(
        await screen.findByText(/Unexpected verification response\. Please try again\./i)
      ).toBeInTheDocument()

      expect(mockSubmitCheckout).not.toHaveBeenCalled()
    })

    test("failed challenge shows verification failed message", async () => {
      renderPage()
      const user = userEvent.setup()

      mockEvaluateSession.mockResolvedValue({
        success: true,
        decision: "hard_puzzle",
        events_processed: 3,
        honeypot_triggered: false,
      })

      await fillValidCheckoutForm(user)
      await user.click(screen.getByRole("button", { name: /purchase/i }))

      expect(await screen.findByText("Challenge Modal")).toBeInTheDocument()

      await user.click(screen.getByRole("button", { name: /fail challenge/i }))

      expect(
        await screen.findByText(/Verification failed\. Please try again\./i)
      ).toBeInTheDocument()

      expect(mockSubmitCheckout).not.toHaveBeenCalled()
    })

    test("passed challenge completes checkout", async () => {
      renderPage()
      const user = userEvent.setup()

      mockEvaluateSession.mockResolvedValue({
        success: true,
        decision: "medium_puzzle",
        events_processed: 3,
        honeypot_triggered: false,
      })

      await fillValidCheckoutForm(user)
      await user.click(screen.getByRole("button", { name: /purchase/i }))

      expect(await screen.findByText("Challenge Modal")).toBeInTheDocument()

      await user.click(screen.getByRole("button", { name: /pass challenge/i }))

      await waitFor(() => {
        expect(mockSubmitCheckout).toHaveBeenCalled()
      })

      expect(await screen.findByText(/Ticket Monarch/i)).toBeInTheDocument();
      expect(await screen.findByText(/Congratulations/i)).toBeInTheDocument();
    })

    test("shows error when submitCheckout returns unsuccessful response", async () => {
      renderPage()
      const user = userEvent.setup()

      mockSubmitCheckout.mockResolvedValue({ success: false })

      await fillValidCheckoutForm(user)
      await user.click(screen.getByRole("button", { name: /purchase/i }))

      expect(await screen.findByText(/^Error$/i)).toBeInTheDocument()
    })

    test("shows error when submitCheckout throws", async () => {
      renderPage()
      const user = userEvent.setup()

      mockSubmitCheckout.mockRejectedValue(new Error("network fail"))

      await fillValidCheckoutForm(user)
      await user.click(screen.getByRole("button", { name: /purchase/i }))

      expect(await screen.findByText(/^Error$/i)).toBeInTheDocument()
    })

    test("honeypot-triggered final evaluation shows hard challenge", async () => {
      renderPage()
      const user = userEvent.setup()

      mockEvaluateSession.mockResolvedValue({
        success: true,
        decision: "allow",
        events_processed: 3,
        honeypot_triggered: true,
      })

      await fillValidCheckoutForm(user)
      await user.click(screen.getByRole("button", { name: /purchase/i }))

      expect(await screen.findByText("Challenge Modal")).toBeInTheDocument()
      expect(screen.getByText(/Difficulty: hard/i)).toBeInTheDocument()
      expect(mockSubmitCheckout).not.toHaveBeenCalled()
    })
  })
})