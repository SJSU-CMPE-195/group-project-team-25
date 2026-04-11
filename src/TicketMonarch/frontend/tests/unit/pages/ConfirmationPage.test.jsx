import { beforeEach, vi, describe, test, expect } from "vitest";
import { screen, within, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderRouterPages } from "./utils/renderRouterPages";

const {
  mockConfirmHumanSession,
  mockResetSession,
} = vi.hoisted(() => ({
  mockConfirmHumanSession: vi.fn(),
  mockResetSession: vi.fn(),
}));

vi.mock("@/services/api", () => ({
  confirmHumanSession: mockConfirmHumanSession,
}));

vi.mock("@/services/tracking", () => ({
  resetSession: mockResetSession,
}));

function renderPage() {
  return renderRouterPages({ initialEntries: ["/confirmation"] })
}

const mockOrderDetails = {
  "concert": {
    "id": 1,
    "name": "Chappell Roan",
    "date": "Feb 14, 2026",
    "eventName": "Midwest Princess Tour",
    "venue": "Aragon Ballroom",
    "city": "Chicago, IL",
    "location": "Chicago, IL • Aragon Ballroom",
    "image": "/src/assets/images/chappell.jpg",
    "price": 100
  },
  "seats": [
    {
      "id": "GA1-1",
      "section": "GA1",
      "row": "4",
      "seat": "9",
      "price": 170
    }
  ],
  "selectedSection": "GA1",
  "sectionPrice": 170,
  "total": 170,
  "customerInfo": {
    "full_name": "Josh",
    "card_number": "1234 1234 1234 1234",
    "card_expiry": "1234",
    "card_cvv": "123",
    "billing_address": "123 main",
    "city": "ney york",
    "state": "California",
    "country": "U.S.A.",
    "zip_code": "10001",
    "company_url": "",
    "fax_number": ""
  },
  "orderDate": "2026-04-10T23:09:43.294Z"
};


describe("Confirmation", () => {
  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear()

    localStorage.setItem(
      "orderDetails",
      JSON.stringify(mockOrderDetails)
    );

    mockConfirmHumanSession.mockReset()
    mockResetSession.mockReset();
  })

  test("confirmation return button works", async () => {
    renderPage()
    const user = userEvent.setup()

    await user.click(screen.getByRole("button", { name: "Return to Home"}))

    expect(screen.getByText(/Ticket Monarch/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/search artists/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/sort/i)).toBeInTheDocument();
    expect(localStorage.getItem("orderDetails")).toBeNull();
  })

  test("return to home if orderDetails is missing", async () => {
    localStorage.clear()

    renderPage()

    expect(localStorage.getItem("orderDetails")).toBeNull();

    expect(screen.getByText(/Ticket Monarch/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/search artists/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/sort/i)).toBeInTheDocument();
  })

  test('calls confirmHumanSession when session exists and user is not a bot', async () => {
    mockConfirmHumanSession.mockResolvedValue({
      success: true,
      updated: true,
      steps: 3,
      metrics: {
        policy_loss: 0.1234,
        value_loss: 0.5678,
        entropy: 0.4321,
      },
      saved_json: true,
    })

    window.sessionStorage.setItem('tm_session_id', 'abc123')

    renderPage()

    await waitFor(() => {
      expect(mockConfirmHumanSession).toHaveBeenCalledWith('abc123')
    })

    await waitFor(() => {
      expect(screen.getByText(/Agent Updated/i)).toBeInTheDocument()
    })

    expect(mockResetSession).toHaveBeenCalledTimes(1)
  })

  test('shows API error message when confirmHumanSession returns success false', async () => {
    mockConfirmHumanSession.mockResolvedValue({
      success: false,
      error: 'Bad session',
    })

    window.sessionStorage.setItem('tm_session_id', 'abc123')

    renderPage()

    await waitFor(() => {
      expect(mockConfirmHumanSession).toHaveBeenCalledWith('abc123')
    })

    expect(
      screen.getByText(/Could not update agent: Bad session/i)
    ).toBeInTheDocument()

    expect(mockResetSession).toHaveBeenCalledTimes(1)
  })

  test('shows fallback API error when confirmHumanSession returns no error', async () => {
    mockConfirmHumanSession.mockResolvedValue({
      success: false,
    })

    window.sessionStorage.setItem('tm_session_id', 'abc123')

    renderPage()

    await waitFor(() => {
      expect(
        screen.getByText(/Could not update agent: Failed to confirm session/i)
      ).toBeInTheDocument()
    })

    expect(mockResetSession).toHaveBeenCalledTimes(1)
  })

  test('shows network error when confirmHumanSession rejects', async () => {
    mockConfirmHumanSession.mockRejectedValue(new Error('Request failed'))

    window.sessionStorage.setItem('tm_session_id', 'abc123')

    renderPage()

    await waitFor(() => {
      expect(
        screen.getByText(/Could not update agent: Request failed/i)
      ).toBeInTheDocument()
    })

    expect(mockResetSession).toHaveBeenCalledTimes(1)
  })

  test('does not call confirmHumanSession when there is no session id', async () => {
    renderPage()

    await waitFor(() => {
      expect(mockResetSession).toHaveBeenCalledTimes(1)
    })

    expect(mockConfirmHumanSession).not.toHaveBeenCalled()
    expect(
      screen.getByText(/Sending session to RL agent/i)
    ).toBeInTheDocument()
  })

  test('does not call confirmHumanSession when session is flagged as bot', async () => {
    window.sessionStorage.setItem('tm_session_id', 'abc123')
    window.sessionStorage.setItem('tm_is_bot', 'true')

    renderPage()

    await waitFor(() => {
      expect(mockResetSession).toHaveBeenCalledTimes(1)
    })

    expect(mockConfirmHumanSession).not.toHaveBeenCalled()
  })
})