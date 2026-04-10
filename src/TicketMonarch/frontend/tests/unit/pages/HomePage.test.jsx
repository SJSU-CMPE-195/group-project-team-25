import { screen, within, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, describe, test, expect } from "vitest";
import { renderRouterPages } from "./utils/renderRouterPages";

vi.mock("@/assets/concerts", async () => {
  const mod = await import("./fixtures/concerts");
  return {
    concerts: mod.mockConcerts,
  };
});

function renderPage() {
  return renderRouterPages();
}

describe("Home", () => {
  test("default render state (header / controls, concert cards in FEATURED order, disclaimer footer", async () => {
    const user = userEvent.setup();
    renderPage();

    expect(screen.getByText(/Ticket Monarch/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/search artists/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/sort/i)).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText(/sort/i), "featured");

    const headings = screen.getAllByRole("heading", { level: 2 });
    const names = headings.map((h) => h.textContent);

    expect(names).toEqual(["Chappell Roan", "Metallica", "Lady Gaga"]);

    expect(screen.getByText(/this is a student project/i)).toBeInTheDocument();
  });

  test("filters concerts by search text", async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByPlaceholderText(/search artists/i), "Metallica");

    expect(screen.getByText("Metallica")).toBeInTheDocument();
    expect(screen.queryByText("Chappell Roan")).not.toBeInTheDocument();
    expect(screen.queryByText("Lady Gaga")).not.toBeInTheDocument();
  });

  test("shows empty state when no artists match the search", async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByPlaceholderText(/search artists/i), "not-a-real-artist");
    expect(screen.getByText(/no artists found/i)).toBeInTheDocument();
  });

  test("sorts concerts by artist name A to Z", async () => {
    const user = userEvent.setup();
    renderPage();

    await user.selectOptions(screen.getByLabelText(/sort/i), "name_asc");

    const headings = screen.getAllByRole("heading", { level: 2 });
    const names = headings.map((h) => h.textContent);

    expect(names).toEqual(["Chappell Roan", "Lady Gaga", "Metallica"]);
  });

  test("sorts concerts by price low to high", async () => {
    const user = userEvent.setup();
    renderPage();

    await user.selectOptions(screen.getByLabelText(/sort/i), "price_asc");

    const headings = screen.getAllByRole("heading", { level: 2 });
    const names = headings.map((h) => h.textContent);

    expect(names).toEqual(["Chappell Roan", "Lady Gaga", "Metallica"]);
  });

  test("sorts concerts by price high to low", async () => {
    const user = userEvent.setup();
    renderPage();

    await user.selectOptions(screen.getByLabelText(/sort/i), "price_desc");

    const headings = screen.getAllByRole("heading", { level: 2 });
    const names = headings.map((h) => h.textContent);

    expect(names).toEqual(["Metallica", "Lady Gaga", "Chappell Roan"]);
  });

  test("sorts concerts by soonest date", async () => {
    const user = userEvent.setup();
    renderPage();

    await user.selectOptions(screen.getByLabelText(/sort/i), "date_asc");

    const headings = screen.getAllByRole("heading", { level: 2 });
    const names = headings.map((h) => h.textContent);

    expect(names).toEqual(["Chappell Roan", "Metallica", "Lady Gaga"]);
  });

  test("test price tooltip info button (hover in, hover out, focus, blur)", async () => {
    renderPage();

    const infoButton = screen.getByRole("button", {
      name: /starting price for Metallica/i,
    });

    fireEvent.mouseEnter(infoButton);
    expect(screen.getByText(/From: \$145/i)).toBeInTheDocument();

    fireEvent.mouseLeave(infoButton);
    expect(screen.queryByText(/From: \$145/i)).not.toBeInTheDocument();

    fireEvent.focus(infoButton);
    expect(screen.getByText(/From: \$145/i)).toBeInTheDocument();

    fireEvent.blur(infoButton);
    expect(screen.queryByText(/From: \$145/i)).not.toBeInTheDocument();
  });

  test("renders ticket links with the correct destination", () => {
    renderPage();

    const heading = screen.getByRole("heading", { name: "Metallica" });
    const card = heading.closest(".concert-card");
    const link = within(card).getByRole("link", { name: /tickets/i });

    expect(link).toHaveAttribute("href", "/seats/2");
  });
});