import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PAGE_INDEX = {
    "home": 0,
    "seat": 1,
    "checkout": 2,
}

PAGE_LABEL = {
    "home": "Home",
    "seat": "Seat Selection",
    "checkout": "Checkout",
}

EVENT_LABEL = {
    "mouse": "Mouse Movement",
    "click": "Mouse Clicks",
}

# Read the file and return tuple of (concert_select_page, section_select_page , checkout_page) based on event type sent ("mouse" || "click")
def read_file(file, eventType):
    folder = Path(file)

    if not folder.exists():
        print(f"Error: folder '{file}' does not exist.")
        sys.exit(1)

    json_files = list(folder.glob("*.json"))

    if not json_files:
        print(f"Error: no JSON files found in '{file}'.")
        sys.exit(1)

    concert_select_page = []
    section_select_page = []
    checkout_page = []

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if "segments" not in data:
            data = {
                "segments": [
                    {
                        "mouse": data.get("mouse", []),
                        "clicks": data.get("clicks", []),
                        "scroll": data.get("scroll", []),
                    }
                ]
            }

        mouse_events = []
        click_events = []
        scroll_events = []

        segments = data.get("segments", [])

        for seg in segments:
            for event in seg.get("mouse", []):
                mouse_events.append({
                    "t": event.get("t"),
                    "x": event.get("x"),
                    "y": event.get("y"),
                })

            for event in seg.get("clicks", []):
                target = event.get("target", {})
                click_events.append({
                    "t": event.get("t"),
                    "x": event.get("x"),
                    "y": event.get("y"),
                    "target_classes": target.get("classes"),
                })

            for event in seg.get("scroll", []):
                scroll_events.append({
                    "t": event.get("t"),
                    "dy": event.get("dy")
                })

        new_concert_select_page, new_section_select_page, new_checkout_page = separate_pages(mouse_events, click_events, scroll_events, eventType)

        concert_select_page += new_concert_select_page
        section_select_page += new_section_select_page
        checkout_page += new_checkout_page

    return concert_select_page, section_select_page , checkout_page

# Helper funciton to separate the events into each page (home, seats, checkout)
def separate_pages(mouse_events, click_events, scroll_events, eventType):
    concert_select_page = []
    section_select_page = []
    checkout_page = []

    # Map button to timestamp (if present)
    button_times = {
        item["target_classes"]: item["t"]
        for item in click_events
        if item.get("target_classes") in {"home-button", "tickets-button", "ss-checkout-btn"}
    }

    t_home = button_times.get("home-button", float("-inf"))
    t_tickets = button_times.get("tickets-button", float("inf"))
    t_checkout = button_times.get("ss-checkout-btn", float("inf"))

    def classify(event, weight):
        t = event["t"]

        # Some of our data is weird, this removes those data points
        if(event["x"] == 0): 
            return

        if t < t_home:
            return  # skip
        elif t < t_tickets:
            concert_select_page.append({**event, "weight": weight})
        elif t < t_checkout:
            section_select_page.append({**event, "weight": weight})
        elif t > t_checkout:
            checkout_page.append({**event, "weight": weight})

    # Tracks cumulative scroll up to current event time
    scroll_idx = 0
    total_dy = 0

    def adjust_y(event):
        nonlocal scroll_idx, total_dy

        # Add all scroll deltas that happened at or before this event's timestamp
        while scroll_idx < len(scroll_events) and scroll_events[scroll_idx]["t"] <= event["t"]:
            total_dy = scroll_events[scroll_idx]["dy"]  # overwrite, not accumulate
            scroll_idx += 1

        return {
            "x": event["x"],
            "y": event["y"] - total_dy,
            "t": event["t"]
        }

    if(eventType == "mouse"):
        events = [("mouse", m, 1) for m in mouse_events]
    elif(eventType == "click"):
        events = [("click", c, 1) for c in click_events]

    for _, event, weight in events:
        adjusted_event = adjust_y(event)
        classify(adjusted_event, weight)

    return concert_select_page, section_select_page, checkout_page

# Helper function to filter out the vertical data: returns a list with the middle 96% remaining
def filter_y_outliers(points, low_q=0.02, high_q=0.98):
    if not points:
        return points

    ys = np.array([p["y"] for p in points])
    low = np.quantile(ys, low_q)
    high = np.quantile(ys, high_q)

    return [p for p in points if low <= p["y"] <= high]

# Takes a list of 3 point lists, title and page then display the stats and heatmap
def plot_heatmaps(points_list, title, page):
    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=125, constrained_layout=True)
    MAP_NAMES = ("Human Events", "Bot Events", "Augmented Bot Events")

    mappable = None
    bins = 60

    points_list = [filter_y_outliers(points) for points in points_list]

    # Compute one shared vmax from normalized, nonzero bin values
    all_nonzero = []
    for i, points in enumerate(points_list):
        xs = np.array([p["x"] for p in points], dtype=float)
        ys = np.array([p["y"] for p in points], dtype=float)
        raw_weights = np.array([p["weight"] for p in points], dtype=float)
        weights = (raw_weights / raw_weights.sum()) * 100 if raw_weights.sum() > 0 else raw_weights

        H, _, _ = np.histogram2d(
            xs,
            ys,
            bins=bins,
            range=[[xs.min(), xs.max()], [ys.min(), ys.max()]],
            weights=weights
        )

        nonzero = H[H > 0]
        if len(nonzero) > 0:
            all_nonzero.extend(nonzero)

        # Print data for heatmap
        print("Heatmap data for", MAP_NAMES[i])
        print("Mean position (x, y | px):", f"({xs.mean():.0f},", f"{ys.mean():.0f})")
        print("Total data points:", len(points))
        p = H / H.sum()
        p = p[p > 0]
        print("Spatial entropy (Higher = more spread out | Lower = more concentrated):", f"{-np.sum(p * np.log2(p)):.2f}")
        print("Occupied bins:", f"{np.sum(H > 0)} / {bins ** 2}", f"= {(np.sum(H > 0)/bins ** 2) * 100:.2f}% of space used")
        print("Peak concentration (max bin):", f"{H.max():.2f}%")
        print("X-axis variance:", f"{np.var(xs):.2f}")
        print("Unique X-axis positions:", len(np.unique(xs)), "\n")

    vmax = np.percentile(all_nonzero, 98) if all_nonzero else 1.0

    for i, points in enumerate(points_list):
        xs = np.array([p["x"] for p in points], dtype=float)
        ys = np.array([p["y"] for p in points], dtype=float)
        raw_weights = np.array([p["weight"] for p in points], dtype=float)
        weights = (raw_weights / raw_weights.sum()) * 100 if raw_weights.sum() > 0 else raw_weights

        h = sns.histplot(
            x=xs,
            y=ys,
            weights=weights,
            bins=bins,
            binrange=[[xs.min(), xs.max()], [ys.min(), ys.max()]],
            cmap="turbo",
            alpha=0.9,
            vmax=vmax,
            cbar=False,
            ax=axes[i]
        )

        if mappable is None:
            mappable = h.collections[0]

        axes[i].set_title(MAP_NAMES[i])
        axes[i].set_xlabel("X position (px)")
        if i == 0:
            axes[i].set_ylabel("Y position (px)")
        else:
            axes[i].set_ylabel("")
            axes[i].tick_params(axis="y", left=False, labelleft=False)

        axes[i].invert_yaxis()
        axes[i].margins(x=0.05)
        
        mean_x = np.mean(xs)
        mean_y = np.mean(ys)
        axes[i].scatter(
            mean_x, mean_y,
            color="white",
            edgecolor="red",
            s=150,
            linewidths=2.5,
            zorder=5
        )

    cbar = fig.colorbar(mappable, ax=axes, location="right", pad=0.02)
    cbar.ax.set_ylabel("Percent of total events per bin", labelpad=10, fontsize=16)
    cbar.ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1f}%")
    cbar.ax.yaxis.set_label_position("left")

    fig.suptitle(f"{title} | {page} Page", fontsize=16, fontweight="bold")
    return fig

def main():
    parser = argparse.ArgumentParser(description="Generate event heatmaps from JSON session data.")
    parser.add_argument(
        "--event",
        choices=["mouse", "click"],
        required=True,
        help="Choose which event type to plot.",
    )
    parser.add_argument(
        "--page",
        choices=["home", "seat", "checkout"],
        required=True,
        help="Choose which page to plot.",
    )

    args = parser.parse_args()

    page_idx = PAGE_INDEX[args.page]
    page_name = PAGE_LABEL[args.page]
    title = EVENT_LABEL[args.event]

    human_pages = read_file("human", args.event)
    bot_pages = read_file("bot", args.event)
    bot_aug_pages = read_file("bot_augmented", args.event)

    points_list = [
        human_pages[page_idx],
        bot_pages[page_idx],
        bot_aug_pages[page_idx],
    ]

    plot_heatmaps(points_list, title, page_name)
    plt.show()

if __name__ == "__main__":
    main()