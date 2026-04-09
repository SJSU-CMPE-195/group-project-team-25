"""Parse online_training.log and generate figures showing live improvement.

Usage:
    python -m rl_captcha.scripts.plot_online --log online_training.log
    python -m rl_captcha.scripts.plot_online --log online_training.log --out figures/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Regex patterns matching agent_service.py online log output ───────────

RE_HEADER = re.compile(
    r"---\s*Online Update\s*#(\d+)\s*\|\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
)
RE_LABEL = re.compile(
    r"True label:\s*(\w+)\s*\|\s*Events:\s*(\d+)(?:\s*\|\s*Windows:\s*(\d+))?"
)
RE_BEFORE = re.compile(
    r"BEFORE:\s*decision=(\S+)\s+p_allow=([\d.]+)\s+p_suspicious=([\d.]+)\s+(CORRECT|WRONG)"
)
RE_AFTER = re.compile(
    r"AFTER:\s*decision=(\S+)\s+p_allow=([\d.]+)\s+p_suspicious=([\d.]+)\s+(CORRECT|WRONG)"
)
RE_RESULT = re.compile(
    r"Result:\s*(IMPROVED|REGRESSED|UNCHANGED)\s*\|\s*Policy loss:\s*([-\d.]+)\s*\|\s*Value loss:\s*([-\d.]+)"
)


def parse_log(path: str) -> list[dict]:
    """Parse online_training.log into a list of update dicts."""
    updates = []
    current = {}

    # handle UTF-16 (PowerShell Tee-Object)
    encoding = "utf-8"
    with open(path, "rb") as fb:
        bom = fb.read(2)
        if bom == b"\xff\xfe":
            encoding = "utf-16-le"
        elif bom == b"\xfe\xff":
            encoding = "utf-16-be"
        elif b"\x00" in fb.read(64):
            encoding = "utf-16"

    with open(path, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            line = line.strip()

            m = RE_HEADER.search(line)
            if m:
                if current:
                    updates.append(current)
                current = {
                    "update_num": int(m.group(1)),
                    "timestamp": m.group(2),
                }
                continue

            m = RE_LABEL.search(line)
            if m and current:
                current["true_label"] = m.group(1)
                current["events"] = int(m.group(2))
                if m.group(3):
                    current["windows"] = int(m.group(3))
                continue

            m = RE_BEFORE.search(line)
            if m and current:
                current["before_decision"] = m.group(1)
                current["before_p_allow"] = float(m.group(2))
                current["before_p_suspicious"] = float(m.group(3))
                current["before_correct"] = m.group(4) == "CORRECT"
                continue

            m = RE_AFTER.search(line)
            if m and current:
                current["after_decision"] = m.group(1)
                current["after_p_allow"] = float(m.group(2))
                current["after_p_suspicious"] = float(m.group(3))
                current["after_correct"] = m.group(4) == "CORRECT"
                continue

            m = RE_RESULT.search(line)
            if m and current:
                current["result"] = m.group(1)
                current["policy_loss"] = float(m.group(2))
                current["value_loss"] = float(m.group(3))
                continue

    if current:
        updates.append(current)

    return updates


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_all(updates: list[dict], out_dir: Path, fmt: str = "png"):
    """Generate all online training figures."""
    out_dir.mkdir(parents=True, exist_ok=True)

    nums = np.array([u["update_num"] for u in updates])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )

    # ── 1. Cumulative accuracy (rolling window) ─────────────────────────
    after_correct = np.array([1.0 if u.get("after_correct") else 0.0 for u in updates])
    before_correct = np.array(
        [1.0 if u.get("before_correct") else 0.0 for u in updates]
    )

    cumulative_after = (
        np.cumsum(after_correct) / np.arange(1, len(after_correct) + 1) * 100
    )
    cumulative_before = (
        np.cumsum(before_correct) / np.arange(1, len(before_correct) + 1) * 100
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        nums,
        cumulative_before,
        color="#e74c3c",
        linewidth=1.8,
        label="Before Update (cumulative)",
    )
    ax.plot(
        nums,
        cumulative_after,
        color="#2ecc71",
        linewidth=1.8,
        label="After Update (cumulative)",
    )
    if len(nums) >= 5:
        window = min(10, len(nums))
        ax.plot(
            nums,
            smooth(after_correct * 100, window),
            color="#27ae60",
            linewidth=1.2,
            linestyle="--",
            alpha=0.6,
            label=f"After (rolling w={window})",
        )
    ax.set_xlabel("Online Update #")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Online Learning — Cumulative Accuracy")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"online_accuracy.{fmt}")
    plt.close(fig)
    print(f"  Saved online_accuracy.{fmt}")

    # ── 2. Before vs After probabilities ─────────────────────────────────
    before_p_sus = np.array([u.get("before_p_suspicious", 0) for u in updates])
    after_p_sus = np.array([u.get("after_p_suspicious", 0) for u in updates])
    before_p_allow = np.array([u.get("before_p_allow", 0) for u in updates])
    after_p_allow = np.array([u.get("after_p_allow", 0) for u in updates])

    # split by label
    human_mask = np.array([u.get("true_label") == "human" for u in updates])
    bot_mask = ~human_mask

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # humans: p_allow should go UP after update
    if human_mask.any():
        h_nums = nums[human_mask]
        ax1.scatter(
            h_nums,
            before_p_allow[human_mask],
            color="#e74c3c",
            alpha=0.5,
            s=25,
            label="Before",
            zorder=3,
        )
        ax1.scatter(
            h_nums,
            after_p_allow[human_mask],
            color="#2ecc71",
            alpha=0.5,
            s=25,
            label="After",
            zorder=3,
        )
        for i in range(len(h_nums)):
            idx = np.where(human_mask)[0][i]
            ax1.annotate(
                "",
                xy=(h_nums[i], after_p_allow[idx]),
                xytext=(h_nums[i], before_p_allow[idx]),
                arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8),
            )
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
    ax1.set_xlabel("Online Update #")
    ax1.set_ylabel("P(allow)")
    ax1.set_title("Human Sessions — P(allow) Shift")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # bots: p_suspicious should go UP after update
    if bot_mask.any():
        b_nums = nums[bot_mask]
        ax2.scatter(
            b_nums,
            before_p_sus[bot_mask],
            color="#e74c3c",
            alpha=0.5,
            s=25,
            label="Before",
            zorder=3,
        )
        ax2.scatter(
            b_nums,
            after_p_sus[bot_mask],
            color="#2ecc71",
            alpha=0.5,
            s=25,
            label="After",
            zorder=3,
        )
        for i in range(len(b_nums)):
            idx = np.where(bot_mask)[0][i]
            ax2.annotate(
                "",
                xy=(b_nums[i], after_p_sus[idx]),
                xytext=(b_nums[i], before_p_sus[idx]),
                arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8),
            )
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("Online Update #")
    ax2.set_ylabel("P(suspicious)")
    ax2.set_title("Bot Sessions — P(suspicious) Shift")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    fig.suptitle(
        "Probability Shift Per Update (Before → After)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"online_prob_shift.{fmt}")
    plt.close(fig)
    print(f"  Saved online_prob_shift.{fmt}")

    # ── 3. Improvement / Regression / Unchanged bar over time ────────────
    improved = np.array(
        [1 if u.get("result") == "IMPROVED" else 0 for u in updates], dtype=float
    )
    regressed = np.array(
        [1 if u.get("result") == "REGRESSED" else 0 for u in updates], dtype=float
    )
    unchanged = np.array(
        [1 if u.get("result") == "UNCHANGED" else 0 for u in updates], dtype=float
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    bar_width = 0.8
    ax.bar(nums, improved, bar_width, color="#2ecc71", label="Improved")
    ax.bar(
        nums, unchanged, bar_width, bottom=improved, color="#95a5a6", label="Unchanged"
    )
    ax.bar(
        nums,
        regressed,
        bar_width,
        bottom=improved + unchanged,
        color="#e74c3c",
        label="Regressed",
    )
    ax.set_xlabel("Online Update #")
    ax.set_ylabel("Outcome")
    ax.set_title("Per-Update Outcome")
    ax.legend()
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(out_dir / f"online_outcomes.{fmt}")
    plt.close(fig)
    print(f"  Saved online_outcomes.{fmt}")

    # ── 4. Loss curves ───────────────────────────────────────────────────
    policy_loss = np.array([u.get("policy_loss", 0) for u in updates])
    value_loss = np.array([u.get("value_loss", 0) for u in updates])

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color1 = "#e74c3c"
    color2 = "#2ecc71"
    ax1.plot(nums, policy_loss, color=color1, linewidth=1.5, alpha=0.4)
    if len(nums) >= 5:
        ax1.plot(
            nums,
            smooth(policy_loss, min(10, len(nums))),
            color=color1,
            linewidth=2,
            label="Policy Loss",
        )
    else:
        ax1.plot(nums, policy_loss, color=color1, linewidth=2, label="Policy Loss")
    ax1.set_xlabel("Online Update #")
    ax1.set_ylabel("Policy Loss", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(nums, value_loss, color=color2, linewidth=1.5, alpha=0.4)
    if len(nums) >= 5:
        ax2.plot(
            nums,
            smooth(value_loss, min(10, len(nums))),
            color=color2,
            linewidth=2,
            label="Value Loss",
        )
    else:
        ax2.plot(nums, value_loss, color=color2, linewidth=2, label="Value Loss")
    ax2.set_ylabel("Value Loss", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Online Learning — Loss per Update")
    ax1.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"online_loss.{fmt}")
    plt.close(fig)
    print(f"  Saved online_loss.{fmt}")

    # ── 5. Summary 2×2 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Online Learning Summary", fontsize=15, fontweight="bold", y=0.98)

    # (a) cumulative accuracy
    ax = axes[0, 0]
    ax.plot(nums, cumulative_before, color="#e74c3c", linewidth=1.8, label="Before")
    ax.plot(nums, cumulative_after, color="#2ecc71", linewidth=1.8, label="After")
    ax.set_xlabel("Update #")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Cumulative Accuracy")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) outcomes bar
    ax = axes[0, 1]
    total = len(updates)
    imp_count = int(improved.sum())
    unch_count = int(unchanged.sum())
    reg_count = int(regressed.sum())
    bars = ax.barh(
        ["Improved", "Unchanged", "Regressed"],
        [imp_count, unch_count, reg_count],
        color=["#2ecc71", "#95a5a6", "#e74c3c"],
    )
    for bar, count in zip(bars, [imp_count, unch_count, reg_count]):
        if count > 0:
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{count} ({count/total*100:.0f}%)",
                va="center",
                fontsize=10,
            )
    ax.set_xlabel("Count")
    ax.set_title(f"(b) Update Outcomes (n={total})")
    ax.grid(True, axis="x", alpha=0.3)

    # (c) losses
    ax = axes[1, 0]
    ax.plot(nums, policy_loss, color="#e74c3c", linewidth=1.5, label="Policy")
    ax.plot(nums, value_loss, color="#2ecc71", linewidth=1.5, label="Value")
    ax.set_xlabel("Update #")
    ax.set_ylabel("Loss")
    ax.set_title("(c) Losses")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) events per session
    events = np.array([u.get("events", 0) for u in updates])
    labels_arr = np.array([u.get("true_label", "") for u in updates])
    ax = axes[1, 1]
    h_mask = labels_arr == "human"
    b_mask = labels_arr == "bot"
    if h_mask.any():
        ax.scatter(
            nums[h_mask],
            events[h_mask],
            color="#3498db",
            alpha=0.6,
            s=30,
            label="Human",
        )
    if b_mask.any():
        ax.scatter(
            nums[b_mask], events[b_mask], color="#e74c3c", alpha=0.6, s=30, label="Bot"
        )
    ax.set_xlabel("Update #")
    ax.set_ylabel("Events in Session")
    ax.set_title("(d) Session Sizes")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"online_summary.{fmt}")
    plt.close(fig)
    print(f"  Saved online_summary.{fmt}")

    # ── Print stats ──────────────────────────────────────────────────────
    print(
        f"\n  Updates: {total} | Improved: {imp_count} | Unchanged: {unch_count} | Regressed: {reg_count}"
    )
    if total > 0:
        print(
            f"  Before accuracy: {before_correct.mean()*100:.1f}% | After accuracy: {after_correct.mean()*100:.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize online training logs")
    parser.add_argument(
        "--log",
        type=str,
        default="online_training.log",
        help="Path to online_training.log",
    )
    parser.add_argument(
        "--out", type=str, default="figures", help="Output directory for figures"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Figure format (pdf recommended for papers)",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: {log_path} not found")
        return

    print(f"Parsing {log_path}...")
    updates = parse_log(str(log_path))

    if not updates:
        print("No online update data found in log file.")
        return

    print(f"Found {len(updates)} online updates")
    plot_all(updates, Path(args.out), fmt=args.format)


if __name__ == "__main__":
    main()
