"""Parse training.log and generate publication-quality figures.

Usage:
    python -m rl_captcha.scripts.plot_training --log training.log
    python -m rl_captcha.scripts.plot_training --log training.log --out figures/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Regex patterns matching train_ppo.py output ──────────────────────────

RE_HEADER = re.compile(
    r"---\s*Rollout\s+(\d+)/(\d+)\s*\|\s*Steps:\s*(\d+)\s*\|\s*Time:\s*([\d.]+)s"
)
RE_EPISODES = re.compile(
    r"Episodes:\s*(\d+)\s*\|\s*Avg reward:\s*([-\d.]+)\s*\|\s*Avg length:\s*([\d.]+)(?:\s*\|\s*Avg windows:\s*([\d.]+))?"
)
RE_LOSSES = re.compile(
    r"Policy loss:\s*([-\d.]+)\s*\|\s*Value loss:\s*([-\d.]+)\s*\|\s*Entropy:\s*([-\d.]+)"
)
RE_OUTCOMES = re.compile(r"Outcomes:\s*(.+)")
RE_VAL_ACC = re.compile(r"\[Val accuracy:\s*([\d.]+)\s+over\s+(\d+)\s+episodes\]")
RE_DG_METRICS = re.compile(r"Delight:\s*([-\d.]+)\s*\|\s*Gate:\s*([-\d.]+)")
RE_SOFT_PPO_METRICS = re.compile(
    r"Alpha:\s*([-\d.]+)\s*\|\s*Alpha loss:\s*([-\d.]+)\s*\|\s*Target H:\s*([-\d.]+)"
)


def parse_log(path: str) -> list[dict]:
    """Parse a training.log file into a list of rollout dicts."""
    rollouts = []
    current = {}

    # PowerShell's Tee-Object writes UTF-16LE on Windows; detect encoding
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
                    rollouts.append(current)
                current = {
                    "rollout": int(m.group(1)),
                    "total_rollouts": int(m.group(2)),
                    "steps": int(m.group(3)),
                    "time_s": float(m.group(4)),
                }
                continue

            m = RE_EPISODES.search(line)
            if m and current:
                current["episodes"] = int(m.group(1))
                current["avg_reward"] = float(m.group(2))
                current["avg_length"] = float(m.group(3))
                if m.group(4):
                    current["avg_windows"] = float(m.group(4))
                continue

            m = RE_LOSSES.search(line)
            if m and current:
                current["policy_loss"] = float(m.group(1))
                current["value_loss"] = float(m.group(2))
                current["entropy"] = float(m.group(3))
                continue

            m = RE_OUTCOMES.search(line)
            if m and current:
                outcomes = {}
                for part in m.group(1).split(","):
                    part = part.strip()
                    if ":" in part:
                        name, pct = part.rsplit(":", 1)
                        outcomes[name.strip()] = float(pct.strip().rstrip("%"))
                current["outcomes"] = outcomes
                continue

            m = RE_DG_METRICS.search(line)
            if m and current:
                current["delight"] = float(m.group(1))
                current["gate"] = float(m.group(2))
                continue

            m = RE_SOFT_PPO_METRICS.search(line)
            if m and current:
                current["alpha"] = float(m.group(1))
                current["alpha_loss"] = float(m.group(2))
                current["target_entropy"] = float(m.group(3))
                continue

            m = RE_VAL_ACC.search(line)
            if m and current:
                current["val_accuracy"] = float(m.group(1))
                current["val_episodes"] = int(m.group(2))
                continue

    if current:
        rollouts.append(current)

    return rollouts


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average for smoothing noisy curves."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    # pad to avoid shrinking the array
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_all(rollouts: list[dict], out_dir: Path, fmt: str = "png"):
    """Generate all figures and save to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = np.array([r["steps"] for r in rollouts])
    steps_k = steps / 1000  # x-axis in thousands

    # ── Style ─────────────────────────────────────────────────────────
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

    # ── 1. Reward curve ───────────────────────────────────────────────
    rewards = np.array([r.get("avg_reward", 0) for r in rollouts])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps_k, rewards, alpha=0.25, color="#4a90e2", linewidth=0.8)
    ax.plot(
        steps_k,
        smooth(rewards, 10),
        color="#4a90e2",
        linewidth=2,
        label="Smoothed (w=10)",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Training Steps (×1K)")
    ax.set_ylabel("Average Episode Reward")
    ax.set_title("Training Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"reward_curve.{fmt}")
    plt.close(fig)
    print(f"  Saved reward_curve.{fmt}")

    # ── 2. Loss curves (policy + value on twin axes) ──────────────────
    policy_loss = np.array([r.get("policy_loss", 0) for r in rollouts])
    value_loss = np.array([r.get("value_loss", 0) for r in rollouts])

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color1 = "#e74c3c"
    color2 = "#2ecc71"
    ax1.plot(
        steps_k,
        smooth(policy_loss, 10),
        color=color1,
        linewidth=1.8,
        label="Policy Loss",
    )
    ax1.set_xlabel("Training Steps (×1K)")
    ax1.set_ylabel("Policy Loss", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(
        steps_k, smooth(value_loss, 10), color=color2, linewidth=1.8, label="Value Loss"
    )
    ax2.set_ylabel("Value Loss", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Policy and Value Loss")
    ax1.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"loss_curves.{fmt}")
    plt.close(fig)
    print(f"  Saved loss_curves.{fmt}")

    # ── 3. Entropy ────────────────────────────────────────────────────
    entropy = np.array([r.get("entropy", 0) for r in rollouts])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps_k, entropy, alpha=0.25, color="#9b59b6", linewidth=0.8)
    ax.plot(
        steps_k,
        smooth(entropy, 10),
        color="#9b59b6",
        linewidth=2,
        label="Smoothed (w=10)",
    )
    ax.set_xlabel("Training Steps (×1K)")
    ax.set_ylabel("Policy Entropy")
    ax.set_title("Policy Entropy Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"entropy.{fmt}")
    plt.close(fig)
    print(f"  Saved entropy.{fmt}")

    # ── 4. Episode length ─────────────────────────────────────────────
    lengths = np.array([r.get("avg_length", 0) for r in rollouts])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps_k, lengths, alpha=0.25, color="#e67e22", linewidth=0.8)
    ax.plot(
        steps_k,
        smooth(lengths, 10),
        color="#e67e22",
        linewidth=2,
        label="Smoothed (w=10)",
    )
    ax.set_xlabel("Training Steps (×1K)")
    ax.set_ylabel("Average Episode Length (windows)")
    ax.set_title("Episode Length — Windows Before Decision")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"episode_length.{fmt}")
    plt.close(fig)
    print(f"  Saved episode_length.{fmt}")

    # ── 5. Outcome distribution (stacked area) ───────────────────────
    # Only terminal outcomes are logged (per-episode, not per-step).
    # With action masking: observation-phase outcomes (continue, honeypot)
    # never appear here — only final classification decisions.
    all_outcomes = set()
    for r in rollouts:
        all_outcomes.update(r.get("outcomes", {}).keys())

    if all_outcomes:
        # Map raw outcome names to display categories
        CATEGORY_MAP = {
            "correct_allow": "Correct Allow (TN)",
            "correct_block": "Correct Block (TP)",
            "bot_blocked_puzzle": "Bot Caught by Puzzle (TP)",
            "false_positive_block": "False Positive (FP)",
            "fp_puzzle": "False Positive Puzzle (FP)",
            "false_negative": "False Negative (FN)",
            "bot_passed_puzzle": "Bot Passed Puzzle (FN)",
            "truncated": "Truncated",
        }
        CATEGORY_COLORS = {
            "Correct Allow (TN)": "#2ecc71",
            "Correct Block (TP)": "#3498db",
            "Bot Caught by Puzzle (TP)": "#1abc9c",
            "False Positive (FP)": "#e74c3c",
            "False Positive Puzzle (FP)": "#e67e22",
            "False Negative (FN)": "#c0392b",
            "Bot Passed Puzzle (FN)": "#d35400",
            "Truncated": "#95a5a6",
        }

        # aggregate into categories
        categories = {}
        for r in rollouts:
            oc = r.get("outcomes", {})
            for raw_name, pct in oc.items():
                cat = CATEGORY_MAP.get(raw_name, raw_name)
                if cat not in categories:
                    categories[cat] = []

        # build arrays per category
        for cat in categories:
            categories[cat] = []
            for r in rollouts:
                oc = r.get("outcomes", {})
                total = 0
                for raw_name, pct in oc.items():
                    if CATEGORY_MAP.get(raw_name, raw_name) == cat:
                        total += pct
                categories[cat].append(total)

        fig, ax = plt.subplots(figsize=(8, 5))
        # sort categories so correct decisions are at the bottom
        order = [
            "Correct Allow (TN)",
            "Correct Block (TP)",
            "Bot Caught by Puzzle (TP)",
            "False Positive (FP)",
            "False Positive Puzzle (FP)",
            "False Negative (FN)",
            "Bot Passed Puzzle (FN)",
            "Truncated",
        ]
        labels = [c for c in order if c in categories]
        # include any unmapped outcomes
        for c in categories:
            if c not in labels:
                labels.append(c)
        data = np.array([smooth(np.array(categories[c]), 10) for c in labels])
        colors = [CATEGORY_COLORS.get(c, "#bdc3c7") for c in labels]

        ax.stackplot(steps_k, data, labels=labels, colors=colors, alpha=0.85)
        ax.set_xlabel("Training Steps (×1K)")
        ax.set_ylabel("Episode Outcome Distribution (%)")
        ax.set_title("Classification Outcome Distribution Over Training")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        fig.savefig(out_dir / f"outcome_distribution.{fmt}", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved outcome_distribution.{fmt}")

    # ── 6. Classification accuracy: train vs validation ─────────────
    correct_pcts = []
    for r in rollouts:
        oc = r.get("outcomes", {})
        correct = (
            oc.get("correct_allow", 0)
            + oc.get("correct_block", 0)
            + oc.get("bot_blocked_puzzle", 0)
        )
        correct_pcts.append(correct)

    correct_arr = np.array(correct_pcts)

    # Extract validation accuracy points (only logged every save_interval)
    val_steps_k = []
    val_accs = []
    for r in rollouts:
        if "val_accuracy" in r:
            val_steps_k.append(r["steps"] / 1000)
            val_accs.append(r["val_accuracy"] * 100)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps_k, correct_arr, alpha=0.2, color="#27ae60", linewidth=0.8)
    ax.plot(
        steps_k,
        smooth(correct_arr, 10),
        color="#27ae60",
        linewidth=2,
        label="Train (smoothed)",
    )
    if val_steps_k:
        ax.plot(
            val_steps_k,
            val_accs,
            "o-",
            color="#e74c3c",
            linewidth=1.8,
            markersize=4,
            label="Validation",
        )
    ax.set_xlabel("Training Steps (×1K)")
    ax.set_ylabel("Correct Decisions (%)")
    ax.set_title("Train vs Validation Accuracy")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"accuracy.{fmt}")
    plt.close(fig)
    print(f"  Saved accuracy.{fmt}")

    # ── 7. DG-specific: delight and gate curves ─────────────────────
    has_dg = any("delight" in r for r in rollouts)
    if has_dg:
        delight_vals = np.array([r.get("delight", 0) for r in rollouts])
        gate_vals = np.array([r.get("gate", 0.5) for r in rollouts])

        fig, (ax_d, ax_g) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("DG-Specific Metrics", fontsize=13, fontweight="bold")

        ax_d.plot(steps_k, delight_vals, alpha=0.25, color="#e67e22", linewidth=0.8)
        ax_d.plot(steps_k, smooth(delight_vals, 10), color="#e67e22", linewidth=2)
        ax_d.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax_d.set_xlabel("Training Steps (×1K)")
        ax_d.set_ylabel("Mean Delight")
        ax_d.set_title("Delight (advantage × surprisal)")
        ax_d.grid(True, alpha=0.3)

        ax_g.plot(steps_k, gate_vals, alpha=0.25, color="#1abc9c", linewidth=0.8)
        ax_g.plot(steps_k, smooth(gate_vals, 10), color="#1abc9c", linewidth=2)
        ax_g.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
        ax_g.set_xlabel("Training Steps (×1K)")
        ax_g.set_ylabel("Mean Gate σ(χ/η)")
        ax_g.set_title("Gate Activation (0.5 = neutral)")
        ax_g.set_ylim(0, 1)
        ax_g.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"dg_metrics.{fmt}")
        plt.close(fig)
        print(f"  Saved dg_metrics.{fmt}")

    # ── 7b. Soft PPO-specific: alpha and alpha_loss curves ──────────
    has_soft_ppo = any("alpha" in r for r in rollouts)
    if has_soft_ppo:
        alpha_vals = np.array([r.get("alpha", 0) for r in rollouts])
        alpha_loss_vals = np.array([r.get("alpha_loss", 0) for r in rollouts])
        target_ent = rollouts[-1].get("target_entropy", 0)

        fig, (ax_a, ax_al) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(
            "Soft PPO — Adaptive Entropy Temperature", fontsize=13, fontweight="bold"
        )

        ax_a.plot(steps_k, alpha_vals, alpha=0.25, color="#e67e22", linewidth=0.8)
        ax_a.plot(steps_k, smooth(alpha_vals, 10), color="#e67e22", linewidth=2)
        ax_a.set_xlabel("Training Steps (×1K)")
        ax_a.set_ylabel("α (entropy temperature)")
        ax_a.set_title(f"Entropy Temperature α (target H={target_ent:.3f})")
        ax_a.grid(True, alpha=0.3)

        ax_al.plot(steps_k, alpha_loss_vals, alpha=0.25, color="#1abc9c", linewidth=0.8)
        ax_al.plot(steps_k, smooth(alpha_loss_vals, 10), color="#1abc9c", linewidth=2)
        ax_al.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax_al.set_xlabel("Training Steps (×1K)")
        ax_al.set_ylabel("α Loss")
        ax_al.set_title("α Dual Loss (>0 → entropy too high)")
        ax_al.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"soft_ppo_metrics.{fmt}")
        plt.close(fig)
        print(f"  Saved soft_ppo_metrics.{fmt}")

    # ── 8. Combined 2×2 summary figure ────────────────────────────────
    algo_name = (
        "Soft-PPO+LSTM" if has_soft_ppo else ("DG+LSTM" if has_dg else "PPO+LSTM")
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"{algo_name} Training Summary", fontsize=15, fontweight="bold", y=0.98
    )

    # reward
    ax = axes[0, 0]
    ax.plot(steps_k, rewards, alpha=0.2, color="#4a90e2", linewidth=0.8)
    ax.plot(steps_k, smooth(rewards, 10), color="#4a90e2", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Steps (×1K)")
    ax.set_ylabel("Avg Reward")
    ax.set_title("(a) Reward")
    ax.grid(True, alpha=0.3)

    # losses
    ax = axes[0, 1]
    ax.plot(
        steps_k, smooth(policy_loss, 10), color="#e74c3c", linewidth=1.8, label="Policy"
    )
    ax_v = ax.twinx()
    ax_v.plot(
        steps_k, smooth(value_loss, 10), color="#2ecc71", linewidth=1.8, label="Value"
    )
    ax.set_xlabel("Steps (×1K)")
    ax.set_ylabel("Policy Loss", color="#e74c3c")
    ax_v.set_ylabel("Value Loss", color="#2ecc71")
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax_v.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right", fontsize=9)
    ax.set_title("(b) Losses")
    ax.grid(True, alpha=0.3)

    # entropy
    ax = axes[1, 0]
    ax.plot(steps_k, smooth(entropy, 10), color="#9b59b6", linewidth=2)
    ax.set_xlabel("Steps (×1K)")
    ax.set_ylabel("Entropy")
    ax.set_title("(c) Policy Entropy")
    ax.grid(True, alpha=0.3)

    # accuracy (train vs val)
    ax = axes[1, 1]
    ax.plot(
        steps_k, smooth(correct_arr, 10), color="#27ae60", linewidth=2, label="Train"
    )
    if val_steps_k:
        ax.plot(
            val_steps_k,
            val_accs,
            "o-",
            color="#e74c3c",
            linewidth=1.8,
            markersize=3,
            label="Validation",
        )
    ax.set_xlabel("Steps (×1K)")
    ax.set_ylabel("Correct (%)")
    ax.set_title("(d) Train vs Validation Accuracy")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"training_summary.{fmt}")
    plt.close(fig)
    print(f"  Saved training_summary.{fmt}")

    print(f"\nDone! {len(rollouts)} rollouts parsed, figures saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PPO training logs for research papers"
    )
    parser.add_argument(
        "--log", type=str, default="training.log", help="Path to training.log"
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
    rollouts = parse_log(str(log_path))

    if not rollouts:
        print("No rollout data found in log file.")
        return

    print(
        f"Found {len(rollouts)} rollouts ({rollouts[0]['steps']} – {rollouts[-1]['steps']} steps)"
    )
    plot_all(rollouts, Path(args.out), fmt=args.format)


if __name__ == "__main__":
    main()
