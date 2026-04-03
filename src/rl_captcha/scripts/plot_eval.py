"""Parse evaluate_ppo.py output and generate publication-quality evaluation figures.

Usage:
    python -m rl_captcha.scripts.evaluate_ppo --agent ... --data-dir data/ | tee eval.log
    python -m rl_captcha.scripts.plot_eval --log eval.log --out figures/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Regex patterns matching evaluate_ppo.py output ─────────────────────

RE_SPLIT = re.compile(
    r"Evaluating on (\w+) split:\s*(\d+) sessions \((\d+) human, (\d+) bot\)"
)
RE_METRICS = re.compile(r"Accuracy:\s*([\d.]+)")
RE_PRECISION = re.compile(r"Precision:\s*([\d.]+)")
RE_RECALL = re.compile(r"Recall:\s*([\d.]+)")
RE_F1 = re.compile(r"F1:\s*([\d.]+)")

RE_TP = re.compile(r"True Positives.*?:\s*(\d+)")
RE_TN = re.compile(r"True Negatives.*?:\s*(\d+)")
RE_FP = re.compile(r"False Positives.*?:\s*(\d+)")
RE_FN = re.compile(r"False Negatives.*?:\s*(\d+)")
RE_TRUNC = re.compile(r"Truncated.*?:\s*(\d+)")

RE_ACTION_LINE = re.compile(r"^\s+(\w[\w_]+)\s+(\d+)\s+\(([\d.]+)%\)")
RE_OUTCOME_LINE = re.compile(r"^\s+([\w_]+)\s+(\d+)\s+\(([\d.]+)%\)")

RE_HUMAN_STEPS = re.compile(r"Avg steps \(human sessions\):\s*([\d.]+)")
RE_BOT_STEPS = re.compile(r"Avg steps \(bot sessions\):\s*([\d.]+)")


def parse_log(path: str) -> dict:
    """Parse evaluation log into a result dict."""
    result = {}
    actions = {}
    outcomes = {}
    in_actions = False
    in_outcomes = False

    encoding = "utf-8"
    with open(path, "rb") as fb:
        bom = fb.read(2)
        if bom == b"\xff\xfe":
            encoding = "utf-16-le"
        elif bom == b"\xfe\xff":
            encoding = "utf-16-be"

    with open(path, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            raw = line.rstrip()

            m = RE_SPLIT.search(raw)
            if m:
                result["split"] = m.group(1)
                result["total_sessions"] = int(m.group(2))
                result["human_sessions"] = int(m.group(3))
                result["bot_sessions"] = int(m.group(4))

            for name, regex in [
                ("accuracy", RE_METRICS),
                ("precision", RE_PRECISION),
                ("recall", RE_RECALL),
                ("f1", RE_F1),
            ]:
                m = regex.search(raw)
                if m:
                    result[name] = float(m.group(1))

            for name, regex in [
                ("tp", RE_TP),
                ("tn", RE_TN),
                ("fp", RE_FP),
                ("fn", RE_FN),
                ("truncated", RE_TRUNC),
            ]:
                m = regex.search(raw)
                if m:
                    result[name] = int(m.group(1))

            m = RE_HUMAN_STEPS.search(raw)
            if m:
                result["human_avg_steps"] = float(m.group(1))
            m = RE_BOT_STEPS.search(raw)
            if m:
                result["bot_avg_steps"] = float(m.group(1))

            if "Final Action Distribution" in raw:
                in_actions = True
                in_outcomes = False
                continue
            if "Outcome Distribution" in raw:
                in_outcomes = True
                in_actions = False
                continue
            if raw.startswith("---") or raw.startswith("==="):
                in_actions = False
                in_outcomes = False
                continue

            if in_actions:
                m = RE_ACTION_LINE.search(raw)
                if m:
                    actions[m.group(1)] = int(m.group(2))

            if in_outcomes:
                m = RE_OUTCOME_LINE.search(raw)
                if m:
                    outcomes[m.group(1)] = int(m.group(2))

    result["actions"] = actions
    result["outcomes"] = outcomes
    return result


def plot_all(result: dict, out_dir: Path, fmt: str = "png"):
    """Generate evaluation figures."""
    out_dir.mkdir(parents=True, exist_ok=True)

    split_name = result.get("split", "test").upper()

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

    # ── 1. Confusion matrix heatmap ─────────────────────────────────
    tp = result.get("tp", 0)
    tn = result.get("tn", 0)
    fp = result.get("fp", 0)
    fn = result.get("fn", 0)
    total = tp + tn + fp + fn or 1

    cm = np.array([[tn, fp], [fn, tp]])
    cm_pct = cm / total * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=cm_pct.max() * 1.2)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nHuman", "Predicted\nBot"])
    ax.set_yticklabels(["Actual\nHuman", "Actual\nBot"])
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = cm_pct[i, j]
            color = "white" if pct > cm_pct.max() * 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=color,
            )
    ax.set_title(f"Confusion Matrix — {split_name} Split")
    fig.colorbar(im, ax=ax, label="% of episodes", shrink=0.8)
    fig.savefig(out_dir / f"eval_confusion_matrix.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_confusion_matrix.{fmt}")

    # ── 2. Metrics bar chart ────────────────────────────────────────
    metrics = {}
    for name in ["accuracy", "precision", "recall", "f1"]:
        if name in result:
            metrics[name.capitalize()] = result[name]

    if metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(metrics.keys())
        values = [metrics[n] for n in names]
        colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"][: len(names)]
        bars = ax.bar(
            names, values, color=colors, width=0.6, edgecolor="white", linewidth=1.5
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title(f"Evaluation Metrics — {split_name} Split")
        ax.grid(True, axis="y", alpha=0.3)
        fig.savefig(out_dir / f"eval_metrics.{fmt}")
        plt.close(fig)
        print(f"  Saved eval_metrics.{fmt}")

    # ── 3. Final action distribution ────────────────────────────────
    actions = result.get("actions", {})
    if actions:
        fig, ax = plt.subplots(figsize=(7, 4))
        action_names = list(actions.keys())
        action_counts = [actions[a] for a in action_names]
        action_colors = {
            "allow": "#2ecc71",
            "block": "#e74c3c",
            "easy_puzzle": "#f1c40f",
            "medium_puzzle": "#e67e22",
            "hard_puzzle": "#d35400",
            "continue": "#95a5a6",
            "deploy_honeypot": "#3498db",
        }
        colors = [action_colors.get(a, "#bdc3c7") for a in action_names]
        bars = ax.barh(
            action_names, action_counts, color=colors, edgecolor="white", linewidth=1.2
        )
        total_actions = sum(action_counts)
        for bar, count in zip(bars, action_counts):
            pct = count / total_actions * 100 if total_actions else 0
            ax.text(
                bar.get_width() + total_actions * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{count} ({pct:.1f}%)",
                va="center",
                fontsize=10,
            )
        ax.set_xlabel("Count")
        ax.set_title(f"Final Action Distribution — {split_name} Split")
        ax.grid(True, axis="x", alpha=0.3)
        fig.savefig(out_dir / f"eval_actions.{fmt}")
        plt.close(fig)
        print(f"  Saved eval_actions.{fmt}")

    # ── 4. Decision timing by label ─────────────────────────────────
    human_steps = result.get("human_avg_steps")
    bot_steps = result.get("bot_avg_steps")
    if human_steps is not None and bot_steps is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        labels = ["Human", "Bot"]
        values = [human_steps, bot_steps]
        bars = ax.bar(
            labels,
            values,
            color=["#3498db", "#e74c3c"],
            width=0.5,
            edgecolor="white",
            linewidth=1.5,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
        ax.set_ylabel("Avg Windows Before Decision")
        ax.set_title(f"Decision Timing — {split_name} Split")
        ax.grid(True, axis="y", alpha=0.3)
        fig.savefig(out_dir / f"eval_timing.{fmt}")
        plt.close(fig)
        print(f"  Saved eval_timing.{fmt}")

    # ── 5. Combined 2×2 summary ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Evaluation Summary — {split_name} Split",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # (a) confusion matrix
    ax = axes[0, 0]
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=cm_pct.max() * 1.2)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Human", "Pred Bot"])
    ax.set_yticklabels(["True Human", "True Bot"])
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = cm_pct[i, j]
            color = "white" if pct > cm_pct.max() * 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=color,
            )
    ax.set_title("(a) Confusion Matrix")

    # (b) metrics
    ax = axes[0, 1]
    if metrics:
        names = list(metrics.keys())
        values = [metrics[n] for n in names]
        colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"][: len(names)]
        bars = ax.bar(names, values, color=colors, width=0.6)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        ax.set_ylim(0, 1.15)
    ax.set_title("(b) Metrics")
    ax.grid(True, axis="y", alpha=0.3)

    # (c) action distribution
    ax = axes[1, 0]
    if actions:
        action_names = list(actions.keys())
        action_counts = [actions[a] for a in action_names]
        colors = [action_colors.get(a, "#bdc3c7") for a in action_names]
        ax.barh(action_names, action_counts, color=colors)
    ax.set_xlabel("Count")
    ax.set_title("(c) Final Actions")
    ax.grid(True, axis="x", alpha=0.3)

    # (d) timing
    ax = axes[1, 1]
    if human_steps is not None and bot_steps is not None:
        ax.bar(
            ["Human", "Bot"],
            [human_steps, bot_steps],
            color=["#3498db", "#e74c3c"],
            width=0.5,
        )
        for i, val in enumerate([human_steps, bot_steps]):
            ax.text(
                i, val + 0.1, f"{val:.1f}", ha="center", fontsize=11, fontweight="bold"
            )
    ax.set_ylabel("Avg Windows")
    ax.set_title("(d) Decision Timing")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"eval_summary.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_summary.{fmt}")

    print(f"\nDone! Evaluation figures saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Visualize PPO evaluation results")
    parser.add_argument(
        "--log", type=str, required=True, help="Path to evaluation log file"
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
    result = parse_log(str(log_path))

    if not result.get("accuracy"):
        print("No evaluation data found in log file.")
        return

    print(
        f"Split: {result.get('split', 'unknown')} | "
        f"Accuracy: {result.get('accuracy', 0):.3f} | "
        f"F1: {result.get('f1', 0):.3f}"
    )
    plot_all(result, Path(args.out), fmt=args.format)


if __name__ == "__main__":
    main()
