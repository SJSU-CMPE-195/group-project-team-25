"""Generate a single combined PNG with all training + evaluation graphs.

Usage:
    python -m rl_captcha.scripts.plot_combined \
        --train-log training.log --eval-log eval.log --out figures/combined.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from rl_captcha.scripts.plot_training import parse_log as parse_train_log, smooth
from rl_captcha.scripts.plot_eval import parse_log as parse_eval_log


def build_combined(train_rollouts: list[dict], eval_result: dict, out_path: Path):
    """Build a single 2×4 figure: top row = training, bottom row = eval."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        "PPO+LSTM Bot Detection — Training & Evaluation Summary",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    steps = np.array([r["steps"] for r in train_rollouts])
    steps_k = steps / 1000

    # ═══════════════════ TOP ROW: TRAINING ═══════════════════

    # (a) Reward
    ax = axes[0, 0]
    rewards = np.array([r.get("avg_reward", 0) for r in train_rollouts])
    ax.plot(steps_k, rewards, alpha=0.2, color="#4a90e2", linewidth=0.8)
    ax.plot(steps_k, smooth(rewards, 10), color="#4a90e2", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Steps (×1K)")
    ax.set_ylabel("Avg Reward")
    ax.set_title("(a) Training Reward")
    ax.grid(True, alpha=0.3)

    # (b) Losses
    ax = axes[0, 1]
    policy_loss = np.array([r.get("policy_loss", 0) for r in train_rollouts])
    value_loss = np.array([r.get("value_loss", 0) for r in train_rollouts])
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
    ax.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right", fontsize=8)
    ax.set_title("(b) Losses")
    ax.grid(True, alpha=0.3)

    # (c) Entropy
    ax = axes[0, 2]
    entropy = np.array([r.get("entropy", 0) for r in train_rollouts])
    ax.plot(steps_k, smooth(entropy, 10), color="#9b59b6", linewidth=2)
    ax.set_xlabel("Steps (×1K)")
    ax.set_ylabel("Entropy")
    ax.set_title("(c) Policy Entropy")
    ax.grid(True, alpha=0.3)

    # (d) Train vs Val Accuracy
    ax = axes[0, 3]
    correct_pcts = []
    for r in train_rollouts:
        oc = r.get("outcomes", {})
        correct = (
            oc.get("correct_allow", 0)
            + oc.get("correct_block", 0)
            + oc.get("bot_blocked_puzzle", 0)
        )
        correct_pcts.append(correct)
    correct_arr = np.array(correct_pcts)
    ax.plot(
        steps_k, smooth(correct_arr, 10), color="#27ae60", linewidth=2, label="Train"
    )
    val_steps_k = []
    val_accs = []
    for r in train_rollouts:
        if "val_accuracy" in r:
            val_steps_k.append(r["steps"] / 1000)
            val_accs.append(r["val_accuracy"] * 100)
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
    ax.set_title("(d) Train vs Val Accuracy")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ═══════════════════ BOTTOM ROW: EVALUATION ═══════════════════

    split_name = eval_result.get("split", "test").upper()
    tp = eval_result.get("tp", 0)
    tn = eval_result.get("tn", 0)
    fp = eval_result.get("fp", 0)
    fn = eval_result.get("fn", 0)
    total = tp + tn + fp + fn or 1
    cm = np.array([[tn, fp], [fn, tp]])
    cm_pct = cm / total * 100

    # (e) Confusion Matrix
    ax = axes[1, 0]
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
                fontsize=11,
                fontweight="bold",
                color=color,
            )
    ax.set_title(f"(e) Confusion Matrix ({split_name})")

    # (f) Metrics
    ax = axes[1, 1]
    metrics = {}
    for name in ["accuracy", "precision", "recall", "f1"]:
        if name in eval_result:
            metrics[name.capitalize()] = eval_result[name]
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
                fontsize=10,
                fontweight="bold",
            )
        ax.set_ylim(0, 1.15)
    ax.set_title(f"(f) Eval Metrics ({split_name})")
    ax.grid(True, axis="y", alpha=0.3)

    # (g) Action Distribution
    ax = axes[1, 2]
    actions = eval_result.get("actions", {})
    action_colors = {
        "allow": "#2ecc71",
        "block": "#e74c3c",
        "easy_puzzle": "#f1c40f",
        "medium_puzzle": "#e67e22",
        "hard_puzzle": "#d35400",
        "continue": "#95a5a6",
        "deploy_honeypot": "#3498db",
    }
    if actions:
        action_names = list(actions.keys())
        action_counts = [actions[a] for a in action_names]
        colors = [action_colors.get(a, "#bdc3c7") for a in action_names]
        ax.barh(action_names, action_counts, color=colors)
    ax.set_xlabel("Count")
    ax.set_title(f"(g) Final Actions ({split_name})")
    ax.grid(True, axis="x", alpha=0.3)

    # (h) Decision Timing
    ax = axes[1, 3]
    human_steps = eval_result.get("human_avg_steps")
    bot_steps = eval_result.get("bot_avg_steps")
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
    ax.set_title(f"(h) Decision Timing ({split_name})")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved combined figure to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Combined training + eval figure")
    parser.add_argument(
        "--train-log", type=str, required=True, help="Path to training.log"
    )
    parser.add_argument("--eval-log", type=str, required=True, help="Path to eval.log")
    parser.add_argument(
        "--out",
        type=str,
        default="figures/combined.png",
        help="Output file path (default: figures/combined.png)",
    )
    args = parser.parse_args()

    train_path = Path(args.train_log)
    eval_path = Path(args.eval_log)

    if not train_path.exists():
        print(f"Error: {train_path} not found")
        return
    if not eval_path.exists():
        print(f"Error: {eval_path} not found")
        return

    print(f"Parsing {train_path}...")
    rollouts = parse_train_log(str(train_path))
    if not rollouts:
        print("No training data found.")
        return

    print(f"Parsing {eval_path}...")
    eval_result = parse_eval_log(str(eval_path))
    if not eval_result.get("accuracy"):
        print("No evaluation data found.")
        return

    print(
        f"Training: {len(rollouts)} rollouts | Eval: acc={eval_result.get('accuracy', 0):.3f}"
    )
    build_combined(rollouts, eval_result, Path(args.out))


if __name__ == "__main__":
    main()
