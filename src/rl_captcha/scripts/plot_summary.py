"""Generate a concise 1×2 summary: training accuracy + eval confusion matrix.

Usage:
    python -m rl_captcha.scripts.plot_summary \
        --train-log training.log --eval-log eval.log --out figures/summary.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from rl_captcha.scripts.plot_training import parse_log as parse_train_log, smooth
from rl_captcha.scripts.plot_eval import parse_log as parse_eval_log


def build_summary(train_rollouts: list[dict], eval_result: dict, out_path: Path):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
        }
    )

    fig, (ax_acc, ax_cm) = plt.subplots(2, 1, figsize=(7, 10))
    fig.suptitle("PPO+LSTM Bot Detection", fontsize=15, fontweight="bold", y=0.98)

    # ── Left: Train vs Val Accuracy ──────────────────────────────
    steps_k = np.array([r["steps"] for r in train_rollouts]) / 1000

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

    ax_acc.plot(steps_k, correct_arr, alpha=0.15, color="#27ae60", linewidth=0.8)
    ax_acc.plot(
        steps_k, smooth(correct_arr, 10), color="#27ae60", linewidth=2.2, label="Train"
    )

    val_steps_k, val_accs = [], []
    for r in train_rollouts:
        if "val_accuracy" in r:
            val_steps_k.append(r["steps"] / 1000)
            val_accs.append(r["val_accuracy"] * 100)
    if val_steps_k:
        ax_acc.plot(
            val_steps_k,
            val_accs,
            "o-",
            color="#e74c3c",
            linewidth=2,
            markersize=4,
            label="Validation",
        )

    ax_acc.set_xlabel("Training Steps (×1K)")
    ax_acc.set_ylabel("Correct Decisions (%)")
    ax_acc.set_title("Train vs Validation Accuracy")
    ax_acc.set_ylim(0, 105)
    ax_acc.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    # ── Right: Confusion Matrix ──────────────────────────────────
    split_name = eval_result.get("split", "test").upper()
    tp = eval_result.get("tp", 0)
    tn = eval_result.get("tn", 0)
    fp = eval_result.get("fp", 0)
    fn = eval_result.get("fn", 0)
    total = tp + tn + fp + fn or 1
    cm = np.array([[tn, fp], [fn, tp]])
    cm_pct = cm / total * 100

    im = ax_cm.imshow(cm_pct, cmap="Blues", vmin=0, vmax=cm_pct.max() * 1.2)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Predicted\nHuman", "Predicted\nBot"])
    ax_cm.set_yticklabels(["Actual\nHuman", "Actual\nBot"])
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = cm_pct[i, j]
            color = "white" if pct > cm_pct.max() * 0.6 else "black"
            ax_cm.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=color,
            )
    ax_cm.set_title(f"Confusion Matrix — {split_name} Split")
    fig.colorbar(im, ax=ax_cm, label="% of episodes", shrink=0.8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved summary to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Concise 2-panel training+eval summary"
    )
    parser.add_argument("--train-log", type=str, required=True)
    parser.add_argument("--eval-log", type=str, required=True)
    parser.add_argument("--out", type=str, default="figures/summary.png")
    args = parser.parse_args()

    train_path = Path(args.train_log)
    eval_path = Path(args.eval_log)

    if not train_path.exists():
        print(f"Error: {train_path} not found")
        return
    if not eval_path.exists():
        print(f"Error: {eval_path} not found")
        return

    rollouts = parse_train_log(str(train_path))
    if not rollouts:
        print("No training data found.")
        return

    eval_result = parse_eval_log(str(eval_path))
    if not eval_result.get("accuracy"):
        print("No evaluation data found.")
        return

    build_summary(rollouts, eval_result, Path(args.out))


if __name__ == "__main__":
    main()
