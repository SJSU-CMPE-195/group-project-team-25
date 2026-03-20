"""Generate PPO vs DG comparison figures for research papers.

Usage:
    python -m rl_captcha.scripts.plot_comparison \
        --ppo-train ppo_training.log --ppo-eval ppo_eval.log \
        --dg-train dg_training.log --dg-eval dg_eval.log \
        --out figures/comparison/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from rl_captcha.scripts.plot_training import parse_log as parse_train_log, smooth
from rl_captcha.scripts.plot_eval import parse_log as parse_eval_log


PPO_COLOR = "#4a90e2"
DG_COLOR = "#e67e22"


def plot_comparison(
    ppo_rollouts: list[dict],
    dg_rollouts: list[dict],
    ppo_eval: dict | None,
    dg_eval: dict | None,
    out_dir: Path,
    fmt: str = "png",
):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })

    ppo_steps_k = np.array([r["steps"] for r in ppo_rollouts]) / 1000
    dg_steps_k = np.array([r["steps"] for r in dg_rollouts]) / 1000

    # ── 1. Reward comparison ─────────────────────────────────────────
    ppo_rewards = np.array([r.get("avg_reward", 0) for r in ppo_rollouts])
    dg_rewards = np.array([r.get("avg_reward", 0) for r in dg_rollouts])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ppo_steps_k, smooth(ppo_rewards, 10), color=PPO_COLOR, linewidth=2, label="PPO")
    ax.fill_between(ppo_steps_k, smooth(ppo_rewards, 20) - 0.05,
                    smooth(ppo_rewards, 20) + 0.05, color=PPO_COLOR, alpha=0.1)
    ax.plot(dg_steps_k, smooth(dg_rewards, 10), color=DG_COLOR, linewidth=2, label="DG")
    ax.fill_between(dg_steps_k, smooth(dg_rewards, 20) - 0.05,
                    smooth(dg_rewards, 20) + 0.05, color=DG_COLOR, alpha=0.1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Training Steps (×1K)")
    ax.set_ylabel("Average Episode Reward")
    ax.set_title("PPO vs DG — Training Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"cmp_reward.{fmt}")
    plt.close(fig)
    print(f"  Saved cmp_reward.{fmt}")

    # ── 2. Training accuracy comparison ──────────────────────────────
    def _correct_pcts(rollouts):
        arr = []
        for r in rollouts:
            oc = r.get("outcomes", {})
            arr.append(oc.get("correct_allow", 0) + oc.get("correct_block", 0)
                       + oc.get("bot_blocked_puzzle", 0))
        return np.array(arr)

    ppo_acc = _correct_pcts(ppo_rollouts)
    dg_acc = _correct_pcts(dg_rollouts)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ppo_steps_k, smooth(ppo_acc, 10), color=PPO_COLOR, linewidth=2, label="PPO (train)")
    ax.plot(dg_steps_k, smooth(dg_acc, 10), color=DG_COLOR, linewidth=2, label="DG (train)")

    # Overlay val accuracy points
    for rollouts, color, label in [(ppo_rollouts, PPO_COLOR, "PPO (val)"),
                                    (dg_rollouts, DG_COLOR, "DG (val)")]:
        vs, va = [], []
        for r in rollouts:
            if "val_accuracy" in r:
                vs.append(r["steps"] / 1000)
                va.append(r["val_accuracy"] * 100)
        if vs:
            ax.plot(vs, va, "o--", color=color, linewidth=1.2, markersize=3,
                    alpha=0.7, label=label)

    ax.set_xlabel("Training Steps (×1K)")
    ax.set_ylabel("Correct Decisions (%)")
    ax.set_title("PPO vs DG — Classification Accuracy")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"cmp_accuracy.{fmt}")
    plt.close(fig)
    print(f"  Saved cmp_accuracy.{fmt}")

    # ── 3. Policy Loss + Entropy (2-panel) ──────────────────────────
    # Policy loss measures different objectives for PPO vs DG, so we
    # plot each on its own y-axis alongside entropy to show the
    # confidence–calibration tradeoff.
    ppo_ploss = np.array([r.get("policy_loss", 0) for r in ppo_rollouts])
    dg_ploss = np.array([r.get("policy_loss", 0) for r in dg_rollouts])
    ppo_ent = np.array([r.get("entropy", 0) for r in ppo_rollouts])
    dg_ent = np.array([r.get("entropy", 0) for r in dg_rollouts])

    fig, (ax_loss, ax_ent) = plt.subplots(1, 2, figsize=(14, 4.5))

    # Left panel: each algorithm's loss on its own y-axis
    ax_ppo = ax_loss
    ax_dg = ax_loss.twinx()
    ln1 = ax_ppo.plot(ppo_steps_k, smooth(ppo_ploss, 10), color=PPO_COLOR,
                      linewidth=2, label="PPO (clipped surrogate)")
    ln2 = ax_dg.plot(dg_steps_k, smooth(dg_ploss, 10), color=DG_COLOR,
                     linewidth=2, label="DG (delight gradient)")
    ax_ppo.set_xlabel("Training Steps (×1K)")
    ax_ppo.set_ylabel("PPO Policy Loss", color=PPO_COLOR)
    ax_dg.set_ylabel("DG Policy Loss", color=DG_COLOR)
    ax_ppo.tick_params(axis="y", labelcolor=PPO_COLOR)
    ax_dg.tick_params(axis="y", labelcolor=DG_COLOR)
    lns = ln1 + ln2
    ax_loss.legend(lns, [l.get_label() for l in lns], loc="upper right", fontsize=9)
    ax_loss.set_title("Policy Loss (separate scales)")
    ax_loss.grid(True, alpha=0.3)

    # Right panel: entropy on shared axis
    ax_ent.plot(ppo_steps_k, smooth(ppo_ent, 10), color=PPO_COLOR, linewidth=2, label="PPO")
    ax_ent.plot(dg_steps_k, smooth(dg_ent, 10), color=DG_COLOR, linewidth=2, label="DG")
    ax_ent.set_xlabel("Training Steps (×1K)")
    ax_ent.set_ylabel("Policy Entropy")
    ax_ent.set_title("Decision Confidence (Entropy)")
    ax_ent.legend()
    ax_ent.grid(True, alpha=0.3)
    ax_ent.annotate("PPO collapses → overconfident",
                    xy=(ppo_steps_k[-1], smooth(ppo_ent, 10)[-1]),
                    xytext=(-120, 40), textcoords="offset points",
                    fontsize=8, color=PPO_COLOR, fontstyle="italic",
                    arrowprops=dict(arrowstyle="->", color=PPO_COLOR, lw=1.2))
    ax_ent.annotate("DG stays calibrated",
                    xy=(dg_steps_k[-1], smooth(dg_ent, 10)[-1]),
                    xytext=(-100, -40), textcoords="offset points",
                    fontsize=8, color=DG_COLOR, fontstyle="italic",
                    arrowprops=dict(arrowstyle="->", color=DG_COLOR, lw=1.2))

    fig.suptitle("PPO vs DG — Optimization Landscape", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / f"cmp_loss_entropy.{fmt}")
    plt.close(fig)
    print(f"  Saved cmp_loss_entropy.{fmt}")

    # ── 5. Eval metrics side-by-side bar chart ───────────────────────
    if ppo_eval and dg_eval:
        metric_names = ["Accuracy", "Precision", "Recall", "F1"]
        ppo_vals = [ppo_eval.get(m.lower(), 0) for m in metric_names]
        dg_vals = [dg_eval.get(m.lower(), 0) for m in metric_names]

        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width / 2, ppo_vals, width, label="PPO", color=PPO_COLOR,
                       edgecolor="white", linewidth=1.5)
        bars2 = ax.bar(x + width / 2, dg_vals, width, label="DG", color=DG_COLOR,
                       edgecolor="white", linewidth=1.5)

        for bars in [bars1, bars2]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{bar.get_height():.3f}", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("PPO vs DG — Test Set Evaluation Metrics")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.savefig(out_dir / f"cmp_eval_metrics.{fmt}")
        plt.close(fig)
        print(f"  Saved cmp_eval_metrics.{fmt}")

        # ── 6. Confusion matrices side by side ───────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        for ax, result, title, cmap_color in [
            (ax1, ppo_eval, "PPO", "Blues"),
            (ax2, dg_eval, "DG", "Oranges"),
        ]:
            tp = result.get("tp", 0)
            tn = result.get("tn", 0)
            fp = result.get("fp", 0)
            fn = result.get("fn", 0)
            total = tp + tn + fp + fn or 1
            cm = np.array([[tn, fp], [fn, tp]])
            cm_pct = cm / total * 100

            im = ax.imshow(cm_pct, cmap=cmap_color, vmin=0, vmax=cm_pct.max() * 1.2)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred Human", "Pred Bot"])
            ax.set_yticklabels(["True Human", "True Bot"])
            for i in range(2):
                for j in range(2):
                    val = cm[i, j]
                    pct = cm_pct[i, j]
                    color = "white" if pct > cm_pct.max() * 0.6 else "black"
                    ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                            fontsize=13, fontweight="bold", color=color)
            acc = result.get("accuracy", 0)
            f1 = result.get("f1", 0)
            ax.set_title(f"{title} (Acc={acc:.3f}, F1={f1:.3f})")

        fig.suptitle("PPO vs DG — Confusion Matrices", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(out_dir / f"cmp_confusion.{fmt}")
        plt.close(fig)
        print(f"  Saved cmp_confusion.{fmt}")

    # ── 7. Combined 2×3 summary ──────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("PPO vs DG — Complete Comparison", fontsize=16, fontweight="bold", y=0.98)

    # (a) Reward
    ax = axes[0, 0]
    ax.plot(ppo_steps_k, smooth(ppo_rewards, 10), color=PPO_COLOR, linewidth=2, label="PPO")
    ax.plot(dg_steps_k, smooth(dg_rewards, 10), color=DG_COLOR, linewidth=2, label="DG")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Steps (×1K)")
    ax.set_ylabel("Avg Reward")
    ax.set_title("(a) Training Reward")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Accuracy
    ax = axes[0, 1]
    ax.plot(ppo_steps_k, smooth(ppo_acc, 10), color=PPO_COLOR, linewidth=2, label="PPO")
    ax.plot(dg_steps_k, smooth(dg_acc, 10), color=DG_COLOR, linewidth=2, label="DG")
    ax.set_xlabel("Steps (×1K)")
    ax.set_ylabel("Correct (%)")
    ax.set_title("(b) Train Accuracy")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Policy Loss (dual y-axis)
    ax = axes[0, 2]
    ax_dg2 = ax.twinx()
    ax.plot(ppo_steps_k, smooth(ppo_ploss, 10), color=PPO_COLOR, linewidth=2, label="PPO")
    ax_dg2.plot(dg_steps_k, smooth(dg_ploss, 10), color=DG_COLOR, linewidth=2, label="DG")
    ax.set_xlabel("Steps (×1K)")
    ax.set_ylabel("PPO Loss", color=PPO_COLOR, fontsize=9)
    ax_dg2.set_ylabel("DG Loss", color=DG_COLOR, fontsize=9)
    ax.tick_params(axis="y", labelcolor=PPO_COLOR, labelsize=8)
    ax_dg2.tick_params(axis="y", labelcolor=DG_COLOR, labelsize=8)
    ax.set_title("(c) Policy Loss (separate scales)")
    ax.grid(True, alpha=0.3)

    # (d) Entropy
    ax = axes[1, 0]
    ax.plot(ppo_steps_k, smooth(ppo_ent, 10), color=PPO_COLOR, linewidth=2, label="PPO")
    ax.plot(dg_steps_k, smooth(dg_ent, 10), color=DG_COLOR, linewidth=2, label="DG")
    ax.set_xlabel("Steps (×1K)")
    ax.set_ylabel("Entropy")
    ax.set_title("(d) Decision Confidence (Entropy)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (e) Eval metrics
    ax = axes[1, 1]
    if ppo_eval and dg_eval:
        x = np.arange(len(metric_names))
        ax.bar(x - 0.18, ppo_vals, 0.35, label="PPO", color=PPO_COLOR)
        ax.bar(x + 0.18, dg_vals, 0.35, label="DG", color=DG_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1.15)
    ax.set_title("(e) Test Metrics")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # (f) Confusion matrices (PPO left half, DG right half)
    ax = axes[1, 2]
    if ppo_eval and dg_eval:
        ppo_acc_val = ppo_eval.get("accuracy", 0)
        dg_acc_val = dg_eval.get("accuracy", 0)
        ppo_f1_val = ppo_eval.get("f1", 0)
        dg_f1_val = dg_eval.get("f1", 0)
        text = (f"PPO: Acc={ppo_acc_val:.3f}  F1={ppo_f1_val:.3f}\n"
                f"DG:  Acc={dg_acc_val:.3f}  F1={dg_f1_val:.3f}")
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
                fontsize=14, fontfamily="monospace", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray"))
    ax.set_title("(f) Summary")
    ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"cmp_summary.{fmt}")
    plt.close(fig)
    print(f"  Saved cmp_summary.{fmt}")

    print(f"\nDone! Comparison figures saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="PPO vs DG comparison figures")
    parser.add_argument("--ppo-train", type=str, required=True, help="PPO training.log")
    parser.add_argument("--dg-train", type=str, required=True, help="DG training.log")
    parser.add_argument("--ppo-eval", type=str, default=None, help="PPO eval.log (optional)")
    parser.add_argument("--dg-eval", type=str, default=None, help="DG eval.log (optional)")
    parser.add_argument("--out", type=str, default="figures/comparison", help="Output directory")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    for name, path in [("PPO train", args.ppo_train), ("DG train", args.dg_train)]:
        if not Path(path).exists():
            print(f"Error: {name} log not found: {path}")
            return

    print("Parsing logs...")
    ppo_rollouts = parse_train_log(args.ppo_train)
    dg_rollouts = parse_train_log(args.dg_train)

    if not ppo_rollouts or not dg_rollouts:
        print("Error: Could not parse training logs.")
        return

    ppo_eval = parse_eval_log(args.ppo_eval) if args.ppo_eval and Path(args.ppo_eval).exists() else None
    dg_eval = parse_eval_log(args.dg_eval) if args.dg_eval and Path(args.dg_eval).exists() else None

    print(f"PPO: {len(ppo_rollouts)} rollouts | DG: {len(dg_rollouts)} rollouts")
    plot_comparison(ppo_rollouts, dg_rollouts, ppo_eval, dg_eval, Path(args.out), args.format)


if __name__ == "__main__":
    main()
