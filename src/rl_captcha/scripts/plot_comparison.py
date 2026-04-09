"""Generate comparison figures across all algorithm variants.

Supports both 3-way (PPO vs DG vs Soft-PPO) and 6-way comparisons
(each algorithm with and without adversarial augmentation).

Usage (6-way comparison):
    python -m rl_captcha.scripts.plot_comparison \
        --logs ppo_noaug=logs/ppo_noaug_training.log \
               ppo_advaug=logs/ppo_advaug_training.log \
               dg_noaug=logs/dg_noaug_training.log \
               dg_advaug=logs/dg_advaug_training.log \
               soft_ppo_noaug=logs/soft_ppo_noaug_training.log \
               soft_ppo_advaug=logs/soft_ppo_advaug_training.log \
        --evals ppo_noaug=logs/ppo_noaug_eval.log \
                ppo_advaug=logs/ppo_advaug_eval.log \
                dg_noaug=logs/dg_noaug_eval.log \
                dg_advaug=logs/dg_advaug_eval.log \
                soft_ppo_noaug=logs/soft_ppo_noaug_eval.log \
                soft_ppo_advaug=logs/soft_ppo_advaug_eval.log \
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

COLORS = {
    "ppo": "#7bb3e0",
    "dg": "#f0a870",
    "soft_ppo": "#8bc7a0",
    "ppo_noaug": "#7bb3e0",
    "dg_noaug": "#f0a870",
    "soft_ppo_noaug": "#8bc7a0",
    "ppo_advaug": "#a0cff0",
    "dg_advaug": "#f5c9a0",
    "soft_ppo_advaug": "#b5dcc5",
}
LABELS = {
    "ppo": "PPO",
    "dg": "DG",
    "soft_ppo": "Soft PPO",
    "ppo_noaug": "PPO (no aug)",
    "dg_noaug": "DG (no aug)",
    "soft_ppo_noaug": "Soft PPO (no aug)",
    "ppo_advaug": "PPO (adv aug)",
    "dg_advaug": "DG (adv aug)",
    "soft_ppo_advaug": "Soft PPO (adv aug)",
}
# Line styles: solid for no-aug, dashed for adv-aug
LINESTYLES = {
    "ppo": "-",
    "dg": "-",
    "soft_ppo": "-",
    "ppo_noaug": "-",
    "dg_noaug": "-",
    "soft_ppo_noaug": "-",
    "ppo_advaug": "--",
    "dg_advaug": "--",
    "soft_ppo_advaug": "--",
}


def _parse_kv_args(args: list[str]) -> dict[str, str]:
    """Parse name=path pairs from CLI args."""
    result = {}
    for arg in args:
        if "=" in arg:
            name, path = arg.split("=", 1)
            result[name.strip()] = path.strip()
        else:
            # Infer name from filename
            p = Path(arg)
            name = p.stem.replace("_training", "").replace("_eval", "")
            result[name] = arg
    return result


def plot_comparison(
    all_rollouts: dict[str, list[dict]],
    all_evals: dict[str, dict] | None,
    out_dir: Path,
    fmt: str = "png",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    algos = list(all_rollouts.keys())

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelcolor": "#444444",
            "legend.fontsize": 9,
            "figure.dpi": 200,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#dddddd",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": "#e0e0e0",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
            "savefig.facecolor": "white",
        }
    )

    # Precompute steps arrays
    steps_k = {}
    for algo in algos:
        steps_k[algo] = np.array([r["steps"] for r in all_rollouts[algo]]) / 1000

    # ── 1. Reward comparison ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo in algos:
        rewards = np.array([r.get("avg_reward", 0) for r in all_rollouts[algo]])
        color = COLORS.get(algo, "#999999")
        label = LABELS.get(algo, algo)
        ls = LINESTYLES.get(algo, "-")
        ax.plot(
            steps_k[algo],
            smooth(rewards, 10),
            color=color,
            linewidth=2,
            label=label,
            linestyle=ls,
        )
        ax.fill_between(
            steps_k[algo],
            smooth(rewards, 20) - 0.05,
            smooth(rewards, 20) + 0.05,
            color=color,
            alpha=0.1,
        )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Training Steps (x1K)")
    ax.set_ylabel("Average Episode Reward")
    ax.set_title("Training Reward Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"cmp_reward.{fmt}")
    plt.close(fig)
    print(f"  Saved cmp_reward.{fmt}")

    # ── 2. Validation accuracy comparison ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo in algos:
        vs, va = [], []
        for r in all_rollouts[algo]:
            if "val_accuracy" in r:
                vs.append(r["steps"] / 1000)
                va.append(r["val_accuracy"] * 100)
        if vs:
            color = COLORS.get(algo, "#999999")
            label = LABELS.get(algo, algo)
            ls = LINESTYLES.get(algo, "-")
            ax.plot(
                vs,
                va,
                "o-",
                color=color,
                linewidth=1.8,
                markersize=3,
                label=label,
                linestyle=ls,
            )

    ax.set_xlabel("Training Steps (x1K)")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Accuracy During Training")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"cmp_accuracy.{fmt}")
    plt.close(fig)
    print(f"  Saved cmp_accuracy.{fmt}")

    # ── 3. Entropy comparison ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo in algos:
        ent = np.array([r.get("entropy", 0) for r in all_rollouts[algo]])
        color = COLORS.get(algo, "#999999")
        label = LABELS.get(algo, algo)
        ls = LINESTYLES.get(algo, "-")
        ax.plot(
            steps_k[algo],
            smooth(ent, 10),
            color=color,
            linewidth=2,
            label=label,
            linestyle=ls,
        )
    ax.set_xlabel("Training Steps (x1K)")
    ax.set_ylabel("Policy Entropy")
    ax.set_title("Decision Confidence (Entropy)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"cmp_entropy.{fmt}")
    plt.close(fig)
    print(f"  Saved cmp_entropy.{fmt}")

    # ── 4. Training Correct Decision Rate ──────────────────────────
    def _correct_pcts(rollouts):
        arr = []
        for r in rollouts:
            oc = r.get("outcomes", {})
            arr.append(
                oc.get("correct_allow", 0)
                + oc.get("correct_block", 0)
                + oc.get("bot_blocked_puzzle", 0)
            )
        return np.array(arr)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for algo in algos:
        acc = _correct_pcts(all_rollouts[algo])
        color = COLORS.get(algo, "#999999")
        label = LABELS.get(algo, algo)
        ls = LINESTYLES.get(algo, "-")
        ax.plot(
            steps_k[algo],
            smooth(acc, 10),
            color=color,
            linewidth=2,
            label=label,
            linestyle=ls,
        )
    ax.set_xlabel("Training Steps (x1K)")
    ax.set_ylabel("Correct Decisions (%)")
    ax.set_title("Training Correct Decision Rate")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"cmp_train_decisions.{fmt}")
    plt.close(fig)
    print(f"  Saved cmp_train_decisions.{fmt}")

    # ── 5. Eval metrics bar chart ────────────────────────────────────
    if all_evals:
        eval_algos = [a for a in algos if a in all_evals]
        if eval_algos:
            metric_names = ["Accuracy", "Precision", "Recall", "F1"]
            x = np.arange(len(metric_names))
            n_algos = len(eval_algos)
            width = 0.8 / n_algos

            fig, ax = plt.subplots(figsize=(9, 5))
            for i, algo in enumerate(eval_algos):
                vals = [all_evals[algo].get(m.lower(), 0) for m in metric_names]
                color = COLORS.get(algo, "#999999")
                label = LABELS.get(algo, algo)
                offset = (i - (n_algos - 1) / 2) * width
                bars = ax.bar(
                    x + offset,
                    vals,
                    width,
                    label=label,
                    color=color,
                    edgecolor="white",
                    linewidth=1.5,
                )
                for bar in bars:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{bar.get_height():.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(metric_names)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Score")
            ax.set_title("Test Set Evaluation Metrics")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            fig.savefig(out_dir / f"cmp_eval_metrics.{fmt}")
            plt.close(fig)
            print(f"  Saved cmp_eval_metrics.{fmt}")

            # ── 6. Confusion matrices (2-row grid, pastel blue) ─────────
            from matplotlib.colors import LinearSegmentedColormap

            teal_cmap = LinearSegmentedColormap.from_list(
                "pastel_blue", ["#f0f4f8", "#c6ddf0", "#8ab8d8", "#5a9bc5"], N=256
            )

            n_eval = len(eval_algos)
            if n_eval > 3:
                n_rows, n_cols = 2, (n_eval + 1) // 2
            else:
                n_rows, n_cols = 1, n_eval
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows)
            )
            axes_flat = np.array(axes).flatten()

            for idx, algo in enumerate(eval_algos):
                ax = axes_flat[idx]
                result = all_evals[algo]
                tp = result.get("tp", 0)
                tn = result.get("tn", 0)
                fp = result.get("fp", 0)
                fn = result.get("fn", 0)
                cm = np.array([[tp, fn], [fp, tn]])

                im = ax.imshow(cm, cmap=teal_cmap, vmin=0, vmax=max(cm.max() * 1.2, 1))
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(["Bot", "Human"], fontsize=10)
                ax.set_yticklabels(["Bot", "Human"], fontsize=10)
                if idx % n_cols == 0:
                    ax.set_ylabel("Actual", fontsize=10, color="#555555")
                ax.set_xlabel("Predicted", fontsize=10, color="#555555")
                for i in range(2):
                    for j in range(2):
                        val = cm[i, j]
                        color = "#222222" if val < cm.max() * 0.6 else "white"
                        ax.text(
                            j,
                            i,
                            f"{val}",
                            ha="center",
                            va="center",
                            fontsize=16,
                            fontweight="bold",
                            color=color,
                        )
                acc = result.get("accuracy", 0)
                label = LABELS.get(algo, algo)
                ax.set_title(
                    f"{label}\nAcc={acc:.3f}",
                    fontsize=11,
                    fontweight="bold",
                    color="#444444",
                    pad=10,
                )

            for idx in range(n_eval, len(axes_flat)):
                axes_flat[idx].axis("off")

            fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
            fig.tight_layout()
            fig.savefig(out_dir / f"cmp_confusion.{fmt}")
            plt.close(fig)
            print(f"  Saved cmp_confusion.{fmt}")

    # ── 7. Combined summary (2×2 training-only) ─────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    fig.suptitle(
        "Algorithm Training Comparison",
        fontsize=16,
        fontweight="bold",
        color="#333333",
        y=0.98,
    )

    # (a) Reward
    ax = axes[0, 0]
    for algo in algos:
        rewards = np.array([r.get("avg_reward", 0) for r in all_rollouts[algo]])
        ax.plot(
            steps_k[algo],
            smooth(rewards, 10),
            color=COLORS.get(algo),
            linewidth=2,
            label=LABELS.get(algo, algo),
            linestyle=LINESTYLES.get(algo, "-"),
        )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Steps (x1K)")
    ax.set_ylabel("Avg Reward")
    ax.set_title("(a) Training Reward")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Validation Accuracy
    ax = axes[0, 1]
    for algo in algos:
        vs, va = [], []
        for r in all_rollouts[algo]:
            if "val_accuracy" in r:
                vs.append(r["steps"] / 1000)
                va.append(r["val_accuracy"] * 100)
        if vs:
            ax.plot(
                vs,
                va,
                "o-",
                color=COLORS.get(algo),
                linewidth=1.8,
                markersize=3,
                label=LABELS.get(algo, algo),
                linestyle=LINESTYLES.get(algo, "-"),
            )
    ax.set_xlabel("Steps (x1K)")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("(b) Validation Accuracy")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Policy Entropy
    ax = axes[1, 0]
    for algo in algos:
        ent = np.array([r.get("entropy", 0) for r in all_rollouts[algo]])
        ax.plot(
            steps_k[algo],
            smooth(ent, 10),
            color=COLORS.get(algo),
            linewidth=2,
            label=LABELS.get(algo, algo),
            linestyle=LINESTYLES.get(algo, "-"),
        )
    ax.set_xlabel("Steps (x1K)")
    ax.set_ylabel("Entropy")
    ax.set_title("(c) Policy Entropy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Training Correct Decision Rate
    ax = axes[1, 1]
    for algo in algos:
        acc = _correct_pcts(all_rollouts[algo])
        ax.plot(
            steps_k[algo],
            smooth(acc, 10),
            color=COLORS.get(algo),
            linewidth=2,
            label=LABELS.get(algo, algo),
            linestyle=LINESTYLES.get(algo, "-"),
        )
    ax.set_xlabel("Steps (x1K)")
    ax.set_ylabel("Correct Decisions (%)")
    ax.set_title("(d) Training Correct Decision Rate")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"cmp_summary.{fmt}")
    plt.close(fig)
    print(f"  Saved cmp_summary.{fmt}")

    print(f"\nDone! Comparison figures saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Algorithm comparison figures (with/without adversarial augmentation)"
    )
    parser.add_argument(
        "--logs",
        type=str,
        nargs="+",
        required=True,
        help="Training logs as name=path pairs (e.g. ppo=logs/ppo_training.log)",
    )
    parser.add_argument(
        "--evals",
        type=str,
        nargs="+",
        default=None,
        help="Eval logs as name=path pairs (optional)",
    )
    parser.add_argument(
        "--out", type=str, default="figures/comparison", help="Output directory"
    )
    parser.add_argument(
        "--format", type=str, default="png", choices=["png", "pdf", "svg"]
    )
    args = parser.parse_args()

    log_map = _parse_kv_args(args.logs)
    for name, path in log_map.items():
        if not Path(path).exists():
            print(f"Error: {name} training log not found: {path}")
            return

    print("Parsing training logs...")
    all_rollouts = {}
    for name, path in log_map.items():
        rollouts = parse_train_log(path)
        if not rollouts:
            print(f"Warning: No rollout data found in {path}")
            continue
        all_rollouts[name] = rollouts
        print(f"  {name}: {len(rollouts)} rollouts")

    if not all_rollouts:
        print("Error: No valid training logs parsed.")
        return

    all_evals = None
    if args.evals:
        eval_map = _parse_kv_args(args.evals)
        all_evals = {}
        for name, path in eval_map.items():
            if Path(path).exists():
                parsed = parse_eval_log(path)
                if parsed:
                    all_evals[name] = parsed
                    print(f"  {name} eval: loaded")

    plot_comparison(all_rollouts, all_evals, Path(args.out), args.format)


if __name__ == "__main__":
    main()
