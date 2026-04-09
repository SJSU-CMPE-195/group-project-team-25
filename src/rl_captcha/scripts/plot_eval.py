"""Parse evaluate_ppo.py output and generate evaluation figures.

Supports both single-agent and multi-agent evaluation logs.
When multiple agents are detected, generates per-agent plots plus
side-by-side comparison figures including per-tier and per-family breakdowns.

Usage:
    # Run eval with Tee-Object to capture log
    python -m rl_captcha.scripts.evaluate_ppo \
        --agent ppo=rl_captcha/agent/checkpoints/ppo_run1 \
               dg=rl_captcha/agent/checkpoints/dg_run1 \
               soft_ppo=rl_captcha/agent/checkpoints/soft_ppo_run1 \
        --episodes 500 --split test \
        2>&1 | Tee-Object -FilePath logs/eval_all.log

    # Plot
    python -m rl_captcha.scripts.plot_eval --log logs/eval_all.log --out figures/eval
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Regex patterns matching evaluate_ppo.py output ─────────────────────

RE_AGENT_HEADER = re.compile(r"Loading agent:\s*(\S+)\s+\(")
RE_AGENT_SECTION = re.compile(
    r"===\s*(\S+)\s*[-\u2014]\s*(\w+)\s+split\s+\((\d+)\s+episodes\)"
)
RE_SPLIT = re.compile(
    r"Evaluating on (\w+) split:\s*(\d+) sessions \((\d+) human, (\d+) bot\)"
)

RE_ACCURACY = re.compile(r"Accuracy:\s*([\d.]+)")
RE_PRECISION = re.compile(r"Precision:\s*([\d.]+)")
RE_RECALL = re.compile(r"Recall:\s*([\d.]+)")
RE_F1 = re.compile(r"F1:\s*([\d.]+)")

RE_TP = re.compile(r"True Positives.*?:\s*(\d+)")
RE_TN = re.compile(r"True Negatives.*?:\s*(\d+)")
RE_FP = re.compile(r"False Positives.*?:\s*(\d+)")
RE_FN = re.compile(r"False Negatives.*?:\s*(\d+)")
RE_TRUNC = re.compile(r"Truncated.*?:\s*(\d+)")

RE_AVG_REWARD = re.compile(r"Avg reward:\s*([\d.+-]+)")
RE_ACTION_LINE = re.compile(r"^\s+(\w[\w_]+)\s+(\d+)\s+\(([\d.]+)%\)")
RE_OUTCOME_LINE = re.compile(r"^\s+([\w_]+)\s+(\d+)\s+\(([\d.]+)%\)")

RE_HUMAN_STEPS = re.compile(r"Avg steps \(human sessions\):\s*([\d.]+)")
RE_BOT_STEPS = re.compile(r"Avg steps \(bot sessions\):\s*([\d.]+)")

# Per-family line: "  stealth             2    25       23     2   92.0%"
RE_FAMILY_LINE = re.compile(
    r"^\s+([\w_]+)\s+(\d+|\?)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%"
)
# Per-tier line: "  Tier 2 (Careful Automation):   45 bots, 88.9% detected"
RE_TIER_LINE = re.compile(
    r"^\s+Tier\s+(\d+)\s+\(([^)]+)\):\s+(\d+)\s+bots?,\s+([\d.]+)%\s+detected"
)

# Algorithm colors — distinct hues, noaug=solid, advaug=lighter tint
ALGO_COLORS = {
    "ppo": "#4a90d9",
    "dg": "#e8813a",
    "soft_ppo": "#8b6bbf",
    "ppo_noaug": "#4a90d9",  # blue
    "dg_noaug": "#e8813a",  # orange
    "soft_ppo_noaug": "#8b6bbf",  # purple
    "ppo_advaug": "#89bbe8",  # light blue
    "dg_advaug": "#f2b07a",  # light orange
    "soft_ppo_advaug": "#b9a3d6",  # light purple
}
ALGO_LABELS = {
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

# Tier colors — warm pastels matching the system diagram style
TIER_COLORS = {
    1: "#7ec8a0",  # green — commodity
    2: "#6aafe6",  # blue — careful automation
    3: "#f5c462",  # warm yellow — semi-automated
    4: "#e88e6e",  # salmon — trace-conditioned
    5: "#b385d1",  # purple — LLM-powered
}
TIER_NAMES = {
    1: "T1: Commodity",
    2: "T2: Careful Automation",
    3: "T3: Semi-Automated",
    4: "T4: Trace-Conditioned",
    5: "T5: LLM-Powered",
}


def _detect_encoding(path: str) -> str:
    with open(path, "rb") as fb:
        bom = fb.read(2)
    if bom == b"\xff\xfe":
        return "utf-16-le"
    elif bom == b"\xfe\xff":
        return "utf-16-be"
    return "utf-8"


def parse_log(path: str) -> dict[str, dict]:
    """Parse evaluation log into per-agent result dicts.

    Returns:
        {"agent_name": {metrics...}, ...}
        Also stores global "split" info in a special "_meta" key.
    """
    encoding = _detect_encoding(path)
    agents: dict[str, dict] = {}
    meta: dict = {}
    current_agent: str | None = None
    current: dict = {}
    in_actions = False
    in_outcomes = False
    in_family = False
    in_tier = False

    def _finalize():
        nonlocal current_agent, current
        if current_agent and current:
            current.setdefault("actions", {})
            current.setdefault("outcomes", {})
            current.setdefault("families", {})
            current.setdefault("tiers", {})
            agents[current_agent] = current
        current_agent = None
        current = {}

    with open(path, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            raw = line.rstrip()

            # Global split info
            m = RE_SPLIT.search(raw)
            if m:
                meta["split"] = m.group(1)
                meta["total_sessions"] = int(m.group(2))
                meta["human_sessions"] = int(m.group(3))
                meta["bot_sessions"] = int(m.group(4))

            # New agent section: "=== PPO — TEST split (500 episodes) ==="
            m = RE_AGENT_SECTION.search(raw)
            if m:
                _finalize()
                current_agent = m.group(1).lower()
                current = {"split": m.group(2), "episodes": int(m.group(3))}
                in_actions = in_outcomes = in_family = in_tier = False
                continue

            # Also detect "Loading agent: ppo (...)"
            m = RE_AGENT_HEADER.search(raw)
            if m and not current_agent:
                _finalize()
                current_agent = m.group(1).lower()
                current = {}
                in_actions = in_outcomes = in_family = in_tier = False
                continue

            if not current_agent:
                continue

            # Scalar metrics
            for name, regex in [
                ("accuracy", RE_ACCURACY),
                ("precision", RE_PRECISION),
                ("recall", RE_RECALL),
                ("f1", RE_F1),
            ]:
                m = regex.search(raw)
                if m:
                    current[name] = float(m.group(1))

            for name, regex in [
                ("tp", RE_TP),
                ("tn", RE_TN),
                ("fp", RE_FP),
                ("fn", RE_FN),
                ("truncated", RE_TRUNC),
            ]:
                m = regex.search(raw)
                if m:
                    current[name] = int(m.group(1))

            m = RE_AVG_REWARD.search(raw)
            if m:
                current["avg_reward"] = float(m.group(1))

            m = RE_HUMAN_STEPS.search(raw)
            if m:
                current["human_avg_steps"] = float(m.group(1))
            m = RE_BOT_STEPS.search(raw)
            if m:
                current["bot_avg_steps"] = float(m.group(1))

            # Section toggles
            if "Per-Family Bot Detection" in raw:
                in_family = True
                in_tier = in_actions = in_outcomes = False
                current.setdefault("families", {})
                continue
            if "Per-Tier Summary" in raw:
                in_tier = True
                in_family = in_actions = in_outcomes = False
                current.setdefault("tiers", {})
                continue
            if "Final Action Distribution" in raw:
                in_actions = True
                in_family = in_tier = in_outcomes = False
                current.setdefault("actions", {})
                continue
            if "Outcome Distribution" in raw:
                in_outcomes = True
                in_family = in_tier = in_actions = False
                current.setdefault("outcomes", {})
                continue
            if raw.strip().startswith("---") and ("Confusion" in raw or not in_actions):
                if not in_actions and not in_outcomes:
                    pass
                else:
                    in_actions = in_outcomes = False
                    continue

            if in_family:
                m = RE_FAMILY_LINE.search(raw)
                if m:
                    family = m.group(1)
                    tier = int(m.group(2)) if m.group(2) != "?" else 0
                    current.setdefault("families", {})[family] = {
                        "tier": tier,
                        "n": int(m.group(3)),
                        "detected": int(m.group(4)),
                        "missed": int(m.group(5)),
                        "rate": float(m.group(6)) / 100.0,
                    }

            if in_tier:
                m = RE_TIER_LINE.search(raw)
                if m:
                    tier_num = int(m.group(1))
                    current.setdefault("tiers", {})[tier_num] = {
                        "name": m.group(2),
                        "n": int(m.group(3)),
                        "rate": float(m.group(4)) / 100.0,
                    }

            if in_actions:
                m = RE_ACTION_LINE.search(raw)
                if m:
                    current.setdefault("actions", {})[m.group(1)] = int(m.group(2))

            if in_outcomes:
                m = RE_OUTCOME_LINE.search(raw)
                if m:
                    current.setdefault("outcomes", {})[m.group(1)] = int(m.group(2))

    _finalize()
    agents["_meta"] = meta
    return agents


def _get_color(name: str) -> str:
    return ALGO_COLORS.get(name.lower(), "#34495e")


def _get_label(name: str) -> str:
    return ALGO_LABELS.get(name.lower(), name.upper())


def _setup_style():
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
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
            "savefig.facecolor": "white",
        }
    )


def plot_single(
    name: str, result: dict, out_dir: Path, fmt: str = "png", test_set_label: str = ""
):
    """Generate per-agent evaluation plots."""
    _setup_style()
    split_name = result.get("split", "test").upper()
    if test_set_label:
        split_name = f"{split_name} — {test_set_label}"
    label = _get_label(name)

    # Confusion matrix (pastel blue style)
    tp = result.get("tp", 0)
    tn = result.get("tn", 0)
    fp = result.get("fp", 0)
    fn = result.get("fn", 0)
    total = tp + tn + fp + fn or 1

    cm = np.array([[tp, fn], [fp, tn]])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    # Fixed color mapping to match classifier visual: TP=medium blue, TN=dark navy, off-diag=light
    cm_visual = np.array(
        [
            [0.5, cm[0, 1] / max(total, 1)],
            [cm[1, 0] / max(total, 1), 1.0],
        ]
    )
    im = ax.imshow(cm_visual, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Bot", "Human"], fontsize=12)
    ax.set_yticklabels(["Bot", "Human"], fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if cm_visual[i, j] > 0.4 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=22, color=color)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    fig.text(
        0.5,
        0.01,
        f"{label}  (n={total})",
        ha="center",
        fontsize=10,
        style="italic",
        color="#888888",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_dir / f"eval_{name}_confusion.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_{name}_confusion.{fmt}")

    # Action distribution
    actions = result.get("actions", {})
    if actions:
        action_colors = {
            "allow": "#2ecc71",
            "block": "#e74c3c",
            "easy_puzzle": "#f1c40f",
            "medium_puzzle": "#e8813a",
            "hard_puzzle": "#c0392b",
            "continue": "#95a5a6",
            "deploy_honeypot": "#8b6bbf",
        }
        fig, ax = plt.subplots(figsize=(7, 4))
        action_names = list(actions.keys())
        action_counts = [actions[a] for a in action_names]
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
        ax.set_title(f"{label} — Final Action Distribution ({split_name})")
        ax.grid(False)
        fig.savefig(out_dir / f"eval_{name}_actions.{fmt}")
        plt.close(fig)
        print(f"  Saved eval_{name}_actions.{fmt}")

    # Per-family detection rate (single agent)
    families = result.get("families", {})
    if families:
        _plot_family_bars(name, families, out_dir, fmt, split_name)

    # Per-tier detection rate (single agent)
    tiers = result.get("tiers", {})
    if tiers:
        _plot_tier_bars_single(name, tiers, out_dir, fmt, split_name)


def _plot_family_bars(
    name: str, families: dict, out_dir: Path, fmt: str, split_name: str
):
    """Per-family horizontal bar chart with detection rates, colored by tier."""
    label = _get_label(name)
    # Sort by tier then name
    sorted_fams = sorted(families.items(), key=lambda x: (x[1]["tier"], x[0]))

    fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_fams) * 0.6 + 1)))

    fam_names = [f for f, _ in sorted_fams]
    rates = [d["rate"] for _, d in sorted_fams]
    counts = [d["n"] for _, d in sorted_fams]
    colors = [TIER_COLORS.get(d["tier"], "#95a5a6") for _, d in sorted_fams]

    y = np.arange(len(fam_names))
    bars = ax.barh(y, rates, color=colors, edgecolor="white", linewidth=1.2, height=0.7)

    for i, (bar, rate, n) in enumerate(zip(bars, rates, counts)):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{rate:.0%} (n={n})",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(fam_names)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Detection Rate")
    ax.set_title(f"{label} — Per-Family Bot Detection ({split_name})")
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.3)

    # Tier legend
    from matplotlib.patches import Patch

    tier_nums = sorted(set(d["tier"] for _, d in sorted_fams))
    legend_handles = [
        Patch(color=TIER_COLORS.get(t, "#95a5a6"), label=TIER_NAMES.get(t, f"Tier {t}"))
        for t in tier_nums
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=len(tier_nums),
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_dir / f"eval_{name}_per_family.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_{name}_per_family.{fmt}")


def _plot_tier_bars_single(
    name: str, tiers: dict, out_dir: Path, fmt: str, split_name: str
):
    """Per-tier bar chart for a single agent."""
    label = _get_label(name)

    tier_nums = sorted(tiers.keys())
    rates = [tiers[t]["rate"] for t in tier_nums]
    counts = [tiers[t]["n"] for t in tier_nums]
    colors = [TIER_COLORS.get(t, "#95a5a6") for t in tier_nums]
    labels = [TIER_NAMES.get(t, f"Tier {t}") for t in tier_nums]

    fig, ax = plt.subplots(figsize=(max(6, len(tier_nums) * 1.8 + 1), 5))
    x = np.arange(len(tier_nums))
    bars = ax.bar(x, rates, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

    for bar, rate, n in zip(bars, rates, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.0%}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Detection Rate")
    ax.set_title(f"{label} — Per-Tier Detection Rate ({split_name})")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_dir / f"eval_{name}_per_tier.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_{name}_per_tier.{fmt}")


def plot_tier_comparison(
    agents: dict[str, dict], out_dir: Path, fmt: str = "png", test_set_label: str = ""
):
    """Per-tier detection rates compared across all agents (grouped bar chart)."""
    names = [n for n in agents if n != "_meta"]
    # Collect all tier numbers across agents
    all_tiers = set()
    for name in names:
        all_tiers.update(agents[name].get("tiers", {}).keys())
    if not all_tiers:
        return

    _setup_style()
    meta = agents.get("_meta", {})
    split_name = meta.get("split", "test").upper()
    if test_set_label:
        split_name = f"{split_name} — {test_set_label}"
    tier_nums = sorted(all_tiers)

    x = np.arange(len(tier_nums))
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(max(8, len(tier_nums) * 2.5), 6))

    for i, name in enumerate(names):
        tiers = agents[name].get("tiers", {})
        rates = [tiers.get(t, {}).get("rate", 0) for t in tier_nums]
        offset = (i - len(names) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            rates,
            width,
            label=_get_label(name),
            color=_get_color(name),
            edgecolor="white",
            linewidth=1,
        )
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{rate:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    tier_labels = [TIER_NAMES.get(t, f"Tier {t}") for t in tier_nums]
    # Add sample counts below tier labels
    for name in names:
        tiers = agents[name].get("tiers", {})
        for t_idx, t in enumerate(tier_nums):
            n = tiers.get(t, {}).get("n", 0)
            if n > 0:
                ax.text(
                    x[t_idx],
                    -0.08,
                    f"n={n}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="gray",
                    transform=ax.get_xaxis_transform(),
                )
                break  # Only show once per tier

    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Detection Rate")
    ax.set_title(f"Per-Tier Detection Rate — Algorithm Comparison ({split_name})")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    ax.grid(False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_dir / f"eval_tier_comparison.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_tier_comparison.{fmt}")


def plot_family_heatmap(
    agents: dict[str, dict], out_dir: Path, fmt: str = "png", test_set_label: str = ""
):
    """Heatmap: agents × bot families, cell = detection rate."""
    names = [n for n in agents if n != "_meta"]
    # Collect all families
    all_families = set()
    for name in names:
        all_families.update(agents[name].get("families", {}).keys())
    if not all_families:
        return

    _setup_style()
    meta = agents.get("_meta", {})
    split_name = meta.get("split", "test").upper()
    if test_set_label:
        split_name = f"{split_name} — {test_set_label}"

    # Sort families by tier then name
    def _sort_key(fam):
        for name in names:
            info = agents[name].get("families", {}).get(fam)
            if info:
                return (info["tier"], fam)
        return (99, fam)

    families = sorted(all_families, key=_sort_key)

    # Build matrix
    matrix = np.zeros((len(names), len(families)))
    counts = np.zeros((len(names), len(families)), dtype=int)
    for i, name in enumerate(names):
        fam_data = agents[name].get("families", {})
        for j, fam in enumerate(families):
            info = fam_data.get(fam, {})
            matrix[i, j] = info.get("rate", 0)
            counts[i, j] = info.get("n", 0)

    fig, ax = plt.subplots(
        figsize=(max(8, len(families) * 1.2 + 2), max(4, len(names) * 1.2 + 1))
    )
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(len(names)):
        for j in range(len(families)):
            rate = matrix[i, j]
            n = counts[i, j]
            color = "white" if rate < 0.4 or rate > 0.85 else "black"
            ax.text(
                j,
                i,
                f"{rate:.0%}\n(n={n})",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(np.arange(len(families)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(families, rotation=35, ha="right")
    ax.set_yticklabels([_get_label(n) for n in names])

    # Add tier separators
    prev_tier = None
    for j, fam in enumerate(families):
        for name in names:
            info = agents[name].get("families", {}).get(fam)
            if info:
                tier = info["tier"]
                if prev_tier is not None and tier != prev_tier:
                    ax.axvline(x=j - 0.5, color="black", linewidth=2, alpha=0.5)
                prev_tier = tier
                break

    ax.set_title(f"Detection Rate by Bot Family × Algorithm ({split_name})")
    fig.colorbar(im, ax=ax, label="Detection Rate", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / f"eval_family_heatmap.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_family_heatmap.{fmt}")


def plot_outcome_comparison(
    agents: dict[str, dict], out_dir: Path, fmt: str = "png", test_set_label: str = ""
):
    """Stacked bar chart of outcome distributions with puzzle difficulty breakdown."""
    names = [n for n in agents if n != "_meta"]
    all_outcomes = set()
    for name in names:
        all_outcomes.update(agents[name].get("outcomes", {}).keys())
    if not all_outcomes:
        return

    _setup_style()
    meta = agents.get("_meta", {})
    split_name = meta.get("split", "test").upper()
    if test_set_label:
        split_name = f"{split_name} — {test_set_label}"

    # Build stacked segments: correct decisions, puzzle breakdown by difficulty, errors
    # Segment: (key, label, color, source)
    #   source="outcomes" reads from outcomes dict, source="actions" reads from actions dict
    # Colors chosen for maximum distinguishability — no two similar shades
    segments = [
        ("correct_allow", "Correct Allow", "#4a90d9", "outcomes"),  # blue
        ("correct_block", "Correct Block", "#2ecc71", "outcomes"),  # green
        ("easy_puzzle", "Easy Puzzle", "#f1c40f", "actions"),  # yellow
        ("medium_puzzle", "Medium Puzzle", "#e8813a", "actions"),  # orange
        ("hard_puzzle", "Hard Puzzle", "#c0392b", "actions"),  # red
        ("human_passed_puzzle", "Human Passed Puzzle", "#7ec8a0", "outcomes"),  # mint
        ("bot_passed_puzzle", "Bot Passed Puzzle", "#e88e8e", "outcomes"),  # salmon
        ("fp_puzzle", "Unnecessary Puzzle (Human)", "#d4a0d4", "outcomes"),  # mauve
        ("false_negative", "False Negative", "#e74c3c", "outcomes"),  # bright red
        ("false_positive", "False Positive", "#f39c12", "outcomes"),  # amber
    ]

    # Filter to segments that actually have data
    active_segments = []
    for key, label, color, source in segments:
        has_data = any(
            agents[n]
            .get("actions" if source == "actions" else "outcomes", {})
            .get(key, 0)
            > 0
            for n in names
        )
        if has_data:
            active_segments.append((key, label, color, source))

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.8), 7))
    x = np.arange(len(names))
    bottoms = np.zeros(len(names))

    for key, label, color, source in active_segments:
        src_key = "actions" if source == "actions" else "outcomes"
        counts = np.array(
            [agents[n].get(src_key, {}).get(key, 0) for n in names], dtype=float
        )
        ax.bar(
            x,
            counts,
            bottom=bottoms,
            label=label,
            color=color,
            edgecolor="#f5f5f5",
            linewidth=0.5,
            width=0.6,
        )
        bottoms += counts

    ax.set_xticks(x)
    ax.set_xticklabels([_get_label(n) for n in names], rotation=15, ha="right")
    ax.set_ylabel("Episode Count")
    ax.set_title(f"Outcome Distribution — {split_name} Split")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=8,
        ncol=3,
        frameon=False,
    )
    ax.grid(False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_dir / f"eval_outcome_comparison.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_outcome_comparison.{fmt}")


def plot_comparison(
    agents: dict[str, dict], out_dir: Path, fmt: str = "png", test_set_label: str = ""
):
    """Generate multi-agent comparison plots."""
    names = [n for n in agents if n != "_meta"]
    if not names:
        return

    _setup_style()
    meta = agents.get("_meta", {})
    split_name = meta.get("split", "test").upper()
    if test_set_label:
        split_name = f"{split_name} — {test_set_label}"

    # ── 1. Grouped metrics bar chart ─────────────────────────────────
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    x = np.arange(len(metric_keys))
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, name in enumerate(names):
        r = agents[name]
        values = [r.get(k, 0) for k in metric_keys]
        offset = (i - len(names) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=_get_label(name),
            color=_get_color(name),
            edgecolor="white",
            linewidth=1,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(f"Evaluation Metrics Comparison — {split_name} Split")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    ax.grid(False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_dir / f"eval_metrics_comparison.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_metrics_comparison.{fmt}")

    # ── 2. Confusion matrices side by side (Blues cmap) ──────────

    n_agents = len(names)
    if n_agents > 3:
        n_rows, n_cols = 2, (n_agents + 1) // 2
    else:
        n_rows, n_cols = 1, n_agents

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    axes_flat = np.array(axes).flatten()

    for idx, name in enumerate(names):
        ax = axes_flat[idx]
        r = agents[name]
        tp = r.get("tp", 0)
        tn = r.get("tn", 0)
        fp = r.get("fp", 0)
        fn = r.get("fn", 0)
        cm = np.array([[tp, fn], [fp, tn]])
        cm_total = tp + tn + fp + fn or 1
        cm_visual = np.array(
            [
                [0.5, cm[0, 1] / max(cm_total, 1)],
                [cm[1, 0] / max(cm_total, 1), 1.0],
            ]
        )

        im = ax.imshow(cm_visual, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Bot", "Human"], fontsize=10)
        ax.set_yticklabels(["Bot", "Human"], fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel("True", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = "white" if cm_visual[i, j] > 0.4 else "black"
                ax.text(
                    j, i, str(val), ha="center", va="center", fontsize=18, color=color
                )
        acc = r.get("accuracy", 0)
        ax.set_title(
            f"{_get_label(name)}\nAcc={acc:.3f}",
            fontsize=11,
            fontweight="bold",
            color="#444444",
            pad=10,
        )

    for idx in range(n_agents, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(
        f"Confusion Matrices — {split_name} Split",
        fontsize=15,
        fontweight="bold",
        color="#333333",
        y=1.01,
    )
    fig.savefig(out_dir / f"eval_confusion_comparison.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_confusion_comparison.{fmt}")

    # ── 3. Decision timing comparison ────────────────────────────────
    has_timing = all(
        agents[n].get("human_avg_steps") is not None
        and agents[n].get("bot_avg_steps") is not None
        for n in names
    )
    if has_timing:
        x = np.arange(2)
        width = 0.8 / len(names)
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, name in enumerate(names):
            r = agents[name]
            values = [r["human_avg_steps"], r["bot_avg_steps"]]
            offset = (i - len(names) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=_get_label(name),
                color=_get_color(name),
                edgecolor="white",
                linewidth=1,
            )
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(["Human", "Bot"])
        ax.set_ylabel("Avg Windows Before Decision")
        ax.set_title(f"Decision Timing — {split_name} Split")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=3,
            fontsize=9,
            frameon=False,
        )
        ax.grid(False)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.18)
        fig.savefig(out_dir / f"eval_timing_comparison.{fmt}")
        plt.close(fig)
        print(f"  Saved eval_timing_comparison.{fmt}")

    # ── 4. Per-tier comparison ────────────────────────────────────────
    plot_tier_comparison(agents, out_dir, fmt, test_set_label=test_set_label)

    # ── 5. Family heatmap ─────────────────────────────────────────────
    plot_family_heatmap(agents, out_dir, fmt, test_set_label=test_set_label)

    # ── 6. Outcome distribution comparison ─────────────────────────────
    plot_outcome_comparison(agents, out_dir, fmt, test_set_label=test_set_label)

    # ── 7. Combined summary (2×3 grid) ───────────────────────────────
    _plot_combined_summary(agents, names, split_name, out_dir, fmt, test_set_label)


def _plot_combined_summary(agents, names, split_name, out_dir, fmt, test_set_label=""):
    """2×3 combined summary figure with tier and family data."""

    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.subplots_adjust(hspace=0.4, wspace=0.35)
    fig.suptitle(
        f"Evaluation Summary — {split_name} Split",
        fontsize=18,
        fontweight="bold",
        color="#333333",
        y=0.99,
    )

    # (a) Metrics comparison
    ax = axes[0, 0]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        r = agents[name]
        values = [r.get(k, 0) for k in metric_keys]
        offset = (i - len(names) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=_get_label(name),
            color=_get_color(name),
            edgecolor="white",
            linewidth=0.8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.9, 1.02)
    ax.set_title("(a) Classification Metrics", fontweight="bold")
    ax.legend(fontsize=6, ncol=3, loc="lower left")

    # (b) Confusion matrix of best agent
    ax = axes[0, 1]
    best_name = max(names, key=lambda n: agents[n].get("f1", 0))
    r = agents[best_name]
    tp, tn, fp, fn = r.get("tp", 0), r.get("tn", 0), r.get("fp", 0), r.get("fn", 0)
    cm = np.array([[tp, fn], [fp, tn]])
    cm_total = tp + tn + fp + fn or 1
    cm_visual = np.array(
        [
            [0.5, cm[0, 1] / max(cm_total, 1)],
            [cm[1, 0] / max(cm_total, 1), 1.0],
        ]
    )
    im = ax.imshow(cm_visual, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Bot", "Human"], fontsize=10)
    ax.set_yticklabels(["Bot", "Human"], fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if cm_visual[i, j] > 0.4 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=18, color=color)
    ax.set_title(f"(b) Confusion Matrix ({_get_label(best_name)})", fontweight="bold")

    # (c) Per-tier detection rates (grouped)
    ax = axes[0, 2]
    all_tiers = set()
    for name in names:
        all_tiers.update(agents[name].get("tiers", {}).keys())
    if all_tiers:
        tier_nums = sorted(all_tiers)
        x = np.arange(len(tier_nums))
        width = 0.8 / len(names)
        for i, name in enumerate(names):
            tiers = agents[name].get("tiers", {})
            rates = [tiers.get(t, {}).get("rate", 0) for t in tier_nums]
            offset = (i - len(names) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                rates,
                width,
                label=_get_label(name),
                color=_get_color(name),
                edgecolor="white",
                linewidth=0.8,
            )
        tier_labels = [TIER_NAMES.get(t, f"T{t}") for t in tier_nums]
        ax.set_xticks(x)
        ax.set_xticklabels(tier_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_title("(c) Per-Tier Detection Rate", fontweight="bold")
    ax.legend(fontsize=6, ncol=3, loc="lower left")

    # (d) Action distributions (grouped)
    ax = axes[1, 0]
    all_actions = set()
    for name in names:
        all_actions.update(agents[name].get("actions", {}).keys())
    all_actions = sorted(all_actions)
    if all_actions:
        y = np.arange(len(all_actions))
        bar_h = 0.8 / len(names)
        for i, name in enumerate(names):
            acts = agents[name].get("actions", {})
            vals = [acts.get(a, 0) for a in all_actions]
            offset = (i - len(names) / 2 + 0.5) * bar_h
            ax.barh(
                y + offset,
                vals,
                bar_h,
                label=_get_label(name),
                color=_get_color(name),
                edgecolor="white",
                linewidth=0.8,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(all_actions, fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title("(d) Final Actions", fontweight="bold")
    ax.legend(fontsize=6, ncol=3, loc="lower right")

    # (e) Family heatmap (all agents)
    ax = axes[1, 1]
    all_families = set()
    for name in names:
        all_families.update(agents[name].get("families", {}).keys())
    if all_families:

        def _fam_sort(f):
            for n in names:
                info = agents[n].get("families", {}).get(f)
                if info:
                    return (info["tier"], f)
            return (99, f)

        families_sorted = sorted(all_families, key=_fam_sort)
        matrix = np.zeros((len(names), len(families_sorted)))
        for i, name in enumerate(names):
            fam_data = agents[name].get("families", {})
            for j, fam in enumerate(families_sorted):
                matrix[i, j] = fam_data.get(fam, {}).get("rate", 0)
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for i in range(len(names)):
            for j in range(len(families_sorted)):
                rate = matrix[i, j]
                color = "white" if rate < 0.4 or rate > 0.85 else "black"
                ax.text(
                    j,
                    i,
                    f"{rate:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                )
        ax.set_xticks(np.arange(len(families_sorted)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(families_sorted, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels([_get_label(n) for n in names], fontsize=8)
    ax.set_title("(e) Per-Family Detection Heatmap", fontweight="bold")

    # (f) Decision timing
    ax = axes[1, 2]
    has_timing = all(
        agents[n].get("human_avg_steps") is not None
        and agents[n].get("bot_avg_steps") is not None
        for n in names
    )
    if has_timing:
        x = np.arange(2)
        width = 0.8 / len(names)
        for i, name in enumerate(names):
            r = agents[name]
            values = [r["human_avg_steps"], r["bot_avg_steps"]]
            offset = (i - len(names) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                values,
                width,
                label=_get_label(name),
                color=_get_color(name),
                edgecolor="white",
                linewidth=0.8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["Human", "Bot"])
    ax.set_ylabel("Avg Windows")
    ax.set_title("(f) Decision Timing", fontweight="bold")
    ax.legend(fontsize=6, ncol=3, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / f"eval_summary.{fmt}")
    plt.close(fig)
    print(f"  Saved eval_summary.{fmt}")


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
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
    parser.add_argument(
        "--augmented",
        action="store_true",
        help="Label figures as using augmented test set",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: {log_path} not found")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {log_path}...")
    agents = parse_log(str(log_path))

    agent_names = [n for n in agents if n != "_meta"]
    if not agent_names:
        print("No evaluation data found in log file.")
        return

    # Determine test-set label
    test_set_label = (
        "With Augmented Data" if args.augmented else "Standard (No Augmented)"
    )
    agents["_meta"]["test_set_label"] = test_set_label

    print(f"Found {len(agent_names)} agent(s): {', '.join(agent_names)}")
    print(f"Test set: {test_set_label}")
    for name in agent_names:
        r = agents[name]
        print(f"  {name}: accuracy={r.get('accuracy', 'N/A')}, f1={r.get('f1', 'N/A')}")
        tiers = r.get("tiers", {})
        if tiers:
            for t in sorted(tiers):
                print(f"    Tier {t}: {tiers[t]['rate']:.0%} ({tiers[t]['n']} bots)")

    # Per-agent plots
    for name in agent_names:
        plot_single(
            name, agents[name], out_dir, fmt=args.format, test_set_label=test_set_label
        )

    # Comparison plots (always generated, even for single agent)
    plot_comparison(agents, out_dir, fmt=args.format, test_set_label=test_set_label)

    print(f"\nDone! Evaluation figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
