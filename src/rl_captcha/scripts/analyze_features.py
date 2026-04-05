"""Analyze which features make missed LLM bots look human.

Loads human sessions and LLM bot sessions, encodes them into
26-dim feature vectors, runs the agent to classify each LLM session
as detected or missed, then compares feature distributions across
three groups: humans, detected LLM bots, and missed LLM bots.

Outputs:
  - Feature comparison table (printed + CSV)
  - Feature distribution heatmap (normalized means)
  - Per-feature violin/box plots for the most discriminative features

Usage (from src/):
    python -m rl_captcha.scripts.analyze_features \
        --agent rl_captcha/agent/checkpoints/ppo_noaug \
        --data-dir data/ \
        --out figures/feature_analysis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add src to path
src_dir = Path(__file__).resolve().parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rl_captcha.config import EventEnvConfig
from rl_captcha.data.loader import (
    Session,
    load_from_directory,
    split_sessions,
    BOT_TYPE_TO_TIER,
)
from rl_captcha.environment.event_env import EventEncoder, EventEnv
from rl_captcha.agent.ppo_lstm import PPOLSTM


FEATURE_NAMES = [
    "Mouse event ratio",
    "Click event ratio",
    "Key event ratio",
    "Scroll event ratio",
    "Mean mouse speed",
    "Mouse speed variance",
    "Mean mouse accel",
    "Path curvature",
    "Mean inter-event dt",
    "Inter-event dt variance",
    "Min inter-event dt",
    "Mean click interval",
    "Click interval variance",
    "Mean keystroke hold",
    "Keystroke hold variance",
    "Mean key-press interval",
    "Key-press interval variance",
    "Scroll total distance",
    "Scroll direction changes",
    "Unique X positions",
    "Unique Y positions",
    "X range",
    "Y range",
    "Interactive click ratio",
    "Window duration",
    "Event count ratio",
]


def encode_session_features(
    session: Session, encoder: EventEncoder, config: EventEnvConfig
) -> np.ndarray:
    """Encode a session into per-window feature vectors and return the mean."""
    timeline = encoder.build_timeline(session)
    if len(timeline) < config.min_events:
        return None

    ws = config.window_size
    stride = ws // 2
    windows = []
    for start in range(0, len(timeline), stride):
        window = timeline[start : start + ws]
        if len(window) >= config.min_events:
            windows.append(window)

    if not windows:
        return None

    vectors = np.array([encoder.encode_window(w) for w in windows])
    return vectors.mean(axis=0)


def classify_session(
    session: Session, env: EventEnv, agent: PPOLSTM
) -> str:
    """Run the agent on a session and return the outcome."""
    # Temporarily set the env to use only this session
    old_sessions = env._sessions
    old_human = env._human_sessions
    old_bot = env._bot_sessions

    env._sessions = [session]
    env._human_sessions = [session] if session.label == 1 else []
    env._bot_sessions = [session] if session.label == 0 else []

    obs, info = env.reset()
    agent.reset_hidden()

    if info.get("too_short"):
        env._sessions = old_sessions
        env._human_sessions = old_human
        env._bot_sessions = old_bot
        return "skipped"

    action_mask = info.get("action_mask")
    done = False
    while not done:
        action, _, _ = agent.select_action(obs, action_mask=action_mask, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        action_mask = step_info.get("action_mask")

    env._sessions = old_sessions
    env._human_sessions = old_human
    env._bot_sessions = old_bot

    return step_info.get("outcome", "unknown")


def main():
    parser = argparse.ArgumentParser(description="Analyze features of missed LLM bots")
    parser.add_argument("--agent", type=str, required=True, help="Path to agent checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/", help="Root data directory")
    parser.add_argument("--out", type=str, default="figures/feature_analysis", help="Output directory")
    parser.add_argument("--split-seed", type=int, default=42, help="Data split seed")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading sessions...")
    sessions = load_from_directory(args.data_dir)
    train_sessions, val_sessions, test_sessions = split_sessions(
        sessions, train=0.70, val=0.15, test=0.15, seed=args.split_seed
    )

    human_sessions = [s for s in test_sessions if s.label == 1]
    llm_sessions = [s for s in test_sessions if s.label == 0 and s.metadata.get("bot_type") == "llm"]

    print(f"  Test set: {len(test_sessions)} sessions")
    print(f"  Humans: {len(human_sessions)}")
    print(f"  LLM bots: {len(llm_sessions)}")

    if not llm_sessions:
        print("  No LLM bot sessions in test set! Try --split-seed or check data.")
        # Fall back to all LLM sessions
        llm_sessions = [s for s in sessions if s.label == 0 and s.metadata.get("bot_type") == "llm"]
        human_sessions = [s for s in sessions if s.label == 1]
        print(f"  Falling back to ALL data: {len(human_sessions)} humans, {len(llm_sessions)} LLM bots")

    # Load agent
    print(f"Loading agent from {args.agent}...")
    config = EventEnvConfig()
    agent = PPOLSTM(obs_dim=config.event_dim, action_dim=7, device="cpu")
    agent.load(args.agent)

    encoder = EventEncoder(config)
    all_eval_sessions = test_sessions if len(test_sessions) > 0 else sessions
    env = EventEnv(all_eval_sessions, config)

    # Classify LLM sessions
    print("Classifying LLM bot sessions...")
    detected_sessions = []
    missed_sessions = []
    for i, s in enumerate(llm_sessions):
        outcome = classify_session(s, env, agent)
        if outcome in ("correct_block", "bot_failed_puzzle"):
            detected_sessions.append(s)
        elif outcome in ("false_negative", "bot_passed_puzzle"):
            missed_sessions.append(s)
        # skip unknown/skipped
        if (i + 1) % 10 == 0:
            sys.stdout.write(f"\r  Classified {i + 1}/{len(llm_sessions)}")
            sys.stdout.flush()
    print(f"\n  Detected: {len(detected_sessions)}, Missed: {len(missed_sessions)}")

    # Encode features for each group
    print("Encoding features...")

    def encode_group(group_sessions: list[Session]) -> np.ndarray:
        vecs = []
        for s in group_sessions:
            v = encode_session_features(s, encoder, config)
            if v is not None:
                vecs.append(v)
        return np.array(vecs) if vecs else np.zeros((0, 26))

    human_features = encode_group(human_sessions)
    detected_features = encode_group(detected_sessions)
    missed_features = encode_group(missed_sessions)

    print(f"  Feature vectors — Human: {len(human_features)}, "
          f"Detected LLM: {len(detected_features)}, Missed LLM: {len(missed_features)}")

    if len(missed_features) == 0:
        print("No missed LLM sessions — agent detects all LLM bots!")
        return

    # ── Print comparison table ─────────────────────────────────────
    print("\n" + "=" * 90)
    print("FEATURE COMPARISON: Human vs Detected LLM vs Missed LLM")
    print("=" * 90)
    print(f"  {'Feature':<28s} {'Human':>10s} {'Detected':>10s} {'Missed':>10s}  {'Miss-Human':>11s}")
    print("-" * 90)

    human_means = human_features.mean(axis=0)
    detected_means = detected_features.mean(axis=0)
    missed_means = missed_features.mean(axis=0)

    # How similar is missed to human (lower = more similar)
    # Normalize by human std to get z-score-like distance
    human_stds = human_features.std(axis=0)
    human_stds[human_stds < 1e-6] = 1e-6
    missed_distance = np.abs(missed_means - human_means) / human_stds
    detected_distance = np.abs(detected_means - human_means) / human_stds

    for i, name in enumerate(FEATURE_NAMES):
        flag = " <--" if missed_distance[i] < 0.5 and detected_distance[i] > 0.5 else ""
        if missed_distance[i] < detected_distance[i] * 0.5:
            flag = " ***"
        print(f"  {name:<28s} {human_means[i]:10.4f} {detected_means[i]:10.4f} "
              f"{missed_means[i]:10.4f}  {missed_distance[i]:10.2f}σ{flag}")

    print("-" * 90)
    print("  *** = missed LLM is much closer to human than detected LLM")
    print("  <-- = missed LLM is within 0.5σ of human (detected is not)")

    # Save CSV
    csv_path = out_dir / "feature_comparison.csv"
    with open(csv_path, "w") as f:
        f.write("feature,human_mean,human_std,detected_mean,detected_std,"
                "missed_mean,missed_std,missed_zscore,detected_zscore\n")
        detected_stds = detected_features.std(axis=0)
        missed_stds = missed_features.std(axis=0)
        for i, name in enumerate(FEATURE_NAMES):
            f.write(f"{name},{human_means[i]:.6f},{human_stds[i]:.6f},"
                    f"{detected_means[i]:.6f},{detected_stds[i]:.6f},"
                    f"{missed_means[i]:.6f},{missed_stds[i]:.6f},"
                    f"{missed_distance[i]:.4f},{detected_distance[i]:.4f}\n")
    print(f"\n  Saved: {csv_path}")

    # ── Plots ──────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })

    # ── Plot 1: Normalized feature heatmap ─────────────────────────
    # Normalize each feature to [0, 1] across the 3 groups
    all_means = np.stack([human_means, detected_means, missed_means])
    feat_min = all_means.min(axis=0, keepdims=True)
    feat_max = all_means.max(axis=0, keepdims=True)
    feat_range = feat_max - feat_min
    feat_range[feat_range < 1e-8] = 1.0
    normalized = (all_means - feat_min) / feat_range  # (3, 26)

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = LinearSegmentedColormap.from_list("custom", ["#dce6f1", "#4472c4", "#1f3864"])
    im = ax.imshow(normalized.T, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Human", "Detected LLM", "Missed LLM"], fontsize=11, fontweight="bold")
    ax.set_yticks(range(26))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=8)

    # Annotate cells with actual values
    for row in range(26):
        for col in range(3):
            val = all_means[col, row]
            color = "white" if normalized[col, row] > 0.6 else "black"
            ax.text(col, row, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title("Feature Comparison: Human vs Detected LLM vs Missed LLM Bots",
                 fontsize=13, fontweight="bold", pad=12)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Normalized value (0=min, 1=max)")
    fig.tight_layout()
    heatmap_path = out_dir / f"feature_heatmap.{args.format}"
    fig.savefig(heatmap_path)
    plt.close(fig)
    print(f"  Saved: {heatmap_path}")

    # ── Plot 2: Z-score distance bar chart ─────────────────────────
    # Shows how far each group's mean is from human mean (in human σ)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(26)
    width = 0.35

    bars1 = ax.bar(x - width / 2, detected_distance, width, label="Detected LLM",
                   color="#4472c4", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, missed_distance, width, label="Missed LLM",
                   color="#c55a5a", alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Distance from human mean (in σ)", fontsize=10)
    ax.set_title("Feature Distance from Human Distribution",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_NAMES, rotation=65, ha="right", fontsize=7)
    ax.legend(frameon=False, fontsize=9)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(25.5, 0.55, "0.5σ threshold", fontsize=7, color="gray", ha="right")

    fig.tight_layout()
    distance_path = out_dir / f"feature_distance.{args.format}"
    fig.savefig(distance_path)
    plt.close(fig)
    print(f"  Saved: {distance_path}")

    # ── Plot 3: Top discriminative features — violin plots ─────────
    # Pick features where missed is much closer to human than detected
    closeness_ratio = np.where(detected_distance > 0.1,
                               missed_distance / detected_distance, 1.0)
    top_indices = np.argsort(closeness_ratio)[:8]  # 8 most discriminative

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    colors = {"Human": "#5b9bd5", "Detected LLM": "#4472c4", "Missed LLM": "#c55a5a"}

    for idx, feat_idx in enumerate(top_indices):
        ax = axes[idx]
        data = [human_features[:, feat_idx],
                detected_features[:, feat_idx],
                missed_features[:, feat_idx]]
        labels = ["Human", "Detected\nLLM", "Missed\nLLM"]

        parts = ax.violinplot(data, positions=[0, 1, 2], showmeans=True, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(list(colors.values())[i])
            pc.set_alpha(0.7)
        parts["cmeans"].set_color("black")

        # Add individual points with jitter
        for i, d in enumerate(data):
            jitter = np.random.normal(0, 0.04, size=len(d))
            ax.scatter(np.full_like(d, i) + jitter, d,
                      color=list(colors.values())[i], s=8, alpha=0.3, edgecolors="none")

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(FEATURE_NAMES[feat_idx], fontsize=9, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Top Features Where Missed LLM Bots Resemble Humans",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    violin_path = out_dir / f"feature_violins.{args.format}"
    fig.savefig(violin_path)
    plt.close(fig)
    print(f"  Saved: {violin_path}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print("\nFeatures where missed LLM bots are closest to human:")
    for rank, idx in enumerate(top_indices[:5], 1):
        print(f"  {rank}. {FEATURE_NAMES[idx]:<28s} "
              f"(missed: {missed_distance[idx]:.2f}σ from human, "
              f"detected: {detected_distance[idx]:.2f}σ)")

    print("\nFeatures where detected LLM bots differ most from human:")
    detected_top = np.argsort(detected_distance)[::-1][:5]
    for rank, idx in enumerate(detected_top, 1):
        print(f"  {rank}. {FEATURE_NAMES[idx]:<28s} "
              f"(detected: {detected_distance[idx]:.2f}σ, "
              f"missed: {missed_distance[idx]:.2f}σ)")

    print(f"\nAll outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
