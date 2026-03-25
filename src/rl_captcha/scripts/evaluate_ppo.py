"""Evaluate one or more trained PPO/DG/Soft-PPO+LSTM agents on the event-level CAPTCHA environment.

Usage (single agent):
    python -m rl_captcha.scripts.evaluate_ppo \
        --agent rl_captcha/agent/checkpoints/ppo_run1 \
        --data-dir data/ \
        --episodes 500

Usage (all three at once):
    python -m rl_captcha.scripts.evaluate_ppo \
        --agent ppo=rl_captcha/agent/checkpoints/ppo_run1 \
               dg=rl_captcha/agent/checkpoints/dg_run1 \
               soft_ppo=rl_captcha/agent/checkpoints/soft_ppo_run1 \
        --episodes 500

By default, evaluation uses the held-out TEST split (15% of data) with
the same seed used during training, ensuring no overlap with training data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from rl_captcha.config import Config
from rl_captcha.data.loader import (
    load_from_directory, split_sessions, split_sessions_by_family,
    bot_type_to_tier, TIER_NAMES,
)
from rl_captcha.environment.event_env import EventEnv, ACTION_NAMES
from rl_captcha.agent.ppo_lstm import PPOLSTM
from rl_captcha.agent.dg_lstm import DGLSTM, DGConfig
from rl_captcha.agent.soft_ppo_lstm import SoftPPOLSTM, SoftPPOConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO/DG/Soft-PPO+LSTM agents")
    p.add_argument("--agent", type=str, nargs="+", required=True,
                    help="Agent checkpoint(s). Either plain paths or name=path pairs "
                         "(e.g. ppo=checkpoints/ppo_run1 dg=checkpoints/dg_run1)")
    p.add_argument("--data-dir", type=str, default="data/",
                    help="Path to data directory with human/ and bot/ subdirs")
    p.add_argument("--episodes", type=int, default=500,
                    help="Number of episodes to evaluate per agent")
    p.add_argument("--split", type=str, default="test",
                    choices=["test", "val", "train", "all"],
                    help="Which data split to evaluate on (default: test)")
    p.add_argument("--split-seed", type=int, default=42,
                    help="Random seed for split (must match training)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--held-out-families", type=str, nargs="*", default=None,
                    help="Bot families to hold out from train/val (test-only)")
    p.add_argument("--held-out-tiers", type=int, nargs="*", default=None,
                    help="Bot tiers to hold out from train/val (test-only)")
    return p.parse_args()


def _parse_agent_specs(agent_args: list[str]) -> list[tuple[str, str]]:
    """Parse agent arguments into (name, path) pairs.

    Supports:
        --agent path/to/checkpoint              → name inferred from dir name
        --agent ppo=path/to/ppo dg=path/to/dg   → explicit names
    """
    specs = []
    for arg in agent_args:
        if "=" in arg:
            name, path = arg.split("=", 1)
            specs.append((name.strip(), path.strip()))
        else:
            # Infer name from last directory component
            from pathlib import Path
            name = Path(arg).name
            specs.append((name, arg))
    return specs


def _create_agent(name: str, cfg: Config, device: str) -> PPOLSTM:
    """Instantiate the correct agent class based on the agent name."""
    kwargs = dict(obs_dim=cfg.event_env.event_dim, action_dim=7, device=device)
    algo = name.lower()
    if algo == "dg":
        return DGLSTM(config=DGConfig(), **kwargs)
    elif algo == "soft_ppo":
        return SoftPPOLSTM(config=SoftPPOConfig(), **kwargs)
    else:
        return PPOLSTM(config=cfg.ppo, **kwargs)


def main():
    args = parse_args()
    cfg = Config()

    # Load data
    print(f"Loading sessions from {args.data_dir}...")
    sessions = load_from_directory(args.data_dir)
    human_count = sum(1 for s in sessions if s.label == 1)
    bot_count = sum(1 for s in sessions if s.label == 0)
    print(f"  Loaded {len(sessions)} sessions ({human_count} human, {bot_count} bot)")

    if not sessions:
        print("ERROR: No sessions found.")
        return

    # Select split
    if args.split == "all":
        eval_sessions = sessions
        print(f"  Evaluating on ALL {len(eval_sessions)} sessions")
    else:
        if args.held_out_families or args.held_out_tiers:
            print(f"  Held-out families: {args.held_out_families or '(none)'}")
            print(f"  Held-out tiers:    {args.held_out_tiers or '(none)'}")
            train_s, val_s, test_s = split_sessions_by_family(
                sessions,
                held_out_families=args.held_out_families,
                held_out_tiers=args.held_out_tiers,
                train=0.70, val=0.15, test=0.15, seed=args.split_seed,
            )
        else:
            train_s, val_s, test_s = split_sessions(
                sessions, train=0.70, val=0.15, test=0.15, seed=args.split_seed,
            )
        splits = {"train": train_s, "val": val_s, "test": test_s}
        eval_sessions = splits[args.split]
        h = sum(1 for s in eval_sessions if s.label == 1)
        b = sum(1 for s in eval_sessions if s.label == 0)
        print(f"  Evaluating on {args.split.upper()} split: "
              f"{len(eval_sessions)} sessions ({h} human, {b} bot)")

    # Create environment (shared across all agents — same eval data)
    from dataclasses import replace
    eval_cfg = replace(cfg.event_env, augment=False)
    env = EventEnv(eval_sessions, config=eval_cfg)

    # Parse agent specs
    agent_specs = _parse_agent_specs(args.agent)
    print(f"\n  Evaluating {len(agent_specs)} agent(s): "
          f"{', '.join(name for name, _ in agent_specs)}")
    print(f"  Episodes per agent: {args.episodes}")
    print()

    # Evaluate each agent
    all_results = {}
    for name, path in agent_specs:
        print(f"{'=' * 60}")
        print(f"  Loading agent: {name} ({path})")
        agent = _create_agent(name, cfg, args.device)
        agent.load(path)
        print(f"  Device: {agent.device}")
        print()

        results = _run_evaluation(env, agent, args.episodes, agent_name=name)
        all_results[name] = results
        _print_results(results, agent_name=name, split_name=args.split)
        _print_per_family_results(results, agent_name=name)

    # Print comparison table if multiple agents
    if len(all_results) > 1:
        _print_comparison(all_results, split_name=args.split)


def _run_evaluation(
    env: EventEnv,
    agent: PPOLSTM,
    num_episodes: int,
    agent_name: str = "",
    eval_seed: int = 42,
) -> dict:
    """Run deterministic evaluation episodes.

    Seeds the RNG before each run so every agent sees the exact same
    sequence of sessions, making results reproducible and comparable.
    """
    import random as _random
    import sys
    import time

    _random.seed(eval_seed)
    episode_data = []
    t_start = time.time()

    for ep in range(num_episodes):
        if (ep + 1) % 10 == 0 or ep == 0:
            elapsed = time.time() - t_start
            eps_per_sec = (ep + 1) / elapsed if elapsed > 0 else 0
            eta = (num_episodes - ep - 1) / eps_per_sec if eps_per_sec > 0 else 0
            sys.stdout.write(
                f"\r  [{agent_name}] Episode {ep + 1}/{num_episodes} "
                f"({eps_per_sec:.1f} ep/s, ETA {eta:.0f}s)"
            )
            sys.stdout.flush()

        obs, info = env.reset()
        agent.reset_hidden()

        # Skip too-short sessions
        while info.get("too_short"):
            obs, info = env.reset()
            agent.reset_hidden()

        true_label = info["true_label"]
        bot_type = info.get("bot_type")
        total_reward = 0.0
        steps = 0
        actions_taken = []

        action_mask = info.get("action_mask")
        done = False
        while not done:
            action, _, _ = agent.select_action(obs, action_mask=action_mask, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            actions_taken.append(action)
            action_mask = step_info.get("action_mask")
            info = step_info

        episode_data.append({
            "true_label": true_label,
            "bot_type": bot_type,
            "outcome": info.get("outcome", "unknown"),
            "reward": total_reward,
            "steps": steps,
            "actions": actions_taken,
            "final_action": actions_taken[-1] if actions_taken else -1,
        })

    elapsed = time.time() - t_start
    sys.stdout.write(
        f"\r  [{agent_name}] Done: {num_episodes} episodes in {elapsed:.1f}s "
        f"({num_episodes / elapsed:.1f} ep/s)      \n"
    )
    sys.stdout.flush()
    return {"episodes": episode_data}


def _compute_metrics(episodes: list[dict]) -> dict:
    """Compute evaluation metrics from episode data."""
    n = len(episodes)
    rewards = [e["reward"] for e in episodes]
    lengths = [e["steps"] for e in episodes]

    tp = sum(1 for e in episodes if e["true_label"] == 0 and e["outcome"] in
             ("correct_block", "bot_blocked_puzzle"))
    tn = sum(1 for e in episodes if e["true_label"] == 1 and e["outcome"] == "correct_allow")
    fp = sum(1 for e in episodes if e["true_label"] == 1 and e["outcome"] in
             ("false_positive_block", "fp_puzzle"))
    fn = sum(1 for e in episodes if e["true_label"] == 0 and e["outcome"] in
             ("false_negative", "bot_passed_puzzle"))
    truncated = sum(1 for e in episodes if e["outcome"] == "truncated")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0

    return {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_length": float(np.mean(lengths)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "truncated": truncated,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _print_results(results: dict, agent_name: str = "agent", split_name: str = "test"):
    """Print evaluation summary for one agent."""
    episodes = results["episodes"]
    n = len(episodes)
    m = _compute_metrics(episodes)

    print(f"=== {agent_name.upper()} - {split_name.upper()} split ({n} episodes) ===")
    print(f"  Avg reward:  {m['avg_reward']:.3f} +/- {m['std_reward']:.3f}")
    print(f"  Avg length:  {m['avg_length']:.1f}")
    print()

    # Confusion matrix
    print("--- Confusion Matrix ---")
    print(f"  True Positives  (bot blocked):   {m['tp']:4d} ({100*m['tp']/n:.1f}%)")
    print(f"  True Negatives  (human allowed): {m['tn']:4d} ({100*m['tn']/n:.1f}%)")
    print(f"  False Positives (human blocked): {m['fp']:4d} ({100*m['fp']/n:.1f}%)")
    print(f"  False Negatives (bot allowed):   {m['fn']:4d} ({100*m['fn']/n:.1f}%)")
    print(f"  Truncated (indecisive):          {m['truncated']:4d} ({100*m['truncated']/n:.1f}%)")
    other = n - m['tp'] - m['tn'] - m['fp'] - m['fn'] - m['truncated']
    if other > 0:
        print(f"  Other:                           {other:4d} ({100*other/n:.1f}%)")
    print()

    print(f"  Accuracy:  {m['accuracy']:.3f}")
    print(f"  Precision: {m['precision']:.3f}")
    print(f"  Recall:    {m['recall']:.3f}")
    print(f"  F1:        {m['f1']:.3f}")
    print()

    # Outcome distribution
    outcome_counts = defaultdict(int)
    for e in episodes:
        outcome_counts[e["outcome"]] += 1

    print("--- Outcome Distribution ---")
    for outcome, count in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        print(f"  {outcome:30s} {count:4d} ({100*count/n:.1f}%)")
    print()

    # Action distribution (final actions)
    action_counts = defaultdict(int)
    for e in episodes:
        fa = e["final_action"]
        if 0 <= fa < len(ACTION_NAMES):
            action_counts[ACTION_NAMES[fa]] += 1
        else:
            action_counts[f"action_{fa}"] += 1

    print("--- Final Action Distribution ---")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action:20s} {count:4d} ({100*count/n:.1f}%)")
    print()

    # Decision timing by label
    human_eps = [e for e in episodes if e["true_label"] == 1]
    bot_eps = [e for e in episodes if e["true_label"] == 0]

    if human_eps:
        avg_human_steps = np.mean([e["steps"] for e in human_eps])
        print(f"  Avg steps (human sessions): {avg_human_steps:.1f}")
    if bot_eps:
        avg_bot_steps = np.mean([e["steps"] for e in bot_eps])
        print(f"  Avg steps (bot sessions):   {avg_bot_steps:.1f}")
    print()


def _print_per_family_results(results: dict, agent_name: str = "agent"):
    """Print per-bot-family and per-tier detection rate breakdowns."""
    episodes = results["episodes"]
    bot_eps = [e for e in episodes if e["true_label"] == 0]

    if not bot_eps:
        return

    # --- Per-family breakdown ---
    by_family: dict[str, list[dict]] = defaultdict(list)
    for e in bot_eps:
        family = e.get("bot_type") or "unknown"
        by_family[family].append(e)

    detected_outcomes = {"correct_block", "bot_blocked_puzzle"}

    print(f"--- Per-Family Bot Detection ({agent_name}) ---")
    print(f"  {'Family':<18s} {'Tier':>4s} {'N':>5s} {'Detect':>7s} {'Miss':>5s} {'Rate':>7s}")
    print(f"  {'-' * 48}")

    for family in sorted(by_family.keys()):
        eps = by_family[family]
        n = len(eps)
        detected = sum(1 for e in eps if e["outcome"] in detected_outcomes)
        missed = n - detected
        rate = detected / n if n > 0 else 0.0
        tier = bot_type_to_tier(family if family != "unknown" else None)
        tier_str = str(tier) if tier > 0 else "?"
        print(f"  {family:<18s} {tier_str:>4s} {n:5d} {detected:7d} {missed:5d} {rate:7.1%}")

    print()

    # --- Per-tier breakdown ---
    by_tier: dict[int, list[dict]] = defaultdict(list)
    for e in bot_eps:
        tier = bot_type_to_tier(e.get("bot_type"))
        by_tier[tier].append(e)

    print(f"--- Per-Tier Summary ({agent_name}) ---")
    for tier_num in sorted(by_tier.keys()):
        eps = by_tier[tier_num]
        n = len(eps)
        detected = sum(1 for e in eps if e["outcome"] in detected_outcomes)
        rate = detected / n if n > 0 else 0.0
        tier_name = TIER_NAMES.get(tier_num, "Unknown")
        print(f"  Tier {tier_num} ({tier_name}): {n:4d} bots, {rate:.1%} detected")
    print()


def _print_comparison(all_results: dict[str, dict], split_name: str = "test"):
    """Print a side-by-side comparison table of all agents."""
    print()
    print("=" * 70)
    print(f"  COMPARISON TABLE - {split_name.upper()} split")
    print("=" * 70)

    metrics = {}
    for name, results in all_results.items():
        metrics[name] = _compute_metrics(results["episodes"])

    # Header
    names = list(metrics.keys())
    col_w = max(12, max(len(n) for n in names) + 2)
    header = f"  {'Metric':<20s}" + "".join(f"{n:>{col_w}s}" for n in names)
    print(header)
    print("  " + "-" * (20 + col_w * len(names)))

    # Rows
    rows = [
        ("Accuracy",  "accuracy",  ".3f"),
        ("Precision",  "precision", ".3f"),
        ("Recall",     "recall",    ".3f"),
        ("F1",         "f1",        ".3f"),
        ("Avg Reward", "avg_reward", ".3f"),
        ("Avg Length", "avg_length", ".1f"),
        ("TP (bot blocked)", "tp",  "d"),
        ("TN (human ok)",    "tn",  "d"),
        ("FP (human bad)",   "fp",  "d"),
        ("FN (bot missed)",  "fn",  "d"),
        ("Truncated",  "truncated", "d"),
    ]

    for label, key, fmt in rows:
        row = f"  {label:<20s}"
        for name in names:
            val = metrics[name][key]
            row += f"{val:>{col_w}{fmt}}"
        print(row)

    print()

    # Highlight best
    best_acc = max(names, key=lambda n: metrics[n]["accuracy"])
    best_f1 = max(names, key=lambda n: metrics[n]["f1"])
    best_reward = max(names, key=lambda n: metrics[n]["avg_reward"])
    print(f"  Best accuracy: {best_acc} ({metrics[best_acc]['accuracy']:.3f})")
    print(f"  Best F1:       {best_f1} ({metrics[best_f1]['f1']:.3f})")
    print(f"  Best reward:   {best_reward} ({metrics[best_reward]['avg_reward']:.3f})")
    print()


if __name__ == "__main__":
    main()
