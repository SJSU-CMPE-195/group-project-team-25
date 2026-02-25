"""Evaluate a trained PPO+LSTM agent on the event-level CAPTCHA environment.

Usage:
    python -m rl_captcha.scripts.evaluate_ppo \
        --agent rl_captcha/agent/checkpoints/ppo_run1 \
        --data-dir data/ \
        --episodes 500
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from rl_captcha.config import Config
from rl_captcha.data.loader import load_from_directory
from rl_captcha.environment.event_env import EventEnv, ACTION_NAMES
from rl_captcha.agent.ppo_lstm import PPOLSTM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO+LSTM agent")
    p.add_argument("--agent", type=str, required=True,
                    help="Path to agent checkpoint directory")
    p.add_argument("--data-dir", type=str, default="data/",
                    help="Path to data directory with human/ and bot/ subdirs")
    p.add_argument("--episodes", type=int, default=500,
                    help="Number of episodes to evaluate")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


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

    # Create environment and agent
    env = EventEnv(sessions, config=cfg.event_env)
    agent = PPOLSTM(
        obs_dim=cfg.event_env.event_dim,
        action_dim=7,
        config=cfg.ppo,
        device=args.device,
    )
    agent.load(args.agent)
    print(f"  Loaded agent from {args.agent}")
    print(f"  Device: {agent.device}")
    print()

    # Run evaluation
    results = _run_evaluation(env, agent, args.episodes)
    _print_results(results)


def _run_evaluation(
    env: EventEnv,
    agent: PPOLSTM,
    num_episodes: int,
) -> dict:
    """Run deterministic evaluation episodes."""
    episode_data = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        agent.reset_hidden()

        # Skip too-short sessions
        while info.get("too_short"):
            obs, info = env.reset()
            agent.reset_hidden()

        true_label = info["true_label"]
        total_reward = 0.0
        steps = 0
        actions_taken = []

        done = False
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            actions_taken.append(action)

        episode_data.append({
            "true_label": true_label,
            "outcome": info.get("outcome", "unknown"),
            "reward": total_reward,
            "steps": steps,
            "actions": actions_taken,
            "final_action": actions_taken[-1] if actions_taken else -1,
        })

    return {"episodes": episode_data}


def _print_results(results: dict):
    """Print evaluation summary."""
    episodes = results["episodes"]
    n = len(episodes)

    rewards = [e["reward"] for e in episodes]
    lengths = [e["steps"] for e in episodes]

    print(f"=== Evaluation Results ({n} episodes) ===")
    print(f"  Avg reward:  {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")
    print(f"  Avg length:  {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
    print()

    # Confusion matrix
    tp = sum(1 for e in episodes if e["true_label"] == 0 and e["outcome"] in
             ("correct_block", "bot_blocked_puzzle"))
    tn = sum(1 for e in episodes if e["true_label"] == 1 and e["outcome"] == "correct_allow")
    fp = sum(1 for e in episodes if e["true_label"] == 1 and e["outcome"] in
             ("false_positive_block", "fp_puzzle"))
    fn = sum(1 for e in episodes if e["true_label"] == 0 and e["outcome"] in
             ("false_negative", "bot_passed_puzzle"))
    truncated = sum(1 for e in episodes if e["outcome"] == "truncated")
    other = n - tp - tn - fp - fn - truncated

    print("--- Confusion Matrix ---")
    print(f"  True Positives  (bot blocked):   {tp:4d} ({100*tp/n:.1f}%)")
    print(f"  True Negatives  (human allowed): {tn:4d} ({100*tn/n:.1f}%)")
    print(f"  False Positives (human blocked): {fp:4d} ({100*fp/n:.1f}%)")
    print(f"  False Negatives (bot allowed):   {fn:4d} ({100*fn/n:.1f}%)")
    print(f"  Truncated (indecisive):          {truncated:4d} ({100*truncated/n:.1f}%)")
    if other > 0:
        print(f"  Other:                           {other:4d} ({100*other/n:.1f}%)")
    print()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0

    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
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


if __name__ == "__main__":
    main()