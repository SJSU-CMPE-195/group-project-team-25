"""Train the PPO+LSTM event-level CAPTCHA agent.

Usage:
    python -m rl_captcha.scripts.train_ppo \
        --data-dir data/ \
        --save-path rl_captcha/agent/checkpoints/ppo_run1 \
        --total-timesteps 500000
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from rl_captcha.config import Config
from rl_captcha.data.loader import load_from_directory, split_sessions
from rl_captcha.environment.event_env import EventEnv
from rl_captcha.agent.ppo_lstm import PPOLSTM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO+LSTM agent")
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Path to data directory with human/ and bot/ subdirs",
    )
    p.add_argument(
        "--save-path",
        type=str,
        default="rl_captcha/agent/checkpoints/ppo_run1",
        help="Directory to save checkpoints",
    )
    p.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total timesteps (default from PPOConfig)",
    )
    p.add_argument(
        "--log-interval", type=int, default=1, help="Print stats every N rollouts"
    )
    p.add_argument(
        "--save-interval", type=int, default=10, help="Save checkpoint every N rollouts"
    )
    p.add_argument(
        "--val-episodes",
        type=int,
        default=100,
        help="Number of validation episodes per checkpoint",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split",
    )
    return p.parse_args()


def _label_counts(sessions):
    h = sum(1 for s in sessions if s.label == 1)
    b = sum(1 for s in sessions if s.label == 0)
    return h, b


def main():
    args = parse_args()
    cfg = Config()

    if args.total_timesteps is not None:
        cfg.ppo.total_timesteps = args.total_timesteps

    # Load data
    print(f"Loading sessions from {args.data_dir}...")
    sessions = load_from_directory(args.data_dir)
    human_count, bot_count = _label_counts(sessions)
    print(f"  Loaded {len(sessions)} sessions ({human_count} human, {bot_count} bot)")

    if not sessions:
        print(
            "ERROR: No sessions found. Place JSON files in data/human/ and data/bot/."
        )
        return

    # Stratified 70/15/15 split
    train_sessions, val_sessions, test_sessions = split_sessions(
        sessions,
        train=0.70,
        val=0.15,
        test=0.15,
        seed=args.split_seed,
    )
    h_tr, b_tr = _label_counts(train_sessions)
    h_va, b_va = _label_counts(val_sessions)
    h_te, b_te = _label_counts(test_sessions)
    print(f"  Train: {len(train_sessions)} ({h_tr} human, {b_tr} bot)")
    print(f"  Val:   {len(val_sessions)} ({h_va} human, {b_va} bot)")
    print(f"  Test:  {len(test_sessions)} ({h_te} human, {b_te} bot)  [held out]")

    # Create environments (augmentation on for training, off for validation)
    train_env = EventEnv(train_sessions, config=cfg.event_env)
    if val_sessions:
        from dataclasses import replace

        val_cfg = replace(cfg.event_env, augment=False)
        val_env = EventEnv(val_sessions, config=val_cfg)
    else:
        val_env = None

    agent = PPOLSTM(
        obs_dim=cfg.event_env.event_dim,
        action_dim=7,
        config=cfg.ppo,
        device=args.device,
    )
    print(f"  Device: {agent.device}")
    print(f"  Rollout steps: {cfg.ppo.rollout_steps}")
    print(f"  Total timesteps: {cfg.ppo.total_timesteps}")
    print()

    total_steps = 0
    rollout_num = 0
    num_rollouts = cfg.ppo.total_timesteps // cfg.ppo.rollout_steps

    while total_steps < cfg.ppo.total_timesteps:
        rollout_num += 1
        t_start = time.time()

        # Collect rollout
        agent.buffer.reset()
        rollout_stats = _collect_rollout(train_env, agent, cfg.ppo.rollout_steps)
        total_steps += agent.buffer.ptr

        # Compute GAE (bootstrap with value of last observation)
        last_value = rollout_stats["last_value"]
        agent.buffer.compute_gae(
            last_value=last_value,
            gamma=cfg.ppo.gamma,
            gae_lambda=cfg.ppo.gae_lambda,
        )

        # PPO update
        update_metrics = agent.update()

        t_elapsed = time.time() - t_start

        # Logging
        if rollout_num % args.log_interval == 0:
            _print_rollout_stats(
                rollout_num,
                num_rollouts,
                total_steps,
                rollout_stats,
                update_metrics,
                t_elapsed,
            )

        # Save checkpoint + validation
        if rollout_num % args.save_interval == 0:
            agent.save(args.save_path)
            print(f"  [Checkpoint saved to {args.save_path}]")

            if val_env and args.val_episodes > 0:
                val_acc = _quick_validate(val_env, agent, args.val_episodes)
                print(
                    f"  [Val accuracy: {val_acc:.3f} over {args.val_episodes} episodes]"
                )
            print()

    # Final save
    agent.save(args.save_path)
    print(f"\nTraining complete. Final checkpoint saved to {args.save_path}")

    # Final validation
    if val_env and args.val_episodes > 0:
        val_acc = _quick_validate(val_env, agent, args.val_episodes)
        print(f"Final val accuracy: {val_acc:.3f}")


def _quick_validate(env: EventEnv, agent: PPOLSTM, num_episodes: int) -> float:
    """Run deterministic episodes on the validation set and return accuracy."""
    correct = 0
    total = 0

    for _ in range(num_episodes):
        obs, info = env.reset()
        agent.reset_hidden()

        while info.get("too_short"):
            obs, info = env.reset()
            agent.reset_hidden()

        true_label = info["true_label"]
        action_mask = info.get("action_mask")
        done = False

        while not done:
            action, _, _ = agent.select_action(
                obs, action_mask=action_mask, deterministic=True
            )
            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            action_mask = step_info.get("action_mask")

        outcome = step_info.get("outcome", "")
        if outcome in ("correct_block", "bot_blocked_puzzle", "correct_allow"):
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def _collect_rollout(
    env: EventEnv,
    agent: PPOLSTM,
    num_steps: int,
) -> dict:
    """Collect transitions into the agent's rollout buffer.

    Returns summary statistics about the rollout.
    """
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_windows: list[int] = []
    # Per-episode final outcomes (the actual classification decisions)
    episode_outcomes: dict[str, int] = defaultdict(int)

    obs, info = env.reset()
    agent.reset_hidden()
    ep_reward = 0.0
    ep_len = 0

    # Skip too-short sessions
    while info.get("too_short"):
        obs, info = env.reset()
        agent.reset_hidden()

    action_mask = info.get("action_mask")

    for _step in range(num_steps):
        action, log_prob, value = agent.select_action(obs, action_mask=action_mask)

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        agent.buffer.push(
            obs, action, reward, done, log_prob, value, action_mask=action_mask
        )

        ep_reward += reward
        ep_len += 1

        if done:
            # Record the FINAL outcome (the actual classification decision)
            episode_outcomes[step_info.get("outcome", "unknown")] += 1
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            episode_windows.append(step_info.get("total_windows", ep_len))
            ep_reward = 0.0
            ep_len = 0

            obs, info = env.reset()
            agent.reset_hidden()
            while info.get("too_short"):
                obs, info = env.reset()
                agent.reset_hidden()
            action_mask = info.get("action_mask")
        else:
            obs = next_obs
            action_mask = step_info.get("action_mask")

    # Bootstrap value for GAE
    last_value = agent.get_value(obs) if not done else 0.0

    return {
        "last_value": last_value,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_windows": episode_windows,
        "outcome_counts": dict(episode_outcomes),
        "steps_collected": agent.buffer.ptr,
    }


def _print_rollout_stats(
    rollout_num: int,
    num_rollouts: int,
    total_steps: int,
    rollout_stats: dict,
    update_metrics: dict,
    elapsed: float,
):
    ep_rewards = rollout_stats["episode_rewards"]
    ep_lengths = rollout_stats["episode_lengths"]
    ep_windows = rollout_stats["episode_windows"]
    outcomes = rollout_stats["outcome_counts"]

    avg_reward = np.mean(ep_rewards) if ep_rewards else 0.0
    avg_length = np.mean(ep_lengths) if ep_lengths else 0.0
    avg_windows = np.mean(ep_windows) if ep_windows else 0.0
    num_episodes = len(ep_rewards)

    print(
        f"--- Rollout {rollout_num}/{num_rollouts} | "
        f"Steps: {total_steps} | "
        f"Time: {elapsed:.1f}s ---"
    )
    print(
        f"  Episodes: {num_episodes} | "
        f"Avg reward: {avg_reward:.3f} | "
        f"Avg length: {avg_length:.1f} | "
        f"Avg windows: {avg_windows:.1f}"
    )

    if update_metrics:
        print(
            f"  Policy loss: {update_metrics.get('policy_loss', 0):.4f} | "
            f"Value loss: {update_metrics.get('value_loss', 0):.4f} | "
            f"Entropy: {update_metrics.get('entropy', 0):.4f}"
        )

    # Outcome breakdown
    total_outcomes = sum(outcomes.values())
    if total_outcomes > 0:
        parts = []
        for outcome, count in sorted(outcomes.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_outcomes
            parts.append(f"{outcome}: {pct:.1f}%")
        print(f"  Outcomes: {', '.join(parts)}")
    print()


if __name__ == "__main__":
    main()
