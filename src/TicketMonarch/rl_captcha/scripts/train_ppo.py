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
from rl_captcha.data.loader import load_from_directory
from rl_captcha.environment.event_env import EventEnv
from rl_captcha.agent.ppo_lstm import PPOLSTM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO+LSTM agent")
    p.add_argument("--data-dir", type=str, default="data/",
                    help="Path to data directory with human/ and bot/ subdirs")
    p.add_argument("--save-path", type=str,
                    default="rl_captcha/agent/checkpoints/ppo_run1",
                    help="Directory to save checkpoints")
    p.add_argument("--total-timesteps", type=int, default=None,
                    help="Override total timesteps (default from PPOConfig)")
    p.add_argument("--log-interval", type=int, default=1,
                    help="Print stats every N rollouts")
    p.add_argument("--save-interval", type=int, default=10,
                    help="Save checkpoint every N rollouts")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    if args.total_timesteps is not None:
        cfg.ppo.total_timesteps = args.total_timesteps

    # Load data
    print(f"Loading sessions from {args.data_dir}...")
    sessions = load_from_directory(args.data_dir)
    human_count = sum(1 for s in sessions if s.label == 1)
    bot_count = sum(1 for s in sessions if s.label == 0)
    print(f"  Loaded {len(sessions)} sessions ({human_count} human, {bot_count} bot)")

    if not sessions:
        print("ERROR: No sessions found. Place JSON files in data/human/ and data/bot/.")
        return

    # Create environment and agent
    env = EventEnv(sessions, config=cfg.event_env)
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
        rollout_stats = _collect_rollout(env, agent, cfg.ppo.rollout_steps)
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
                rollout_num, num_rollouts, total_steps,
                rollout_stats, update_metrics, t_elapsed,
            )

        # Save checkpoint
        if rollout_num % args.save_interval == 0:
            agent.save(args.save_path)
            print(f"  [Checkpoint saved to {args.save_path}]")

    # Final save
    agent.save(args.save_path)
    print(f"\nTraining complete. Final checkpoint saved to {args.save_path}")


def _collect_rollout(
    env: EventEnv,
    agent: PPOLSTM,
    num_steps: int,
) -> dict:
    """Collect transitions into the agent's rollout buffer.

    Returns summary statistics about the rollout.
    """
    stats: dict = defaultdict(list)
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    outcome_counts: dict[str, int] = defaultdict(int)

    obs, info = env.reset()
    agent.reset_hidden()
    ep_reward = 0.0
    ep_len = 0

    # Skip too-short sessions
    while info.get("too_short"):
        obs, info = env.reset()
        agent.reset_hidden()

    for _step in range(num_steps):
        action, log_prob, value = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.buffer.push(obs, action, reward, done, log_prob, value)

        ep_reward += reward
        ep_len += 1
        outcome_counts[info.get("outcome", "unknown")] += 1

        if done:
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            ep_reward = 0.0
            ep_len = 0

            obs, info = env.reset()
            agent.reset_hidden()
            while info.get("too_short"):
                obs, info = env.reset()
                agent.reset_hidden()
        else:
            obs = next_obs

    # Bootstrap value for GAE
    last_value = agent.get_value(obs) if not done else 0.0

    return {
        "last_value": last_value,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "outcome_counts": dict(outcome_counts),
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
    outcomes = rollout_stats["outcome_counts"]

    avg_reward = np.mean(ep_rewards) if ep_rewards else 0.0
    avg_length = np.mean(ep_lengths) if ep_lengths else 0.0
    num_episodes = len(ep_rewards)

    print(f"--- Rollout {rollout_num}/{num_rollouts} | "
          f"Steps: {total_steps} | "
          f"Time: {elapsed:.1f}s ---")
    print(f"  Episodes: {num_episodes} | "
          f"Avg reward: {avg_reward:.3f} | "
          f"Avg length: {avg_length:.1f}")

    if update_metrics:
        print(f"  Policy loss: {update_metrics.get('policy_loss', 0):.4f} | "
              f"Value loss: {update_metrics.get('value_loss', 0):.4f} | "
              f"Entropy: {update_metrics.get('entropy', 0):.4f}")

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