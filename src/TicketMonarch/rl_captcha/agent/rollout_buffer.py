"""On-policy rollout buffer for PPO with LSTM hidden state storage.

Stores transitions collected during one rollout period. Data is consumed
once (across multiple PPO epochs) and then discarded.

For LSTM: stores the initial hidden state at each episode boundary so
the PPO update can reconstruct hidden states by processing each episode
segment sequentially.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """Fixed-capacity on-policy buffer for PPO + LSTM."""

    capacity: int = 2048
    obs_dim: int = 13

    # Transition arrays (allocated in __post_init__)
    observations: np.ndarray = field(init=False)
    actions: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)
    dones: np.ndarray = field(init=False)
    log_probs: np.ndarray = field(init=False)
    values: np.ndarray = field(init=False)

    # GAE-computed
    advantages: np.ndarray = field(init=False)
    returns: np.ndarray = field(init=False)

    # LSTM hidden states at episode-start boundaries
    # List of (step_index, h_0, c_0)
    episode_starts: list = field(default_factory=list)

    ptr: int = 0

    def __post_init__(self):
        self.observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.advantages = np.zeros(self.capacity, dtype=np.float32)
        self.returns = np.zeros(self.capacity, dtype=np.float32)

    def reset(self):
        """Clear buffer for next rollout."""
        self.ptr = 0
        self.episode_starts = []

    def mark_episode_start(self, h: torch.Tensor, c: torch.Tensor):
        """Record LSTM hidden state at the start of a new episode."""
        self.episode_starts.append((
            self.ptr,
            h.detach().cpu().clone(),
            c.detach().cpu().clone(),
        ))

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        """Store one transition."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Compute GAE advantages and returns."""
        n = self.ptr
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_episode_segments(self) -> list[dict]:
        """Return data grouped by episode for sequential LSTM processing.

        Each segment has all transitions for one episode (or partial episode
        at rollout boundary), plus the initial LSTM hidden state.
        """
        segments = []
        n = self.ptr

        for i, (start_idx, h0, c0) in enumerate(self.episode_starts):
            if i + 1 < len(self.episode_starts):
                end_idx = self.episode_starts[i + 1][0]
            else:
                end_idx = n

            if end_idx <= start_idx:
                continue

            segments.append({
                "obs": torch.from_numpy(self.observations[start_idx:end_idx].copy()),
                "actions": torch.from_numpy(self.actions[start_idx:end_idx].copy()),
                "old_log_probs": torch.from_numpy(self.log_probs[start_idx:end_idx].copy()),
                "advantages": torch.from_numpy(self.advantages[start_idx:end_idx].copy()),
                "returns": torch.from_numpy(self.returns[start_idx:end_idx].copy()),
                "h0": h0,
                "c0": c0,
            })

        return segments
