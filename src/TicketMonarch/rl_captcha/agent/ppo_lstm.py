"""PPO agent with LSTM recurrence for event-level CAPTCHA defence.

Collects on-policy rollouts with hidden-state tracking, then performs
multiple epochs of clipped surrogate updates over episode segments
(processed sequentially to reconstruct LSTM hidden states).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_captcha.config import PPOConfig

from .lstm_networks import LSTMActorCritic
from .rollout_buffer import RolloutBuffer


class PPOLSTM:
    """PPO agent wrapping an LSTMActorCritic network."""

    def __init__(
        self,
        obs_dim: int = 13,
        action_dim: int = 7,
        config: PPOConfig | None = None,
        device: str = "auto",
    ):
        self.config = config or PPOConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network = LSTMActorCritic(
            input_dim=obs_dim,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            action_dim=action_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.lr)

        self.buffer = RolloutBuffer(
            capacity=self.config.rollout_steps,
            obs_dim=obs_dim,
        )

        # Current LSTM hidden state (used during rollout collection)
        self._hidden: Tuple[torch.Tensor, torch.Tensor] | None = None

    # ── Hidden state management ─────────────────────────────────────

    def reset_hidden(self) -> None:
        """Reset LSTM hidden state and mark episode start in buffer."""
        self._hidden = self.network.init_hidden(batch_size=1, device=self.device)
        self.buffer.mark_episode_start(self._hidden[0], self._hidden[1])

    def get_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._hidden is None:
            self.reset_hidden()
        return self._hidden

    # ── Action selection ────────────────────────────────────────────

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Pick an action for one observation during rollout.

        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            h, c = self.get_hidden()

            logits, values, new_hidden = self.network(obs_t, (h, c))
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            value = values.squeeze(-1)

            self._hidden = new_hidden

        return int(action.item()), float(log_prob.item()), float(value.item())

    def get_value(self, obs: np.ndarray) -> float:
        """Estimate V(s) for the current observation (for GAE bootstrap)."""
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            h, c = self.get_hidden()
            _, values, _ = self.network(obs_t, (h, c))
            return float(values.squeeze(-1).item())

    # ── PPO update ──────────────────────────────────────────────────

    def update(self) -> dict[str, float]:
        """Run PPO clipped surrogate update over the rollout buffer.

        Processes episode segments sequentially through the LSTM to
        reconstruct hidden states (no minibatch shuffling).

        Returns:
            Dictionary of loss metrics averaged over all epochs.
        """
        cfg = self.config
        segments = self.buffer.get_episode_segments()

        if not segments:
            return {}

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_steps = 0
        num_updates = 0

        for _epoch in range(cfg.num_epochs):
            for seg in segments:
                obs = seg["obs"].to(self.device)            # (T, obs_dim)
                actions = seg["actions"].to(self.device)     # (T,)
                old_log_probs = seg["old_log_probs"].to(self.device)  # (T,)
                advantages = seg["advantages"].to(self.device)        # (T,)
                returns = seg["returns"].to(self.device)              # (T,)

                h0 = seg["h0"].to(self.device)  # (num_layers, 1, hidden)
                c0 = seg["c0"].to(self.device)

                # Normalize advantages within segment
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Forward pass: process entire segment sequentially
                # obs shape: (T, obs_dim) → add batch dim → (1, T, obs_dim)
                obs_seq = obs.unsqueeze(0)
                logits, values, _ = self.network(obs_seq, (h0, c0))

                # logits: (1, T, action_dim), values: (1, T, 1)
                logits = logits.squeeze(0)   # (T, action_dim)
                values = values.squeeze(0).squeeze(-1)  # (T,)

                # Compute new log probs and entropy
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # PPO clipped surrogate
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, returns)

                # Total loss
                loss = (
                    policy_loss
                    + cfg.value_loss_coeff * value_loss
                    - cfg.entropy_coeff * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), cfg.max_grad_norm,
                )
                self.optimizer.step()

                seg_len = len(actions)
                total_policy_loss += policy_loss.item() * seg_len
                total_value_loss += value_loss.item() * seg_len
                total_entropy += entropy.item() * seg_len
                total_steps += seg_len
                num_updates += 1

        if total_steps == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / total_steps,
            "value_loss": total_value_loss / total_steps,
            "entropy": total_entropy / total_steps,
            "num_segments": len(segments),
            "num_updates": num_updates,
        }

    # ── Save / Load ─────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save network and optimizer state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "config": {
                    "lr": self.config.lr,
                    "gamma": self.config.gamma,
                    "gae_lambda": self.config.gae_lambda,
                    "clip_eps": self.config.clip_eps,
                    "value_loss_coeff": self.config.value_loss_coeff,
                    "entropy_coeff": self.config.entropy_coeff,
                    "max_grad_norm": self.config.max_grad_norm,
                    "lstm_hidden_size": self.config.lstm_hidden_size,
                    "lstm_num_layers": self.config.lstm_num_layers,
                    "rollout_steps": self.config.rollout_steps,
                    "num_epochs": self.config.num_epochs,
                },
            },
            path / "ppo_lstm_checkpoint.pt",
        )

    def load(self, path: str | Path) -> None:
        """Load network and optimizer state."""
        path = Path(path)
        checkpoint = torch.load(
            path / "ppo_lstm_checkpoint.pt", map_location=self.device,
        )
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])