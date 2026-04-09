"""Soft PPO agent with LSTM recurrence and adaptive entropy tuning.

Extends standard PPO with an automatically tuned entropy temperature α,
inspired by Soft Actor-Critic (SAC). Instead of a fixed entropy coefficient,
Soft PPO learns α via dual gradient descent to maintain a target entropy
level throughout training.

Key differences from standard PPO:
  - Learnable log-temperature parameter α (optimized separately)
  - Target entropy H* ≈ −ratio · log(1/|A|) drives exploration adaptively
  - Policy loss: L_clip − α · H(π)  (α replaces fixed entropy_coeff)
  - α update: minimise  α · (H(π) − H*).detach()

This yields more stable exploration than fixed entropy coefficients:
α increases when the policy becomes too deterministic (H < H*) and
decreases when it's too random (H > H*).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_captcha.config import PPOConfig
from .ppo_lstm import PPOLSTM


@dataclass
class SoftPPOConfig(PPOConfig):
    """PPO config extended with Soft PPO (adaptive entropy) hyperparameters."""

    # Target entropy as a fraction of maximum entropy log(|A|).
    # 0.5 means the policy should maintain ~50% of uniform-random entropy.
    target_entropy_ratio: float = 0.5

    # Learning rate for the entropy temperature α.
    alpha_lr: float = 3e-4

    # Initial value of log(α). exp(−2) ≈ 0.135.
    init_log_alpha: float = -2.0

    # Clamp α to avoid runaway values.
    alpha_min: float = 0.001
    alpha_max: float = 1.0


class SoftPPOLSTM(PPOLSTM):
    """Soft PPO agent with adaptive entropy temperature.

    Identical to PPOLSTM for rollout collection, GAE, value loss, and
    save/load. Overrides update() to use a learnable entropy coefficient
    and adds the α optimiser.
    """

    def __init__(
        self,
        obs_dim: int = 26,
        action_dim: int = 7,
        config: SoftPPOConfig | None = None,
        device: str = "auto",
    ):
        if config is None:
            config = SoftPPOConfig()
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config,
            device=device,
        )

        # Target entropy: H* = ratio * log(|A|)
        import math

        self.target_entropy = config.target_entropy_ratio * math.log(action_dim)

        # Learnable log-temperature
        self.log_alpha = torch.tensor(
            config.init_log_alpha,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.alpha_min = config.alpha_min
        self.alpha_max = config.alpha_max

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy temperature, clamped to [alpha_min, alpha_max]."""
        return self.log_alpha.exp().clamp(self.alpha_min, self.alpha_max)

    def update(self) -> dict[str, float]:
        """Run Soft PPO update over the rollout buffer.

        Same clipped surrogate as PPO, but the entropy bonus uses a
        learnable α instead of a fixed coefficient. α is updated via
        dual gradient descent targeting a desired entropy level.

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
        total_alpha_loss = 0.0
        total_steps = 0
        num_updates = 0

        for _epoch in range(cfg.num_epochs):
            random.shuffle(segments)
            for seg in segments:
                obs = seg["obs"].to(self.device)
                actions = seg["actions"].to(self.device)
                old_log_probs = seg["old_log_probs"].to(self.device)
                advantages = seg["advantages"].to(self.device)
                returns = seg["returns"].to(self.device)
                masks = seg.get("action_masks")

                h0 = seg["h0"].to(self.device)
                c0 = seg["c0"].to(self.device)

                # Forward pass
                obs_seq = obs.unsqueeze(0)
                mask_t = None
                if masks is not None:
                    mask_t = masks.to(self.device).unsqueeze(0)

                logits, values, _ = self.network(obs_seq, (h0, c0), action_mask=mask_t)
                logits = logits.squeeze(0)
                values = values.squeeze(0).squeeze(-1)

                # New log probs and entropy
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # ── PPO clipped surrogate (unchanged) ────────────
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - cfg.clip_eps,
                        1.0 + cfg.clip_eps,
                    )
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Clipped value loss (unchanged) ───────────────
                old_values = seg["old_values"].to(self.device)
                v_clipped = old_values + torch.clamp(
                    values - old_values,
                    -cfg.clip_eps,
                    cfg.clip_eps,
                )
                vl_unclipped = (values - returns) ** 2
                vl_clipped = (v_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()

                # ── Soft PPO: use adaptive α instead of fixed entropy_coeff ──
                alpha = self.alpha
                loss = (
                    policy_loss
                    + cfg.value_loss_coeff * value_loss
                    - alpha.detach() * entropy  # detach α for policy update
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                # ── α update: dual gradient descent ──────────────
                # Increase α when entropy is below target, decrease when above
                alpha_loss = self.alpha * (entropy.detach() - self.target_entropy)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                seg_len = len(actions)
                total_policy_loss += policy_loss.item() * seg_len
                total_value_loss += value_loss.item() * seg_len
                total_entropy += entropy.item() * seg_len
                total_alpha_loss += alpha_loss.item() * seg_len
                total_steps += seg_len
                num_updates += 1

        if total_steps == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / total_steps,
            "value_loss": total_value_loss / total_steps,
            "entropy": total_entropy / total_steps,
            "alpha": self.alpha.item(),
            "alpha_loss": total_alpha_loss / total_steps,
            "target_entropy": self.target_entropy,
            "num_segments": len(segments),
            "num_updates": num_updates,
        }

    # ── Save / Load (extend parent to include α state) ───────────

    def save(self, path) -> None:
        """Save network, optimizer, and α state."""
        from pathlib import Path as P

        path = P(path)
        # Let parent save the main checkpoint
        super().save(path)
        # Save α state alongside
        torch.save(
            {
                "log_alpha": self.log_alpha.detach().cpu(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
                "target_entropy": self.target_entropy,
            },
            path / "soft_ppo_alpha.pt",
        )

    def load(self, path) -> None:
        """Load network, optimizer, and α state from checkpoint."""
        from pathlib import Path as P

        path = P(path)
        super().load(path)
        alpha_path = path / "soft_ppo_alpha.pt"
        if alpha_path.exists():
            ckpt = torch.load(alpha_path, map_location=self.device, weights_only=False)
            self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
            self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
            # Move optimizer state tensors to correct device
            for state in self.alpha_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.target_entropy = ckpt.get("target_entropy", self.target_entropy)
