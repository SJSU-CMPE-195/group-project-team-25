"""Delightful Policy Gradient (DG) agent with LSTM recurrence.

Inherits the full PPO+LSTM infrastructure (network, buffer, rollout,
save/load) and overrides only the policy loss computation with the
delight-gated surrogate from:

    "Delightful Policy Gradients" (arXiv:2603.14608, 2025)

Key idea (Algorithm 1 from the paper): each gradient term is gated by
σ(delight / η) where
    delight  χ_t = U_t · ℓ_t
    surprisal ℓ_t = −log π_θ(A_t | H_t)   (CURRENT policy, not old)

This amplifies rare successes ("delightful" outcomes) and suppresses
rare failures, fixing two pathologies of standard policy gradients:
  1. Noisy gradients from low-probability bad actions (within-context)
  2. Gradient budget wasted on easy contexts (across-context)

Per the paper, DG is a drop-in replacement for the policy gradient
and does NOT use importance ratios or PPO-style clipping by default.
An optional PPO blend is available via dg_baseline_weight for
practical stability.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rl_captcha.config import PPOConfig
from .ppo_lstm import PPOLSTM


@dataclass
class DGConfig(PPOConfig):
    """PPO config extended with DG-specific hyperparameters."""

    dg_temperature: float = 1.0       # η — controls sigmoid sharpness (paper uses 1.0)
    dg_baseline_weight: float = 0.0   # blend: 0 = pure DG (paper default), 1 = pure PPO


class DGLSTM(PPOLSTM):
    """Delightful Policy Gradient agent wrapping an LSTMActorCritic network.

    Identical to PPOLSTM except for the policy loss in update().
    The value loss, entropy bonus, GAE, and rollout collection are unchanged.
    """

    def __init__(
        self,
        obs_dim: int = 26,
        action_dim: int = 7,
        config: DGConfig | None = None,
        device: str = "auto",
    ):
        if config is None:
            config = DGConfig()
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config,
            device=device,
        )
        self.dg_temperature = config.dg_temperature
        self.dg_baseline_weight = config.dg_baseline_weight

    def update(self) -> dict[str, float]:
        """Run DG update over the rollout buffer.

        Implements Algorithm 1 from arXiv:2603.14608:

            ℓ_t ← −log π_θ(A_t | H_t)       (current policy surprisal)
            χ_t ← U_t · ℓ_t                   (delight)
            w_t ← σ(χ_t / η)                  (gate)
            Δθ ← Δθ + w_t · U_t · ∇_θ log π_θ(A_t | H_t)

        The value loss and entropy bonus remain identical to PPO.
        """
        cfg = self.config
        segments = self.buffer.get_episode_segments()

        if not segments:
            return {}

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_delight_mean = 0.0
        total_gate_mean = 0.0
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

                # Current policy log probs and entropy
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # ── Delight-gated policy loss (Algorithm 1) ────────────
                # surprisal ℓ_t = −log π_θ(A_t | H_t) — CURRENT policy
                surprisal = -new_log_probs

                # delight χ_t = U_t · ℓ_t
                delight = advantages * surprisal

                # Gate: w_t = σ(χ_t / η) ∈ (0, 1)
                gate = torch.sigmoid(delight / self.dg_temperature)

                # DG gradient: w_t · U_t · ∇_θ log π_θ(A_t | H_t)
                # As a loss to minimize: −(gate · advantage · log_prob)
                dg_loss = -(gate.detach() * advantages * new_log_probs).mean()

                # Optionally blend with PPO clipped surrogate for stability
                if self.dg_baseline_weight > 0:
                    ratio = (new_log_probs - old_log_probs).exp()
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(
                        ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
                    ) * advantages
                    ppo_loss = -torch.min(surr1, surr2).mean()

                    w = self.dg_baseline_weight
                    policy_loss = (1 - w) * dg_loss + w * ppo_loss
                else:
                    policy_loss = dg_loss

                # Clipped value loss (unchanged from PPO — paper only gates policy)
                old_values = seg["old_values"].to(self.device)
                v_clipped = old_values + torch.clamp(
                    values - old_values, -cfg.clip_eps, cfg.clip_eps
                )
                vl_unclipped = (values - returns) ** 2
                vl_clipped = (v_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()

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
                total_delight_mean += delight.detach().mean().item() * seg_len
                total_gate_mean += gate.detach().mean().item() * seg_len
                total_steps += seg_len
                num_updates += 1

        if total_steps == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / total_steps,
            "value_loss": total_value_loss / total_steps,
            "entropy": total_entropy / total_steps,
            "delight_mean": total_delight_mean / total_steps,
            "gate_mean": total_gate_mean / total_steps,
            "num_segments": len(segments),
            "num_updates": num_updates,
        }
