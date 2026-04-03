"""Agent inference service — wraps PPO+LSTM for live evaluation.

Loads the trained agent once on first use, then evaluates sessions
by replaying events through the LSTM and returning the decision.

Action masking: non-final windows only allow continue/honeypot (0,1).
The final window only allows terminal actions (2-6: puzzles, allow, block).
This matches the training environment exactly.
"""

from __future__ import annotations

import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rl_captcha.agent.ppo_lstm import PPOLSTM
from rl_captcha.config import EventEnvConfig
from rl_captcha.data.loader import Session
from rl_captcha.environment.event_env import EventEncoder, ACTION_NAMES

# Action masks matching EventEnv
NON_FINAL_MASK = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
FINAL_MASK = np.array([0, 0, 1, 1, 1, 1, 1], dtype=np.float32)


class AgentService:
    """Singleton service that loads PPO+LSTM agent for inference and online learning."""

    def __init__(self, checkpoint_path: str | None = None):
        if checkpoint_path is None:
            checkpoint_path = str(
                PROJECT_ROOT / "rl_captcha" / "agent" / "checkpoints" / "ppo_run1"
            )

        self.checkpoint_path = checkpoint_path
        self.env_config = EventEnvConfig()
        self._online_update_count = 0

        # online training logger
        log_path = PROJECT_ROOT / "online_training.log"
        self._online_logger = logging.getLogger("online_training")
        self._online_logger.setLevel(logging.INFO)
        self._online_logger.propagate = False
        if not self._online_logger.handlers:
            fh = logging.FileHandler(str(log_path), encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(message)s"))
            self._online_logger.addHandler(fh)

        self._lock = threading.Lock()
        self.agent = PPOLSTM(
            obs_dim=self.env_config.event_dim,
            action_dim=7,
            device="cpu",
        )

        cp = Path(checkpoint_path) / "ppo_lstm_checkpoint.pt"
        if cp.exists():
            self.agent.load(checkpoint_path)
            self.agent.network.eval()
            self._loaded = True
            print(f"[AgentService] Loaded checkpoint from {checkpoint_path}")
        else:
            self._loaded = False
            print(
                f"[AgentService] WARNING: No checkpoint at {checkpoint_path}, agent will allow all"
            )

    def _build_windows(self, session: Session) -> tuple[list[dict], list[list[dict]]]:
        """Build timeline and split into overlapping windows."""
        encoder = EventEncoder(self.env_config)
        timeline = encoder.build_timeline(session)

        ws = self.env_config.window_size
        stride = ws // 2
        windows = []
        for start in range(0, len(timeline), stride):
            window = timeline[start : start + ws]
            if len(window) >= self.env_config.min_events:
                windows.append(window)

        return timeline, windows

    def evaluate_session(self, session: Session) -> dict:
        """Run agent over all windows with action masking, decide on final window."""
        with self._lock:
            return self._evaluate_session(session)

    def _evaluate_session(self, session: Session) -> dict:
        """Process all windows with proper action masking.

        Non-final windows: mask to continue/honeypot only (observe phase).
        Final window: mask to terminal actions only (decision phase).
        Matches training environment exactly.
        """
        if not self._loaded:
            return {
                "decision": "allow",
                "action_index": 5,
                "events_processed": 0,
                "total_events": 0,
                "action_history": [],
                "final_probs": [0] * 7,
                "final_value": 0.0,
                "reason": "no_checkpoint",
            }

        encoder = EventEncoder(self.env_config)
        timeline, windows = self._build_windows(session)

        if not windows:
            return {
                "decision": "allow",
                "action_index": 5,
                "events_processed": len(timeline),
                "total_events": len(timeline),
                "action_history": [],
                "final_probs": [0] * 7,
                "final_value": 0.0,
                "reason": "too_few_events",
            }

        self.agent.reset_hidden()
        action_history = []

        with torch.no_grad():
            for i, window in enumerate(windows):
                is_final = i == len(windows) - 1
                mask = FINAL_MASK if is_final else NON_FINAL_MASK
                mask_t = (
                    torch.from_numpy(mask).float().unsqueeze(0).to(self.agent.device)
                )

                obs = encoder.encode_window(window)
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.agent.device)
                h, c = self.agent.get_hidden()

                logits, values, new_hidden = self.agent.network(
                    obs_t, (h, c), action_mask=mask_t
                )
                probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze()
                value = float(values.squeeze().item())
                action = int(np.argmax(probs))
                self.agent._hidden = new_hidden

                action_history.append(
                    {
                        "window_idx": i,
                        "window_events": len(window),
                        "action": ACTION_NAMES[action],
                        "action_index": action,
                        "probs": [round(float(p), 4) for p in probs],
                        "value": round(value, 4),
                        "is_final": is_final,
                    }
                )

        last = action_history[-1]
        final_probs_arr = np.array(last["probs"])
        chosen = last["action_index"]

        probs_list = [round(float(p), 4) for p in final_probs_arr]
        p_allow = float(final_probs_arr[5])
        p_block = float(final_probs_arr[6])
        p_puzzles = float(final_probs_arr[2] + final_probs_arr[3] + final_probs_arr[4])
        p_suspicious = p_block + p_puzzles

        return {
            "decision": ACTION_NAMES[chosen],
            "action_index": chosen,
            "confidence": round(max(p_allow, p_suspicious), 4),
            "events_processed": len(timeline),
            "total_events": len(timeline),
            "num_windows": len(windows),
            "windows_processed": len(action_history),
            "final_probs": probs_list,
            "final_value": round(last["value"], 4),
            "action_history": action_history,
            "p_allow": round(p_allow, 4),
            "p_suspicious": round(p_suspicious, 4),
        }

    def rolling_evaluate(self, session: Session) -> dict:
        """Rolling evaluation — bot probability from final window."""
        with self._lock:
            return self._rolling_evaluate(session)

    def _rolling_evaluate(self, session: Session) -> dict:
        """Process all windows with masking, return bot probability."""
        if not self._loaded:
            return {
                "bot_probability": 0.0,
                "deploy_honeypot": False,
                "events_processed": 0,
            }

        encoder = EventEncoder(self.env_config)
        timeline, windows = self._build_windows(session)

        if not windows:
            return {
                "bot_probability": 0.0,
                "deploy_honeypot": False,
                "events_processed": len(timeline),
            }

        self.agent.reset_hidden()
        deploy_honeypot = False
        last_probs = None

        with torch.no_grad():
            for i, window in enumerate(windows):
                is_final = i == len(windows) - 1
                mask = FINAL_MASK if is_final else NON_FINAL_MASK
                mask_t = (
                    torch.from_numpy(mask).float().unsqueeze(0).to(self.agent.device)
                )

                obs = encoder.encode_window(window)
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.agent.device)
                h, c = self.agent.get_hidden()

                logits, _, new_hidden = self.agent.network(
                    obs_t, (h, c), action_mask=mask_t
                )
                probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze()
                action = int(np.argmax(probs))
                self.agent._hidden = new_hidden

                if not is_final:
                    last_probs = probs  # track non-final for honeypot
                    if action == 1:
                        deploy_honeypot = True
                else:
                    last_probs = probs  # final window probs for decision

        # suspicious = puzzle + block probabilities from final window
        bot_probability = float(
            last_probs[2] + last_probs[3] + last_probs[4] + last_probs[6]
        )

        return {
            "bot_probability": round(bot_probability, 4),
            "deploy_honeypot": deploy_honeypot,
            "events_processed": len(timeline),
            "num_windows": len(windows),
            "action_distribution": {
                ACTION_NAMES[i]: round(float(last_probs[i]), 4) for i in range(7)
            },
        }

    def online_learn(self, session: Session, true_label: int) -> dict:
        """One-session PPO update with action masking (matches offline training)."""
        with self._lock:
            return self._online_learn(session, true_label)

    def _online_learn(self, session: Session, true_label: int) -> dict:
        """Replay session windows with action masking and do a PPO update.

        Matches offline training exactly:
        - Non-final windows: masked to continue/honeypot, get continue_penalty
        - Final window: masked to terminal actions, get true-label reward
        - Same GAE computation
        """
        if not self._loaded:
            return {"updated": False, "reason": "no_checkpoint"}

        encoder = EventEncoder(self.env_config)
        timeline, windows = self._build_windows(session)

        if not windows:
            return {
                "updated": False,
                "reason": "too_few_events",
                "event_count": len(timeline),
            }

        cfg = self.env_config
        label_str = "human" if true_label == 1 else "bot"

        # evaluate before update
        before = self._evaluate_session(session)

        # encode all windows
        obs_list = [encoder.encode_window(w) for w in windows]

        # online learning rate (60% of training LR)
        original_lr = self.agent.config.lr
        online_lr = original_lr * 0.6
        for pg in self.agent.optimizer.param_groups:
            pg["lr"] = online_lr

        original_epochs = self.agent.config.num_epochs
        self.agent.config.num_epochs = 3

        self.agent.network.train()
        self.agent.buffer.reset()
        self.agent.reset_hidden()

        # replay all windows with proper action masking
        for i, obs in enumerate(obs_list):
            is_last = i == len(obs_list) - 1
            mask = FINAL_MASK if is_last else NON_FINAL_MASK

            action, log_prob, value = self.agent.select_action(obs, action_mask=mask)

            if is_last:
                # terminal reward based on true label
                if action == 5:  # allow
                    reward = (
                        cfg.reward_correct_allow
                        if true_label == 1
                        else cfg.penalty_false_negative
                    )
                elif action == 6:  # block
                    reward = (
                        cfg.reward_correct_block
                        if true_label == 0
                        else cfg.penalty_false_positive
                    )
                elif action in (2, 3, 4):  # puzzle
                    reward = (
                        cfg.penalty_false_positive * 0.7
                        if true_label == 1
                        else cfg.reward_correct_block * 0.9
                    )
                else:
                    reward = -0.3  # should not happen with masking
                done = True
            else:
                reward = -cfg.continue_penalty
                done = False

            self.agent.buffer.push(
                obs, action, reward, done, log_prob, value, action_mask=mask
            )

        self.agent.buffer.compute_gae(
            last_value=0.0,
            gamma=self.agent.config.gamma,
            gae_lambda=self.agent.config.gae_lambda,
        )

        metrics = self.agent.update()

        # restore settings
        self.agent.config.num_epochs = original_epochs
        for pg in self.agent.optimizer.param_groups:
            pg["lr"] = original_lr

        self.agent.network.eval()

        # evaluate after update
        after = self._evaluate_session(session)

        self._online_update_count += 1
        before_correct = (before["decision"] == "allow" and true_label == 1) or (
            before["decision"] != "allow" and true_label == 0
        )
        after_correct = (after["decision"] == "allow" and true_label == 1) or (
            after["decision"] != "allow" and true_label == 0
        )

        improvement = (
            "IMPROVED"
            if (not before_correct and after_correct)
            else "REGRESSED" if (before_correct and not after_correct) else "UNCHANGED"
        )

        self.agent.save(self.checkpoint_path)

        log = self._online_logger
        log.info(
            f"--- Online Update #{self._online_update_count} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} ---"
        )
        log.info(
            f"  True label: {label_str} | Events: {len(timeline)} | Windows: {len(windows)}"
        )
        log.info(
            f"  BEFORE: decision={before['decision']} p_allow={before.get('p_allow', 0):.4f} p_suspicious={before.get('p_suspicious', 0):.4f} {'CORRECT' if before_correct else 'WRONG'}"
        )
        log.info(
            f"  AFTER:  decision={after['decision']} p_allow={after.get('p_allow', 0):.4f} p_suspicious={after.get('p_suspicious', 0):.4f} {'CORRECT' if after_correct else 'WRONG'}"
        )
        log.info(
            f"  Result: {improvement} | Policy loss: {metrics.get('policy_loss', 0):.4f} | Value loss: {metrics.get('value_loss', 0):.4f}"
        )
        log.info("")

        return {
            "updated": True,
            "steps": len(obs_list),
            "true_label": true_label,
            "online_lr": online_lr,
            "metrics": metrics,
            "before_decision": before["decision"],
            "after_decision": after["decision"],
            "improvement": improvement,
        }

    def get_hidden_state_info(self) -> dict:
        """Return LSTM hidden state for visualization (last layer only)."""
        h, c = self.agent.get_hidden()
        # h shape: (num_layers, 1, hidden_size) — use last layer for viz
        last_h = h[-1].squeeze(0).cpu().numpy()  # (hidden_size,)
        return {
            "lstm_hidden_norm": round(float(h.norm().item()), 4),
            "lstm_cell_norm": round(float(c.norm().item()), 4),
            "lstm_hidden_values": [round(float(v), 4) for v in last_h.tolist()],
        }


_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service
