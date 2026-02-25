"""Agent inference service — wraps PPO+LSTM for live evaluation.

Loads the trained agent once on first use, then evaluates sessions
by replaying events through the LSTM and returning the decision.
"""

from __future__ import annotations

import sys
import threading
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

ACTION_NAMES = [
    "continue", "deploy_honeypot",
    "easy_puzzle", "medium_puzzle", "hard_puzzle",
    "allow", "block",
]

EVENT_MOUSE = 0
EVENT_CLICK = 1
EVENT_KEY_DOWN = 2
EVENT_KEY_UP = 3
EVENT_SCROLL = 4

INTERACTIVE_TAGS = {"INPUT", "BUTTON", "A", "SELECT", "TEXTAREA"}


class EventEncoder:
    """Standalone encoder matching EventEnv._encode_event() logic."""

    def __init__(self, config: EventEnvConfig | None = None):
        self.config = config or EventEnvConfig()
        self.prev_mouse_x = 0.0
        self.prev_mouse_y = 0.0
        self.prev_mouse_t = 0.0

    def reset(self):
        self.prev_mouse_x = 0.0
        self.prev_mouse_y = 0.0
        self.prev_mouse_t = 0.0

    def build_timeline(self, session: Session) -> list[dict]:
        """Merge all event types, subsample mouse, sort by time."""
        cfg = self.config
        events = []

        for i, evt in enumerate(session.mouse):
            if i % cfg.mouse_subsample == 0:
                events.append({"_type": EVENT_MOUSE, **evt})

        for evt in session.clicks:
            events.append({"_type": EVENT_CLICK, **evt})

        for evt in session.keystrokes:
            etype = EVENT_KEY_DOWN if evt.get("type") == "down" else EVENT_KEY_UP
            events.append({"_type": etype, **evt})

        for evt in session.scroll:
            events.append({"_type": EVENT_SCROLL, **evt})

        events.sort(key=lambda e: e.get("t", e.get("timestamp", 0)))

        if len(events) > cfg.max_events:
            events = events[: cfg.max_events]

        return events

    def encode(self, evt: dict) -> np.ndarray:
        """Encode one event into 13-dim vector."""
        cfg = self.config
        vec = np.zeros(cfg.event_dim, dtype=np.float32)
        etype = evt["_type"]

        vec[etype] = 1.0

        x = evt.get("x", evt.get("pageX", 0.0)) or 0.0
        y = evt.get("y", evt.get("pageY", 0.0)) or 0.0
        vec[5] = x / cfg.max_coord_x
        vec[6] = y / cfg.max_coord_y

        dt = evt.get("dt_since_last")
        if dt is None or not isinstance(dt, (int, float)):
            dt = 0.0
        dt = max(dt, 0.0)
        vec[7] = np.log1p(min(dt, cfg.max_dt_ms)) / np.log1p(cfg.max_dt_ms)

        if etype == EVENT_MOUSE:
            t = evt.get("t", 0.0)
            if self.prev_mouse_t > 0 and t > self.prev_mouse_t:
                dt_s = (t - self.prev_mouse_t) / 1000.0
                dist = np.sqrt(
                    (x - self.prev_mouse_x) ** 2 + (y - self.prev_mouse_y) ** 2
                )
                speed = dist / max(dt_s, 1e-6)
                vec[8] = min(speed, cfg.max_speed) / cfg.max_speed
            self.prev_mouse_x = x
            self.prev_mouse_y = y
            self.prev_mouse_t = t

        if etype == EVENT_SCROLL:
            dy = evt.get("dy", 0.0) or 0.0
            vec[9] = np.clip(dy / cfg.max_scroll_dy, -1.0, 1.0)

        if etype in (EVENT_KEY_DOWN, EVENT_KEY_UP):
            vec[10] = 1.0 if evt.get("key") is not None else 0.0

        if etype == EVENT_CLICK:
            vec[11] = 1.0 if evt.get("button") == "left" else 0.0

        if etype == EVENT_CLICK:
            target = evt.get("target", {})
            if isinstance(target, dict):
                tag = (target.get("tag") or "").upper()
                vec[12] = 1.0 if tag in INTERACTIVE_TAGS else 0.0

        return vec


class AgentService:
    """Singleton service that loads PPO+LSTM agent for inference and online learning."""

    def __init__(self, checkpoint_path: str | None = None):
        if checkpoint_path is None:
            checkpoint_path = str(
                PROJECT_ROOT / "rl_captcha" / "agent" / "checkpoints" / "ppo_run1"
            )

        self.checkpoint_path = checkpoint_path
        self.env_config = EventEnvConfig()
        # Lock prevents concurrent network access (Flask is threaded;
        # React StrictMode double-fires useEffect → duplicate requests)
        self._lock = threading.Lock()
        # CPU is faster for this model (~100K params). CUDA init alone takes 20-30s on Windows.
        self.agent = PPOLSTM(obs_dim=13, action_dim=7, device="cpu")

        cp = Path(checkpoint_path) / "ppo_lstm_checkpoint.pt"
        if cp.exists():
            self.agent.load(checkpoint_path)
            self.agent.network.eval()
            self._loaded = True
            print(f"[AgentService] Loaded checkpoint from {checkpoint_path}")
        else:
            self._loaded = False
            print(f"[AgentService] WARNING: No checkpoint at {checkpoint_path}, agent will allow all")

    def evaluate_session(self, session: Session) -> dict:
        """Run agent over ALL events, decide using final LSTM output.
        use the LAST N events' averaged probabilities — the LSTM's
        most informed assessment after processing all evidence.
        """
        with self._lock:
            return self._evaluate_session(session)

    def _evaluate_session(self, session: Session) -> dict:
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
        timeline = encoder.build_timeline(session)

        if len(timeline) < self.env_config.min_events:
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

        event_types_map = ["mouse", "click", "key_down", "key_up", "scroll"]
        obs_list = [encoder.encode(evt) for evt in timeline]
        obs_batch = np.stack(obs_list, axis=0)  # (T, 13)

        self.agent.reset_hidden()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_batch).float().unsqueeze(0).to(self.agent.device)
            h, c = self.agent.get_hidden()
            all_logits, all_values, new_hidden = self.agent.network(obs_t, (h, c))
            all_probs = F.softmax(all_logits.squeeze(0), dim=-1).cpu().numpy()  # (T, 7)
            all_vals = all_values.squeeze(0).squeeze(-1).cpu().numpy()  # (T,)
            all_actions = np.argmax(all_probs, axis=-1)  # (T,)
            self.agent._hidden = new_hidden

        # build per-event action history for dashboard
        action_history = []
        for i in range(len(timeline)):
            action_history.append({
                "event_idx": i,
                "event_type": event_types_map[timeline[i]["_type"]],
                "action": ACTION_NAMES[int(all_actions[i])],
                "action_index": int(all_actions[i]),
                "probs": [round(float(p), 4) for p in all_probs[i]],
                "value": round(float(all_vals[i]), 4),
            })

        # Decision: average the last 10% of events (min 5) 
        # The LSTM accumulates evidence; its final outputs are most reliable.
        n_tail = max(5, len(timeline) // 10)
        tail_probs = all_probs[-n_tail:].mean(axis=0)  # (7,)

        p_allow = float(tail_probs[5])
        p_block = float(tail_probs[6])
        p_puzzles = float(tail_probs[2] + tail_probs[3] + tail_probs[4])
        p_suspicious = p_block + p_puzzles

        # Confidence gating: need strong suspicious signal to challenge.
        # Default is ALLOW, challenge when confident.
        CHALLENGE_THRESHOLD = 0.45  # >45% suspicious to challenge

        if p_suspicious > CHALLENGE_THRESHOLD:
            if p_block > 0.4 or tail_probs[4] > 0.25:
                chosen = 4  # hard puzzle
            elif tail_probs[3] > tail_probs[2]:
                chosen = 3  # medium
            else:
                chosen = 2  # easy
            confidence = round(p_suspicious, 4)
        else:
            chosen = 5  # allow
            confidence = round(p_allow, 4)

        final_probs = [round(float(p), 4) for p in tail_probs]

        return {
            "decision": ACTION_NAMES[chosen],
            "action_index": chosen,
            "confidence": confidence,
            "events_processed": len(timeline),
            "total_events": len(timeline),
            "final_probs": final_probs,
            "final_value": round(float(all_vals[-1]), 4),
            "action_history": action_history,
            "p_allow": round(p_allow, 4),
            "p_suspicious": round(p_suspicious, 4),
        }

    def rolling_evaluate(self, session: Session) -> dict:
        """Rolling evaluation using the LAST events' probabilities."""
        with self._lock:
            return self._rolling_evaluate(session)

    def _rolling_evaluate(self, session: Session) -> dict:
        if not self._loaded:
            return {
                "bot_probability": 0.0,
                "deploy_honeypot": False,
                "events_processed": 0,
            }

        encoder = EventEncoder(self.env_config)
        timeline = encoder.build_timeline(session)

        if len(timeline) < self.env_config.min_events:
            return {
                "bot_probability": 0.0,
                "deploy_honeypot": False,
                "events_processed": len(timeline),
            }

        obs_list = [encoder.encode(evt) for evt in timeline]
        obs_batch = np.stack(obs_list, axis=0)

        self.agent.reset_hidden()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_batch).float().unsqueeze(0).to(self.agent.device)
            h, c = self.agent.get_hidden()
            all_logits, _, _ = self.agent.network(obs_t, (h, c))
            all_probs = F.softmax(all_logits.squeeze(0), dim=-1).cpu().numpy()  # (T, 7)
            all_actions = np.argmax(all_probs, axis=-1)

        # use last 20% of events (min 5) for bot probability
        n_tail = max(5, len(timeline) // 5)
        tail_probs = all_probs[-n_tail:]

        # exponential weighting: most recent events count more
        weights = np.exp(np.linspace(-2, 0, len(tail_probs)))
        weights /= weights.sum()

        # suspicious = puzzle + block probabilities
        suspicious_per_event = tail_probs[:, [2, 3, 4, 6]].sum(axis=1)
        bot_probability = float(np.dot(weights, suspicious_per_event))

        deploy_honeypot = bool(1 in all_actions)
        honeypot_steps = int((all_actions == 1).sum())

        return {
            "bot_probability": round(bot_probability, 4),
            "deploy_honeypot": deploy_honeypot,
            "honeypot_steps": honeypot_steps,
            "events_processed": len(timeline),
            "action_distribution": {
                ACTION_NAMES[i]: round(float(tail_probs[:, i].mean()), 4)
                for i in range(7)
            },
        }

    def online_learn(self, session: Session, true_label: int) -> dict:
        """Run one session through the environment and do an aggressive PPO update."""
        with self._lock:
            return self._online_learn(session, true_label)

    def _online_learn(self, session: Session, true_label: int) -> dict:
        """Replay ALL events and do a PPO update with known true label.

        process ALL events directly. For each event, the agent picks an
        action. Non-terminal events get a small continue reward. The final
        event gets the true-label reward (correct allow/block = positive,
        wrong = negative). LSTM sees the entire sequence.

        Online updates use 60% of the training LR and 3 epochs so the
        agent learns meaningfully from every confirmed session.
        """
        if not self._loaded:
            return {"updated": False, "reason": "no_checkpoint"}

        encoder = EventEncoder(self.env_config)
        timeline = encoder.build_timeline(session)

        if len(timeline) < self.env_config.min_events:
            return {"updated": False, "reason": "too_few_events",
                    "event_count": len(timeline)}

        cfg = self.env_config

        # encode all events
        obs_list = [encoder.encode(evt) for evt in timeline]

        # aggressive learning rate for online updates (60% of training LR)
        original_lr = self.agent.config.lr
        online_lr = original_lr * 0.6
        for pg in self.agent.optimizer.param_groups:
            pg['lr'] = online_lr

        # 3 epochs per online update for stronger gradient signal
        original_epochs = self.agent.config.num_epochs
        self.agent.config.num_epochs = 3

        self.agent.network.train()
        self.agent.buffer.reset()
        self.agent.reset_hidden()

        # replay ALL events through the agent
        for i, obs in enumerate(obs_list):
            action, log_prob, value = self.agent.select_action(obs)
            is_last = (i == len(obs_list) - 1)

            if is_last:
                # final event: reward based on true label vs agent's decision
                if action == 5:  # allow
                    reward = cfg.reward_correct_allow if true_label == 1 else cfg.penalty_false_negative
                elif action == 6:  # block
                    reward = cfg.reward_correct_block if true_label == 0 else cfg.penalty_false_positive
                elif action in (2, 3, 4):  # puzzle
                    # strong penalty for challenging a human, strong reward for catching a bot
                    reward = cfg.penalty_false_positive * 0.7 if true_label == 1 else cfg.reward_correct_block * 0.9
                else:
                    # continue/honeypot on last event — stronger penalty (should decide)
                    reward = -0.3
                done = True
            else:
                # non-terminal: small penalty for continuing (encourages eventual decision)
                reward = -cfg.continue_penalty
                done = False

            self.agent.buffer.push(obs, action, reward, done, log_prob, value)

        self.agent.buffer.compute_gae(
            last_value=0.0,  # episode ended
            gamma=self.agent.config.gamma,
            gae_lambda=self.agent.config.gae_lambda,
        )

        metrics = self.agent.update()

        # restore original settings
        self.agent.config.num_epochs = original_epochs
        for pg in self.agent.optimizer.param_groups:
            pg['lr'] = original_lr

        self.agent.save(self.checkpoint_path)
        self.agent.network.eval()

        return {
            "updated": True,
            "steps": len(obs_list),
            "true_label": true_label,
            "online_lr": online_lr,
            "metrics": metrics,
        }

    def get_hidden_state_info(self) -> dict:
        """Return LSTM hidden state for visualization."""
        h, c = self.agent.get_hidden()
        return {
            "lstm_hidden_norm": round(float(h.norm().item()), 4),
            "lstm_cell_norm": round(float(c.norm().item()), 4),
            "lstm_hidden_values": [round(v, 4) for v in h.squeeze().cpu().numpy().tolist()],
        }


_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service