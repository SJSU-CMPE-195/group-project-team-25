"""Agent inference service — wraps PPO+LSTM for live evaluation.

Loads the trained agent once on first use, then evaluates sessions
by replaying events through the LSTM and returning the decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root so rl_captcha imports work
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
    """Singleton service that loads PPO+LSTM agent for inference."""

    def __init__(self, checkpoint_path: str | None = None):
        if checkpoint_path is None:
            checkpoint_path = str(
                PROJECT_ROOT / "rl_captcha" / "agent" / "checkpoints" / "ppo_run1"
            )

        self.checkpoint_path = checkpoint_path
        self.env_config = EventEnvConfig()
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
        """Run agent over all events in a single batched forward pass."""
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

        # Encode all events into a single tensor
        event_types_map = ["mouse", "click", "key_down", "key_up", "scroll"]
        obs_list = []
        for evt in timeline:
            obs_list.append(encoder.encode(evt))

        obs_batch = np.stack(obs_list, axis=0)  # (T, 13)

        # Single batched forward pass through the LSTM
        self.agent.reset_hidden()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_batch).float().unsqueeze(0).to(self.agent.device)  # (1, T, 13)
            h, c = self.agent.get_hidden()
            all_logits, all_values, new_hidden = self.agent.network(obs_t, (h, c))
            # all_logits: (1, T, 7), all_values: (1, T, 1)
            all_probs = F.softmax(all_logits.squeeze(0), dim=-1).cpu().numpy()  # (T, 7)
            all_vals = all_values.squeeze(0).squeeze(-1).cpu().numpy()  # (T,)
            all_actions = np.argmax(all_probs, axis=-1)  # (T,)
            self.agent._hidden = new_hidden

        # Scan for the first terminal action
        terminal_actions = {2, 3, 4, 5, 6}
        final_decision = None

        # Build action history (lightweight — just store per-event data)
        action_history = []
        for i in range(len(timeline)):
            action = int(all_actions[i])
            probs = all_probs[i].tolist()
            step_info = {
                "event_idx": i,
                "event_type": event_types_map[timeline[i]["_type"]],
                "action": ACTION_NAMES[action],
                "action_index": action,
                "probs": [round(p, 4) for p in probs],
                "value": round(float(all_vals[i]), 4),
            }
            action_history.append(step_info)

            if action in terminal_actions and final_decision is None:
                chosen_action = action
                confidence = probs[action]

                # Confidence gating — never hard-block, always
                # give a solvable puzzle so humans can get through.
                if action == 6:
                    # Block → downgrade to hard puzzle (solvable)
                    chosen_action = 4
                elif action in (3, 4) and confidence < 0.6:
                    # Medium/hard puzzle with low confidence → easy puzzle
                    chosen_action = 2

                final_decision = {
                    "decision": ACTION_NAMES[chosen_action],
                    "action_index": chosen_action,
                    "original_action": ACTION_NAMES[action],
                    "confidence": round(confidence, 4),
                    "events_processed": i + 1,
                    "total_events": len(timeline),
                    "final_probs": [round(p, 4) for p in probs],
                    "final_value": round(float(all_vals[i]), 4),
                }
                # Keep building history for dashboard but decision is locked

        if final_decision is None:
            last_probs = all_probs[-1].tolist()
            final_decision = {
                "decision": "allow",
                "action_index": 5,
                "events_processed": len(timeline),
                "total_events": len(timeline),
                "final_probs": [round(p, 4) for p in last_probs],
                "final_value": round(float(all_vals[-1]), 4),
                "reason": "no_terminal_action",
            }

        final_decision["action_history"] = action_history
        return final_decision

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
