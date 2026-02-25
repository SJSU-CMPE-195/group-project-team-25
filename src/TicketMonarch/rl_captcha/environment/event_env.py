"""Event-level Gymnasium environment for CAPTCHA defender with LSTM agent.

Each timestep = one telemetry event (mouse move, click, keystroke, scroll).
The agent processes raw events through its LSTM and decides an action at
every event. Most actions should be 'continue' (action 0). Terminal actions
end the episode.

No feature extraction, no classifier — the LSTM learns its own features.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from rl_captcha.config import EventEnvConfig
from rl_captcha.data.loader import Session

ACTION_NAMES = [
    "continue", "deploy_honeypot",
    "easy_puzzle", "medium_puzzle", "hard_puzzle",
    "allow", "block",
]

# Event type indices for one-hot encoding
EVENT_MOUSE = 0
EVENT_CLICK = 1
EVENT_KEY_DOWN = 2
EVENT_KEY_UP = 3
EVENT_SCROLL = 4

INTERACTIVE_TAGS = {"INPUT", "BUTTON", "A", "SELECT", "TEXTAREA"}


class EventEnv(gym.Env):
    """Event-level CAPTCHA environment.

    Observation: 13-dim encoded event vector.
    Action: 7 discrete (continue, honeypot, 3 puzzles, allow, block).
    Episode: one user session's events presented sequentially.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        sessions: Sequence[Session],
        config: EventEnvConfig | None = None,
    ):
        super().__init__()
        self.config = config or EventEnvConfig()
        self._sessions = list(sessions)

        # Separate by label for balanced sampling
        self._human_sessions = [s for s in self._sessions if s.label == 1]
        self._bot_sessions = [s for s in self._sessions if s.label == 0]

        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.config.event_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(7)

        # Per-episode state
        self._timeline: list[dict] = []
        self._event_idx: int = 0
        self._current_session: Session | None = None
        self._honeypot_deployed: bool = False
        self._honeypot_triggered: bool = False
        self._num_honeypots: int = 0
        self._honeypot_info_bonus_pending: float = 0.0
        self._prev_mouse_x: float = 0.0
        self._prev_mouse_y: float = 0.0
        self._prev_mouse_t: float = 0.0

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Balanced 50/50 sampling
        if self._human_sessions and self._bot_sessions:
            if random.random() < 0.5:
                session = random.choice(self._human_sessions)
            else:
                session = random.choice(self._bot_sessions)
        else:
            session = random.choice(self._sessions)

        self._current_session = session
        self._event_idx = 0
        self._honeypot_deployed = False
        self._honeypot_triggered = False
        self._num_honeypots = 0
        self._honeypot_info_bonus_pending = 0.0
        self._prev_mouse_x = 0.0
        self._prev_mouse_y = 0.0
        self._prev_mouse_t = 0.0

        self._timeline = self._build_timeline(session)

        info = {
            "session_id": session.session_id,
            "true_label": session.label,
            "total_events": len(self._timeline),
        }

        if len(self._timeline) < self.config.min_events:
            info["too_short"] = True
            return np.zeros(self.config.event_dim, dtype=np.float32), info

        obs = self._encode_event(self._timeline[0])
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._current_session is not None
        cfg = self.config
        true_label = self._current_session.label  # 1=human, 0=bot

        reward = 0.0
        terminated = False
        truncated = False
        outcome = "continue"

        # Collect pending honeypot bonus
        reward += self._honeypot_info_bonus_pending
        self._honeypot_info_bonus_pending = 0.0

        if action == 0:  # continue
            reward -= cfg.continue_penalty
            outcome = "continue"

        elif action == 1:  # deploy_honeypot
            reward -= cfg.action_costs[1]
            if self._num_honeypots >= cfg.max_honeypots:
                reward -= cfg.continue_penalty
                outcome = "honeypot_maxed"
            else:
                self._honeypot_deployed = True
                self._num_honeypots += 1
                if true_label == 0:
                    triggered = random.random() < cfg.honeypot_trigger_rate_bot
                else:
                    triggered = random.random() < cfg.honeypot_trigger_rate_human
                self._honeypot_triggered = triggered
                if triggered and true_label == 0:
                    self._honeypot_info_bonus_pending = cfg.honeypot_info_bonus
                    outcome = "honeypot_bot_triggered"
                elif triggered:
                    outcome = "honeypot_human_triggered"
                else:
                    outcome = "honeypot_no_trigger"

        elif action in (2, 3, 4):  # puzzle
            terminated = True
            human_pass, bot_pass = cfg.puzzle_pass_rates[action]
            if true_label == 1:
                reward = cfg.penalty_false_positive * (1.0 - human_pass)
                outcome = "fp_puzzle"
            else:
                if random.random() < bot_pass:
                    reward = cfg.penalty_false_negative * 0.5
                    outcome = "bot_passed_puzzle"
                else:
                    reward = cfg.reward_correct_block
                    outcome = "bot_blocked_puzzle"
            reward -= cfg.action_costs[action]

        elif action == 5:  # allow
            terminated = True
            if true_label == 1:
                reward = cfg.reward_correct_allow
                outcome = "correct_allow"
            else:
                reward = cfg.penalty_false_negative
                outcome = "false_negative"

        elif action == 6:  # block
            terminated = True
            if true_label == 0:
                reward = cfg.reward_correct_block
                outcome = "correct_block"
            else:
                reward = cfg.penalty_false_positive
                outcome = "false_positive_block"

        # Advance to next event for non-terminal actions
        if not terminated:
            self._event_idx += 1
            if self._event_idx >= len(self._timeline):
                truncated = True
                reward += cfg.truncation_penalty
                outcome = "truncated"

        # Build next observation
        if terminated or truncated:
            obs = np.zeros(cfg.event_dim, dtype=np.float32)
        else:
            obs = self._encode_event(self._timeline[self._event_idx])

        info = {
            "session_id": self._current_session.session_id,
            "true_label": true_label,
            "action": ACTION_NAMES[action],
            "outcome": outcome,
            "reward": reward,
            "event_idx": self._event_idx,
            "total_events": len(self._timeline),
            "honeypot_deployed": self._honeypot_deployed,
            "honeypot_triggered": self._honeypot_triggered,
        }
        return obs, reward, terminated, truncated, info

    def _build_timeline(self, session: Session) -> list[dict]:
        """Merge all events, subsample mouse, sort by timestamp."""
        events = []

        for i, evt in enumerate(session.mouse):
            if i % self.config.mouse_subsample == 0:
                events.append({"_type": EVENT_MOUSE, **evt})

        for evt in session.clicks:
            events.append({"_type": EVENT_CLICK, **evt})

        for evt in session.keystrokes:
            etype = EVENT_KEY_DOWN if evt.get("type") == "down" else EVENT_KEY_UP
            events.append({"_type": etype, **evt})

        for evt in session.scroll:
            events.append({"_type": EVENT_SCROLL, **evt})

        events.sort(key=lambda e: e.get("t", e.get("timestamp", 0)))

        if len(events) > self.config.max_events:
            events = events[: self.config.max_events]

        return events

    def _encode_event(self, evt: dict) -> np.ndarray:
        """Encode a single raw event into a 13-dim vector."""
        cfg = self.config
        vec = np.zeros(cfg.event_dim, dtype=np.float32)
        etype = evt["_type"]

        # Dims 0-4: event type one-hot
        vec[etype] = 1.0

        # Dims 5-6: normalized coordinates
        x = evt.get("x", evt.get("pageX", 0.0)) or 0.0
        y = evt.get("y", evt.get("pageY", 0.0)) or 0.0
        vec[5] = x / cfg.max_coord_x
        vec[6] = y / cfg.max_coord_y

        # Dim 7: log-normalized dt_since_last
        dt = evt.get("dt_since_last")
        if dt is None or not isinstance(dt, (int, float)):
            dt = 0.0
        dt = max(dt, 0.0)
        vec[7] = np.log1p(min(dt, cfg.max_dt_ms)) / np.log1p(cfg.max_dt_ms)

        # Dim 8: mouse speed (computed incrementally from consecutive mouse events)
        if etype == EVENT_MOUSE:
            t = evt.get("t", 0.0)
            if self._prev_mouse_t > 0 and t > self._prev_mouse_t:
                dt_s = (t - self._prev_mouse_t) / 1000.0
                dist = np.sqrt(
                    (x - self._prev_mouse_x) ** 2 + (y - self._prev_mouse_y) ** 2
                )
                speed = dist / max(dt_s, 1e-6)
                vec[8] = min(speed, cfg.max_speed) / cfg.max_speed
            self._prev_mouse_x = x
            self._prev_mouse_y = y
            self._prev_mouse_t = t

        # Dim 9: scroll dy (normalized)
        if etype == EVENT_SCROLL:
            dy = evt.get("dy", 0.0) or 0.0
            vec[9] = np.clip(dy / cfg.max_scroll_dy, -1.0, 1.0)

        # Dim 10: is_special_key
        if etype in (EVENT_KEY_DOWN, EVENT_KEY_UP):
            vec[10] = 1.0 if evt.get("key") is not None else 0.0

        # Dim 11: button_is_left
        if etype == EVENT_CLICK:
            vec[11] = 1.0 if evt.get("button") == "left" else 0.0

        # Dim 12: target_is_interactive
        if etype == EVENT_CLICK:
            target = evt.get("target", {})
            if isinstance(target, dict):
                tag = (target.get("tag") or "").upper()
                vec[12] = 1.0 if tag in INTERACTIVE_TAGS else 0.0

        return vec
