"""Windowed Gymnasium environment for CAPTCHA defender with LSTM agent.

Each timestep = one WINDOW of telemetry events (e.g., 30 events grouped).
The agent processes statistical features computed over each window through
its LSTM and decides an action per window. Terminal actions end the episode.

Window features capture discriminative behavioral patterns (speed variance,
click timing regularity, path curvature, etc.) that single events cannot.
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from rl_captcha.config import EventEnvConfig
from rl_captcha.data.loader import Session, bot_type_to_tier

ACTION_NAMES = [
    "continue", "deploy_honeypot",
    "easy_puzzle", "medium_puzzle", "hard_puzzle",
    "allow", "block",
]

# Event type indices
EVENT_MOUSE = 0
EVENT_CLICK = 1
EVENT_KEY_DOWN = 2
EVENT_KEY_UP = 3
EVENT_SCROLL = 4

INTERACTIVE_TAGS = {"INPUT", "BUTTON", "A", "SELECT", "TEXTAREA"}


def _honeypot_bot_trigger_prob(cfg: EventEnvConfig, meta: dict) -> float:
    """Per-tier honeypot trigger rate for bots; fallback when tier unknown."""
    tier = meta.get("tier")
    if tier is None:
        tier = bot_type_to_tier(meta.get("bot_type"))
    try:
        t = int(tier)
    except (TypeError, ValueError):
        t = None
    rates = cfg.honeypot_trigger_rates_by_tier
    if t is not None and t in rates:
        return rates[t]
    return cfg.honeypot_trigger_rate_bot_fallback


def compute_terminal_reward(
    cfg: EventEnvConfig,
    action: int,
    true_label: int,
    metadata: dict,
    rng: random.Random,
) -> tuple[float, str]:
    """Reward for terminal actions 2--6 (puzzles, allow, block). Shared with online training."""
    if action in (2, 3, 4):
        human_pass, bot_pass = cfg.puzzle_pass_rates[action]
        if true_label == 1:
            if rng.random() < human_pass:
                return cfg.human_puzzle_friction[action], "human_passed_puzzle"
            return cfg.penalty_human_puzzle_fail, "fp_puzzle"
        if rng.random() < bot_pass:
            return cfg.penalty_bot_passes_puzzle, "bot_passed_puzzle"
        return cfg.puzzle_catch_rewards[action], "bot_blocked_puzzle"
    if action == 5:
        if true_label == 1:
            return cfg.reward_correct_allow, "correct_allow"
        return cfg.penalty_bot_missed_allow, "false_negative"
    if action == 6:
        if true_label == 0:
            return cfg.reward_direct_block_bot, "correct_block"
        return cfg.penalty_block_human, "false_positive_block"
    raise ValueError(f"Not a terminal action: {action}")


def _safe_var(arr: list[float]) -> float:
    """Variance with fallback for empty/single-element lists."""
    if len(arr) < 2:
        return 0.0
    mean = sum(arr) / len(arr)
    return sum((x - mean) ** 2 for x in arr) / (len(arr) - 1)


def _safe_mean(arr: list[float]) -> float:
    return sum(arr) / len(arr) if arr else 0.0


class EventEncoder:
    """Encodes a window of raw events into a fixed-size feature vector.

    26-dimensional windowed feature vector:
        [0]  mouse_event_ratio      — fraction of events that are mouse moves
        [1]  click_event_ratio      — fraction that are clicks
        [2]  key_event_ratio        — fraction that are keystrokes
        [3]  scroll_event_ratio     — fraction that are scrolls
        [4]  mean_mouse_speed       — avg mouse speed (px/s, normalized)
        [5]  var_mouse_speed        — speed variance (bots have LOW variance)
        [6]  mean_mouse_accel       — avg acceleration (direction changes)
        [7]  path_curvature         — ratio of path length to displacement
        [8]  mean_dt                — avg time between events (normalized)
        [9]  var_dt                 — timing variance (bots have LOW variance)
        [10] min_dt                 — minimum inter-event time
        [11] mean_click_dt          — avg time between consecutive clicks
        [12] var_click_dt           — click timing variance
        [13] mean_key_hold          — avg keystroke hold duration
        [14] var_key_hold           — keystroke hold variance
        [15] mean_key_interval      — avg time between key presses
        [16] var_key_interval       — key interval variance
        [17] scroll_total_dy        — total scroll distance in window
        [18] scroll_direction_changes — number of scroll direction reversals
        [19] unique_x_positions     — number of distinct x coords (normalized)
        [20] unique_y_positions     — number of distinct y coords (normalized)
        [21] x_range                — max-min x spread (normalized)
        [22] y_range                — max-min y spread (normalized)
        [23] interactive_click_ratio — fraction of clicks on interactive elements
        [24] window_duration        — total time span of this window (normalized)
        [25] event_count_norm       — actual event count / window_size
    """

    def __init__(self, config: EventEnvConfig):
        self.config = config

    def build_timeline(self, session: Session) -> list[dict]:
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
        return events

    def encode_window(self, events: list[dict]) -> np.ndarray:
        """Encode a window of events into a 26-dim feature vector."""
        cfg = self.config
        vec = np.zeros(cfg.event_dim, dtype=np.float32)
        n = len(events)
        if n == 0:
            return vec

        # Count event types
        n_mouse = sum(1 for e in events if e["_type"] == EVENT_MOUSE)
        n_click = sum(1 for e in events if e["_type"] == EVENT_CLICK)
        n_key = sum(1 for e in events if e["_type"] in (EVENT_KEY_DOWN, EVENT_KEY_UP))
        n_scroll = sum(1 for e in events if e["_type"] == EVENT_SCROLL)

        # [0-3] Event type ratios
        vec[0] = n_mouse / n
        vec[1] = n_click / n
        vec[2] = n_key / n
        vec[3] = n_scroll / n

        # ── Mouse features [4-7] ─────────────────────────────────────
        mouse_events = [e for e in events if e["_type"] == EVENT_MOUSE]
        speeds = []
        accels = []
        path_length = 0.0
        if len(mouse_events) >= 2:
            prev_speed = 0.0
            for i in range(1, len(mouse_events)):
                prev = mouse_events[i - 1]
                curr = mouse_events[i]
                x0 = prev.get("x", prev.get("pageX", 0)) or 0
                y0 = prev.get("y", prev.get("pageY", 0)) or 0
                x1 = curr.get("x", curr.get("pageX", 0)) or 0
                y1 = curr.get("y", curr.get("pageY", 0)) or 0
                t0 = prev.get("t", prev.get("timestamp", 0)) or 0
                t1 = curr.get("t", curr.get("timestamp", 0)) or 0
                dt_s = max((t1 - t0) / 1000.0, 1e-6)
                dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                path_length += dist
                speed = dist / dt_s
                speeds.append(speed)
                accels.append(abs(speed - prev_speed) / dt_s)
                prev_speed = speed

            vec[4] = min(_safe_mean(speeds), cfg.max_speed) / cfg.max_speed
            vec[5] = min(_safe_var(speeds), cfg.max_speed ** 2) / (cfg.max_speed ** 2)
            vec[6] = min(_safe_mean(accels), cfg.max_speed * 10) / (cfg.max_speed * 10)

            # Path curvature: path_length / straight-line displacement
            first = mouse_events[0]
            last = mouse_events[-1]
            x0 = first.get("x", first.get("pageX", 0)) or 0
            y0 = first.get("y", first.get("pageY", 0)) or 0
            x1 = last.get("x", last.get("pageX", 0)) or 0
            y1 = last.get("y", last.get("pageY", 0)) or 0
            displacement = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            if displacement > 1.0:
                vec[7] = min(path_length / displacement, 10.0) / 10.0
            elif path_length > 0:
                vec[7] = 1.0  # moved but ended up in same place

        # ── Timing features [8-10] ───────────────────────────────────
        dts = []
        for i in range(1, n):
            t0 = events[i - 1].get("t", events[i - 1].get("timestamp", 0)) or 0
            t1 = events[i].get("t", events[i].get("timestamp", 0)) or 0
            dts.append(max(t1 - t0, 0))
        if dts:
            vec[8] = math.log1p(min(_safe_mean(dts), cfg.max_dt_ms)) / math.log1p(cfg.max_dt_ms)
            vec[9] = min(_safe_var(dts), cfg.max_dt_ms ** 2) / (cfg.max_dt_ms ** 2)
            vec[10] = math.log1p(min(min(dts), cfg.max_dt_ms)) / math.log1p(cfg.max_dt_ms)

        # ── Click timing [11-12] ─────────────────────────────────────
        click_times = [
            e.get("t", e.get("timestamp", 0)) or 0
            for e in events if e["_type"] == EVENT_CLICK
        ]
        if len(click_times) >= 2:
            click_dts = [click_times[i] - click_times[i - 1] for i in range(1, len(click_times))]
            vec[11] = math.log1p(min(_safe_mean(click_dts), cfg.max_dt_ms)) / math.log1p(cfg.max_dt_ms)
            vec[12] = min(_safe_var(click_dts), cfg.max_dt_ms ** 2) / (cfg.max_dt_ms ** 2)

        # ── Keystroke features [13-16] ───────────────────────────────
        key_downs: dict[str, list[float]] = {}
        key_holds = []
        key_down_times = []
        for e in events:
            if e["_type"] == EVENT_KEY_DOWN:
                key = e.get("field") or e.get("key") or ""
                t = e.get("t", e.get("timestamp", 0)) or 0
                key_downs.setdefault(key, []).append(t)
                key_down_times.append(t)
            elif e["_type"] == EVENT_KEY_UP:
                key = e.get("field") or e.get("key") or ""
                t = e.get("t", e.get("timestamp", 0)) or 0
                pending = key_downs.get(key)
                if pending:
                    hold = t - pending.pop(0)
                    if 0 < hold < 2000:
                        key_holds.append(hold)

        if key_holds:
            vec[13] = min(_safe_mean(key_holds), 1000) / 1000
            vec[14] = min(_safe_var(key_holds), 1e6) / 1e6
        if len(key_down_times) >= 2:
            key_intervals = [key_down_times[i] - key_down_times[i - 1] for i in range(1, len(key_down_times))]
            vec[15] = math.log1p(min(_safe_mean(key_intervals), cfg.max_dt_ms)) / math.log1p(cfg.max_dt_ms)
            vec[16] = min(_safe_var(key_intervals), cfg.max_dt_ms ** 2) / (cfg.max_dt_ms ** 2)

        # ── Scroll features [17-18] ──────────────────────────────────
        scroll_dys = [
            (e.get("dy", 0) or 0) for e in events if e["_type"] == EVENT_SCROLL
        ]
        if scroll_dys:
            vec[17] = np.clip(sum(abs(d) for d in scroll_dys) / (cfg.max_scroll_dy * 10), 0, 1)
            direction_changes = sum(
                1 for i in range(1, len(scroll_dys))
                if scroll_dys[i] * scroll_dys[i - 1] < 0
            )
            vec[18] = min(direction_changes, 10) / 10

        # ── Spatial features [19-22] ─────────────────────────────────
        all_x = []
        all_y = []
        for e in events:
            x = e.get("x", e.get("pageX", None))
            y = e.get("y", e.get("pageY", None))
            if x is not None and y is not None:
                all_x.append(x)
                all_y.append(y)
        if all_x:
            unique_x = len(set(int(x / 10) for x in all_x))  # bin to 10px
            unique_y = len(set(int(y / 10) for y in all_y))
            vec[19] = min(unique_x, 100) / 100
            vec[20] = min(unique_y, 100) / 100
            vec[21] = (max(all_x) - min(all_x)) / cfg.max_coord_x
            vec[22] = (max(all_y) - min(all_y)) / cfg.max_coord_y

        # ── Interaction quality [23] ─────────────────────────────────
        clicks_on_interactive = 0
        total_clicks = 0
        for e in events:
            if e["_type"] == EVENT_CLICK:
                total_clicks += 1
                target = e.get("target", {})
                if isinstance(target, dict):
                    tag = (target.get("tag") or "").upper()
                    if tag in INTERACTIVE_TAGS:
                        clicks_on_interactive += 1
        if total_clicks > 0:
            vec[23] = clicks_on_interactive / total_clicks

        # ── Window metadata [24-25] ──────────────────────────────────
        t_first = events[0].get("t", events[0].get("timestamp", 0)) or 0
        t_last = events[-1].get("t", events[-1].get("timestamp", 0)) or 0
        duration = max(t_last - t_first, 0)
        vec[24] = math.log1p(min(duration, 60000)) / math.log1p(60000)
        vec[25] = len(events) / cfg.window_size

        return vec


def _augment_timeline(
    events: list[dict],
    config: EventEnvConfig,
    position_noise_std: float,
    timing_jitter_std: float,
    speed_warp_range: tuple,
) -> list[dict]:
    """Perturb a session's timeline with configurable noise levels.

    Applied stochastically during training so the agent can't
    rely on trivially separable features (zero variance, locked positions).

    Perturbations:
      1. Position noise — Gaussian jitter on x/y coordinates
      2. Timing jitter  — Gaussian noise on timestamps
      3. Speed warp     — random time stretch/compress to vary inter-event dt
    """
    if not events:
        return events

    rng = random.Random()

    # --- 1. Speed warp: stretch/compress all timestamps ----------------
    lo, hi = speed_warp_range
    warp_factor = rng.uniform(lo, hi)

    # Anchor at the first timestamp so the session start doesn't drift
    t0 = events[0].get("t", events[0].get("timestamp", 0)) or 0

    augmented = []
    for evt in events:
        e = dict(evt)  # shallow copy

        # Warp timestamp
        t_key = "t" if "t" in e else "timestamp"
        t_orig = e.get(t_key, 0) or 0
        t_warped = t0 + (t_orig - t0) * warp_factor

        # --- 2. Timing jitter ------------------------------------------
        t_warped += rng.gauss(0, timing_jitter_std)
        e[t_key] = max(0, t_warped)

        # --- 3. Position noise ------------------------------------------
        if "x" in e and e["x"] is not None:
            e["x"] = max(0, min(config.max_coord_x,
                                e["x"] + rng.gauss(0, position_noise_std)))
        if "y" in e and e["y"] is not None:
            e["y"] = max(0, min(config.max_coord_y,
                                e["y"] + rng.gauss(0, position_noise_std)))
        if "pageX" in e and e["pageX"] is not None:
            e["pageX"] = max(0, min(config.max_coord_x,
                                    e["pageX"] + rng.gauss(0, position_noise_std)))
        if "pageY" in e and e["pageY"] is not None:
            e["pageY"] = max(0, min(config.max_coord_y,
                                    e["pageY"] + rng.gauss(0, position_noise_std)))

        augmented.append(e)

    # Re-sort by time (jitter may have swapped a few neighbors)
    t_key = "t" if "t" in augmented[0] else "timestamp"
    augmented.sort(key=lambda e: e.get(t_key, 0))

    return augmented


class EventEnv(gym.Env):
    """Windowed CAPTCHA environment.

    Observation: 26-dim windowed feature vector.
    Action: 7 discrete (continue, honeypot, 3 puzzles, allow, block).
    Episode: one user session's event windows presented sequentially.
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
        self._encoder = EventEncoder(self.config)

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
        self._windows: list[list[dict]] = []
        self._window_idx: int = 0
        self._current_session: Session | None = None
        self._honeypot_deployed: bool = False
        self._honeypot_triggered: bool = False
        self._num_honeypots: int = 0
        self._honeypot_info_bonus_pending: float = 0.0

    # Action masks: which actions are valid on non-final vs final windows
    # Non-final: only continue (0) and honeypot (1)
    NON_FINAL_MASK = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    # Final: only terminal actions — puzzles (2,3,4), allow (5), block (6)
    FINAL_MASK = np.array([0, 0, 1, 1, 1, 1, 1], dtype=np.float32)

    def _get_action_mask(self) -> np.ndarray:
        """Return valid action mask for current window position."""
        is_final = self._window_idx >= len(self._windows) - 1
        return self.FINAL_MASK.copy() if is_final else self.NON_FINAL_MASK.copy()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Balanced 50/50 sampling
        if not self._sessions:
            raise RuntimeError("EventEnv has no sessions — load data before calling reset()")
        if self._human_sessions and self._bot_sessions:
            if random.random() < 0.5:
                session = random.choice(self._human_sessions)
            else:
                session = random.choice(self._bot_sessions)
        else:
            session = random.choice(self._sessions)

        self._current_session = session
        self._window_idx = 0
        self._honeypot_deployed = False
        self._honeypot_triggered = False
        self._num_honeypots = 0
        self._honeypot_info_bonus_pending = 0.0

        timeline = self._encoder.build_timeline(session)

        # Augment sessions stochastically during training
        if self.config.augment and random.random() < self.config.augment_prob:
            if session.label == 0:
                # Bot: heavier perturbation
                timeline = _augment_timeline(
                    timeline, self.config,
                    self.config.aug_position_noise_std,
                    self.config.aug_timing_jitter_std,
                    self.config.aug_speed_warp_range,
                )
            elif self.config.augment_human:
                # Human: lighter perturbation to simulate natural variation
                timeline = _augment_timeline(
                    timeline, self.config,
                    self.config.aug_human_position_noise_std,
                    self.config.aug_human_timing_jitter_std,
                    self.config.aug_human_speed_warp_range,
                )

        # Split timeline into overlapping windows
        ws = self.config.window_size
        stride = ws // 2  # 50% overlap for smoother transitions
        all_windows = []
        for start in range(0, len(timeline), stride):
            window = timeline[start:start + ws]
            if len(window) >= self.config.min_events:
                all_windows.append(window)

        # Subsample to max_windows (evenly spaced) so LSTM sees manageable sequences
        max_w = self.config.max_windows
        if len(all_windows) > max_w:
            indices = np.linspace(0, len(all_windows) - 1, max_w, dtype=int)
            self._windows = [all_windows[i] for i in indices]
        else:
            self._windows = all_windows

        info = {
            "session_id": session.session_id,
            "true_label": session.label,
            "total_events": len(timeline),
            "total_windows": len(self._windows),
            "bot_type": session.metadata.get("bot_type"),
            "tier": session.metadata.get("tier"),
        }

        if not self._windows:
            info["too_short"] = True
            mask = self.FINAL_MASK.copy()
            info["action_mask"] = mask
            return np.zeros(self.config.event_dim, dtype=np.float32), info

        obs = self._encoder.encode_window(self._windows[0])
        info["action_mask"] = self._get_action_mask()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step. Action masking is enforced by the agent, not here.

        Non-final windows: agent picks continue (0) or honeypot (1).
        Final window: agent picks a terminal action (2-6).
        """
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

        if action == 0:  # continue — observe more
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
                meta = self._current_session.metadata or {}
                if true_label == 0:
                    p_trig = _honeypot_bot_trigger_prob(cfg, meta)
                    triggered = random.random() < p_trig
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

        elif action in (2, 3, 4, 5, 6):  # terminal decisions
            terminated = True
            meta = self._current_session.metadata or {}
            reward, outcome = compute_terminal_reward(
                cfg, action, true_label, meta, random.Random(),
            )

        # Advance to next window for non-terminal actions
        if not terminated:
            self._window_idx += 1
            if self._window_idx >= len(self._windows):
                # Should not happen with proper masking (final window forces terminal)
                truncated = True
                reward += cfg.truncation_penalty
                outcome = "truncated"

        # Build next observation and action mask
        if terminated or truncated:
            obs = np.zeros(cfg.event_dim, dtype=np.float32)
            action_mask = self.FINAL_MASK.copy()
        else:
            obs = self._encoder.encode_window(self._windows[self._window_idx])
            action_mask = self._get_action_mask()

        info = {
            "session_id": self._current_session.session_id,
            "true_label": true_label,
            "action": ACTION_NAMES[action],
            "outcome": outcome,
            "reward": reward,
            "window_idx": self._window_idx,
            "total_windows": len(self._windows),
            "honeypot_deployed": self._honeypot_deployed,
            "honeypot_triggered": self._honeypot_triggered,
            "action_mask": action_mask,
            "bot_type": self._current_session.metadata.get("bot_type"),
            "tier": self._current_session.metadata.get("tier"),
        }
        return obs, reward, terminated, truncated, info
