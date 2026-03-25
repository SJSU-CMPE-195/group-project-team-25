"""Session-level feature extraction for the hidden scoring classifier.

Aggregates raw telemetry events from a Session into a fixed-size numeric
feature vector. Features are derived from the actual fields present in the
live-confirm telemetry export format.

Confirmed field structure (from data/human/ and data/bot/):
    Mouse:      x, y, t, pageX, pageY          (no dt_since_last)
    Clicks:     button, dt_since_last, t, target{tag,classes,text,id}, x, y
    Keystrokes: dt_since_last, field, key, t, type (down/up)
    Scroll:     dt_since_last, dx, dy, scrollX, scrollY, t

Feature groups (22 total):
    Mouse (8):      count, avg_speed, std_speed, avg_dt, std_dt,
                    direction_change_ratio, straightness, jitter_ratio
    Click (4):      count, avg_interval, std_interval, interactive_ratio
    Keystroke (5):  count, avg_interval, std_interval,
                    unique_fields, field_switch_ratio
    Scroll (5):     count, avg_dy, std_dy, total_abs_scroll,
                    avg_scroll_speed
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_captcha.config import FeatureConfig
from classifier.data_loader import Session


FEATURE_NAMES = [
    # Mouse (9)
    "mouse_count",
    "mouse_avg_speed",
    "mouse_std_speed",
    "mouse_avg_dt",
    "mouse_std_dt",
    "mouse_direction_change_ratio",
    "mouse_straightness",
    "mouse_jitter_ratio",
    "mouse_acceleration_std",
    # Click (4)
    "click_count",
    "click_avg_interval",
    "click_std_interval",
    "click_interactive_ratio",
    # Keystroke (6)
    "keystroke_count",
    "keystroke_avg_interval",
    "keystroke_std_interval",
    "keystroke_unique_fields",
    "keystroke_field_switch_ratio",
    "keystroke_rhythm_regularity",
    # Scroll (6)
    "scroll_count",
    "scroll_avg_dy",
    "scroll_std_dy",
    "scroll_total_abs",
    "scroll_avg_speed",
    "scroll_direction_change_ratio",
    # Session-level (1)
    "session_duration",
]

FEATURE_DIM = len(FEATURE_NAMES)  # 26

INTERACTIVE_TAGS = {"INPUT", "BUTTON", "A", "SELECT", "TEXTAREA"}


class SessionFeatureExtractor:
    """Converts a Session into a 22-dimensional feature vector.

    Usage::

        extractor = SessionFeatureExtractor()
        vec = extractor.extract(session)        # np.ndarray shape (22,)
        X   = extractor.extract_many(sessions)  # np.ndarray shape (N, 22)
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, session: Session) -> np.ndarray:
        """Return a (26,) float32 feature vector for one session."""
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)

        vec[0:9]   = self._mouse_features(session.mouse)
        vec[9:13]  = self._click_features(session.clicks)
        vec[13:19] = self._keystroke_features(session.keystrokes)
        vec[19:25] = self._scroll_features(session.scroll)
        vec[25]    = self._session_duration(session)

        # Replace any NaN/Inf that slipped through with 0
        np.nan_to_num(vec, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return vec

    def extract_many(self, sessions: list[Session]) -> np.ndarray:
        """Return (N, 22) feature matrix for a list of sessions."""
        return np.stack([self.extract(s) for s in sessions], axis=0)

    # ------------------------------------------------------------------
    # Mouse features
    # Mouse events only have: x, y, t, pageX, pageY
    # dt and speed must be derived from consecutive event pairs.
    # ------------------------------------------------------------------

    def _mouse_features(self, events: list[dict]) -> list[float]:
        cfg = self.config
        count = len(events)
        if count == 0:
            return [0.0] * 9

        speeds, dts = [], []
        dir_changes = 0
        jitter_count = 0
        prev_dx = prev_dy = None

        # For straightness: sum of step distances vs start-to-end distance
        total_step_dist = 0.0
        start_x = start_y = end_x = end_y = None

        for i, evt in enumerate(events):
            x = float(evt.get("x", evt.get("pageX", 0)) or 0)
            y = float(evt.get("y", evt.get("pageY", 0)) or 0)
            t = float(evt.get("t", 0) or 0)

            if i == 0:
                start_x, start_y = x, y

            end_x, end_y = x, y

            if i > 0:
                prev = events[i - 1]
                px = float(prev.get("x", prev.get("pageX", 0)) or 0)
                py = float(prev.get("y", prev.get("pageY", 0)) or 0)
                pt = float(prev.get("t", 0) or 0)

                dt_ms = t - pt
                if dt_ms > 0:
                    dts.append(dt_ms)
                    dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
                    total_step_dist += dist
                    dt_s = dt_ms / 1000.0
                    speed = dist / max(dt_s, 1e-6)
                    speeds.append(min(speed, cfg.mouse_speed_cap))

                dx = x - px
                dy = y - py

                if prev_dx is not None:
                    dot = dx * prev_dx + dy * prev_dy
                    if dot < 0:
                        dir_changes += 1

                if math.sqrt((x - px) ** 2 + (y - py) ** 2) < cfg.jitter_threshold:
                    jitter_count += 1

                prev_dx, prev_dy = dx, dy

        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        std_speed = float(np.std(speeds))  if speeds else 0.0
        avg_dt    = float(np.mean(dts))    if dts    else 0.0
        std_dt    = float(np.std(dts))     if dts    else 0.0

        steps = max(count - 1, 1)
        dir_change_ratio = dir_changes / steps
        jitter_ratio     = jitter_count / steps

        # Straightness: ratio of direct distance to total path length.
        # 1.0 = perfectly straight (bot-like), <1.0 = curved (human-like).
        if start_x is not None and total_step_dist > 0:
            direct_dist = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
            straightness = direct_dist / total_step_dist
        else:
            straightness = 0.0

        # Acceleration std: variability in speed changes between consecutive
        # steps. Humans naturally accelerate/decelerate; bots move at constant
        # speed or with mechanical regularity.
        if len(speeds) >= 2:
            accels = [speeds[i] - speeds[i - 1] for i in range(1, len(speeds))]
            accel_std = float(np.std(accels))
        else:
            accel_std = 0.0

        return [
            float(count),
            avg_speed,
            std_speed,
            avg_dt,
            std_dt,
            dir_change_ratio,
            straightness,
            jitter_ratio,
            accel_std,
        ]

    # ------------------------------------------------------------------
    # Click features
    # Click events: button, dt_since_last (may be null), t, target, x, y
    # ------------------------------------------------------------------

    def _click_features(self, events: list[dict]) -> list[float]:
        count = len(events)
        if count == 0:
            return [0.0] * 4

        intervals, interactive = [], 0
        prev_t = None

        for evt in events:
            t = float(evt.get("t", 0) or 0)

            # Prefer dt_since_last if non-null, otherwise compute from prev t
            dt = evt.get("dt_since_last")
            if dt is not None and isinstance(dt, (int, float)) and dt > 0:
                intervals.append(float(dt))
            elif prev_t is not None:
                diff = t - prev_t
                if diff > 0:
                    intervals.append(diff)

            prev_t = t

            target = evt.get("target", {})
            if isinstance(target, dict):
                tag = (target.get("tag") or "").upper()
                if tag in INTERACTIVE_TAGS:
                    interactive += 1

        avg_interval = float(np.mean(intervals)) if intervals else 0.0
        std_interval = float(np.std(intervals))  if intervals else 0.0
        interactive_ratio = interactive / count

        return [float(count), avg_interval, std_interval, interactive_ratio]

    # ------------------------------------------------------------------
    # Keystroke features
    # Keystroke events: dt_since_last (may be null), field, key (may be
    # null — filtered by extension), t, type (down/up)
    # Hold duration is unreliable since key is often null, so we focus
    # on inter-keystroke intervals and field-switching patterns.
    # ------------------------------------------------------------------

    def _keystroke_features(self, events: list[dict]) -> list[float]:
        downs = [e for e in events if e.get("type") == "down"]
        count = len(downs)
        if count == 0:
            return [0.0] * 6

        intervals = []
        fields_seen = []
        field_switches = 0
        prev_field = None
        prev_t = None

        for evt in sorted(downs, key=lambda e: e.get("t", 0)):
            t = float(evt.get("t", 0) or 0)
            field = evt.get("field", "")

            dt = evt.get("dt_since_last")
            if dt is not None and isinstance(dt, (int, float)) and dt > 0:
                intervals.append(float(dt))
            elif prev_t is not None:
                diff = t - prev_t
                if diff > 0:
                    intervals.append(diff)

            if field not in fields_seen:
                fields_seen.append(field)

            if prev_field is not None and field != prev_field:
                field_switches += 1

            prev_field = field
            prev_t = t

        avg_interval  = float(np.mean(intervals)) if intervals else 0.0
        std_interval  = float(np.std(intervals))  if intervals else 0.0
        unique_fields = float(len(fields_seen))
        field_switch_ratio = field_switches / max(count - 1, 1)

        # Rhythm regularity: coefficient of variation (std / mean) of
        # inter-keystroke intervals. Bots type with suspiciously uniform
        # timing (low CV); humans have natural variability (higher CV).
        if avg_interval > 0 and len(intervals) >= 2:
            rhythm_regularity = std_interval / avg_interval
        else:
            rhythm_regularity = 0.0

        return [
            float(count),
            avg_interval,
            std_interval,
            unique_fields,
            field_switch_ratio,
            rhythm_regularity,
        ]

    # ------------------------------------------------------------------
    # Scroll features
    # Scroll events: dt_since_last (may be null), dx, dy, scrollX,
    # scrollY, t
    # ------------------------------------------------------------------

    def _scroll_features(self, events: list[dict]) -> list[float]:
        count = len(events)
        if count == 0:
            return [0.0] * 6

        abs_dys, speeds = [], []
        raw_dys = []
        prev_t = None
        prev_scroll_y = None

        for evt in events:
            t  = float(evt.get("t", 0) or 0)
            raw_dy = float(evt.get("dy", 0) or 0)
            dy = abs(raw_dy)
            sy = float(evt.get("scrollY", 0) or 0)
            abs_dys.append(dy)
            raw_dys.append(raw_dy)

            dt = evt.get("dt_since_last")
            if dt is not None and isinstance(dt, (int, float)) and dt > 0:
                dt_s = dt / 1000.0
                if prev_scroll_y is not None:
                    travel = abs(sy - prev_scroll_y)
                    speeds.append(travel / max(dt_s, 1e-6))
            elif prev_t is not None:
                diff = (t - prev_t) / 1000.0
                if diff > 0 and prev_scroll_y is not None:
                    travel = abs(sy - prev_scroll_y)
                    speeds.append(travel / max(diff, 1e-6))

            prev_t = t
            prev_scroll_y = sy

        avg_dy       = float(np.mean(abs_dys)) if abs_dys else 0.0
        std_dy       = float(np.std(abs_dys))  if abs_dys else 0.0
        total_abs    = float(np.sum(abs_dys))
        avg_speed    = float(np.mean(speeds))  if speeds  else 0.0

        # Scroll direction change ratio: fraction of consecutive scroll
        # pairs where dy sign flips. Humans scroll up and down erratically;
        # bots tend to scroll in one direction.
        dir_changes = 0
        for i in range(1, len(raw_dys)):
            if raw_dys[i] != 0 and raw_dys[i - 1] != 0:
                if (raw_dys[i] > 0) != (raw_dys[i - 1] > 0):
                    dir_changes += 1
        scroll_dir_change_ratio = dir_changes / max(count - 1, 1)

        return [float(count), avg_dy, std_dy, total_abs, avg_speed, scroll_dir_change_ratio]

    # ------------------------------------------------------------------
    # Session-level features
    # ------------------------------------------------------------------

    @staticmethod
    def _session_duration(session: Session) -> float:
        """Total time span (ms) across all event types in the session."""
        all_times: list[float] = []
        for evt in session.mouse:
            t = evt.get("t")
            if t is not None:
                all_times.append(float(t))
        for evt in session.clicks:
            t = evt.get("t")
            if t is not None:
                all_times.append(float(t))
        for evt in session.keystrokes:
            t = evt.get("t")
            if t is not None:
                all_times.append(float(t))
        for evt in session.scroll:
            t = evt.get("t")
            if t is not None:
                all_times.append(float(t))

        if len(all_times) < 2:
            return 0.0
        return max(all_times) - min(all_times)
