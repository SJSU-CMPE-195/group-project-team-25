"""Adversarial augmentation: humanize bot sessions at the raw telemetry level.

Implements the HumanProfiler and tiered augmentation pipeline described in
Section 3.5.3 of the paper.  Bot sessions are transformed at three difficulty
levels (easy, medium, hard) to progressively resemble human behavior, forcing
the classifier to learn deeper behavioral patterns rather than relying on
trivially separable artifacts such as near-zero key-hold durations or
perfectly straight mouse paths.

Pipeline (per bot session, default 2 copies x 3 levels = 6 augmented copies):

    Easy   – fix key-hold durations, inject mouse micro-jitter
    Medium – easy + compress mouse Δt toward human rates, smooth mouse paths
    Hard   – medium with tighter parameters (near-human sessions)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from classifier.data_loader import Session


# ---------------------------------------------------------------------------
# Human profile
# ---------------------------------------------------------------------------

@dataclass
class HumanProfile:
    """Statistical profile learned from real human sessions."""

    # Key-hold duration (ms)
    hold_mean: float = 120.0
    hold_std: float = 40.0
    # Mouse inter-event Δt (ms)
    mouse_dt_mean: float = 16.0
    mouse_dt_std: float = 8.0
    # Mouse speed (px/s)
    speed_mean: float = 500.0
    speed_std: float = 300.0
    # Jitter ratio (fraction of small movements)
    jitter_mean: float = 0.15
    jitter_std: float = 0.05
    # Direction-change ratio
    dir_change_mean: float = 0.30
    dir_change_std: float = 0.10
    # Event-type ratios
    mouse_ratio_mean: float = 0.70
    click_ratio_mean: float = 0.05
    key_ratio_mean: float = 0.15
    scroll_ratio_mean: float = 0.10


class HumanProfiler:
    """Learns statistical profiles from human sessions across six signal
    categories: key-hold duration, mouse inter-event Δt, jitter ratio,
    mouse speed, direction-change frequency, and event-type ratios.

    Usage::

        profiler = HumanProfiler()
        profile = profiler.fit(human_sessions)
    """

    JITTER_THRESHOLD = 3.0  # px — movements smaller than this are jitter

    def fit(self, human_sessions: list[Session]) -> HumanProfile:
        """Learn μ and σ for each signal category from labeled human data."""
        all_holds: list[float] = []
        all_mouse_dts: list[float] = []
        all_speeds: list[float] = []
        all_jitter_ratios: list[float] = []
        all_dir_change_ratios: list[float] = []
        all_event_ratios: list[tuple[float, ...]] = []

        for session in human_sessions:
            all_holds.extend(self._hold_durations(session.keystrokes))
            dts, speeds, jitter_r, dir_r = self._mouse_stats(session.mouse)
            all_mouse_dts.extend(dts)
            all_speeds.extend(speeds)
            if jitter_r is not None:
                all_jitter_ratios.append(jitter_r)
            if dir_r is not None:
                all_dir_change_ratios.append(dir_r)
            ratios = self._event_ratios(session)
            if ratios is not None:
                all_event_ratios.append(ratios)

        def _mean(xs: list[float], default: float) -> float:
            return float(np.mean(xs)) if xs else default

        def _std(xs: list[float], default: float) -> float:
            return float(np.std(xs)) if xs else default

        profile = HumanProfile(
            hold_mean=_mean(all_holds, 120.0),
            hold_std=_std(all_holds, 40.0),
            mouse_dt_mean=_mean(all_mouse_dts, 16.0),
            mouse_dt_std=_std(all_mouse_dts, 8.0),
            speed_mean=_mean(all_speeds, 500.0),
            speed_std=_std(all_speeds, 300.0),
            jitter_mean=_mean(all_jitter_ratios, 0.15),
            jitter_std=_std(all_jitter_ratios, 0.05),
            dir_change_mean=_mean(all_dir_change_ratios, 0.30),
            dir_change_std=_std(all_dir_change_ratios, 0.10),
            mouse_ratio_mean=_mean([r[0] for r in all_event_ratios], 0.70),
            click_ratio_mean=_mean([r[1] for r in all_event_ratios], 0.05),
            key_ratio_mean=_mean([r[2] for r in all_event_ratios], 0.15),
            scroll_ratio_mean=_mean([r[3] for r in all_event_ratios], 0.10),
        )
        return profile

    # --- internal helpers ---------------------------------------------------

    @staticmethod
    def _hold_durations(keystrokes: list[dict]) -> list[float]:
        """Pair key-down / key-up events and return hold durations (ms)."""
        downs = sorted(
            [e for e in keystrokes if e.get("type") == "down"],
            key=lambda e: e.get("t", 0),
        )
        ups = sorted(
            [e for e in keystrokes if e.get("type") == "up"],
            key=lambda e: e.get("t", 0),
        )
        pending: dict[str, list[float]] = {}
        for evt in downs:
            key_id = evt.get("field") or evt.get("key") or ""
            pending.setdefault(key_id, []).append(float(evt.get("t", 0) or 0))

        holds: list[float] = []
        for evt in ups:
            key_id = evt.get("field") or evt.get("key") or ""
            t = float(evt.get("t", 0) or 0)
            if key_id in pending and pending[key_id]:
                hold = t - pending[key_id].pop(0)
                if 0 < hold < 2000:
                    holds.append(hold)
        return holds

    def _mouse_stats(
        self, mouse_events: list[dict]
    ) -> tuple[list[float], list[float], float | None, float | None]:
        """Return (dts, speeds, jitter_ratio, dir_change_ratio)."""
        dts: list[float] = []
        speeds: list[float] = []
        jitter_count = 0
        dir_changes = 0
        prev_dx = prev_dy = None

        for i in range(1, len(mouse_events)):
            curr, prev = mouse_events[i], mouse_events[i - 1]
            cx = float(curr.get("x", curr.get("pageX", 0)) or 0)
            cy = float(curr.get("y", curr.get("pageY", 0)) or 0)
            ct = float(curr.get("t", 0) or 0)
            px = float(prev.get("x", prev.get("pageX", 0)) or 0)
            py = float(prev.get("y", prev.get("pageY", 0)) or 0)
            pt = float(prev.get("t", 0) or 0)

            dt = ct - pt
            if dt > 0:
                dts.append(dt)
                dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                speeds.append(dist / (dt / 1000.0))
                if dist < self.JITTER_THRESHOLD:
                    jitter_count += 1
                dx, dy = cx - px, cy - py
                if prev_dx is not None and (dx * prev_dx + dy * prev_dy) < 0:
                    dir_changes += 1
                prev_dx, prev_dy = dx, dy

        steps = max(len(mouse_events) - 1, 1)
        jitter_r = jitter_count / steps if len(mouse_events) > 1 else None
        dir_r = dir_changes / steps if len(mouse_events) > 1 else None
        return dts, speeds, jitter_r, dir_r

    @staticmethod
    def _event_ratios(
        session: Session,
    ) -> tuple[float, float, float, float] | None:
        n_mouse = len(session.mouse)
        n_click = len(session.clicks)
        n_key = len([e for e in session.keystrokes if e.get("type") == "down"])
        n_scroll = len(session.scroll)
        total = n_mouse + n_click + n_key + n_scroll
        if total == 0:
            return None
        return (n_mouse / total, n_click / total, n_key / total, n_scroll / total)


# ---------------------------------------------------------------------------
# Augmentation configs for the three difficulty tiers
# ---------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    """Parameters for a single augmentation difficulty level."""

    # --- Easy transforms ---
    fix_hold_durations: bool = True
    jitter_std: float = 3.0       # px — Gaussian noise on mouse (x, y)

    # --- Medium transforms (cumulative with easy) ---
    compress_timing: bool = False
    timing_beta: float = 0.7      # blend factor: Δt' = β·Δt + (1-β)·μ_h
    smooth_paths: bool = False
    smoothing_alpha: float = 0.8  # EMA coefficient: p' = α·p + (1-α)·p_prev


EASY_CONFIG = AugmentConfig(
    fix_hold_durations=True,
    jitter_std=3.0,
    compress_timing=False,
    smooth_paths=False,
)

MEDIUM_CONFIG = AugmentConfig(
    fix_hold_durations=True,
    jitter_std=2.0,
    compress_timing=True,
    timing_beta=0.7,
    smooth_paths=True,
    smoothing_alpha=0.8,
)

HARD_CONFIG = AugmentConfig(
    fix_hold_durations=True,
    jitter_std=1.0,
    compress_timing=True,
    timing_beta=0.4,       # stronger compression toward human Δt
    smooth_paths=True,
    smoothing_alpha=0.6,   # stronger path smoothing
)

LEVEL_CONFIGS = [
    ("easy", EASY_CONFIG),
    ("medium", MEDIUM_CONFIG),
    ("hard", HARD_CONFIG),
]


# ---------------------------------------------------------------------------
# Per-session augmentation transforms
# ---------------------------------------------------------------------------

def _humanize_hold_durations(
    keystrokes: list[dict],
    profile: HumanProfile,
    rng: np.random.RandomState,
) -> list[dict]:
    r"""Resample key-hold durations from a clipped Gaussian centred on the
    human profile:  d_{hold}' ~ N(μ_h^{hold}, σ_h^{hold}),  clamped to
    [20, 500] ms.
    """
    keystrokes = [copy.deepcopy(e) for e in keystrokes]
    downs = sorted(
        [e for e in keystrokes if e.get("type") == "down"],
        key=lambda e: e.get("t", 0),
    )
    ups = sorted(
        [e for e in keystrokes if e.get("type") == "up"],
        key=lambda e: e.get("t", 0),
    )

    pending: dict[str, list[dict]] = {}
    for evt in downs:
        key_id = evt.get("field") or evt.get("key") or ""
        pending.setdefault(key_id, []).append(evt)

    for evt in ups:
        key_id = evt.get("field") or evt.get("key") or ""
        if key_id in pending and pending[key_id]:
            down_evt = pending[key_id].pop(0)
            down_t = float(down_evt.get("t", 0) or 0)
            new_hold = rng.normal(profile.hold_mean, profile.hold_std)
            new_hold = float(np.clip(new_hold, 20.0, 500.0))
            evt["t"] = down_t + new_hold

    return keystrokes


def _inject_jitter(
    mouse_events: list[dict],
    jitter_std: float,
    rng: np.random.RandomState,
) -> list[dict]:
    r"""Add micro-jitter:  p_t' = p_t + ε_t,  ε_t ~ N(0, σ_{jitter}^2 I)."""
    mouse_events = [copy.deepcopy(e) for e in mouse_events]
    for evt in mouse_events:
        noise_x = rng.normal(0, jitter_std)
        noise_y = rng.normal(0, jitter_std)
        if "x" in evt:
            evt["x"] = float(evt["x"]) + noise_x
        if "y" in evt:
            evt["y"] = float(evt["y"]) + noise_y
        if "pageX" in evt:
            evt["pageX"] = float(evt["pageX"]) + noise_x
        if "pageY" in evt:
            evt["pageY"] = float(evt["pageY"]) + noise_y
    return mouse_events


def _compress_timing(
    mouse_events: list[dict],
    profile: HumanProfile,
    beta: float,
) -> list[dict]:
    r"""Compress mouse Δt toward human rates:
    Δt_k' = β · Δt_k + (1 - β) · μ_h^{Δt}.
    Timestamps are rebuilt forward from the first event.
    """
    if len(mouse_events) < 2:
        return mouse_events
    mouse_events = [copy.deepcopy(e) for e in mouse_events]
    mouse_events.sort(key=lambda e: float(e.get("t", 0) or 0))

    for i in range(1, len(mouse_events)):
        prev_t = float(mouse_events[i - 1].get("t", 0) or 0)
        curr_t = float(mouse_events[i].get("t", 0) or 0)
        original_dt = curr_t - prev_t
        if original_dt > 0:
            new_dt = beta * original_dt + (1 - beta) * profile.mouse_dt_mean
            mouse_events[i]["t"] = prev_t + max(new_dt, 1.0)
    return mouse_events


def _smooth_paths(
    mouse_events: list[dict],
    alpha: float,
) -> list[dict]:
    r"""Exponential smoothing on positions to soften abrupt direction changes:
    p_t' = α_s · p_t + (1 - α_s) · p_{t-1}'.
    """
    if len(mouse_events) < 2:
        return mouse_events
    mouse_events = [copy.deepcopy(e) for e in mouse_events]

    for i in range(1, len(mouse_events)):
        for coord in ("x", "y"):
            if coord in mouse_events[i] and coord in mouse_events[i - 1]:
                curr = float(mouse_events[i][coord])
                prev = float(mouse_events[i - 1][coord])
                mouse_events[i][coord] = alpha * curr + (1 - alpha) * prev
        for coord in ("pageX", "pageY"):
            if coord in mouse_events[i] and coord in mouse_events[i - 1]:
                curr = float(mouse_events[i][coord])
                prev = float(mouse_events[i - 1][coord])
                mouse_events[i][coord] = alpha * curr + (1 - alpha) * prev
    return mouse_events


def augment_session(
    session: Session,
    profile: HumanProfile,
    config: AugmentConfig,
    level_name: str,
    rng: np.random.RandomState,
) -> Session:
    """Create a single humanized copy of a bot session at one difficulty level."""
    aug = copy.deepcopy(session)

    # Easy: fix key-hold durations
    if config.fix_hold_durations and aug.keystrokes:
        aug.keystrokes = _humanize_hold_durations(aug.keystrokes, profile, rng)

    # Easy: inject mouse micro-jitter
    if config.jitter_std > 0 and aug.mouse:
        aug.mouse = _inject_jitter(aug.mouse, config.jitter_std, rng)

    # Medium: compress mouse timing toward human rates
    if config.compress_timing and aug.mouse:
        aug.mouse = _compress_timing(aug.mouse, profile, config.timing_beta)

    # Medium: smooth mouse paths
    if config.smooth_paths and aug.mouse:
        aug.mouse = _smooth_paths(aug.mouse, config.smoothing_alpha)

    aug.metadata = {**aug.metadata, "augmented": True, "aug_level": level_name}
    return aug


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def adversarial_augment_sessions(
    bot_sessions: list[Session],
    human_sessions: list[Session],
    n_copies_per_level: int = 2,
    random_state: int = 42,
) -> list[Session]:
    """Generate humanized copies of bot sessions at three difficulty levels.

    For each original bot session, produces
    ``n_copies_per_level × 3 levels`` augmented sessions (default 6).
    All augmented sessions retain ``label=0`` (bot) so the classifier must
    learn to detect them despite their human-like characteristics.

    Parameters
    ----------
    bot_sessions : list[Session]
        Original bot sessions to humanize.
    human_sessions : list[Session]
        Real human sessions used to learn the HumanProfile.
    n_copies_per_level : int
        Number of augmented copies per bot per difficulty level.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    list[Session]
        Only the augmented sessions (originals are not included).
    """
    if not bot_sessions or not human_sessions:
        return []

    # Step 1: learn human behavioral profiles
    profiler = HumanProfiler()
    profile = profiler.fit(human_sessions)
    rng = np.random.RandomState(random_state)

    print(f"  [HumanProfiler] Learned profile from {len(human_sessions)} human sessions:")
    print(f"    key-hold   μ={profile.hold_mean:.1f} ms  σ={profile.hold_std:.1f} ms")
    print(f"    mouse Δt   μ={profile.mouse_dt_mean:.1f} ms  σ={profile.mouse_dt_std:.1f} ms")
    print(f"    mouse spd  μ={profile.speed_mean:.0f} px/s  σ={profile.speed_std:.0f} px/s")
    print(f"    jitter     μ={profile.jitter_mean:.3f}  σ={profile.jitter_std:.3f}")
    print(f"    dir-change μ={profile.dir_change_mean:.3f}  σ={profile.dir_change_std:.3f}")

    # Step 2: generate augmented copies at each level
    augmented: list[Session] = []
    for session in bot_sessions:
        for level_name, config in LEVEL_CONFIGS:
            for copy_idx in range(n_copies_per_level):
                aug = augment_session(session, profile, config, level_name, rng)
                aug.session_id = f"{session.session_id}_aug_{level_name}_{copy_idx}"
                augmented.append(aug)

    n_per_level = len(bot_sessions) * n_copies_per_level
    print(
        f"  [HumanProfiler] Generated {len(augmented)} augmented bot sessions "
        f"({n_per_level} easy + {n_per_level} medium + {n_per_level} hard)"
    )
    return augmented
