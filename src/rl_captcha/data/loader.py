"""Unified data loading from MySQL, JSON exports, and webapp CSV exports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mysql.connector

from rl_captcha.config import DBConfig


# ---------------------------------------------------------------------------
# Bot type → adversarial tier mapping
# ---------------------------------------------------------------------------

BOT_TYPE_TO_TIER: dict[str, int] = {
    # Tier 1 — Commodity: obviously robotic, easy to detect
    "linear": 1,
    "tabber": 1,
    "speedrun": 1,
    # Tier 2 — Careful automation: more sophisticated behavioral mimicry
    "scripted": 2,
    "stealth": 2,
    "slow": 2,
    "erratic": 2,
    "replay": 2,
    # Tier 3 — Semi-automated: bot handles some steps, human handles others
    "semi_auto": 3,
    # Tier 4 — Trace-conditioned: bot replays/perturbs real human traces
    "trace_conditioned": 4,
}

TIER_NAMES: dict[int, str] = {
    1: "Commodity",
    2: "Careful Automation",
    3: "Semi-Automated",
    4: "Trace-Conditioned",
}


def bot_type_to_tier(bot_type: str | None) -> int:
    """Map a bot_type string to its adversarial tier. Unknown types → 0."""
    if bot_type is None:
        return 0
    return BOT_TYPE_TO_TIER.get(bot_type, 2)


@dataclass
class Session:
    """Normalized telemetry session — the universal data format for the entire pipeline."""

    session_id: str
    label: int | None = None  # 1 = human, 0 = bot, None = unlabeled
    mouse: list[dict] = field(default_factory=list)
    clicks: list[dict] = field(default_factory=list)
    keystrokes: list[dict] = field(default_factory=list)
    scroll: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MySQL loader (reads the webapp's user_sessions table directly)
# ---------------------------------------------------------------------------

def load_from_mysql(
    config: DBConfig | None = None,
    limit: int = 10_000,
    label: int | None = 1,
) -> list[Session]:
    """Load sessions from the TicketMonarch MySQL database.

    All webapp sessions are assumed human (label=1) unless overridden.
    """
    if config is None:
        config = DBConfig()

    conn = mysql.connector.connect(
        host=config.host,
        user=config.user,
        password=config.password,
        database=config.database,
        port=config.port,
    )
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT session_id, page, mouse_movements, click_events,
                   keystroke_data, scroll_events, browser_info, session_metadata
            FROM user_sessions
            ORDER BY session_start DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        cursor.close()
    finally:
        conn.close()

    sessions: list[Session] = []
    for row in rows:
        sessions.append(
            Session(
                session_id=row["session_id"],
                label=label,
                mouse=_parse_json(row.get("mouse_movements")),
                clicks=_parse_json(row.get("click_events")),
                keystrokes=_parse_json(row.get("keystroke_data")),
                scroll=_parse_json(row.get("scroll_events")),
                metadata={
                    "page": row.get("page"),
                    "browser_info": _parse_json(row.get("browser_info")),
                    "session_metadata": _parse_json(row.get("session_metadata")),
                },
            )
        )
    return sessions


# ---------------------------------------------------------------------------
# Webapp CSV export loader
# ---------------------------------------------------------------------------

def load_from_csv(
    path: str | Path,
    label: int | None = 1,
) -> list[Session]:
    """Load sessions from the webapp's ``tracking_sessions.csv`` export."""
    path = Path(path)
    sessions: list[Session] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sessions.append(
                Session(
                    session_id=row.get("session_id", ""),
                    label=label,
                    mouse=_parse_json(row.get("mouse_movements")),
                    clicks=_parse_json(row.get("click_events")),
                    keystrokes=_parse_json(row.get("keystroke_data")),
                    scroll=_parse_json(row.get("scroll_events")),
                    metadata={
                        "page": row.get("page"),
                        "browser_info": _parse_json(row.get("browser_info")),
                    },
                )
            )
    return sessions


# ---------------------------------------------------------------------------
# Directory-based loaders (data/human/ and data/bot/)
# ---------------------------------------------------------------------------

def load_from_directory(
    data_dir: str | Path,
) -> list[Session]:
    """Load all sessions from the standard data directory layout.

    Expected structure:
        data_dir/
        ├── human/   ← Human session exports (label=1)
        └── bot/     ← External bot data (label=0)

    ``human/*.json`` files use the live-confirm format (single session with segments).
    ``bot/*.json`` files use either:
      - Live-confirm format (single session with segments), or
      - Flat array format (list of session objects with ``session_id``).
    """
    data_dir = Path(data_dir)
    sessions: list[Session] = []

    human_dir = data_dir / "human"
    bot_dir = data_dir / "bot"

    if human_dir.is_dir():
        for f in sorted(human_dir.glob("*.json")):
            try:
                sessions.extend(_load_flexible_json(f, label=1))
            except Exception as e:
                print(f"Warning: skipping {f} — {e}")

    if bot_dir.is_dir():
        for f in sorted(bot_dir.glob("*.json")):
            try:
                sessions.extend(_load_flexible_json(f, label=0))
            except Exception as e:
                print(f"Warning: skipping {f} — {e}")

    return sessions


def _load_flexible_json(path: Path, label: int) -> list[Session]:
    """Load a JSON file in live-confirm or flat-array format."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sessions: list[Session] = []

    if isinstance(data, list):
        # Flat array format: [{ session_id, mouse, clicks, ... }, ...]
        for item in data:
            sid = item.get("session_id", item.get("sessionId", f"unknown_{id(item)}"))
            item_meta = item.get("metadata", {})
            if not isinstance(item_meta, dict):
                item_meta = {}
            item_meta.setdefault("source_file", str(path.name))
            item_meta.setdefault("bot_type", item.get("bot_type"))
            item_meta.setdefault("tier", item.get("tier"))
            sessions.append(Session(
                session_id=sid,
                label=item.get("label", label),
                mouse=_ensure_list(item.get("mouse")),
                clicks=_ensure_list(item.get("clicks")),
                keystrokes=_ensure_list(item.get("keystrokes")),
                scroll=_ensure_list(item.get("scroll")),
                metadata=item_meta,
            ))
    elif isinstance(data, dict):
        if "segments" in data and isinstance(data.get("segments"), list):
            # Live-confirm / webapp export format:
            # { "sessionId": "...", "segments": [{ "mouse": [...], ... }] }
            sid = data.get("session_id", data.get("sessionId", path.stem))
            mouse, clicks, keystrokes, scroll = [], [], [], []
            for seg in data["segments"]:
                mouse.extend(seg.get("mouse", []))
                clicks.extend(seg.get("clicks", []))
                keystrokes.extend(seg.get("keystrokes", []))
                scroll.extend(seg.get("scroll", []))
            sessions.append(Session(
                session_id=sid,
                label=data.get("label", label),
                mouse=mouse,
                clicks=clicks,
                keystrokes=keystrokes,
                scroll=scroll,
                metadata={
                    "source": data.get("source", "live_confirm"),
                    "source_file": str(path.name),
                    "bot_type": data.get("bot_type"),
                    "tier": data.get("tier"),
                },
            ))
        else:
            # Single flat session object
            sid = data.get("session_id", data.get("sessionId", path.stem))
            raw_meta = data.get("metadata", {})
            if not isinstance(raw_meta, dict):
                raw_meta = {}
            raw_meta.setdefault("source_file", str(path.name))
            raw_meta.setdefault("bot_type", data.get("bot_type"))
            raw_meta.setdefault("tier", data.get("tier"))
            sessions.append(Session(
                session_id=sid,
                label=data.get("label", label),
                mouse=_ensure_list(data.get("mouse")),
                clicks=_ensure_list(data.get("clicks")),
                keystrokes=_ensure_list(data.get("keystrokes")),
                scroll=_ensure_list(data.get("scroll")),
                metadata=raw_meta,
            ))

    return sessions


# ---------------------------------------------------------------------------
# Session slicing (for windowed feature extraction)
# ---------------------------------------------------------------------------

def slice_session(
    session: Session,
    t_start: float,
    t_end: float,
    keystroke_up_extend_ms: float = 2000.0,
) -> Session:
    """Return a new Session containing only events within [t_start, t_end].

    Keystroke 'up' events are included if their matching 'down' is in range,
    even if the 'up' timestamp exceeds t_end by up to *keystroke_up_extend_ms*.
    This prevents orphaned key-down events from losing their hold duration.
    """
    def _in_range(evt: dict) -> bool:
        t = evt.get("t", evt.get("timestamp", -1))
        return t_start <= t <= t_end

    # Mouse, clicks, scroll — simple time filter
    mouse = [e for e in session.mouse if _in_range(e)]
    clicks = [e for e in session.clicks if _in_range(e)]
    scroll = [e for e in session.scroll if _in_range(e)]

    # Keystrokes — keep downs in range, and their matching ups even if slightly past t_end
    down_fields_in_range: set[tuple[str, float]] = set()
    keystrokes: list[dict] = []

    for evt in session.keystrokes:
        t = evt.get("t", evt.get("timestamp", -1))
        evt_type = evt.get("type", "")
        field = evt.get("field", "")

        if evt_type == "down" and _in_range(evt):
            keystrokes.append(evt)
            down_fields_in_range.add((field, t))
        elif evt_type == "up":
            if _in_range(evt):
                keystrokes.append(evt)
            elif t <= t_end + keystroke_up_extend_ms:
                # Check if there's a matching down in range for this field
                for df, dt in down_fields_in_range:
                    if df == field and dt <= t:
                        keystrokes.append(evt)
                        break
        else:
            # Unknown type — include if in range
            if _in_range(evt):
                keystrokes.append(evt)

    return Session(
        session_id=session.session_id,
        label=session.label,
        mouse=mouse,
        clicks=clicks,
        keystrokes=keystrokes,
        scroll=scroll,
        metadata=session.metadata,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_list(value: Any) -> list:
    """Coerce a value to a list."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _parse_json(value: Any) -> list | dict:
    """Safely parse a JSON field that may be a string, dict, list, or None."""
    if value is None:
        return []
    if isinstance(value, (list, dict)):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, (list, dict)) else []
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Train / validation / test splitting
# ---------------------------------------------------------------------------

def split_sessions(
    sessions: list[Session],
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> tuple[list[Session], list[Session], list[Session]]:
    """Stratified split of sessions into train / val / test sets.

    Both human and bot sessions are split independently so each set
    maintains the same label proportions as the full dataset.
    """
    import random as _rng

    assert abs(train + val + test - 1.0) < 1e-6, "Ratios must sum to 1.0"

    human = [s for s in sessions if s.label == 1]
    bot = [s for s in sessions if s.label == 0]

    def _split_group(group: list[Session]) -> tuple[list[Session], list[Session], list[Session]]:
        rng = _rng.Random(seed)
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train)
        n_val = int(n * (train + val))
        return shuffled[:n_train], shuffled[n_train:n_val], shuffled[n_val:]

    h_train, h_val, h_test = _split_group(human)
    b_train, b_val, b_test = _split_group(bot)

    return (
        h_train + b_train,
        h_val + b_val,
        h_test + b_test,
    )


def split_sessions_by_family(
    sessions: list[Session],
    held_out_families: list[str] | None = None,
    held_out_tiers: list[int] | None = None,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> tuple[list[Session], list[Session], list[Session]]:
    """Split sessions with held-out bot families for generalization testing.

    Bot sessions matching *held_out_families* or *held_out_tiers* go entirely
    into the **test** set so the model is evaluated on unseen bot strategies.
    Remaining sessions are split normally with stratification.
    Human sessions are always split across all sets.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Ratios must sum to 1.0"

    held_families = set(held_out_families or [])
    held_tiers = set(held_out_tiers or [])

    def _is_held_out(s: Session) -> bool:
        if s.label != 0:
            return False
        bt = s.metadata.get("bot_type")
        if bt and bt in held_families:
            return True
        tier = s.metadata.get("tier") or bot_type_to_tier(bt)
        if tier in held_tiers:
            return True
        return False

    held_out_bots = [s for s in sessions if _is_held_out(s)]
    seen_sessions = [s for s in sessions if not _is_held_out(s)]

    # Split the seen sessions normally
    seen_train, seen_val, seen_test = split_sessions(
        seen_sessions, train=train, val=val, test=test, seed=seed,
    )

    # Add held-out bots to test set only
    return seen_train, seen_val, seen_test + held_out_bots
