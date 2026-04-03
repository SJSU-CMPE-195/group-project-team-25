"""Unified data loading from MySQL, Chrome extension JSON exports, and webapp CSV exports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mysql.connector

from rl_captcha.config import DBConfig


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
# Chrome extension JSON export loader
# ---------------------------------------------------------------------------


def load_from_json(
    path: str | Path,
    label: int | None = 1,
) -> list[Session]:
    """Load sessions from a Chrome extension JSON export.

    The export format has top-level keys = session IDs, each containing
    ``segments`` (list of telemetry batches) and ``pageMeta``.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    sessions: list[Session] = []
    for sid, session_data in data.items():
        # Merge all segments into flat lists
        mouse: list[dict] = []
        clicks: list[dict] = []
        keystrokes: list[dict] = []
        scroll: list[dict] = []

        for seg in session_data.get("segments", []):
            mouse.extend(seg.get("mouse", []))
            clicks.extend(seg.get("clicks", []))
            keystrokes.extend(seg.get("keystrokes", []))
            scroll.extend(seg.get("scroll", []))

        sessions.append(
            Session(
                session_id=sid,
                label=label,
                mouse=mouse,
                clicks=clicks,
                keystrokes=keystrokes,
                scroll=scroll,
                metadata={
                    "source": "chrome_extension",
                    "page_meta": session_data.get("pageMeta", []),
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
        ├── human/   ← Chrome extension exports (label=1)
        └── bot/     ← External bot data (label=0)

    ``human/*.json`` files use the Chrome extension export format (dict keyed
    by session ID).
    ``bot/*.json`` files use either:
      - Chrome extension format (dict keyed by session ID), or
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
    """Load a JSON file that may be in chrome-extension format or flat-array format."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sessions: list[Session] = []

    if isinstance(data, list):
        # Flat array format: [{ session_id, mouse, clicks, ... }, ...]
        for item in data:
            sid = item.get("session_id", item.get("sessionId", f"unknown_{id(item)}"))
            sessions.append(
                Session(
                    session_id=sid,
                    label=item.get("label", label),
                    mouse=_ensure_list(item.get("mouse")),
                    clicks=_ensure_list(item.get("clicks")),
                    keystrokes=_ensure_list(item.get("keystrokes")),
                    scroll=_ensure_list(item.get("scroll")),
                    metadata=item.get("metadata", {"source_file": str(path.name)}),
                )
            )
    elif isinstance(data, dict):
        # Check if this is chrome extension format (keys are session IDs with
        # nested segments) or a single flat session object.
        first_val = next(iter(data.values()), None) if data else None

        if isinstance(first_val, dict) and "segments" in first_val:
            # Chrome extension export format: { "<sessionId>": { segments: [...] } }
            for sid, session_data in data.items():
                mouse, clicks, keystrokes, scroll = [], [], [], []
                for seg in session_data.get("segments", []):
                    mouse.extend(seg.get("mouse", []))
                    clicks.extend(seg.get("clicks", []))
                    keystrokes.extend(seg.get("keystrokes", []))
                    scroll.extend(seg.get("scroll", []))
                sessions.append(
                    Session(
                        session_id=sid,
                        label=label,
                        mouse=mouse,
                        clicks=clicks,
                        keystrokes=keystrokes,
                        scroll=scroll,
                        metadata={
                            "source": "chrome_extension",
                            "source_file": str(path.name),
                            "page_meta": session_data.get("pageMeta", []),
                        },
                    )
                )
        elif "segments" in data and isinstance(data.get("segments"), list):
            # Live-confirm / webapp export format:
            # { "sessionId": "...", "segments": [{ "mouse": [...], ... }] }
            sid = data.get("session_id", data.get("sessionId", path.stem))
            mouse, clicks, keystrokes, scroll = [], [], [], []
            for seg in data["segments"]:
                mouse.extend(seg.get("mouse", []))
                clicks.extend(seg.get("clicks", []))
                keystrokes.extend(seg.get("keystrokes", []))
                scroll.extend(seg.get("scroll", []))
            sessions.append(
                Session(
                    session_id=sid,
                    label=data.get("label", label),
                    mouse=mouse,
                    clicks=clicks,
                    keystrokes=keystrokes,
                    scroll=scroll,
                    metadata={
                        "source": data.get("source", "live_confirm"),
                        "source_file": str(path.name),
                    },
                )
            )
        else:
            # Single flat session object
            sid = data.get("session_id", data.get("sessionId", path.stem))
            sessions.append(
                Session(
                    session_id=sid,
                    label=data.get("label", label),
                    mouse=_ensure_list(data.get("mouse")),
                    clicks=_ensure_list(data.get("clicks")),
                    keystrokes=_ensure_list(data.get("keystrokes")),
                    scroll=_ensure_list(data.get("scroll")),
                    metadata=data.get("metadata", {"source_file": str(path.name)}),
                )
            )

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

    def _split_group(
        group: list[Session],
    ) -> tuple[list[Session], list[Session], list[Session]]:
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
