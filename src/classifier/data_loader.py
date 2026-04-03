"""Data loader for the classifier — handles all JSON formats in data/human/ and data/bot/.

Replaces the dependency on rl_captcha.data.loader.load_from_directory() which
doesn't correctly parse the live-confirm export format (single flat session
object with a top-level ``segments`` key).

Supported formats:
    1. Single flat session: ``{ "sessionId": "...", "mouse": [...], ... }``
    2. Single session with segments: ``{ "sessionId": "...", "segments": [{ "mouse": [...], ... }] }``
    3. Chrome extension export: ``{ "<session_id>": { "segments": [...] }, ... }``
    4. Flat array: ``[ { "session_id": "...", "mouse": [...] }, ... ]``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Session:
    """Normalized telemetry session."""

    session_id: str
    label: int | None = None  # 1 = human, 0 = bot
    mouse: list[dict] = field(default_factory=list)
    clicks: list[dict] = field(default_factory=list)
    keystrokes: list[dict] = field(default_factory=list)
    scroll: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def load_from_directory(data_dir: str | Path) -> list[Session]:
    """Load all sessions from data/human/ (label=1) and data/bot/ (label=0).

    Handles all JSON formats found in the project.
    """
    data_dir = Path(data_dir)
    sessions: list[Session] = []

    for subdir, label in [("human", 1), ("bot", 0)]:
        folder = data_dir / subdir
        if not folder.is_dir():
            continue
        for f in sorted(folder.glob("*.json")):
            try:
                sessions.extend(_load_json_file(f, label=label))
            except Exception as e:
                print(f"Warning: skipping {f} — {e}")

    return sessions


def _merge_segments(segments: list[dict]) -> tuple[list, list, list, list]:
    """Merge telemetry from a list of segment dicts into flat lists."""
    mouse, clicks, keystrokes, scroll = [], [], [], []
    for seg in segments:
        mouse.extend(seg.get("mouse", []))
        clicks.extend(seg.get("clicks", []))
        keystrokes.extend(seg.get("keystrokes", []))
        scroll.extend(seg.get("scroll", []))
    return mouse, clicks, keystrokes, scroll


def _load_json_file(path: Path, label: int) -> list[Session]:
    """Load sessions from a single JSON file, auto-detecting the format."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sessions: list[Session] = []

    if isinstance(data, list):
        # Format 4: flat array of session objects
        for item in data:
            sid = item.get("session_id", item.get("sessionId", f"unknown_{id(item)}"))
            if "segments" in item:
                mouse, clicks, keystrokes, scroll = _merge_segments(item["segments"])
            else:
                mouse = _ensure_list(item.get("mouse"))
                clicks = _ensure_list(item.get("clicks"))
                keystrokes = _ensure_list(item.get("keystrokes"))
                scroll = _ensure_list(item.get("scroll"))
            sessions.append(
                Session(
                    session_id=sid,
                    label=item.get("label", label),
                    mouse=mouse,
                    clicks=clicks,
                    keystrokes=keystrokes,
                    scroll=scroll,
                    metadata={"source_file": path.name},
                )
            )

    elif isinstance(data, dict):
        first_val = next(iter(data.values()), None) if data else None

        if isinstance(first_val, dict) and "segments" in first_val:
            # Format 3: chrome extension export (keys are session IDs)
            for sid, session_data in data.items():
                mouse, clicks, keystrokes, scroll = _merge_segments(
                    session_data.get("segments", [])
                )
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
                            "source_file": path.name,
                        },
                    )
                )
        else:
            # Format 1 or 2: single session object
            sid = data.get("session_id", data.get("sessionId", path.stem))
            if "segments" in data:
                mouse, clicks, keystrokes, scroll = _merge_segments(data["segments"])
            else:
                mouse = _ensure_list(data.get("mouse"))
                clicks = _ensure_list(data.get("clicks"))
                keystrokes = _ensure_list(data.get("keystrokes"))
                scroll = _ensure_list(data.get("scroll"))
            sessions.append(
                Session(
                    session_id=sid,
                    label=data.get("label", label),
                    mouse=mouse,
                    clicks=clicks,
                    keystrokes=keystrokes,
                    scroll=scroll,
                    metadata={"source_file": path.name},
                )
            )

    return sessions


def _ensure_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]
