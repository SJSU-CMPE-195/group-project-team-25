"""Pre-generate adversarially augmented bot sessions for the classifier.

Applies the exact same HumanProfiler + 3-tier augmentation pipeline used at
training time (Section 3.5.3).  For every bot session in data/bot/, generates
``n_copies_per_level × 3`` humanized copies and saves them as individual JSON
files in data/bot_augmented/.

Usage (from repo root):
    python src/classifier/scripts/generate_augmented_data.py \
        --data-dir data/ \
        --n-copies 2 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or scripts/ directory
_SRC_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_SRC_ROOT))

from classifier.augmentation import adversarial_augment_sessions
from classifier.data_loader import Session, load_from_directory


def _session_to_dict(session: Session) -> dict:
    """Serialize a Session object to the flat JSON format the loader expects."""
    return {
        "session_id": session.session_id,
        "label": session.label,
        "mouse": session.mouse,
        "clicks": session.clicks,
        "keystrokes": session.keystrokes,
        "scroll": session.scroll,
        "metadata": session.metadata,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate adversarially augmented bot sessions "
        "(same pipeline as classifier Section 3.5.3)"
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Path to data directory with human/ and bot/ subdirs",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <data-dir>/bot_augmented)",
    )
    p.add_argument(
        "--n-copies",
        type=int,
        default=2,
        help="Number of augmented copies per bot per difficulty level (default: 2)",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "bot_augmented"

    # Load sessions
    print(f"Loading sessions from {data_dir}...")
    sessions = load_from_directory(data_dir)
    human_sessions = [s for s in sessions if s.label == 1]
    bot_sessions = [s for s in sessions if s.label == 0]
    print(f"  {len(human_sessions)} human sessions, {len(bot_sessions)} bot sessions")

    if not human_sessions or not bot_sessions:
        print("ERROR: Need both human and bot sessions.")
        return

    # Generate augmented sessions (identical to classifier training pipeline)
    print(f"\nGenerating augmented sessions (n_copies_per_level={args.n_copies})...")
    augmented = adversarial_augment_sessions(
        bot_sessions=bot_sessions,
        human_sessions=human_sessions,
        n_copies_per_level=args.n_copies,
        random_state=args.seed,
    )

    if not augmented:
        print("ERROR: No augmented sessions generated.")
        return

    # Save to output directory (one JSON file per augmented session)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing files first
    existing = list(out_dir.glob("*.json"))
    if existing:
        print(f"  Clearing {len(existing)} existing files in {out_dir}/")
        for f in existing:
            f.unlink()

    print(f"  Saving {len(augmented)} augmented sessions to {out_dir}/")
    for session in augmented:
        filepath = out_dir / f"{session.session_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(_session_to_dict(session), f)

    # Summary by level
    levels: dict[str, int] = {}
    for s in augmented:
        lvl = s.metadata.get("aug_level", "unknown")
        levels[lvl] = levels.get(lvl, 0) + 1
    print(f"\n  Summary:")
    for lvl, count in sorted(levels.items()):
        print(f"    {lvl}: {count} sessions")
    print(f"    total: {len(augmented)} sessions")
    print(f"\nDone! Augmented data saved to {out_dir}/")


if __name__ == "__main__":
    main()
