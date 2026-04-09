import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from database import (
    init_database,
    save_order,
    export_to_csv,
    save_user_session,
    export_tracking_data_to_csv,
    get_user_session,
    get_recent_session_ids,
    get_session_summaries,
    ensure_indexes,
)

# Add project root to sys.path so rl_captcha imports work everywhere
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent_service import get_agent_service as _init_agent_service

# Agent loads once at startup — guaranteed ready before any request.
print("[startup] Loading RL agent (PyTorch + LSTM checkpoint)...")

_agent_ref = _init_agent_service()
print("[startup] Agent ready.")


def _get_agent_service():
    return _agent_ref


def _ACTION_NAMES():
    from agent_service import ACTION_NAMES

    return ACTION_NAMES


app = Flask(__name__)
# Enable CORS for Vite frontend (default port 5173)
CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])

# Initialize database when app starts
init_database()
ensure_indexes()


@app.route("/api/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"}), 200


# Dev dashboard CAPTCHA flag setter / getter
flagState = {"flag": "inactive"}
VALID_STATES = {"inactive", "on", "off"}


@app.route("/api/set_flag", methods=["POST"])
def set_flag():
    data = request.json or {}
    state = data.get("flag")

    if state not in VALID_STATES:
        return jsonify({"success": False, "error": "Invalid flag state"}), 400

    flagState["flag"] = state

    return jsonify({"success": True, "flag": flagState["flag"]}), 200


@app.route("/api/get_flag", methods=["GET"])
def get_flag():
    return jsonify({"success": True, "flag": flagState["flag"]}), 200


@app.route("/api/checkout", methods=["POST"])
def checkout():
    """Process checkout form submission and save to database"""
    try:
        data = request.json or {}

        # Prepare data - use empty strings if fields are missing
        order_data = {
            "full_name": data.get("full_name", "") or "",
            "email": data.get("email", "") or "",
            "card_number": data.get("card_number", "") or "",
            "card_expiry": data.get("card_expiry", "") or "",
            "card_cvv": data.get("card_cvv", "") or "",
            "billing_address": data.get("billing_address", "") or "",
            "city": data.get("city", "") or "",
            "state": data.get("state", "") or "",
            "zip_code": data.get("zip_code", "") or "",
        }

        order_id = save_order(order_data)

        return jsonify({"success": True, "id": order_id}), 201

    except Exception as e:
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/tracking/mouse", methods=["POST"])
def tracking_mouse():
    """
    Receive batched mouse movement samples for a session.
    Expects payload:
    {
        "session_id": "...",
        "page": "home|seat_selection|checkout|confirmation",
        "samples": [{ x, y, t }, ...]
    }
    """
    try:
        data = request.json or {}
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"success": False, "error": "session_id is required"}), 400

        telemetry = {
            "page": data.get("page"),
            "mouse_movements": data.get("samples"),
        }

        save_user_session(session_id, telemetry)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/tracking/clicks", methods=["POST"])
def tracking_clicks():
    """
    Receive batched click events for a session.
    Expects payload:
    {
        "session_id": "...",
        "page": "...",
        "clicks": [{ t, x, y, button, target, dt_since_last }, ...]
    }
    """
    try:
        data = request.json or {}
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"success": False, "error": "session_id is required"}), 400

        telemetry = {
            "page": data.get("page"),
            "click_events": data.get("clicks"),
        }

        save_user_session(session_id, telemetry)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/tracking/keystrokes", methods=["POST"])
def tracking_keystrokes():
    """
    Receive batched keystroke timing events for a session.
    Expects payload:
    {
        "session_id": "...",
        "page": "...",
        "keystrokes": [{ field, type: "down|up", t, dt_since_last? }, ...]
    }
    Only timing and field identifiers are tracked (no actual key values).
    """
    try:
        data = request.json or {}
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"success": False, "error": "session_id is required"}), 400

        telemetry = {
            "page": data.get("page"),
            "keystroke_data": data.get("keystrokes"),
        }

        save_user_session(session_id, telemetry)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/tracking/scroll", methods=["POST"])
def tracking_scroll():
    """
    Receive batched scroll events for a session.
    Expects payload:
    {
        "session_id": "...",
        "page": "...",
        "scrolls": [{ t, scrollX, scrollY, dy, dt_since_last }, ...]
    }
    """
    try:
        data = request.json or {}
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"success": False, "error": "session_id is required"}), 400

        telemetry = {
            "page": data.get("page"),
            "scroll_events": data.get("scrolls"),
        }

        save_user_session(session_id, telemetry)
        return jsonify({"success": True}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/export/tracking", methods=["GET"])
def export_tracking():
    """
    Export all user session telemetry data to CSV for RL/ML training.

    Returns:
        {
            "success": true,
            "file_path": "...",
            "message": "..."
        }
    """
    try:
        csv_path = export_tracking_data_to_csv()
        return (
            jsonify(
                {
                    "success": True,
                    "file_path": csv_path,
                    "message": "Tracking data exported successfully.",
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/export", methods=["GET"])
def export_checkouts():
    """
    Export all checkout data to CSV for analysis.
    """
    try:
        csv_path = export_to_csv()
        return (
            jsonify(
                {
                    "success": True,
                    "file_path": csv_path,
                    "message": "Checkout data exported successfully.",
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Honeypot detection
# ---------------------------------------------------------------------------

HONEYPOT_FIELDS = {"company_url", "fax_number"}


def _check_honeypot(keystroke_data, click_events):
    """Return True if any honeypot field was interacted with."""
    for ks in keystroke_data or []:
        if ks.get("field") in HONEYPOT_FIELDS:
            return True
    for click in click_events or []:
        target = click.get("target")
        if isinstance(target, dict):
            field_id = target.get("name") or target.get("id") or ""
            if field_id in HONEYPOT_FIELDS:
                return True
    return False


# ---------------------------------------------------------------------------
# RL Agent endpoints
# ---------------------------------------------------------------------------


@app.route("/api/agent/rolling", methods=["POST"])
def agent_rolling():
    """Rolling inference — called periodically during form fill-out.

    Runs the LSTM on all events so far and returns:
    - bot_probability: how suspicious the session looks (0-1)
    - deploy_honeypot: whether the agent wants to deploy a honeypot
    - events_processed: how many events were analyzed

    This does NOT make a terminal decision — that happens at checkout.
    """
    try:
        data = request.json or {}
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"success": False, "error": "session_id required"}), 400

        db_session = get_user_session(session_id)
        if not db_session:
            return (
                jsonify(
                    {
                        "success": True,
                        "bot_probability": 0.0,
                        "deploy_honeypot": False,
                        "events_processed": 0,
                        "honeypot_triggered": False,
                    }
                ),
                200,
            )

        # Check if honeypot was already triggered
        honeypot_triggered = _check_honeypot(
            db_session.get("keystroke_data") or [],
            db_session.get("click_events") or [],
        )

        from rl_captcha.data.loader import Session

        session = Session(
            session_id=session_id,
            label=None,
            mouse=db_session.get("mouse_movements") or [],
            clicks=db_session.get("click_events") or [],
            keystrokes=db_session.get("keystroke_data") or [],
            scroll=db_session.get("scroll_events") or [],
        )

        agent_svc = _get_agent_service()
        result = agent_svc.rolling_evaluate(session)
        result["success"] = True
        result["honeypot_triggered"] = honeypot_triggered
        return jsonify(result), 200

    except Exception as e:
        import traceback

        traceback.print_exc()
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "bot_probability": 0.0,
                    "deploy_honeypot": False,
                    "events_processed": 0,
                    "honeypot_triggered": False,
                }
            ),
            500,
        )


@app.route("/api/agent/evaluate", methods=["POST"])
def agent_evaluate():
    """Evaluate a session with the RL agent. Called at checkout."""
    try:
        data = request.json or {}
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"success": False, "error": "session_id required"}), 400

        db_session = get_user_session(session_id)
        if not db_session:
            return (
                jsonify(
                    {
                        "success": True,
                        "decision": "allow",
                        "action_index": 5,
                        "reason": "no_session_data",
                    }
                ),
                200,
            )

        # Honeypot pre-check: if a bot typed in a hidden field, skip RL
        honeypot_triggered = _check_honeypot(
            db_session.get("keystroke_data") or [],
            db_session.get("click_events") or [],
        )
        if honeypot_triggered:
            print(
                f"[honeypot] Session {session_id} triggered honeypot — instant hard_puzzle"
            )
            return (
                jsonify(
                    {
                        "success": True,
                        "decision": "hard_puzzle",
                        "action_index": 4,
                        "confidence": 1.0,
                        "events_processed": 0,
                        "total_events": 0,
                        "reason": "honeypot_triggered",
                        "honeypot_triggered": True,
                        "action_history": [],
                        "final_probs": [0, 0, 0, 0, 1, 0, 0],
                        "final_value": 0.0,
                    }
                ),
                200,
            )

        from rl_captcha.data.loader import Session

        session = Session(
            session_id=session_id,
            label=None,
            mouse=db_session.get("mouse_movements") or [],
            clicks=db_session.get("click_events") or [],
            keystrokes=db_session.get("keystroke_data") or [],
            scroll=db_session.get("scroll_events") or [],
        )

        agent_svc = _get_agent_service()
        result = agent_svc.evaluate_session(session)
        result["success"] = True
        result["honeypot_triggered"] = False
        return jsonify(result), 200

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/agent/dashboard/<session_id>", methods=["GET"])
def agent_dashboard(session_id):
    """Return detailed agent analysis for the dev dashboard."""
    try:
        db_session = get_user_session(session_id)
        if not db_session:
            return jsonify({"success": False, "error": "session not found"}), 404

        from rl_captcha.data.loader import Session

        session = Session(
            session_id=session_id,
            label=None,
            mouse=db_session.get("mouse_movements") or [],
            clicks=db_session.get("click_events") or [],
            keystrokes=db_session.get("keystroke_data") or [],
            scroll=db_session.get("scroll_events") or [],
        )

        agent_svc = _get_agent_service()
        result = agent_svc.evaluate_session(session)

        # Add LSTM hidden state for visualization
        hidden_info = agent_svc.get_hidden_state_info()
        result.update(hidden_info)

        result["telemetry_summary"] = {
            "mouse_count": len(session.mouse),
            "click_count": len(session.clicks),
            "keystroke_count": len(session.keystrokes),
            "scroll_count": len(session.scroll),
        }

        result["honeypot_triggered"] = _check_honeypot(
            db_session.get("keystroke_data") or [],
            db_session.get("click_events") or [],
        )

        result["success"] = True
        return jsonify(result), 200

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/agent/sessions", methods=["GET"])
def agent_sessions():
    """List recent sessions for the dev dashboard (lightweight, no JSON blobs)."""
    try:
        limit = request.args.get("limit", 20, type=int)
        rows = get_session_summaries(limit=limit)

        summary = []
        for s in rows:
            summary.append(
                {
                    "session_id": s["session_id"],
                    "session_start": str(s.get("session_start", "")),
                    "page": s.get("page"),
                    "event_counts": {
                        "mouse": s.get("mouse_count") or 0,
                        "clicks": s.get("click_count") or 0,
                        "keystrokes": s.get("keystroke_count") or 0,
                    },
                }
            )

        return jsonify({"success": True, "sessions": summary}), 200
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/agent/session-ids", methods=["GET"])
def agent_session_ids():
    """Lightweight endpoint returning just session IDs (no JSON parsing)."""
    try:
        limit = request.args.get("limit", 10, type=int)
        ids = get_recent_session_ids(limit=limit)
        return jsonify({"success": True, "session_ids": ids}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/agent/live/<session_id>", methods=["GET"])
def agent_live(session_id):
    """Lightweight live telemetry endpoint — no agent inference, just counts."""
    try:
        db_session = get_user_session(session_id)
        if not db_session:
            return (
                jsonify(
                    {
                        "success": True,
                        "found": False,
                        "mouse_count": 0,
                        "click_count": 0,
                        "keystroke_count": 0,
                        "scroll_count": 0,
                        "page": None,
                    }
                ),
                200,
            )

        return (
            jsonify(
                {
                    "success": True,
                    "found": True,
                    "session_id": session_id,
                    "page": db_session.get("page"),
                    "session_start": str(db_session.get("session_start", "")),
                    "mouse_count": len(db_session.get("mouse_movements") or []),
                    "click_count": len(db_session.get("click_events") or []),
                    "keystroke_count": len(db_session.get("keystroke_data") or []),
                    "scroll_count": len(db_session.get("scroll_events") or []),
                    "honeypot_keystrokes": sum(
                        1
                        for ks in (db_session.get("keystroke_data") or [])
                        if ks.get("field") in HONEYPOT_FIELDS
                    ),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/agent/confirm", methods=["POST"])
def agent_confirm():
    """Confirm a session's true label and trigger online RL learning.

    Called by bot scripts (label=0) or human confirmation (label=1).
    The RL agent replays the session with the true label and does a
    PPO gradient update so it learns from its mistakes in real time.

    Expects: { "session_id": "...", "true_label": 0 or 1 }
    """
    try:
        data = request.json or {}
        session_id = data.get("session_id")
        true_label = data.get("true_label")

        if not session_id:
            return jsonify({"success": False, "error": "session_id required"}), 400
        if true_label not in (0, 1):
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "true_label must be 0 (bot) or 1 (human)",
                    }
                ),
                400,
            )

        db_session = get_user_session(session_id)
        if not db_session:
            print(f"[agent_confirm] ERROR: Session {session_id} not found in database!")
            return jsonify({"success": False, "error": "session not found"}), 404

        from rl_captcha.data.loader import Session

        session = Session(
            session_id=session_id,
            label=true_label,
            mouse=db_session.get("mouse_movements") or [],
            clicks=db_session.get("click_events") or [],
            keystrokes=db_session.get("keystroke_data") or [],
            scroll=db_session.get("scroll_events") or [],
        )

        # ── Save session to JSON for training data ──
        # Set DISABLE_HUMAN_SAVE=1 when running LLM bots to prevent
        # evasive bots from being saved as human data.
        json_path = None
        if true_label == 1 and not os.environ.get("DISABLE_HUMAN_SAVE"):
            data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "human"
            data_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")
            json_path = data_dir / f"session_{session_id[:12]}_{ts}.json"

            export_payload = {
                "sessionId": session_id,
                "label": true_label,
                "exportedAt": datetime.now(timezone.utc).isoformat(),
                "source": "live_confirm",
                "segments": [
                    {
                        "mouse": db_session.get("mouse_movements") or [],
                        "clicks": db_session.get("click_events") or [],
                        "keystrokes": db_session.get("keystroke_data") or [],
                        "scroll": db_session.get("scroll_events") or [],
                    }
                ],
            }
            try:
                json_path.write_text(json.dumps(export_payload, indent=2))
                print(f"[agent_confirm] Saved human session to {json_path.name}")
            except Exception as save_err:
                print(f"[agent_confirm] WARNING: Failed to save JSON: {save_err}")
                json_path = None

        agent_svc = _get_agent_service()
        # Skip online learning for human labels when DISABLE_HUMAN_SAVE is set
        # (LLM bot mode) — the frontend auto-confirms with label=1 but it's actually a bot
        if true_label == 1 and os.environ.get("DISABLE_HUMAN_SAVE"):
            result = {"updated": False, "reason": "human_save_disabled"}
        else:
            result = agent_svc.online_learn(session, true_label)
        result["success"] = True
        result["session_id"] = session_id
        result["saved_json"] = str(json_path.name) if json_path else None
        return jsonify(result), 200

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/session/raw/<session_id>", methods=["GET"])
def session_raw(session_id):
    """Return raw telemetry arrays for a session (used by bots for export)."""
    try:
        db_session = get_user_session(session_id)
        if not db_session:
            return jsonify({"success": False, "error": "session not found"}), 404

        return (
            jsonify(
                {
                    "success": True,
                    "session_id": session_id,
                    "page": db_session.get("page"),
                    "session_start": str(db_session.get("session_start", "")),
                    "mouse": db_session.get("mouse_movements") or [],
                    "clicks": db_session.get("click_events") or [],
                    "keystrokes": db_session.get("keystroke_data") or [],
                    "scroll": db_session.get("scroll_events") or [],
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # use_reloader=False so the module only loads once (agent doesn't load twice)
    app.run(debug=True, use_reloader=False, port=5000)
