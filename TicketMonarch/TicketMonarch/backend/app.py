import sys
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
    get_user_sessions,
)

# Add project root to sys.path so rl_captcha imports work everywhere
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# agent_service is imported lazily to avoid slow PyTorch/LSTM loading at startup
def _get_agent_service():
    from agent_service import get_agent_service
    return get_agent_service()

def _ACTION_NAMES():
    from agent_service import ACTION_NAMES
    return ACTION_NAMES

app = Flask(__name__)
# Enable CORS for Vite frontend (default port 5173)
CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])

# Initialize database when app starts
init_database()


@app.route('/api/health', methods=['GET'])
def health():
    """Simple health check endpoint."""
    return jsonify({'status': 'ok'}), 200


@app.route('/api/checkout', methods=['POST'])
def checkout():
    """Process checkout form submission and save to database"""
    try:
        data = request.json or {}

        # Prepare data - use empty strings if fields are missing
        order_data = {
            'full_name': data.get('full_name', '') or '',
            'email': data.get('email', '') or '',
            'card_number': data.get('card_number', '') or '',
            'card_expiry': data.get('card_expiry', '') or '',
            'card_cvv': data.get('card_cvv', '') or '',
            'billing_address': data.get('billing_address', '') or '',
            'city': data.get('city', '') or '',
            'state': data.get('state', '') or '',
            'zip_code': data.get('zip_code', '') or ''
        }

        order_id = save_order(order_data)

        return jsonify({
            'success': True,
            'id': order_id
        }), 201

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/tracking/mouse', methods=['POST'])
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
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'session_id is required'}), 400

        telemetry = {
            'page': data.get('page'),
            'mouse_movements': data.get('samples'),
        }

        save_user_session(session_id, telemetry)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/tracking/clicks', methods=['POST'])
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
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'session_id is required'}), 400

        telemetry = {
            'page': data.get('page'),
            'click_events': data.get('clicks'),
        }

        save_user_session(session_id, telemetry)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/tracking/keystrokes', methods=['POST'])
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
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'session_id is required'}), 400

        telemetry = {
            'page': data.get('page'),
            'keystroke_data': data.get('keystrokes'),
        }

        save_user_session(session_id, telemetry)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/tracking/scroll', methods=['POST'])
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
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'session_id is required'}), 400

        telemetry = {
            'page': data.get('page'),
            'scroll_events': data.get('scrolls'),
        }

        save_user_session(session_id, telemetry)
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/export/tracking', methods=['GET'])
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
        return jsonify(
            {
                'success': True,
                'file_path': csv_path,
                'message': 'Tracking data exported successfully.',
            }
        ), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/export', methods=['GET'])
def export_checkouts():
    """
    Export all checkout data to CSV for analysis.
    """
    try:
        csv_path = export_to_csv()
        return jsonify(
            {
                'success': True,
                'file_path': csv_path,
                'message': 'Checkout data exported successfully.',
            }
        ), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ---------------------------------------------------------------------------
# RL Agent endpoints
# ---------------------------------------------------------------------------

@app.route('/api/agent/evaluate', methods=['POST'])
def agent_evaluate():
    """Evaluate a session with the RL agent. Called at checkout."""
    try:
        data = request.json or {}
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'error': 'session_id required'}), 400

        db_session = get_user_session(session_id)
        if not db_session:
            return jsonify({
                'success': True,
                'decision': 'allow',
                'action_index': 5,
                'reason': 'no_session_data',
            }), 200

        from rl_captcha.data.loader import Session
        session = Session(
            session_id=session_id,
            label=None,
            mouse=db_session.get('mouse_movements') or [],
            clicks=db_session.get('click_events') or [],
            keystrokes=db_session.get('keystroke_data') or [],
            scroll=db_session.get('scroll_events') or [],
        )

        agent_svc = _get_agent_service()
        result = agent_svc.evaluate_session(session)
        result['success'] = True
        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/agent/dashboard/<session_id>', methods=['GET'])
def agent_dashboard(session_id):
    """Return detailed agent analysis for the dev dashboard."""
    try:
        db_session = get_user_session(session_id)
        if not db_session:
            return jsonify({'success': False, 'error': 'session not found'}), 404

        from rl_captcha.data.loader import Session
        session = Session(
            session_id=session_id,
            label=None,
            mouse=db_session.get('mouse_movements') or [],
            clicks=db_session.get('click_events') or [],
            keystrokes=db_session.get('keystroke_data') or [],
            scroll=db_session.get('scroll_events') or [],
        )

        agent_svc = _get_agent_service()
        result = agent_svc.evaluate_session(session)

        # Add LSTM hidden state for visualization
        hidden_info = agent_svc.get_hidden_state_info()
        result.update(hidden_info)

        result['telemetry_summary'] = {
            'mouse_count': len(session.mouse),
            'click_count': len(session.clicks),
            'keystroke_count': len(session.keystrokes),
            'scroll_count': len(session.scroll),
        }

        result['success'] = True
        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/agent/sessions', methods=['GET'])
def agent_sessions():
    """List recent sessions for the dev dashboard."""
    try:
        limit = request.args.get('limit', 20, type=int)
        sessions = get_user_sessions(limit=limit)

        summary = []
        for s in sessions:
            summary.append({
                'session_id': s['session_id'],
                'session_start': str(s.get('session_start', '')),
                'page': s.get('page'),
                'event_counts': {
                    'mouse': len(s.get('mouse_movements') or []),
                    'clicks': len(s.get('click_events') or []),
                    'keystrokes': len(s.get('keystroke_data') or []),
                },
            })

        return jsonify({'success': True, 'sessions': summary}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/agent/live/<session_id>', methods=['GET'])
def agent_live(session_id):
    """Lightweight live telemetry endpoint — no agent inference, just counts."""
    try:
        db_session = get_user_session(session_id)
        if not db_session:
            return jsonify({
                'success': True,
                'found': False,
                'mouse_count': 0,
                'click_count': 0,
                'keystroke_count': 0,
                'scroll_count': 0,
                'page': None,
            }), 200

        return jsonify({
            'success': True,
            'found': True,
            'session_id': session_id,
            'page': db_session.get('page'),
            'session_start': str(db_session.get('session_start', '')),
            'mouse_count': len(db_session.get('mouse_movements') or []),
            'click_count': len(db_session.get('click_events') or []),
            'keystroke_count': len(db_session.get('keystroke_data') or []),
            'scroll_count': len(db_session.get('scroll_events') or []),
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    # Pre-warm agent in background so first request isn't slow
    import threading
    def _warmup():
        try:
            _get_agent_service()
            print("[warmup] Agent service ready.")
        except Exception as e:
            print(f"[warmup] Agent load failed: {e}")
    threading.Thread(target=_warmup, daemon=True).start()

    app.run(debug=True, port=5000)
