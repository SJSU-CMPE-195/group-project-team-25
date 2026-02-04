from flask import Flask, request, jsonify
from flask_cors import CORS
from database import (
    init_database,
    save_order,
    export_to_csv,
    save_user_session,
    export_tracking_data_to_csv,
)

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


if __name__ == '__main__':
    app.run(debug=True, port=5000)
