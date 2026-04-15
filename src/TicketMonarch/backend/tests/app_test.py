import importlib
import sys
import types

import pytest


@pytest.fixture
def client():
    # ---- fake database module ----
    fake_db = types.ModuleType("TicketMonarch.backend.database")

    session_store = {
        "sess-1": {
            "session_id": "sess-1",
            "page": "checkout",
            "session_start": "2026-04-11 12:00:00",
            "mouse_movements": [{"x": 10, "y": 20, "t": 1}],
            "click_events": [{"target": {"id": "submit-btn"}}],
            "keystroke_data": [{"field": "email", "type": "down", "t": 2}],
            "scroll_events": [{"scrollY": 200, "t": 3}],
        },
        "honeypot-1": {
            "session_id": "honeypot-1",
            "page": "checkout",
            "session_start": "2026-04-11 12:00:00",
            "mouse_movements": [],
            "click_events": [],
            "keystroke_data": [{"field": "company_url", "type": "down", "t": 1}],
            "scroll_events": [],
        },
    }

    fake_db.init_database = lambda: None
    fake_db.ensure_indexes = lambda: None
    fake_db.save_order = lambda order_data: 123
    fake_db.export_to_csv = lambda: "/tmp/checkouts.csv"
    fake_db.export_tracking_data_to_csv = lambda: "/tmp/tracking.csv"
    fake_db.save_user_session = lambda session_id, telemetry: None
    fake_db.get_user_session = lambda session_id: session_store.get(session_id)
    fake_db.get_recent_session_ids = lambda limit=10: ["sess-1", "sess-2"][:limit]
    fake_db.get_session_summaries = lambda limit=20: [
        {
            "session_id": "sess-1",
            "session_start": "2026-04-11 12:00:00",
            "page": "checkout",
            "mouse_count": 1,
            "click_count": 1,
            "keystroke_count": 1,
            "scroll_count": 1,
        }
    ][:limit]

    # ---- fake agent_service module ----
    fake_agent_service = types.ModuleType("TicketMonarch.backend.agent_service")

    class DummyAgentService:
        def rolling_evaluate(self, session):
            return {
                "bot_probability": 0.2,
                "deploy_honeypot": False,
                "events_processed": 3,
                "num_windows": 1,
                "action_distribution": {"allow": 0.8, "block": 0.2},
            }

        def evaluate_session(self, session):
            return {
                "decision": "allow",
                "action_index": 5,
                "confidence": 0.8,
                "events_processed": 3,
                "total_events": 3,
                "num_windows": 1,
                "windows_processed": 1,
                "final_probs": [0, 0, 0, 0, 0, 1, 0],
                "final_value": 0.5,
                "action_history": [],
                "p_allow": 0.8,
                "p_suspicious": 0.2,
                "algorithm": "ppo",
            }

        def get_hidden_state_info(self):
            return {
                "lstm_hidden_norm": 0.1,
                "lstm_cell_norm": 0.2,
                "lstm_hidden_values": [0.1, 0.2],
            }

        def online_learn(self, session, true_label):
            return {
                "updated": True,
                "steps": 1,
                "true_label": true_label,
                "before_decision": "allow",
                "after_decision": "allow",
                "improvement": "UNCHANGED",
            }

    fake_agent_service.get_agent_service = lambda: DummyAgentService()
    fake_agent_service.ACTION_NAMES = [
        "continue",
        "honeypot",
        "easy_puzzle",
        "medium_puzzle",
        "hard_puzzle",
        "allow",
        "block",
    ]

    # ---- fake rl_captcha.data.loader.Session ----
    fake_loader = types.ModuleType("rl_captcha.data.loader")

    class Session:
        def __init__(self, session_id, label, mouse, clicks, keystrokes, scroll):
            self.session_id = session_id
            self.label = label
            self.mouse = mouse
            self.clicks = clicks
            self.keystrokes = keystrokes
            self.scroll = scroll

    fake_loader.Session = Session

    # install stubs before importing app
    sys.modules["TicketMonarch.backend.database"] = fake_db
    sys.modules["TicketMonarch.backend.agent_service"] = fake_agent_service
    sys.modules["rl_captcha.data.loader"] = fake_loader

    if "TicketMonarch.backend.app" in sys.modules:
        del sys.modules["TicketMonarch.backend.app"]

    app_module = importlib.import_module("TicketMonarch.backend.app")
    flask_app = app_module.app
    flask_app.config["TESTING"] = True

    with flask_app.test_client() as client:
        yield client


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_get_flag(client):
    r = client.get("/api/get_flag")
    assert r.status_code == 200
    data = r.get_json()
    assert data["success"] is True
    assert data["flag"] == "inactive"


def test_set_flag_valid(client):
    r = client.post("/api/set_flag", json={"flag": "on"})
    assert r.status_code == 200
    assert r.get_json()["flag"] == "on"


def test_set_flag_invalid(client):
    r = client.post("/api/set_flag", json={"flag": "bad"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_checkout_success(client):
    r = client.post(
        "/api/checkout",
        json={
            "full_name": "A User",
            "email": "a@example.com",
            "card_number": "4111111111111111",
        },
    )
    assert r.status_code == 201
    data = r.get_json()
    assert data["success"] is True
    assert data["id"] == 123


def test_checkout_internal_server_error(client, monkeypatch):
    def fake_save_order(order_data):
        raise Exception("DB failed")

    monkeypatch.setattr("TicketMonarch.backend.app.save_order", fake_save_order)

    r = client.post(
        "/api/checkout",
        json={
            "full_name": "Test User",
            "email": "test@example.com",
        },
    )

    assert r.status_code == 500
    assert r.get_json() == {
        "success": False,
        "error": "Internal server error",
    }


TRACKING_CASES = [
    ("/api/tracking/mouse", "samples", [{"x": 1, "y": 2, "t": 1}]),
    ("/api/tracking/clicks", "clicks", [{"t": 1}]),
    ("/api/tracking/keystrokes", "keystrokes", [{"field": "email", "t": 1}]),
    ("/api/tracking/scroll", "scrolls", [{"scrollY": 100, "t": 1}]),
]


@pytest.mark.parametrize("route,payload_key,payload_value", TRACKING_CASES)
def test_tracking_success(client, route, payload_key, payload_value):
    r = client.post(
        route,
        json={"session_id": "sess-1", "page": "home", payload_key: payload_value},
    )
    assert r.status_code == 200
    assert r.get_json()["success"] is True


@pytest.mark.parametrize("route,payload_key,payload_value", TRACKING_CASES)
def test_tracking_requires_session_id(client, route, payload_key, payload_value):
    r = client.post(
        route,
        json={"page": "home", payload_key: payload_value},
    )
    assert r.status_code == 400
    assert "session_id is required" in r.get_json()["error"]


@pytest.mark.parametrize("route,payload_key,payload_value", TRACKING_CASES)
def test_tracking_internal_server_error(
    client, monkeypatch, route, payload_key, payload_value
):
    def fake_save_user_session(session_id, telemetry):
        raise Exception("DB failed")

    monkeypatch.setattr(
        "TicketMonarch.backend.app.save_user_session",
        fake_save_user_session,
    )

    r = client.post(
        route,
        json={"session_id": "sess-1", "page": "home", payload_key: payload_value},
    )

    assert r.status_code == 500
    assert r.get_json() == {
        "success": False,
        "error": "DB failed",
    }


EXPORT_CASES = [
    ("/api/export/tracking", "export_tracking_data_to_csv"),
    ("/api/export", "export_to_csv"),
]


@pytest.mark.parametrize("route,db_call", EXPORT_CASES)
def test_export(client, route, db_call):
    r = client.get(route)
    assert r.status_code == 200
    data = r.get_json()
    assert data["success"] is True
    assert "file_path" in data


@pytest.mark.parametrize("route,db_call", EXPORT_CASES)
def test_export_internal_server_error(client, route, db_call, monkeypatch):
    def fake_get_export():
        raise Exception("DB failed")

    monkeypatch.setattr(
        f"TicketMonarch.backend.app.{db_call}",
        fake_get_export,
    )

    r = client.get(route)
    assert r.status_code == 500
    assert r.get_json() == {
        "success": False,
        "error": "DB failed",
    }


def test_agent_rolling_missing_session_id(client):
    r = client.post("/api/agent/rolling", json={})
    assert r.status_code == 400


def test_agent_rolling_no_session_data(client):
    r = client.post("/api/agent/rolling", json={"session_id": "missing"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["success"] is True
    assert data["bot_probability"] == 0.0


def test_agent_rolling_success(client):
    r = client.post("/api/agent/rolling", json={"session_id": "sess-1"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["success"] is True
    assert "bot_probability" in data


def test_agent_evaluate_missing_session_id(client):
    r = client.post("/api/agent/evaluate", json={})
    assert r.status_code == 400


def test_agent_evaluate_no_session_data(client):
    r = client.post("/api/agent/evaluate", json={"session_id": "missing"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["decision"] == "allow"
    assert data["reason"] == "no_session_data"


def test_agent_evaluate_honeypot_triggered(client):
    r = client.post("/api/agent/evaluate", json={"session_id": "honeypot-1"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["decision"] == "hard_puzzle"
    assert data["honeypot_triggered"] is True


def test_agent_evaluate_success(client):
    r = client.post("/api/agent/evaluate", json={"session_id": "sess-1"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["success"] is True
    assert data["decision"] == "allow"


def test_agent_dashboard_not_found(client):
    r = client.get("/api/agent/dashboard/missing")
    assert r.status_code == 404


def test_agent_dashboard_success(client):
    r = client.get("/api/agent/dashboard/sess-1")
    assert r.status_code == 200
    data = r.get_json()
    assert data["success"] is True
    assert "telemetry_summary" in data


def test_agent_sessions(client):
    r = client.get("/api/agent/sessions")
    assert r.status_code == 200
    data = r.get_json()
    assert data["success"] is True
    assert isinstance(data["sessions"], list)


def test_agent_session_ids(client):
    r = client.get("/api/agent/session-ids")
    assert r.status_code == 200
    data = r.get_json()
    assert data["success"] is True
    assert "session_ids" in data


def test_agent_live_missing_session(client):
    r = client.get("/api/agent/live/missing")
    assert r.status_code == 200
    data = r.get_json()
    assert data["found"] is False


def test_agent_live_success(client):
    r = client.get("/api/agent/live/sess-1")
    assert r.status_code == 200
    data = r.get_json()
    assert data["success"] is True
    assert data["found"] is True


def test_agent_confirm_missing_session_id(client):
    r = client.post("/api/agent/confirm", json={"true_label": 1})
    assert r.status_code == 400


def test_agent_confirm_invalid_label(client):
    r = client.post(
        "/api/agent/confirm", json={"session_id": "sess-1", "true_label": 3}
    )
    assert r.status_code == 400


def test_agent_confirm_session_not_found(client):
    r = client.post(
        "/api/agent/confirm", json={"session_id": "missing", "true_label": 1}
    )
    assert r.status_code == 404


def test_agent_confirm_success(client):
    r = client.post(
        "/api/agent/confirm", json={"session_id": "sess-1", "true_label": 1}
    )
    assert r.status_code == 200
    data = r.get_json()
    assert data["success"] is True
    assert data["session_id"] == "sess-1"


def test_session_raw_not_found(client):
    r = client.get("/api/session/raw/missing")
    assert r.status_code == 404


def test_session_raw_success(client):
    r = client.get("/api/session/raw/sess-1")
    assert r.status_code == 200
    data = r.get_json()
    assert data["success"] is True
    assert data["session_id"] == "sess-1"
