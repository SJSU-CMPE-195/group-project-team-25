import mysql.connector
import pytest
import json

from TicketMonarch.backend.app import app
from TicketMonarch.backend.config import get_db_config


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_checkout_route_saves_order(client):
    payload = {
        "full_name": "Test User",
        "email": "test@example.com",
        "card_number": "4111111111111111",
        "card_expiry": "12/30",
        "card_cvv": "123",
        "billing_address": "123 Test St",
        "city": "San Jose",
        "state": "CA",
        "zip_code": "95112",
    }

    r = client.post("/api/checkout", json=payload)

    assert r.status_code == 201
    body = r.get_json()
    assert body["success"] is True

    # verify it really reached the database
    config = get_db_config()
    conn = mysql.connector.connect(
        host=config["host"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
        port=config["port"],
    )
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT * FROM checkouts WHERE email = %s ORDER BY id DESC LIMIT 1",
        ("test@example.com",),
    )
    row = cursor.fetchone()

    cursor.close()
    conn.close()

    assert row is not None
    assert row["full_name"] == "Test User"
    assert row["email"] == "test@example.com"


def test_tracking_route_saves_data(client):
    payload = {"session_id": "1", "page": "home", "samples": [{"x": 1, "y": 2, "t": 1}]}

    r = client.post("/api/tracking/mouse", json=payload)

    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True

    # verify it really reached the database
    config = get_db_config()
    conn = mysql.connector.connect(
        host=config["host"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
        port=config["port"],
    )
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT * FROM user_sessions WHERE session_id = %s ORDER BY session_id DESC LIMIT 1",
        ("1",),
    )
    row = cursor.fetchone()

    cursor.close()
    conn.close()

    assert row is not None
    assert row["page"] == "home"
    assert any(
        m["x"] == 1 and m["y"] == 2 and m["t"] == 1
        for m in json.loads(row["mouse_movements"])
    )


def test_agent_route_returns_data(client):
    payload = {"session_id": "1", "page": "home", "samples": [{"x": 1, "y": 2, "t": 1}]}

    r = client.post("/api/tracking/mouse", json=payload)

    r = client.get("/api/agent/dashboard/1")

    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True

    # verify it really reached the database
    config = get_db_config()
    conn = mysql.connector.connect(
        host=config["host"],
        user=config["user"],
        password=config["password"],
        database=config["database"],
        port=config["port"],
    )
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT * FROM user_sessions WHERE session_id = %s ORDER BY session_id DESC LIMIT 1",
        ("1",),
    )
    row = cursor.fetchone()

    cursor.close()
    conn.close()

    assert row is not None
    assert row["page"] == "home"
    assert any(
        m["x"] == 1 and m["y"] == 2 and m["t"] == 1
        for m in json.loads(row["mouse_movements"])
    )
