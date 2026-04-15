import json
import os

from TicketMonarch.backend import database


class DummyCursor:
    def __init__(self, fetchone_result=None, fetchall_result=None, description=None):
        self.executed = []
        self.fetchone_result = fetchone_result
        self.fetchall_result = fetchall_result or []
        self.description = description or []
        self.lastrowid = 123

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        return self.fetchone_result

    def fetchall(self):
        return self.fetchall_result

    def close(self):
        pass


class DummyConnection:
    def __init__(self, cursor_obj):
        self.cursor_obj = cursor_obj
        self.committed = False

    def cursor(self, dictionary=False):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def close(self):
        pass


def test_get_connection(monkeypatch):
    called = {}

    def fake_connect(**kwargs):
        called.update(kwargs)
        return "CONN"

    monkeypatch.setattr(
        database,
        "get_db_config",
        lambda: {
            "host": "localhost",
            "user": "root",
            "password": "",
            "database": "ticketmonarch_db",
            "port": 3306,
        },
    )
    monkeypatch.setattr(database.mysql.connector, "connect", fake_connect)

    conn = database.get_connection()

    assert conn == "CONN"
    assert called["database"] == "ticketmonarch_db"


def test_ensure_indexes_creates_index_when_missing(monkeypatch):
    cursor = DummyCursor(fetchone_result=(0,))
    conn = DummyConnection(cursor)

    monkeypatch.setattr(database, "get_connection", lambda: conn)

    database.ensure_indexes()

    assert conn.committed is True
    assert len(cursor.executed) == 2
    assert "SELECT COUNT(*)" in cursor.executed[0][0]
    assert "CREATE INDEX idx_session_start" in cursor.executed[1][0]


def test_ensure_indexes_skips_create_when_index_exists(monkeypatch):
    cursor = DummyCursor(fetchone_result=(1,))
    conn = DummyConnection(cursor)

    monkeypatch.setattr(database, "get_connection", lambda: conn)

    database.ensure_indexes()

    assert len(cursor.executed) == 1
    assert conn.committed is False


# def test_init_database()


def test_save_order(monkeypatch):
    cursor = DummyCursor()
    conn = DummyConnection(cursor)

    monkeypatch.setattr(database, "get_connection", lambda: conn)

    order_id = database.save_order(
        {
            "full_name": "A User",
            "email": "a@example.com",
            "card_number": "4111111111111111",
        }
    )

    assert order_id == 123
    assert conn.committed is True
    assert len(cursor.executed) == 1


def test_export_to_csv_masks_sensitive_values(monkeypatch, tmp_path):
    cursor = DummyCursor(
        fetchall_result=[
            (
                1,
                "Alice",
                "a@example.com",
                "4111111111111111",
                "12/30",
                "123",
                "addr",
                "SJ",
                "CA",
                "95112",
                "now",
            )
        ],
        description=[
            ("id",),
            ("full_name",),
            ("email",),
            ("card_number",),
            ("card_expiry",),
            ("card_cvv",),
            ("billing_address",),
            ("city",),
            ("state",),
            ("zip_code",),
            ("timestamp",),
        ],
    )
    conn = DummyConnection(cursor)

    monkeypatch.setattr(database, "get_connection", lambda: conn)
    monkeypatch.setattr(database, "DATA_DIR", str(tmp_path))

    csv_path = database.export_to_csv()

    assert os.path.exists(csv_path)
    content = open(csv_path, encoding="utf-8").read()
    assert "************1111" in content
    assert "****" in content


def test_export_tracking_data_to_csv(tmp_path, monkeypatch):
    columns = [
        ("session_id",),
        ("session_start",),
        ("page",),
        ("mouse_movements",),
        ("click_events",),
        ("keystroke_data",),
        ("scroll_events",),
        ("form_completion_time",),
        ("browser_info",),
        ("session_metadata",),
    ]

    rows = [
        (
            "sess-1",
            "2026-01-01 12:00:00",
            "home",
            [{"x": 1}],
            [],
            None,
            [],
            None,
            None,
            {"a": 1},
        )
    ]

    cursor = DummyCursor(fetchall_result=rows, description=columns)
    conn = DummyConnection(cursor)

    monkeypatch.setattr(database, "get_connection", lambda: conn)
    monkeypatch.setattr(database, "DATA_DIR", str(tmp_path))

    csv_path = database.export_tracking_data_to_csv()

    assert os.path.exists(csv_path)

    content = open(csv_path, encoding="utf-8").read()
    assert "sess-1" in content
    assert "mouse_movements" in content
    assert '"[{""x"": 1}]"' in content
    assert '"{""a"": 1}"' in content


def test_save_user_session(monkeypatch):
    cursor = DummyCursor()
    conn = DummyConnection(cursor)

    monkeypatch.setattr(database, "get_connection", lambda: conn)

    database.save_user_session(
        "sess-1",
        {
            "page": "home",
            "mouse_movements": [{"x": 1, "y": 2, "t": 1}],
        },
    )

    assert conn.committed is True
    assert len(cursor.executed) == 1
    _, params = cursor.executed[0]
    assert params[0] == "sess-1"
    assert params[1] == "home"
    assert json.loads(params[2]) == [{"x": 1, "y": 2, "t": 1}]


def test_get_user_session_parses_json(monkeypatch):
    row = {
        "session_id": "sess-1",
        "session_start": "2026-01-01",
        "page": "home",
        "mouse_movements": '[{"x":1}]',
        "click_events": "[]",
        "keystroke_data": None,
        "scroll_events": "[]",
        "form_completion_time": None,
        "browser_info": None,
        "session_metadata": '{"a":1}',
    }
    cursor = DummyCursor(fetchone_result=row)
    conn = DummyConnection(cursor)

    monkeypatch.setattr(database, "get_connection", lambda: conn)

    result = database.get_user_session("sess-1")

    assert result == row


def test_get_user_sessions(monkeypatch):
    row = {
        "session_id": "sess-1",
        "session_start": "2026-01-01",
        "page": "home",
        "mouse_movements": '[{"x":1}]',
        "click_events": "[]",
        "keystroke_data": None,
        "scroll_events": "[]",
        "form_completion_time": None,
        "browser_info": None,
        "session_metadata": '{"a":1}',
    }
    row2 = {
        "session_id": "sess-2",
        "session_start": "2026-01-01",
        "page": "home",
        "mouse_movements": '[{"x":5}]',
        "click_events": "[]",
        "keystroke_data": None,
        "scroll_events": "[]",
        "form_completion_time": None,
        "browser_info": None,
        "session_metadata": '{"a":1}',
    }
    cursor = DummyCursor(fetchall_result=[row, row2])
    conn = DummyConnection(cursor)

    monkeypatch.setattr(database, "get_connection", lambda: conn)

    result = database.get_user_sessions()

    assert len(result) == 2
    assert len(result[0]) == 10
    assert result[0]["mouse_movements"] == [{"x": 1}]
    assert result[1]["mouse_movements"] == [{"x": 5}]


def test_get_recent_session_ids_returns_list(monkeypatch):
    cursor = DummyCursor(fetchall_result=[("sess-1",), ("sess-2",)])
    conn = DummyConnection(cursor)

    monkeypatch.setattr(database, "get_connection", lambda: conn)

    result = database.get_recent_session_ids(limit=2)

    assert result == ["sess-1", "sess-2"]


def test_get_session_summaries(monkeypatch):
    row = {
        "session_id": "sess-1",
        "session_start": "2026-01-01",
        "page": "home",
        "mouse_movements": '[{"x":1}]',
        "click_events": "[]",
        "keystroke_data": None,
        "scroll_events": "[]",
        "form_completion_time": None,
        "browser_info": None,
        "session_metadata": '{"a":1}',
    }
    row2 = {
        "session_id": "sess-2",
        "session_start": "2026-01-01",
        "page": "home",
        "mouse_movements": '[{"x":5}]',
        "click_events": "[]",
        "keystroke_data": None,
        "scroll_events": "[]",
        "form_completion_time": None,
        "browser_info": None,
        "session_metadata": '{"a":1}',
    }
    cursor = DummyCursor(fetchall_result=[row, row2])
    conn = DummyConnection(cursor)

    monkeypatch.setattr(database, "get_connection", lambda: conn)

    result = database.get_session_summaries()

    assert len(result) == 2
    assert result[0]["session_id"] == "sess-1"
    assert result[1]["session_id"] == "sess-2"
