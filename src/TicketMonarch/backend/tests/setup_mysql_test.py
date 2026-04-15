import pytest

from TicketMonarch.backend import setup_mysql


class DummyCursor:
    def __init__(self):
        self.executed = []

    def execute(self, query):
        self.executed.append(query)

    def close(self):
        pass


class DummyConnection:
    def __init__(self):
        self.cursor_obj = DummyCursor()
        self.committed = False
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


VALID_DB_CONFIG = {
    "host": "localhost",
    "database": "ticketmonarch_db",
    "user": "root",
    "password": "",
    "port": 3306,
}


class FailingSecondExecuteCursor(DummyCursor):
    def execute(self, query):
        self.executed.append(query)
        if len(self.executed) == 2:
            raise Exception("index already exists")


def test_create_database_if_not_exists(monkeypatch):
    conn = DummyConnection()

    monkeypatch.setattr(setup_mysql, "get_db_config", lambda: VALID_DB_CONFIG)
    monkeypatch.setattr(setup_mysql.mysql.connector, "connect", lambda **kwargs: conn)

    setup_mysql.create_database_if_not_exists()

    assert conn.committed is True
    assert any("CREATE DATABASE IF NOT EXISTS" in q for q in conn.cursor_obj.executed)


def test_create_database_if_not_exists_invalid_name(monkeypatch):
    monkeypatch.setattr(
        setup_mysql,
        "get_db_config",
        lambda: {
            "host": "localhost",
            "database": "bad-name;",
            "user": "root",
            "password": "",
            "port": 3306,
        },
    )

    with pytest.raises(ValueError, match="Invalid database name"):
        setup_mysql.create_database_if_not_exists()


def test_create_orders_table(monkeypatch):
    conn = DummyConnection()

    monkeypatch.setattr(setup_mysql, "get_db_config", lambda: VALID_DB_CONFIG)
    monkeypatch.setattr(setup_mysql.mysql.connector, "connect", lambda **kwargs: conn)

    setup_mysql.create_orders_table()

    assert conn.committed is True
    assert any(
        "CREATE TABLE IF NOT EXISTS orders" in q for q in conn.cursor_obj.executed
    )


def test_create_user_sessions_table(monkeypatch):
    conn = DummyConnection()

    monkeypatch.setattr(setup_mysql, "get_db_config", lambda: VALID_DB_CONFIG)
    monkeypatch.setattr(setup_mysql.mysql.connector, "connect", lambda **kwargs: conn)

    setup_mysql.create_user_sessions_table()

    assert conn.committed is True
    assert any(
        "CREATE TABLE IF NOT EXISTS user_sessions" in q
        for q in conn.cursor_obj.executed
    )


def test_create_user_sessions_table_index_already_exists(monkeypatch):
    conn = DummyConnection()
    conn.cursor_obj = FailingSecondExecuteCursor()

    monkeypatch.setattr(setup_mysql, "get_db_config", lambda: VALID_DB_CONFIG)
    monkeypatch.setattr(setup_mysql.mysql.connector, "connect", lambda **kwargs: conn)

    setup_mysql.create_user_sessions_table()

    assert conn.committed is True
    assert any(
        "CREATE TABLE IF NOT EXISTS user_sessions" in q
        for q in conn.cursor_obj.executed
    )
    assert any("CREATE INDEX idx_session_start" in q for q in conn.cursor_obj.executed)


def test_main_calls_all_setup_steps(monkeypatch):
    called = []

    monkeypatch.setattr(
        setup_mysql, "create_database_if_not_exists", lambda: called.append("db")
    )
    monkeypatch.setattr(
        setup_mysql, "create_orders_table", lambda: called.append("orders")
    )
    monkeypatch.setattr(
        setup_mysql, "create_user_sessions_table", lambda: called.append("sessions")
    )

    setup_mysql.main()

    assert called == ["db", "orders", "sessions"]
