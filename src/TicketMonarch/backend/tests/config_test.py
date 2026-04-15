from TicketMonarch.backend.config import get_db_config


def test_get_db_config_defaults(monkeypatch):
    monkeypatch.delenv("MYSQL_HOST", raising=False)
    monkeypatch.delenv("MYSQL_DATABASE", raising=False)
    monkeypatch.delenv("MYSQL_USER", raising=False)
    monkeypatch.delenv("MYSQL_PASSWORD", raising=False)
    monkeypatch.delenv("MYSQL_PORT", raising=False)

    config = get_db_config()

    assert config["host"] == "localhost"
    assert config["database"] == "ticketmonarch_db"
    assert config["user"] == "root"
    assert config["password"] == ""
    assert config["port"] == 3306


def test_get_db_config_env_override(monkeypatch):
    monkeypatch.setenv("MYSQL_HOST", "db.example.com")
    monkeypatch.setenv("MYSQL_DATABASE", "mydb")
    monkeypatch.setenv("MYSQL_USER", "alice")
    monkeypatch.setenv("MYSQL_PASSWORD", "secret")
    monkeypatch.setenv("MYSQL_PORT", "3307")

    config = get_db_config()

    assert config == {
        "host": "db.example.com",
        "database": "mydb",
        "user": "alice",
        "password": "secret",
        "port": 3307,
    }
