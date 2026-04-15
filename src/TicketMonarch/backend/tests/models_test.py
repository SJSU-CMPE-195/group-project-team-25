import os
from datetime import datetime

import pandas as pd
import pytest

from TicketMonarch.backend import models
from TicketMonarch.backend.models import Order, Checkout


def test_order_to_dict():
    order = Order(
        id=1,
        customer_name="Alice",
        email="a@example.com",
        product_name="Ticket",
        quantity=2,
        price=10.0,
        total=20.0,
        order_date=datetime(2026, 1, 1, 12, 0, 0),
    )

    data = order.to_dict()

    assert data["id"] == 1
    assert data["customer_name"] == "Alice"
    assert data["total"] == 20.0
    assert data["order_date"].startswith("2026-01-01T12:00:00")


def test_checkout_to_dict():
    checkout = Checkout(
        id=2,
        full_name="Bob",
        email="b@example.com",
        card_number="4111111111111111",
        card_expiry="12/30",
        card_cvv="123",
        billing_address="123 Main",
        city="San Jose",
        state="CA",
        zip_code="95112",
        timestamp=datetime(2026, 1, 2, 12, 0, 0),
    )

    data = checkout.to_dict()

    assert data["id"] == 2
    assert data["full_name"] == "Bob"
    assert data["zip_code"] == "95112"
    assert data["timestamp"].startswith("2026-01-02T12:00:00")


CHECKOUT_DATA = {
    "full_name": "Alice",
    "email": "a@example.com",
    "card_number": "4111",
    "card_expiry": "12/30",
    "card_cvv": "123",
    "billing_address": "123 Main",
    "city": "San Jose",
    "state": "CA",
    "zip_code": "95112",
}


class DummyDB:
    def __init__(self):
        self.added = []
        self.committed = False
        self.rolled_back = False
        self.closed = False
        self.refreshed = []

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def refresh(self, obj):
        self.refreshed.append(obj)

    def close(self):
        self.closed = True

    def query(self, model):
        return DummyQuery([])


class DummyQuery:
    def __init__(self, results, first_result=None):
        self.results = results
        self.first_result = first_result

    def all(self):
        return self.results

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.first_result


def test_get_db_closes_session(monkeypatch):
    db = DummyDB()
    monkeypatch.setattr(models, "SessionLocal", lambda: db)

    gen = models.get_db()
    returned = next(gen)

    assert returned is db

    with pytest.raises(StopIteration):
        next(gen)

    assert db.closed is True


def test_save_checkout_to_db_success(monkeypatch):
    db = DummyDB()
    monkeypatch.setattr(models, "SessionLocal", lambda: db)

    result = models.save_checkout_to_db(CHECKOUT_DATA)

    assert isinstance(result, Checkout)
    assert result.full_name == "Alice"
    assert db.committed is True
    assert len(db.added) == 1
    assert len(db.refreshed) == 1
    assert db.closed is True


def test_save_checkout_to_db_rollback_on_error(monkeypatch):
    class FailingDB(DummyDB):
        def commit(self):
            raise Exception("DB failed")

    db = FailingDB()
    monkeypatch.setattr(models, "SessionLocal", lambda: db)

    with pytest.raises(Exception, match="DB failed"):
        models.save_checkout_to_db(CHECKOUT_DATA)

    assert db.rolled_back is True
    assert db.closed is True


def test_export_checkouts_to_csv_empty(tmp_path, monkeypatch):
    db = DummyDB()
    db.query = lambda model: DummyQuery([])
    monkeypatch.setattr(models, "SessionLocal", lambda: db)

    csv_path = tmp_path / "checkouts.csv"
    result = models.export_checkouts_to_csv(str(csv_path))

    assert result == str(csv_path)
    assert os.path.exists(result)
    df = pd.read_csv(result)
    assert list(df.columns) == [
        "id",
        "full_name",
        "email",
        "card_number",
        "card_expiry",
        "card_cvv",
        "billing_address",
        "city",
        "state",
        "zip_code",
        "timestamp",
    ]
    assert db.closed is True


def test_export_checkouts_to_csv_with_data(tmp_path, monkeypatch):
    db = DummyDB()
    db.query = lambda model: DummyQuery(
        [
            Checkout(
                id=1,
                full_name="Alice",
                email="a@example.com",
                card_number="4111",
                card_expiry="12/30",
                card_cvv="123",
                billing_address="123 Main",
                city="San Jose",
                state="CA",
                zip_code="95112",
                timestamp=datetime(2026, 1, 1, 12, 0, 0),
            )
        ]
    )
    monkeypatch.setattr(models, "SessionLocal", lambda: db)

    csv_path = tmp_path / "checkouts.csv"
    result = models.export_checkouts_to_csv(str(csv_path))

    df = pd.read_csv(result)
    assert len(df) == 1
    assert df.loc[0, "full_name"] == "Alice"
    assert db.closed is True


def test_import_checkouts_from_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        models.import_checkouts_from_csv("does_not_exist.csv")


def test_import_checkouts_from_csv_missing_columns(tmp_path, monkeypatch):
    db = DummyDB()
    monkeypatch.setattr(models, "SessionLocal", lambda: db)

    csv_path = tmp_path / "bad.csv"
    pd.DataFrame([{"full_name": "Alice"}]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        models.import_checkouts_from_csv(str(csv_path))

    assert db.rolled_back is True
    assert db.closed is True


def test_import_checkouts_from_csv_success(tmp_path, monkeypatch):
    db = DummyDB()
    db.query = lambda model: DummyQuery([], first_result=None)
    monkeypatch.setattr(models, "SessionLocal", lambda: db)

    csv_path = tmp_path / "good.csv"
    pd.DataFrame([CHECKOUT_DATA]).to_csv(csv_path, index=False)

    imported_count, skipped_count, errors = models.import_checkouts_from_csv(
        str(csv_path)
    )

    assert imported_count == 1
    assert skipped_count == 0
    assert errors == []
    assert len(db.added) == 1
    assert db.committed is True
    assert db.closed is True


def test_import_checkouts_from_csv_skips_duplicates(tmp_path, monkeypatch):
    db = DummyDB()
    existing = object()
    db.query = lambda model: DummyQuery([], first_result=existing)
    monkeypatch.setattr(models, "SessionLocal", lambda: db)

    csv_path = tmp_path / "dupe.csv"
    pd.DataFrame([CHECKOUT_DATA]).to_csv(csv_path, index=False)

    imported_count, skipped_count, errors = models.import_checkouts_from_csv(
        str(csv_path)
    )

    assert imported_count == 0
    assert skipped_count == 1
    assert errors == []
    assert len(db.added) == 0
    assert db.committed is True
    assert db.closed is True
