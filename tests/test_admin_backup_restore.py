"""Admin backup/restore: экспорт/импорт пользователей и платежей (CSV/JSON), сброс БД.

ВАЖНО: гонять с изоляцией от боевой БД:
    DATABASE_URL=sqlite:///./tests/_test_br.db TESTING=0 pytest tests/test_admin_backup_restore.py
иначе recreate_tables() затронет сконфигурированную БД.
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from gex.auth.config import settings
from gex.auth.models import User, SubscriptionStatus
from gex.adapters.persistence.database import SessionLocal, recreate_tables
from gex.auth.router import seed_master_admin
from gex.auth.payment_models import Payment, PaymentMethod, PaymentStatus

from main import app


@pytest.fixture(autouse=True)
def clean_db():
    recreate_tables()
    db = SessionLocal()
    db.query(User).delete()
    db.commit()
    db.close()
    yield


@pytest.fixture
def client():
    recreate_tables()
    db = SessionLocal()
    db.query(User).delete()
    db.commit()
    db.close()
    return TestClient(app)


def _admin_token(client) -> str:
    db = SessionLocal()
    seed_master_admin(db)
    db.close()
    r = client.post("/auth/login", json={"email": "sadisting", "password": settings.MASTER_PASSWORD})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(client) -> dict:
    return {"Authorization": "Bearer " + _admin_token(client)}


def _register(client, email="user@test.com", tg="@usernick") -> dict:
    r = client.post("/auth/register", json={
        "email": email, "password": "pass1234", "telegram_username": tg, "accept_terms": True,
    })
    assert r.status_code == 201, r.text
    return r.json()

def _user_id(client, email) -> str:
    db = SessionLocal()
    u = db.query(User).filter(User.email == email).first()
    db.close()
    assert u is not None
    return u.id


class TestUsersExport:
    def test_json_export_has_all_fields(self, client):
        _register(client, "user@test.com")
        uid = _user_id(client, "user@test.com")
        db = SessionLocal()
        u = db.query(User).filter(User.id == uid).first()
        u.subscription_status = SubscriptionStatus.EXTENDED
        u.is_email_verified = True
        db.commit()
        db.close()

        r = client.get("/auth/admin/users/export?format=json", headers=_auth(client))
        assert r.status_code == 200
        data = r.json()
        assert len(data) >= 1
        row = next(x for x in data if x["email"] == "user@test.com")
        assert row["subscription_status"] == "EXTENDED"
        assert row["is_email_verified"] is True
        # AUTH-04: секретов в экспорте нет. Раньше здесь проверялось обратное
        # (`password_hash` начинается на "$2"), но именно это и было дефектом: утечка
        # файла бэкапа = компрометация аккаунтов. Проверяем инвариант защиты.
        assert "password_hash" not in row
        assert "verification_token" not in row
        assert "telegram_connect_token" not in row
        assert row["telegram_username"] == "@usernick"
        for f in ["id", "is_blocked", "telegram_chat_id", "oauth_id", "created_at", "updated_at"]:
            assert f in row, f"поле {f} отсутствует в экспорте"

    def test_csv_export_columns(self, client):
        _register(client)
        r = client.get("/auth/admin/users/export?format=csv", headers=_auth(client))
        assert r.status_code == 200
        header = r.content.decode("utf-8-sig").splitlines()[0]
        cols = header.split(",")
        for f in ["id", "email", "subscription_status", "created_at", "updated_at"]:
            assert f in cols, f"колонка {f} отсутствует"
        # AUTH-04: секретных колонок быть не должно (обратная проверка к прежней).
        for secret in ["password_hash", "verification_token", "telegram_connect_token"]:
            assert secret not in cols, f"секрет {secret} попал в CSV-экспорт"

    def test_export_requires_admin(self, client):
        assert client.get("/auth/admin/users/export?format=json").status_code == 401


class TestUsersImport:
    def test_import_roundtrip_after_delete(self, client):
        """Экспорт → удаление → импорт: пользователи восстановлены, пароль работает."""
        _register(client, "user@test.com", tg="@usernick")
        _register(client, "second@test.com", tg="@second")
        db = SessionLocal()
        u = db.query(User).filter(User.email == "user@test.com").first()
        u.subscription_status = SubscriptionStatus.EXTENDED
        u.is_email_verified = True
        db.commit()
        db.close()

        # Экспорт JSON
        r = client.get("/auth/admin/users/export?format=json", headers=_auth(client))
        payload = r.content

        # Удаляем всех пользователей (кроме мастера, его не трогаем)
        db = SessionLocal()
        db.query(User).filter(User.email != "sadisting").delete()
        db.commit()
        db.close()

        # Импорт
        r = client.post(
            "/auth/admin/users/import?format=json",
            content=payload,
            headers={**_auth(client), "Content-Type": "application/json"},
        )
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["imported"] >= 2
        # Master Admin (sadisting без @) корректно пропускается
        assert res["skipped"] <= 1

        # Восстановлены поля
        db = SessionLocal()
        u = db.query(User).filter(User.email == "user@test.com").first()
        db.close()
        assert u is not None
        assert u.subscription_status == "EXTENDED"
        assert u.is_email_verified is True
        assert u.telegram_username == "@usernick"

        # Пароль НЕ восстановлен — и это ожидаемое следствие AUTH-04.
        # Экспорт намеренно не содержит ``password_hash`` (иначе утечка файла бэкапа =
        # компрометация аккаунтов), поэтому roundtrip восстанавливает профиль, но не
        # секрет: пользователь задаёт пароль заново. Раньше здесь проверялся успешный
        # вход, то есть тест закреплял ровно тот дефект, который потом закрыли —
        # строка ниже фиксирует новое, безопасное поведение.
        r = client.post("/auth/login", json={"email": "user@test.com", "password": "pass1234"})
        assert r.status_code == 401, (
            "пароль не должен уцелеть после импорта из экспорта без password_hash "
            f"(получено {r.status_code})"
        )
        # Профиль при этом жив: пользователь существует и узнаётся по email.
        assert u.email == "user@test.com"

    def test_import_restores_password_hash_from_a_legacy_backup(self, client):
        """Старый бэкап (с ``password_hash``) по-прежнему импортируется — обратная совместимость.

        AUTH-04 запретил **экспортировать** секреты, но чтение их из прежнего файла
        сохранено: иначе существующие бэкапы администраторов перестали бы работать.
        """
        bcrypt_hash = "$2b$12$" + "x" * 53  # форма bcrypt, значение не важно для импорта
        payload = json.dumps([{
            "id": "legacy-1", "email": "legacy@test.com", "password_hash": bcrypt_hash,
            "is_email_verified": True, "is_blocked": False, "subscription_status": "EXTENDED",
            "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        }]).encode()
        _admin_token(client)  # пользователь-админ существует
        r = client.post(
            "/auth/admin/users/import?format=json",
            content=payload,
            headers={**_auth(client), "Content-Type": "application/json"},
        )
        assert r.status_code == 200, r.text
        db = SessionLocal()
        u = db.query(User).filter(User.email == "legacy@test.com").first()
        db.close()
        assert u is not None, "пользователь из старого бэкапа не импортирован"
        assert u.password_hash == bcrypt_hash, "хэш из старого бэкапа не восстановлен"

    def test_import_csv(self, client):
        csv_data = (
            "id,email,password_hash,is_email_verified,is_blocked,telegram_chat_id,telegram_username,"
            "telegram_connect_token,telegram_notify,telegram_test_at,subscription_status,"
            "subscription_activated_at,subscription_expires_at,subscription_expiry_notified_for,"
            "verification_token,oauth_provider,oauth_id,created_at,updated_at\n"
            ",csv@test.com,,true,false,,@csvuser,,false,,EXTENDED,,2027-01-01T00:00:00Z,,,,,2026-01-01T00:00:00Z,2026-01-01T00:00:00Z\n"
        )
        r = client.post(
            "/auth/admin/users/import?format=csv",
            content=csv_data.encode("utf-8"),
            headers={**_auth(client), "Content-Type": "text/csv"},
        )
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["imported"] == 1
        db = SessionLocal()
        u = db.query(User).filter(User.email == "csv@test.com").first()
        db.close()
        assert u is not None
        assert u.subscription_status == "EXTENDED"

    def test_import_skips_bad_rows(self, client):
        csv_data = (
            "id,email,password_hash,is_email_verified,subscription_status,created_at,updated_at\n"
            ",good@test.com,,true,BASIC,,2026-01-01T00:00:00Z\n"
            ",, ,true,BASIC,,\n"          # нет email
            ",bad@test.com,,true,NOPE,,2026-01-01T00:00:00Z\n"   # плохой статус
        )
        r = client.post(
            "/auth/admin/users/import?format=csv",
            content=csv_data.encode("utf-8"),
            headers={**_auth(client), "Content-Type": "text/csv"},
        )
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["imported"] == 1
        assert res["skipped"] == 2
        assert len(res["errors"]) == 2

    def test_import_upsert_by_email(self, client):
        _register(client, "user@test.com")
        # Повторный импорт с тем же email — обновление, а не дубль
        row = {
            "email": "user@test.com", "subscription_status": "EXTENDED",
            "is_email_verified": True,
        }
        r = client.post(
            "/auth/admin/users/import?format=json",
            content=json.dumps([row]),
            headers={**_auth(client), "Content-Type": "application/json"},
        )
        assert r.status_code == 200
        res = r.json()
        assert res["imported"] == 0
        assert res["updated"] == 1
        db = SessionLocal()
        cnt = db.query(User).filter(User.email == "user@test.com").count()
        u = db.query(User).filter(User.email == "user@test.com").first()
        db.close()
        assert cnt == 1
        assert u.subscription_status == "EXTENDED"

    def test_import_requires_admin(self, client):
        r = client.post("/auth/admin/users/import?format=json", content=b"[]")
        assert r.status_code == 401


class TestPaymentsExportImport:
    def _make_payment(self, user_id: str) -> Payment:
        db = SessionLocal()
        p = Payment(
            id="pay-0001",
            user_id=user_id,
            user_email="user@test.com",
            plan="EXTENDED",
            amount_rub=5900.0,
            amount_usd=60.0,
            method=PaymentMethod.SBP,
            status=PaymentStatus.CONFIRMED,
            admin_note="test",
        )
        db.add(p)
        db.commit()
        db.refresh(p)
        db.close()
        return p

    def test_payments_export_import_roundtrip(self, client):
        _register(client, "user@test.com")
        uid = _user_id(client, "user@test.com")
        self._make_payment(uid)

        r = client.get("/auth/admin/payments/export?format=json", headers=_auth(client))
        assert r.status_code == 200
        payload = r.content
        data = json.loads(payload)
        assert len(data) == 1
        assert data[0]["user_id"] == uid
        assert data[0]["status"] == "CONFIRMED"

        # Удаляем платёж
        db = SessionLocal()
        db.query(Payment).delete()
        db.commit()
        db.close()

        # Импорт
        r = client.post(
            "/auth/admin/payments/import?format=json",
            content=payload,
            headers={**_auth(client), "Content-Type": "application/json"},
        )
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["imported"] == 1

        db = SessionLocal()
        p = db.query(Payment).first()
        db.close()
        assert p is not None
        assert p.user_id == uid
        assert p.plan == "EXTENDED"
        assert p.amount_usd == 60.0
        assert p.status == "CONFIRMED"
        assert p.admin_note == "test"

    def test_payments_import_skips_unknown_user(self, client):
        row = [{
            "id": "pay-x", "user_id": "no-such-user", "user_email": "ghost@test.com",
            "plan": "BASIC", "amount_rub": 100.0, "amount_usd": 1.0,
            "method": "SBP", "status": "PENDING",
        }]
        r = client.post(
            "/auth/admin/payments/import?format=json",
            content=json.dumps(row),
            headers={**_auth(client), "Content-Type": "application/json"},
        )
        assert r.status_code == 200
        res = r.json()
        assert res["imported"] == 0
        assert res["skipped"] == 1
        assert "не найден" in res["errors"][0]

    def test_payments_csv_export(self, client):
        _register(client, "user@test.com")
        uid = _user_id(client, "user@test.com")
        self._make_payment(uid)
        r = client.get("/auth/admin/payments/export?format=csv", headers=_auth(client))
        assert r.status_code == 200
        header = r.content.decode("utf-8-sig").splitlines()[0]
        for f in ["id", "user_id", "plan", "amount_rub", "status", "expires_at"]:
            assert f in header.split(","), f"колонка {f} отсутствует"


class TestDbReset:
    def test_reset_requires_confirm(self, client):
        _register(client, "user@test.com")
        # без параметра confirm — FastAPI-валидация: 422
        r = client.post("/auth/admin/db/reset", headers=_auth(client))
        assert r.status_code == 422
        # с неверным значением — наш guard: 400
        r = client.post("/auth/admin/db/reset?confirm=yes", headers=_auth(client))
        assert r.status_code == 400
        # данные на месте
        db = SessionLocal()
        cnt = db.query(User).count()
        db.close()
        assert cnt >= 1

    def test_reset_wipes_and_reseeds_master(self, client):
        _register(client, "user@test.com")
        _register(client, "second@test.com", tg="@second")
        r = client.post("/auth/admin/db/reset?confirm=DROP", headers=_auth(client))
        assert r.status_code == 200, r.text
        db = SessionLocal()
        users = db.query(User).all()
        db.close()
        assert len(users) == 1
        assert users[0].email == "sadisting"
        assert users[0].subscription_status == "ADMIN"

        # Master может снова войти
        r = client.post("/auth/login", json={"email": "sadisting", "password": settings.MASTER_PASSWORD})
        assert r.status_code == 200, r.text

    def test_reset_requires_admin(self, client):
        assert client.post("/auth/admin/db/reset?confirm=DROP").status_code == 401
