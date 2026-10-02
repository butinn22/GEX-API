"""Admin moderation: resend email/telegram, активация подписки, блокировка, SMTP-конфиг."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from gex.auth.config import settings
from gex.auth.models import User, SubscriptionStatus
from gex.adapters.persistence.database import SessionLocal, recreate_tables
from gex.auth.router import seed_master_admin

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


class TestBlockUser:
    def test_block_user(self, client):
        _register(client)
        uid = _user_id(client, "user@test.com")
        token = _admin_token(client)
        r = client.post(f"/auth/admin/users/{uid}/block", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["is_blocked"] is True

    def test_blocked_user_cannot_login(self, client):
        _register(client)
        uid = _user_id(client, "user@test.com")
        token = _admin_token(client)
        client.post(f"/auth/admin/users/{uid}/block", headers={"Authorization": f"Bearer {token}"})
        # Верифицируем email, чтобы дойти до проверки блокировки
        db = SessionLocal()
        u = db.query(User).filter(User.email == "user@test.com").first()
        u.is_email_verified = True
        db.commit()
        db.close()
        r = client.post("/auth/login", json={"email": "user@test.com", "password": "pass1234"})
        assert r.status_code == 403
        assert "заблокирован" in r.json()["detail"].lower()

    def test_blocked_user_api_403(self, client):
        reg = _register(client)
        uid = _user_id(client, "user@test.com")
        admin_tok = _admin_token(client)
        client.post(f"/auth/admin/users/{uid}/block", headers={"Authorization": f"Bearer {admin_tok}"})
        # access-токен пользователя (выдан при регистрации)
        r = client.get("/auth/me", headers={"Authorization": f"Bearer {reg['access_token']}"})
        assert r.status_code == 403

    def test_unblock_restores_access(self, client):
        _register(client)
        uid = _user_id(client, "user@test.com")
        token = _admin_token(client)
        client.post(f"/auth/admin/users/{uid}/block", headers={"Authorization": f"Bearer {token}"})
        r = client.post(f"/auth/admin/users/{uid}/unblock", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["is_blocked"] is False

    def test_cannot_block_master_admin(self, client):
        token = _admin_token(client)
        db = SessionLocal()
        master = db.query(User).filter(User.email == "sadisting").first()
        db.close()
        r = client.post(f"/auth/admin/users/{master.id}/block", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 403


class TestActivateSubscription:
    def test_activate_basic(self, client):
        _register(client)
        uid = _user_id(client, "user@test.com")
        token = _admin_token(client)
        r = client.post(f"/auth/admin/users/{uid}/activate", json={"plan": "BASIC", "days": 30},
                        headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["subscription_status"] == "BASIC"
        assert r.json()["subscription_expires_at"] is not None

    def test_activate_extended(self, client):
        _register(client)
        uid = _user_id(client, "user@test.com")
        token = _admin_token(client)
        r = client.post(f"/auth/admin/users/{uid}/activate", json={"plan": "EXTENDED", "days": 30},
                        headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["subscription_status"] == "EXTENDED"

    def test_activate_invalid_plan(self, client):
        _register(client)
        uid = _user_id(client, "user@test.com")
        token = _admin_token(client)
        r = client.post(f"/auth/admin/users/{uid}/activate", json={"plan": "ADMIN", "days": 30},
                        headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 422


class TestResend:
    def test_resend_email(self, client):
        _register(client)
        uid = _user_id(client, "user@test.com")
        token = _admin_token(client)
        r = client.post(f"/auth/admin/users/{uid}/resend-email", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert "Письмо отправлено" in r.json()["message"]

    def test_resend_email_already_verified(self, client):
        _register(client)
        uid = _user_id(client, "user@test.com")
        token = _admin_token(client)
        db = SessionLocal()
        u = db.query(User).filter(User.email == "user@test.com").first()
        u.is_email_verified = True
        db.commit()
        db.close()
        r = client.post(f"/auth/admin/users/{uid}/resend-email", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 400

    def test_resend_telegram(self, client):
        _register(client)
        uid = _user_id(client, "user@test.com")
        token = _admin_token(client)
        r = client.post(f"/auth/admin/users/{uid}/resend-telegram", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert "Telegram" in r.json()["message"]


class TestRuntimeConfig:
    @pytest.fixture(autouse=True)
    def _tmp_config(self, tmp_path, monkeypatch):
        from gex.auth import runtime_config
        monkeypatch.setattr(runtime_config, "_CONFIG_PATH", str(tmp_path / "rt.json"))
        yield

    def test_email_config_roundtrip(self, client):
        token = _admin_token(client)
        r = client.put("/auth/admin/email-config", json={
            "smtp_host": "smtp.mail.ru", "smtp_port": 465,
            "smtp_user": "corp@mail.ru", "smtp_pass": "secret123", "from_email": "corp@mail.ru",
        }, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["smtp_host"] == "smtp.mail.ru"
        # пароль замаскирован
        assert "secret123" not in r.json()["smtp_pass"]

        g = client.get("/auth/admin/email-config", headers={"Authorization": f"Bearer {token}"})
        assert g.status_code == 200
        assert g.json()["smtp_user"] == "corp@mail.ru"

    def test_email_config_requires_admin(self, client):
        r = client.get("/auth/admin/email-config")
        assert r.status_code == 401

    def test_telegram_config_roundtrip(self, client):
        token = _admin_token(client)
        r = client.put("/auth/admin/telegram-config", json={
            "bot_token": "123456:ABCDEF_verysecret", "bot_username": "mybot", "chat_id": "999",
        }, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["bot_username"] == "mybot"
        # токен замаскирован: полное значение не возвращается
        assert "verysecret" not in r.json()["bot_token"]
        assert r.json()["bot_token"].startswith("••••")


class TestAuthRedirectGuard:
    """Заблокированный пользователь не проходит даже через refresh/me."""

    def test_is_blocked_field_in_me(self, client):
        reg = _register(client)
        # verify + check is_blocked в ответе /me
        r = client.get("/auth/me", headers={"Authorization": f"Bearer {reg['access_token']}"})
        assert r.status_code == 200
        assert r.json()["is_blocked"] is False
