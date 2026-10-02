"""Tests for Telegram long-polling service (fallback when no webhook/domain)."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from gex.auth.models import TelegramStart
from gex.adapters.persistence.database import SessionLocal, recreate_tables
from gex.adapters.notifications.telegram_polling import (
    TelegramPollingService,
    get_bot_token,
    set_telegram_webhook,
)


@pytest.fixture(autouse=True)
def clean_db():
    recreate_tables()
    db = SessionLocal()
    db.query(TelegramStart).delete()
    db.commit()
    db.close()
    yield


def _updates_response(updates):
    return {"ok": True, "result": updates}


def _update(update_id, text, chat_id="123456", username="polluser"):
    return {
        "update_id": update_id,
        "message": {
            "text": text,
            "chat": {"id": chat_id, "username": username, "type": "private"},
        },
    }


class TestPollingService:
    def test_skips_without_token(self, monkeypatch):
        monkeypatch.setattr("gex.adapters.notifications.telegram_polling.get_bot_token", lambda: "")
        svc = TelegramPollingService()
        svc.start()
        assert svc.is_running is False

    def test_skips_when_webhook_active(self, monkeypatch):
        monkeypatch.setattr("gex.adapters.notifications.telegram_polling.get_bot_token", lambda: "tok")
        monkeypatch.setattr(
            "gex.adapters.notifications.telegram_polling.get_webhook_info",
            lambda token: {"url": "https://example.com/telegram/webhook"},
        )
        svc = TelegramPollingService()
        svc.start()
        assert svc.is_running is False

    def test_starts_when_no_webhook(self, monkeypatch):
        monkeypatch.setattr("gex.adapters.notifications.telegram_polling.get_bot_token", lambda: "tok")
        monkeypatch.setattr(
            "gex.adapters.notifications.telegram_polling.get_webhook_info",
            lambda token: {"url": ""},
        )
        svc = TelegramPollingService(process_update=lambda upd: None)
        svc.start()
        try:
            assert svc.is_running is True
        finally:
            svc.stop()
        assert svc.is_running is False

    def test_loop_processes_updates_and_stops_on_409(self, monkeypatch):
        """Сервис обрабатывает обновления, двигает offset и выключается при 409."""
        seen = []
        calls = {"n": 0}

        def fake_api(method, token, payload=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return _updates_response([_update(1, "/start"), _update(2, "/test")])
            if calls["n"] == 2:
                return _updates_response([])
            # 3-й вызов — конфликт с webhook → сервис должен остановиться
            return {"ok": False, "error_code": 409, "description": "Conflict: webhook"}

        monkeypatch.setattr("gex.adapters.notifications.telegram_polling.get_bot_token", lambda: "tok")
        monkeypatch.setattr("gex.adapters.notifications.telegram_polling.get_webhook_info", lambda token: {"url": ""})
        monkeypatch.setattr("gex.adapters.notifications.telegram_polling._api_call", fake_api)

        svc = TelegramPollingService(process_update=lambda upd: seen.append(upd))
        svc.start()
        # Даём потоку время обработать
        for _ in range(50):
            if not svc.is_running and calls["n"] >= 3:
                break
            import time
            time.sleep(0.05)
        svc.stop()

        assert len(seen) == 2, f"обработано обновлений: {len(seen)}"
        assert seen[0]["update_id"] == 1
        assert svc._offset == 3  # offset после update_id=2
        assert calls["n"] >= 3

    def test_default_process_records_start(self, monkeypatch):
        """_default_process → handle_telegram_message: запись telegram_starts."""
        replies = []
        monkeypatch.setattr(
            "gex.auth.telegram_router._send_telegram_reply",
            lambda chat_id, text: replies.append((chat_id, text)),
        )
        TelegramPollingService._default_process(_update(10, "/start", chat_id="777", username="polluser"))

        db = SessionLocal()
        row = db.query(TelegramStart).filter(TelegramStart.username == "polluser").first()
        db.close()
        assert row is not None
        assert row.chat_id == "777"
        assert len(replies) == 1
        assert "Добро пожаловать" in replies[0][1]

    def test_default_process_activation_reply(self, monkeypatch):
        """/start activate_<token> через polling — активация аккаунта."""
        from gex.auth.models import User

        replies = []
        monkeypatch.setattr(
            "gex.auth.telegram_router._send_telegram_reply",
            lambda chat_id, text: replies.append((chat_id, text)),
        )
        db = SessionLocal()
        db.add(User(
            id="usr-poll-1",
            email="pollact@test.com",
            password_hash="x",
            is_email_verified=False,
            telegram_username="@pollact",
            verification_token="tok1234567890abcdef",
        ))
        db.commit()
        db.close()

        TelegramPollingService._default_process(_update(11, "/start activate_tok1234567890abcdef", chat_id="888", username="pollact"))

        db = SessionLocal()
        user = db.query(User).filter(User.email == "pollact@test.com").first()
        db.close()
        assert user.is_email_verified is True
        assert user.telegram_chat_id == "888"
        assert len(replies) == 1
        assert "Регистрация подтверждена" in replies[0][1]


class TestWebhookHelpers:
    def test_set_webhook_posts_url(self, monkeypatch):
        posted = {}

        def fake_api(method, token, payload=None):
            posted["method"] = method
            posted["token"] = token
            posted["payload"] = payload
            return {"ok": True, "result": {"url": payload["url"]}}

        monkeypatch.setattr("gex.adapters.notifications.telegram_polling._api_call", fake_api)
        res = set_telegram_webhook("https://gex.example.com/telegram/webhook", token="TOK1")
        assert res["ok"] is True
        assert posted["method"] == "setWebhook"
        assert posted["payload"]["url"] == "https://gex.example.com/telegram/webhook"
        assert posted["payload"]["allowed_updates"] == ["message"]

    def test_set_webhook_without_token(self, monkeypatch):
        monkeypatch.setattr("gex.adapters.notifications.telegram_polling.get_bot_token", lambda: "")
        res = set_telegram_webhook("https://x.example.com/tg")
        assert res["ok"] is False
        assert "токен" in res["description"].lower() or "token" in res["description"].lower()

    def test_get_bot_token_falls_back_to_settings(self, monkeypatch):
        """Без runtime-конфигурации токен берётся из settings/.env.

        На машине с заполненной runtime-конфигурацией (бот сохранён из админки)
        ``get_bot_token()`` возвращает значение из runtime-хранилища — это штатная
        семантика «runtime поверх settings». Поэтому тест обязан очистить
        runtime-слой и проверять именно фолбэк на settings.
        """
        from gex.auth.config import settings

        monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "TEST_TOKEN_123")
        monkeypatch.setattr(
            "gex.auth.runtime_config.get_telegram_config",
            lambda: {"bot_token": "", "bot_username": "", "chat_id": ""},
        )
        assert get_bot_token() == "TEST_TOKEN_123"
