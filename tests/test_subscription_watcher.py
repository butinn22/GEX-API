"""Tests: уведомление за 24ч до окончания подписки + гейты сканеров по сроку.

Покрывает:
  * gex.auth.models.subscription_is_active — правила активности подписки;
  * gex.subscription_watcher.SubscriptionExpiryWatcher — выборка истекающих
    за 24ч, отправка в Telegram, дедупликация, повтор после продления;
  * gex.signal_scanner_service — get_all_active_users исключает истёкшие,
    _tg_chat_id/_apply_changes не шлют при истёкшей подписке/выключенном notify.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from gex.adapters.persistence.database import recreate_tables, SessionLocal
from gex.auth.models import User, SubscriptionStatus, subscription_is_active
from gex.auth.user_instrument import UserInstrument
from gex.application.subscription_watcher import SubscriptionExpiryWatcher, _format_message
import gex.application.subscription_watcher as watcher_module
import gex.application.signal_scanner_service as scanner_module
from gex.application.signal_scanner_service import SignalScannerService, ScannerInstrument
from gex.application.signal_service import SignalService
from gex.application.service import GEXService


def now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture(autouse=True)
def clean_db():
    recreate_tables()
    db = SessionLocal()
    db.query(User).delete()
    db.commit()
    db.close()
    yield


def make_user(
    db,
    *,
    status: str = SubscriptionStatus.INACTIVE,
    expires: datetime | None = None,
    chat: str | None = None,
    notify: bool = True,
) -> User:
    u = User(
        id=str(uuid.uuid4()),
        email=f"{uuid.uuid4().hex[:12]}@test.local",
        password_hash="x",
        subscription_status=status,
        subscription_expires_at=expires,
        telegram_chat_id=chat,
        telegram_notify=notify,
        is_email_verified=True,
    )
    db.add(u)
    db.commit()
    return u


# ================================================================= #
#  subscription_is_active
# ================================================================= #
def test_active_admin_always():
    u = SimpleNamespace(subscription_status="ADMIN", subscription_expires_at=now() - timedelta(days=1))
    assert subscription_is_active(u)


def test_active_basic_future():
    u = SimpleNamespace(subscription_status="BASIC", subscription_expires_at=now() + timedelta(days=5))
    assert subscription_is_active(u)


def test_inactive_when_expired():
    u = SimpleNamespace(subscription_status="EXTENDED", subscription_expires_at=now() - timedelta(hours=1))
    assert not subscription_is_active(u)


def test_active_when_no_expiry():
    u = SimpleNamespace(subscription_status="EXTENDED", subscription_expires_at=None)
    assert subscription_is_active(u)


def test_inactive_status():
    u = SimpleNamespace(subscription_status="INACTIVE", subscription_expires_at=None)
    assert not subscription_is_active(u)


def test_naive_datetime_handled():
    """Naive datetime (UTC wall time, как из SQLite) — не должно падать и считаться верно."""
    u = SimpleNamespace(
        subscription_status="EXTENDED",
        # UTC wall time без tzinfo — как приходят значения из SQLite
        subscription_expires_at=(datetime.now(timezone.utc) - timedelta(hours=2)).replace(tzinfo=None),
    )
    assert not subscription_is_active(u)
    u2 = SimpleNamespace(
        subscription_status="EXTENDED",
        subscription_expires_at=(datetime.now(timezone.utc) + timedelta(hours=2)).replace(tzinfo=None),
    )
    assert subscription_is_active(u2)


# ================================================================= #
#  SubscriptionExpiryWatcher
# ================================================================= #
def _watcher(monkeypatch, sent: list):
    monkeypatch.setattr(
        watcher_module,
        "send_telegram_message",
        lambda text, parse_mode=None, chat_id=None: sent.append({"text": text, "chat_id": chat_id}),
    )
    return SubscriptionExpiryWatcher()


def test_sends_for_expiring_user(monkeypatch):
    sent: list = []
    watcher = _watcher(monkeypatch, sent)
    db = SessionLocal()
    user = make_user(db, status="EXTENDED", expires=now() + timedelta(hours=12), chat="123456")
    user_id = user.id
    db.close()

    assert watcher._check() == 1
    assert len(sent) == 1
    assert sent[0]["chat_id"] == "123456"
    assert "истекает завтра" in sent[0]["text"]
    assert "EXTENDED" in sent[0]["text"]

    db = SessionLocal()
    fresh = db.query(User).filter(User.id == user_id).first()
    assert fresh is not None
    assert fresh.subscription_expiry_notified_for is not None
    db.close()


def test_dedup_no_resend(monkeypatch):
    sent: list = []
    watcher = _watcher(monkeypatch, sent)
    db = SessionLocal()
    make_user(db, status="BASIC", expires=now() + timedelta(hours=6), chat="1")
    db.close()

    assert watcher._check() == 1
    assert watcher._check() == 0
    assert len(sent) == 1


def test_skips_far_expiry(monkeypatch):
    sent: list = []
    watcher = _watcher(monkeypatch, sent)
    db = SessionLocal()
    make_user(db, status="EXTENDED", expires=now() + timedelta(hours=30), chat="1")
    db.close()
    assert watcher._check() == 0
    assert sent == []


def test_skips_no_chat(monkeypatch):
    sent: list = []
    watcher = _watcher(monkeypatch, sent)
    db = SessionLocal()
    make_user(db, status="EXTENDED", expires=now() + timedelta(hours=12), chat=None)
    db.close()
    assert watcher._check() == 0


def test_skips_admin(monkeypatch):
    sent: list = []
    watcher = _watcher(monkeypatch, sent)
    db = SessionLocal()
    make_user(db, status="ADMIN", expires=now() + timedelta(hours=12), chat="1")
    db.close()
    assert watcher._check() == 0


def test_skips_inactive(monkeypatch):
    sent: list = []
    watcher = _watcher(monkeypatch, sent)
    db = SessionLocal()
    make_user(db, status="INACTIVE", expires=now() + timedelta(hours=12), chat="1")
    db.close()
    assert watcher._check() == 0


def test_notifies_again_after_renewal(monkeypatch):
    """После продления (новая дата expires_at) уведомление уходит повторно."""
    sent: list = []
    watcher = _watcher(monkeypatch, sent)
    db = SessionLocal()
    user = make_user(db, status="EXTENDED", expires=now() + timedelta(hours=10), chat="1")
    user_id = user.id
    db.close()

    assert watcher._check() == 1
    assert watcher._check() == 0  # дедуп

    db = SessionLocal()
    fresh = db.query(User).filter(User.id == user_id).first()
    fresh.subscription_expires_at = now() + timedelta(hours=20)  # продлили на новую дату
    db.commit()
    db.close()

    assert watcher._check() == 1  # для новой даты снова уведомляем
    assert len(sent) == 2


def test_format_message_contains_plan_and_date():
    u = SimpleNamespace(
        subscription_status="EXTENDED",
        subscription_expires_at=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc),
    )
    text = _format_message(u)
    assert "EXTENDED" in text
    assert "20.08.2026" in text
    assert "истекает завтра" in text


# ================================================================= #
#  SignalScannerService: гейты по сроку подписки
# ================================================================= #
def _add_instrument(db, user_id: str, ticker: str = "SPY", timeframe: str = "1d"):
    db.add(UserInstrument(user_id=user_id, ticker=ticker, timeframe=timeframe))
    db.commit()


def test_get_all_active_users_excludes_expired(monkeypatch):
    from gex.adapters.persistence.database import SessionLocal as RealSession

    scanner = SignalScannerService(signal_service=SignalService(GEXService()))
    db = RealSession()
    u_future = make_user(db, status="EXTENDED", expires=now() + timedelta(days=2))
    u_expired = make_user(db, status="EXTENDED", expires=now() - timedelta(days=1))
    u_admin = make_user(db, status="ADMIN", expires=now() - timedelta(days=30))  # срок в прошлом, но ADMIN
    u_noexp = make_user(db, status="EXTENDED", expires=None)
    u_inactive = make_user(db, status="INACTIVE", expires=now() + timedelta(days=2))
    ids = {
        "future": u_future.id, "expired": u_expired.id, "admin": u_admin.id,
        "noexp": u_noexp.id, "inactive": u_inactive.id,
    }
    for u in (u_future, u_expired, u_admin, u_noexp, u_inactive):
        _add_instrument(db, u.id, ticker=u.id[:5].upper())
    db.close()

    active = set(scanner.get_all_active_users())
    assert ids["future"] in active
    assert ids["noexp"] in active
    assert ids["admin"] in active
    assert ids["expired"] not in active  # истёкший EXTENDED — НЕ сканируется
    assert ids["inactive"] not in active


def test_notify_user_gate_expired(monkeypatch):
    """Гейт отправки (подписка + chat_id + telegram_notify) в новом пайплайне событий."""
    sent: list = []
    import gex.adapters.notifications.telegram_sender as tg_sender

    monkeypatch.setattr(
        tg_sender,
        "send_telegram_message",
        lambda text, parse_mode=None, chat_id=None: sent.append(chat_id),
    )
    from gex.adapters.persistence.database import SessionLocal as RealSession

    db = RealSession()
    u_expired = make_user(db, status="EXTENDED", expires=now() - timedelta(hours=1), chat="100", notify=True)
    u_active = make_user(db, status="EXTENDED", expires=now() + timedelta(days=3), chat="200", notify=True)
    expired_id, active_id = u_expired.id, u_active.id
    db.close()

    scanner = SignalScannerService(signal_service=SignalService(GEXService()))
    # Состояние уведомлений — локально, per-user (как Redis: uid → dict),
    # иначе dev-Redis «протекает» между тестами/прогонами.
    states: dict = {}
    monkeypatch.setattr(scanner, "_load_state", lambda uid: states.get(uid, {}))
    monkeypatch.setattr(scanner, "_save_state", lambda uid, s: states.update({uid: dict(s)}))
    # Оба юзера уже «инициализированы», сигнала нет — событие «появился сигнал».
    states[expired_id] = {"__init": "1", "SPY:1d": ""}
    states[active_id] = {"__init": "1", "SPY:1d": ""}

    # Инструмент с сигналом — дал бы контент, если бы не гейт подписки.
    sig = SimpleNamespace(
        action="SELL", order_type="entry_short", price=100.0,
        timestamp=now(), entry_score=0.5,
    )
    instr = [ScannerInstrument(ticker="SPY", timeframe="1d", latest_signals=[sig])]

    scanner._apply_changes(expired_id, instr, send=True)
    assert sent == []  # истёкшая подписка — молчим

    scanner._apply_changes(active_id, instr, send=True)
    assert sent == ["200"]  # активная — шлём
