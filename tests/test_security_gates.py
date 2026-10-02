"""Tests: гейты безопасности — подписка (уровень + срок) и Master Admin.

Проверяет напрямую зависимости из gex.auth.dependencies:
  * require_subscription(level) — статус, истёкший срок, мастер-байпас;
  * require_master_admin — только мастер.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import HTTPException

from gex.auth.dependencies import require_subscription, require_master_admin


def now() -> datetime:
    return datetime.now(timezone.utc)


def mkuser(status: str, expires=None, email="user@test.local"):
    return SimpleNamespace(
        email=email,
        subscription_status=status,
        subscription_expires_at=expires,
        is_email_verified=True,
        is_blocked=False,
    )


def test_require_subscription_rejects_lower_level():
    dep = require_subscription("EXTENDED")
    with pytest.raises(HTTPException) as ei:
        dep(mkuser("BASIC", now() + timedelta(days=30)))
    assert ei.value.status_code == 403


def test_require_subscription_rejects_expired():
    dep = require_subscription("EXTENDED")
    with pytest.raises(HTTPException) as ei:
        dep(mkuser("EXTENDED", now() - timedelta(hours=1)))
    assert ei.value.status_code == 403
    assert "истёк" in ei.value.detail


def test_require_subscription_accepts_active():
    dep = require_subscription("EXTENDED")
    user = dep(mkuser("EXTENDED", now() + timedelta(days=5)))
    assert user.subscription_status == "EXTENDED"


def test_require_subscription_accepts_no_expiry():
    """Срок не задан — считаем активным (старые записи)."""
    dep = require_subscription("BASIC")
    assert dep(mkuser("BASIC", None))


def test_require_subscription_basic_accepts_extended():
    dep = require_subscription("BASIC")
    assert dep(mkuser("EXTENDED", now() + timedelta(days=1)))


def test_require_subscription_master_bypasses_expired():
    """Мастер проходит даже с истёкшим сроком."""
    dep = require_subscription("EXTENDED")
    user = dep(mkuser("ADMIN", now() - timedelta(days=30), email="sadisting"))
    assert user.subscription_status == "ADMIN"


def test_require_master_admin_rejects_regular():
    with pytest.raises(HTTPException) as ei:
        require_master_admin(mkuser("EXTENDED", now() + timedelta(days=5)))
    assert ei.value.status_code == 403


def test_require_master_admin_accepts_master():
    user = require_master_admin(mkuser("ADMIN", None, email="sadisting"))
    assert user.email == "sadisting"
