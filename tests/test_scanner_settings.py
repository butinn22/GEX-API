"""Tests for /auth/settings/scanner (персональные настройки авто-сканера).

Проверяем:
- GET для анонима → дефолты (слайдер 0.5, фильтр вкл, пустые тикеры).
- PUT/GET для авторизованного пользователя → сохранение и чтение.
- Валидация: слайдеры клипятся в 0..1, тикеры фильтруются (формат/дедуп/кап).
- Отдельная таблица: настройки дашборда не затирают настройки сканера.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from gex.auth.models import User
from gex.auth.user_scanner_settings import UserScannerSettings
from gex.adapters.persistence.database import SessionLocal, recreate_tables
from gex.auth.settings_router import (
    _normalize_scanner,
    MAX_SCANNER_TICKERS,
    SCANNER_UNIVERSES,
)

from main import app


@pytest.fixture
def client():
    recreate_tables()
    db = SessionLocal()
    db.query(User).delete()
    db.commit()
    db.close()
    return TestClient(app)


def _register(client, email="scanner@test.dev", password="Passw0rd!123"):
    base = email.split("@")[0].replace(".", "_")[:20]
    if len(base) < 5:
        base = (base + "user")[:20]
    nick = "@" + base
    r = client.post("/auth/register", json={"email": email, "password": password, "telegram_username": nick, "accept_terms": True})
    assert r.status_code == 201, r.text
    return r.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# ================================================================= #
#  1. UNIT: нормализация
# ================================================================= #
class TestNormalize:
    def test_defaults(self):
        out = _normalize_scanner({})
        assert out["flat_slider"] == 0.5
        assert out["slider_atr"] is None
        assert out["slider_bbw"] is None
        assert out["slider_pct"] is None
        assert out["flat_score_threshold"] == 60.0
        assert out["trend_strength_low"] == 30.0
        assert out["filter_signals"] is True
        assert out["gate_exits"] is False
        assert out["tickers"] == {}

    def test_sliders_clamped(self):
        out = _normalize_scanner({"flat_slider": 7, "slider_atr": -1, "slider_pct": 0.25})
        assert out["flat_slider"] == 1.0
        assert out["slider_atr"] == 0.0
        assert out["slider_pct"] == 0.25

    def test_invalid_sliders_fall_back(self):
        out = _normalize_scanner({"flat_slider": "abc", "slider_bbw": None})
        assert out["flat_slider"] == 0.5
        assert out["slider_bbw"] is None

    def test_thresholds_clamped(self):
        out = _normalize_scanner({"flat_score_threshold": 500, "trend_strength_low": -3})
        assert out["flat_score_threshold"] == 100.0
        assert out["trend_strength_low"] == 0.0

    def test_filter_flags_bool_parsing(self):
        assert _normalize_scanner({"filter_signals": False})["filter_signals"] is False
        assert _normalize_scanner({"filter_signals": "false"})["filter_signals"] is False
        assert _normalize_scanner({"filter_signals": "on"})["filter_signals"] is True
        assert _normalize_scanner({"gate_exits": True})["gate_exits"] is True

    def test_tickers_validated_deduped_capped(self):
        out = _normalize_scanner({
            "tickers": {
                "us": ["spy", "SPY", "AAPL", "bad ticker!", "BRK.B", "^GSPC"],
                "zzz": ["X"],  # неизвестный универсум — отбрасывается
                "crypto": ["BTC"],
            }
        })
        assert out["tickers"]["us"] == ["SPY", "AAPL", "BRK.B", "^GSPC"]
        assert "zzz" not in out["tickers"]
        assert out["tickers"]["crypto"] == ["BTC"]

    def test_tickers_capped(self):
        many = [f"T{i}" for i in range(200)]
        out = _normalize_scanner({"tickers": {"us": many}})
        assert len(out["tickers"]["us"]) == MAX_SCANNER_TICKERS

    def test_universes(self):
        assert SCANNER_UNIVERSES == ("us", "ru", "crypto", "fx", "sectors")


# ================================================================= #
#  2. API: персистентность
# ================================================================= #
class TestApi:
    def test_get_anonymous_defaults(self, client):
        r = client.get("/auth/settings/scanner")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["flat_slider"] == 0.5
        assert data["filter_signals"] is True
        assert data["tickers"] == {}

    def test_put_requires_auth(self, client):
        r = client.put("/auth/settings/scanner", json={"flat_slider": 0.8})
        assert r.status_code in (401, 403), r.text

    def test_put_get_roundtrip(self, client):
        token = _register(client)
        h = _auth(token)
        r = client.put("/auth/settings/scanner", json={
            "flat_slider": 0.8,
            "slider_pct": 0.3,
            "flat_score_threshold": 65,
            "trend_strength_low": 25,
            "filter_signals": True,
            "gate_exits": False,
            "tickers": {"us": ["SPY", "QQQ"], "crypto": ["BTC"]},
        }, headers=h)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["flat_slider"] == 0.8
        assert data["slider_pct"] == 0.3
        assert data["flat_score_threshold"] == 65.0
        assert data["tickers"]["us"] == ["SPY", "QQQ"]

        r2 = client.get("/auth/settings/scanner", headers=h)
        assert r2.status_code == 200
        assert r2.json()["flat_slider"] == 0.8
        assert r2.json()["tickers"]["crypto"] == ["BTC"]

    def test_settings_isolated_between_users(self, client):
        t1 = _register(client, "scanner1@test.dev")
        t2 = _register(client, "scanner2@test.dev")
        client.put("/auth/settings/scanner", json={"flat_slider": 0.9}, headers=_auth(t1))
        r2 = client.get("/auth/settings/scanner", headers=_auth(t2))
        assert r2.json()["flat_slider"] == 0.5  # у второго — дефолт

    def test_scanner_settings_separate_from_dashboard(self, client):
        """PUT дашборда не должен затирать настройки сканера."""
        token = _register(client)
        h = _auth(token)
        client.put("/auth/settings/scanner", json={"flat_slider": 0.75, "tickers": {"us": ["AAPL"]}}, headers=h)
        client.put("/auth/settings/dashboard", json={"emas": [20, 50], "instruments": ["SPY"]}, headers=h)
        r = client.get("/auth/settings/scanner", headers=h)
        assert r.json()["flat_slider"] == 0.75
        assert r.json()["tickers"]["us"] == ["AAPL"]

    def test_db_row_stored(self, client):
        token = _register(client)
        client.put("/auth/settings/scanner", json={"flat_slider": 0.6}, headers=_auth(token))
        db = SessionLocal()
        try:
            rows = db.query(UserScannerSettings).all()
            assert len(rows) == 1
            assert rows[0].data["flat_slider"] == 0.6
        finally:
            db.close()
