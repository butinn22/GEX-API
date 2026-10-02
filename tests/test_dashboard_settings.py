"""Tests for /auth/settings/dashboard (персональные настройки дашборда).

Покрытие:
- GET без авторизации → дефолты (пустые настройки)
- GET/PUT с авторизацией → сохранение и чтение настроек
- Нормализация: EMA, инструменты, линии (лимит 5, невалидные отбрасываются)
- Изоляция между пользователями
- Невалидные значения не роняют эндпоинт
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from gex.auth.models import User
from gex.auth.user_settings import UserDashboardSettings
from gex.adapters.persistence.database import SessionLocal, recreate_tables
from gex.auth.settings_router import (
    _normalize,
    ALLOWED_EMAS,
    TICKER_RE,
    MAX_INSTRUMENTS,
    MAX_CUSTOM_LINES,
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


def _register(client, email="u@test.dev", password="Passw0rd!123"):
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
    def test_emas_filtered_sorted_deduped(self):
        out = _normalize({"emas": [200, 50, 20, 999, 20, "x"]})
        assert out["emas"] == [20, 50, 200]

    def test_instruments_uppercased_format_validated(self):
        # Любой валидный тикер проходит (акции/индексы/крипта), мусор — нет
        out = _normalize({"instruments": ["spy", "BTC", "brk.b", "^GSPC", "SOL", "bad ticker!", "тикер", ""]})
        assert out["instruments"] == ["SPY", "BTC", "BRK.B", "^GSPC", "SOL"]

    def test_instruments_deduped_and_capped(self):
        out = _normalize({"instruments": ["AAPL", "AAPL", "NVDA"] * 20})
        assert out["instruments"] == ["AAPL", "NVDA"]
        many = [f"T{i}" for i in range(50)]
        out = _normalize({"instruments": many})
        assert len(out["instruments"]) == MAX_INSTRUMENTS

    def test_ticker_re_allows_real_symbols(self):
        for t in ["AAPL", "BRK.B", "^GSPC", "BTC-USD", "RTS", "MIX", "SI", "NVDA"]:
            assert TICKER_RE.match(t), t
        for t in ["", "AA PL", "TICKER!", "АППЛ", "TOO-LONG-TICKER-NAME-123"]:
            assert not TICKER_RE.match(t), t

    def test_trendlines_limit_and_validation(self):
        lines = [
            {"x1": i, "y1": 10.5, "x2": i + 10, "y2": 11.0, "color": "#ffd166"}
            for i in range(8)
        ]
        lines.append({"x1": "bad", "y1": 1, "x2": 2, "y2": 3})
        out = _normalize({"trendlines": {"SPY": lines}})
        assert len(out["trendlines"]["SPY"]) == MAX_CUSTOM_LINES
        for ln in out["trendlines"]["SPY"]:
            assert set(ln) == {"x1", "y1", "x2", "y2", "color"}

    def test_empty_input(self):
        assert _normalize({}) == {
            "emas": [], "instruments": [], "trendlines": {}, "walls": {}, "bands": {},
            "timeframe": "1d", "weights": {},
        }

    def test_nan_lines_rejected(self):
        out = _normalize({"trendlines": {"SPY": [
            {"x1": 1, "y1": float("nan"), "x2": 5, "y2": 6},
        ]}})
        assert out["trendlines"] == {}

    def test_wall_line_fields_kept(self):
        """GEX-стена: opacity/label/kind проходят нормализацию."""
        out = _normalize({"walls": {"NVDA": [{
            "x1": 0, "y1": 975.5, "x2": 999999, "y2": 975.5,
            "color": "#ef4444", "opacity": 0.2,
            "label": "Call Wall 975.5", "kind": "resistance",
        }]}})
        ln = out["walls"]["NVDA"][0]
        assert ln["opacity"] == 0.2
        assert ln["label"] == "Call Wall 975.5"
        assert ln["kind"] == "resistance"

    def test_wall_line_invalid_fields_stripped(self):
        """opacity вне 0..1, пустой label, неизвестный kind — отбрасываются."""
        out = _normalize({"walls": {"SPY": [{
            "x1": 0, "y1": 1, "x2": 10, "y2": 1,
            "opacity": 5, "label": "   ", "kind": "weird",
        }]}})
        ln = out["walls"]["SPY"][0]
        assert set(ln) == {"x1", "y1", "x2", "y2", "color"}

    def test_walls_do_not_leak_into_trendlines(self):
        """Стены — отдельный словарь, не смешиваются с кастомными линиями."""
        out = _normalize({
            "trendlines": {"SPY": [{"x1": 1, "y1": 900, "x2": 5, "y2": 920, "color": "#ffd166"}]},
            "walls": {"NVDA": [{"x1": 0, "y1": 975.5, "x2": 999999, "y2": 975.5,
                                  "opacity": 0.2, "kind": "support"}]},
        })
        assert "SPY" in out["trendlines"] and "NVDA" not in out["trendlines"]
        assert "NVDA" in out["walls"] and "SPY" not in out["walls"]

    def test_timeframe_allowed_values(self):
        """Таймфрейм: только из разрешённого набора, иначе дефолт 1d."""
        assert _normalize({"timeframe": "4h"})["timeframe"] == "4h"
        assert _normalize({"timeframe": "4H"})["timeframe"] == "4h"
        assert _normalize({"timeframe": "1h"})["timeframe"] == "1h"
        for bad in ("5m", "1w", "", None, 4, "; drop table"):
            assert _normalize({"timeframe": bad})["timeframe"] == "1d"

    def test_line_time_anchors_kept(self):
        """Метки времени t1/t2 (epoch мс) сохраняются — без них линия
        не перенесётся на другой таймфрейм."""
        out = _normalize({"trendlines": {"SPY": [{
            "x1": 10, "y1": 500, "x2": 120, "y2": 520,
            "t1": 1735689600000, "t2": 1743465600000,
        }]}})
        ln = out["trendlines"]["SPY"][0]
        assert ln["t1"] == 1735689600000.0
        assert ln["t2"] == 1743465600000.0

    def test_line_time_anchors_partial_or_invalid_dropped(self):
        """Одна метка, NaN или мусор — отбрасываются обе (частичная
        привязка бессмысленна; линия остаётся bar_index-линией)."""
        cases = [
            {"t1": 1735689600000},
            {"t1": 1735689600000, "t2": float("nan")},
            {"t1": 1735689600000, "t2": -5},
            {"t1": "yesterday", "t2": "today"},
        ]
        for extra in cases:
            line = {"x1": 1, "y1": 2, "x2": 3, "y2": 4}
            line.update(extra)
            ln = _normalize({"trendlines": {"SPY": [line]}})["trendlines"]["SPY"][0]
            assert "t1" not in ln and "t2" not in ln, extra

    def test_band_normalize_keeps_valid(self):
        """Доверительный интервал: low/high/label/color проходят."""
        out = _normalize({"bands": {"NVDA": {
            "low": 910.25, "high": 975.5, "label": "CI 910-975", "color": "#8b6cf5",
        }}})
        b = out["bands"]["NVDA"]
        assert b == {"low": 910.25, "high": 975.5, "label": "CI 910-975", "color": "#8b6cf5"}

    def test_band_invalid_stripped(self):
        """high <= low, NaN, мусор — отбрасываются целиком."""
        out = _normalize({"bands": {
            "SPY": {"low": 5, "high": 2},
            "AAPL": {"low": float("nan"), "high": 3},
            "TSLA": "not-a-dict",
        }})
        assert out["bands"] == {}

    def test_band_roundtrip(self, client):
        """Интеграция: интервал сохраняется и возвращается через PUT/GET."""
        token = _register(client)
        r = client.put("/auth/settings/dashboard", json={
            "emas": [],
            "instruments": [],
            "trendlines": {},
            "walls": {},
            "bands": {"NVDA": {"low": 920, "high": 970, "label": "GEX CI 920-970", "color": "#8b6cf5"}},
        }, headers=_auth(token))
        assert r.status_code == 200, r.text
        r = client.get("/auth/settings/dashboard", headers=_auth(token))
        b = r.json()["bands"]["NVDA"]
        assert b["low"] == 920.0 and b["high"] == 970.0
        assert b["label"] == "GEX CI 920-970"

    def test_put_wall_line_roundtrip(self, client):
        """Интеграция: стена сохраняется и возвращается через PUT/GET."""
        token = _register(client)
        r = client.put("/auth/settings/dashboard", json={
            "emas": [20],
            "instruments": ["NVDA"],
            "trendlines": {},
            "walls": {"NVDA": [{
                "x1": 0, "y1": 910.25, "x2": 999999, "y2": 910.25,
                "color": "#16c784", "opacity": 0.2,
                "label": "Put Wall 910.25", "kind": "support",
            }]},
        }, headers=_auth(token))
        assert r.status_code == 200, r.text
        r = client.get("/auth/settings/dashboard", headers=_auth(token))
        ln = r.json()["walls"]["NVDA"][0]
        assert ln["opacity"] == 0.2
        assert ln["label"] == "Put Wall 910.25"
        assert ln["kind"] == "support"


# ================================================================= #
#  2. INTEGRATION: GET /auth/settings/dashboard
# ================================================================= #
class TestGetSettings:
    def test_anonymous_gets_defaults(self, client):
        r = client.get("/auth/settings/dashboard")
        assert r.status_code == 200
        assert r.json() == {
            "emas": [], "instruments": [], "trendlines": {}, "walls": {}, "bands": {},
            "timeframe": "1d", "weights": {},
        }

    def test_new_user_gets_defaults(self, client):
        token = _register(client)
        r = client.get("/auth/settings/dashboard", headers=_auth(token))
        assert r.status_code == 200
        assert r.json() == {
            "emas": [], "instruments": [], "trendlines": {}, "walls": {}, "bands": {},
            "timeframe": "1d", "weights": {},
        }

    def test_returns_saved_settings(self, client):
        token = _register(client)
        body = {
            "emas": [20, 50, 200],
            "instruments": ["SPY", "QQQ"],
            "trendlines": {"SPY": [{"x1": 1, "y1": 500.5, "x2": 120, "y2": 512.25, "color": "#ffd166"}]},
            "walls": {},
            "bands": {},
            "timeframe": "4h",
            "weights": {},
        }
        r = client.put("/auth/settings/dashboard", json=body, headers=_auth(token))
        assert r.status_code == 200
        got = client.get("/auth/settings/dashboard", headers=_auth(token)).json()
        assert got == body


# ================================================================= #
#  3. INTEGRATION: PUT /auth/settings/dashboard
# ================================================================= #
class TestPutSettings:
    def test_timeframe_roundtrip(self, client):
        """Интеграция: выбранный 4h закрепляется за аккаунтом."""
        token = _register(client)
        r = client.put("/auth/settings/dashboard", json={
            "instruments": ["SPY"], "timeframe": "4h",
        }, headers=_auth(token))
        assert r.status_code == 200, r.text
        assert r.json()["timeframe"] == "4h"
        assert client.get("/auth/settings/dashboard",
                          headers=_auth(token)).json()["timeframe"] == "4h"

    def test_line_time_anchors_roundtrip(self, client):
        """Линия с метками времени выживает PUT/GET целиком."""
        token = _register(client)
        r = client.put("/auth/settings/dashboard", json={
            "trendlines": {"SPY": [{
                "x1": 10, "y1": 500.5, "x2": 150, "y2": 530.25,
                "color": "#ffd166", "t1": 1735689600000, "t2": 1743465600000,
            }]},
            "timeframe": "4h",
        }, headers=_auth(token))
        assert r.status_code == 200, r.text
        ln = client.get("/auth/settings/dashboard",
                        headers=_auth(token)).json()["trendlines"]["SPY"][0]
        assert ln["t1"] == 1735689600000.0 and ln["t2"] == 1743465600000.0
        assert ln["y1"] == 500.5 and ln["y2"] == 530.25

    def test_legacy_row_without_timeframe_gets_default(self, client):
        """Старая запись в БД (без timeframe) читается с дефолтным 1d."""
        from gex.auth.user_settings import UserDashboardSettings
        from gex.adapters.persistence.database import SessionLocal

        token = _register(client)
        me = client.get("/auth/me", headers=_auth(token)).json()
        db = SessionLocal()
        try:
            db.add(UserDashboardSettings(user_id=me["id"], data={
                "emas": [50], "instruments": ["SPY"],
                "trendlines": {}, "walls": {}, "bands": {},
            }))
            db.commit()
        finally:
            db.close()
        got = client.get("/auth/settings/dashboard", headers=_auth(token)).json()
        assert got["timeframe"] == "1d"
        assert got["emas"] == [50]

    def test_requires_auth(self, client):
        r = client.put("/auth/settings/dashboard", json={"emas": [20]})
        assert r.status_code == 401

    def test_persists_and_normalizes(self, client):
        token = _register(client)
        r = client.put("/auth/settings/dashboard", json={
            "emas": [999, 50, 20],
            "instruments": ["bad", "btc", "nvda"],
            "trendlines": {"SPY": [{"x1": 1, "y1": 2, "x2": 3, "y2": 4}]},
        }, headers=_auth(token))
        assert r.status_code == 200
        assert r.json() == {
            "emas": [20, 50],
            "instruments": ["BAD", "BTC", "NVDA"],
            "trendlines": {"SPY": [{"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0, "color": None}]},
            "walls": {},
            "bands": {},
            "timeframe": "1d",
            "weights": {},
        }

    def test_custom_ticker_saved_to_account(self, client):
        """Пользовательский тикер (не из базового набора) закрепляется за аккаунтом."""
        token = _register(client)
        r = client.put("/auth/settings/dashboard", json={
            "instruments": ["SPY", "NVDA", "SOL"],
        }, headers=_auth(token))
        assert r.status_code == 200
        got = client.get("/auth/settings/dashboard", headers=_auth(token)).json()
        assert got["instruments"] == ["SPY", "NVDA", "SOL"]
        # Другой пользователь своих настроек не видит
        t2 = _register(client, "b@test.dev")
        assert client.get("/auth/settings/dashboard", headers=_auth(t2)).json()["instruments"] == []

    def test_update_overwrites(self, client):
        token = _register(client)
        client.put("/auth/settings/dashboard", json={"emas": [20]}, headers=_auth(token))
        r = client.put("/auth/settings/dashboard", json={"emas": [100, 200]}, headers=_auth(token))
        assert r.status_code == 200
        got = client.get("/auth/settings/dashboard", headers=_auth(token)).json()
        assert got["emas"] == [100, 200]

    def test_row_is_unique_per_user(self, client):
        t1 = _register(client, "a@test.dev")
        t2 = _register(client, "b@test.dev")
        client.put("/auth/settings/dashboard", json={"emas": [20]}, headers=_auth(t1))
        client.put("/auth/settings/dashboard", json={"emas": [50, 100]}, headers=_auth(t2))
        db = SessionLocal()
        try:
            rows = db.query(UserDashboardSettings).all()
            assert len(rows) == 2
        finally:
            db.close()
        assert client.get("/auth/settings/dashboard", headers=_auth(t1)).json()["emas"] == [20]
        assert client.get("/auth/settings/dashboard", headers=_auth(t2)).json()["emas"] == [50, 100]

    def test_isolated_between_users(self, client):
        t1 = _register(client, "a@test.dev")
        t2 = _register(client, "b@test.dev")
        client.put("/auth/settings/dashboard", json={"emas": [20]}, headers=_auth(t1))
        assert client.get("/auth/settings/dashboard", headers=_auth(t2)).json()["emas"] == []

    def test_many_lines_capped_at_five(self, client):
        token = _register(client)
        lines = [{"x1": i, "y1": i, "x2": i + 1, "y2": i + 1} for i in range(7)]
        r = client.put("/auth/settings/dashboard", json={"trendlines": {"QQQ": lines}}, headers=_auth(token))
        assert r.status_code == 200
        assert len(r.json()["trendlines"]["QQQ"]) == MAX_CUSTOM_LINES

    def test_garbage_types_rejected_by_validation(self, client):
        """Неверные типы отклоняются на границе (422) — защита от мусора."""
        token = _register(client)
        r = client.put("/auth/settings/dashboard", json={
            "emas": ["abc", None],
            "instruments": [1, None],
            "trendlines": {"SPY": [{"x1": None}, "junk", {"x1": 1, "y1": 2, "x2": 3, "y2": 4}]},
        }, headers=_auth(token))
        assert r.status_code == 422
        # Невалидные строки в линиях ("junk") тоже не проходят валидацию модели
        r2 = client.put("/auth/settings/dashboard", json={"trendlines": {"SPY": ["junk"]}}, headers=_auth(token))
        assert r2.status_code == 422
