"""Route-level tests for AUTO mode (design doc §5, AC2).

Проверяется именно проводка роутера:
  * ``/ext/gex``: ``mode=auto`` → ``analyze_auto`` + AUTO-ключ кэша (TTL 600) + блок ``auto``;
  * ``/ext/gex``: ``mode=manual`` (и без ``mode``) → обычный ``analyze`` + ручной ключ;
  * ``/ext/gex``: недопустимый ``mode`` → 422 (валидация Query-паттерна);
  * ``/moex/gex``: ``mode=auto`` → days=90, expiries=0, auto=True; manual — без auto.

Ext-роутер тестируется через TestClient (нужна валидация 422). MOEX-роутер —
прямым вызовом обработчика: это проверяет проводку аргументов без тяжёлой
сборки полного GEXAnalysisOut и без сети.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gex.auth.dependencies as auth_deps
from gex.auth.dependencies import get_current_user
from gex.deps import provide_extended_gex_service, provide_gex_service
from gex.domain.data_loader import OptionSnapshot
from gex.application.auto_scope import AutoCoverage
from gex.application.extended import ExtendedGEXAnalyzer
from gex.routers import extended_router as ext_mod
from gex.routers import moex_router as moex_mod


# ====================================================================== #
#  Фикстуры
# ====================================================================== #
def _snapshot(
    spot: float = 100.0, symbol: str = "AAPL",
    T: float | tuple[float, ...] = 30.0 / 365.0,
) -> OptionSnapshot:
    """Синтетическая цепочка. ``T`` — одиночная доля года или их кортеж (несколько экспираций)."""
    ts = (T,) if isinstance(T, (int, float)) else tuple(T)
    rows = []
    for T_i in ts:
        for k in np.linspace(80.0, 120.0, 11):
            diff = float(k) - spot
            rows.append({"strike": float(k), "type": "C", "oi": max(100.0, 1000.0 + diff * 40.0),
                         "iv": 0.2, "T": T_i})
            rows.append({"strike": float(k), "type": "P", "oi": max(100.0, 1000.0 - diff * 40.0),
                         "iv": 0.2, "T": T_i})
    chain = pd.DataFrame(rows)
    return OptionSnapshot(symbol=symbol, spot=spot, as_of=pd.Timestamp.now(tz="UTC"), chain=chain)


def _make_report(source: str = "stock"):
    rep = ExtendedGEXAnalyzer().analyze("AAPL", snapshot=_snapshot())
    rep.source = source
    return rep


class FakeExtService:
    """Фейковый ExtendedGEXAnalyzer: пишет вызовы, отдаёт валидный отчёт."""

    def __init__(self):
        self.calls: list[tuple] = []

    def analyze(self, ticker, **kw):
        self.calls.append(("manual", ticker, kw))
        return _make_report("stock")

    def analyze_auto(self, ticker, **kw):
        self.calls.append(("auto", ticker, kw))
        src = kw.get("source")
        primary = src if src in ("webull", "yfinance") else "webull"
        rep = _make_report(primary)
        rep.coverage = AutoCoverage(
            mode="auto", resolved_days=90.0, resolved_expiries=20,
            sources_used=[primary], primary_source=primary,
            fallback_used=False, escalated=False, partial=False,
            expirations_merged=1, strike_count=len(rep.per_strike),
            total_oi=sum(s.oi_call + s.oi_put for s in rep.per_strike),
            sparse=False, sparse_reasons=[], elapsed_ms=7,
        )
        return rep


class _RecordingCache:
    def __init__(self):
        self.key = ""
        self.ttl = -1

    def get(self, key, ttl, compute):
        self.key = key
        self.ttl = ttl
        return compute()


@pytest.fixture
def ext_client(monkeypatch):
    # Пропускаем auth-барьер: get_current_user подменён, bypass всегда True.
    monkeypatch.setattr(auth_deps, "can_bypass_barriers", lambda user: True)

    svc = FakeExtService()
    cache = _RecordingCache()
    monkeypatch.setattr(ext_mod, "result_cache", cache)

    app = FastAPI()
    app.include_router(ext_mod.router)
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[provide_extended_gex_service] = lambda: svc
    client = TestClient(app)
    return client, svc, cache


# ====================================================================== #
#  /ext/gex
# ====================================================================== #
class TestExtendedRoute:

    def test_auto_mode_uses_analyze_auto_and_auto_key(self, ext_client):
        client, svc, cache = ext_client
        r = client.get("/ext/gex/AAPL?mode=auto")
        assert r.status_code == 200
        body = r.json()
        assert body["auto"]["mode"] == "auto"
        assert body["auto"]["resolved_days"] == 90.0
        assert body["auto"]["resolved_expiries"] == 20
        assert body["days"] == 90.0
        # выбран auto-путь и auto-ключ кэша (дизъюнктный сегмент)
        assert svc.calls[-1][0] == "auto"
        assert "EXTGEXA" in cache.key
        assert cache.ttl == 600

    def test_auto_ignores_client_days_expiries(self, ext_client):
        client, svc, _ = ext_client
        r = client.get("/ext/gex/AAPL?mode=auto&days=7&expiries=1")
        assert r.status_code == 200
        assert r.json()["days"] == 90.0

    def test_manual_mode_regression(self, ext_client):
        client, svc, cache = ext_client
        r = client.get("/ext/gex/AAPL?mode=manual&days=30&expiries=5")
        assert r.status_code == 200
        body = r.json()
        # Ручной режим тоже отдаёт охват (аудит 2026-09-17), но помеченный
        # 'manual' и без эскалации/fallback — AUTO-путь не затронут.
        assert body["auto"] is not None
        assert body["auto"]["mode"] == "manual"
        assert body["auto"]["escalated"] is False
        assert body["auto"]["fallback_used"] is False
        assert body["days"] == 30.0
        assert svc.calls[-1][0] == "manual"
        assert "EXTGEXA" not in cache.key
        assert cache.ttl == 600

    def test_default_mode_is_manual(self, ext_client):
        client, svc, _ = ext_client
        r = client.get("/ext/gex/AAPL")
        assert r.status_code == 200
        assert r.json()["auto"]["mode"] == "manual"
        assert svc.calls[-1][0] == "manual"

    def test_invalid_mode_422(self, ext_client):
        client, _, _ = ext_client
        r = client.get("/ext/gex/AAPL?mode=fast")
        assert r.status_code == 422


# ====================================================================== #
#  /moex/gex — прямая проводка обработчика
# ====================================================================== #
class FakeMoexService:
    def __init__(self):
        self.calls: list[tuple] = []

    def analyze_moex(self, asset, **kw):
        self.calls.append((asset, kw))
        return {"ok": True, "asset": asset, "kw": kw}


class TestMoexRoute:

    def test_auto_wiring_days90_expiries0(self):
        svc = FakeMoexService()
        moex_mod.get_moex_gex("RTS", mode="auto", svc=svc, notify=False, background_tasks=None)
        asset, kw = svc.calls[-1]
        assert asset == "RTS"
        assert kw == {"days": 90.0, "max_expiries": 0, "auto": True}

    def test_manual_wiring_has_no_auto(self):
        svc = FakeMoexService()
        moex_mod.get_moex_gex("RTS", mode="manual", days=30, expiries=5,
                              svc=svc, notify=False, background_tasks=None)
        _asset, kw = svc.calls[-1]
        assert kw["days"] == 30
        assert kw["max_expiries"] == 5
        assert "auto" not in kw


# ====================================================================== #
#  MOEX coverage builder (без сети)
# ====================================================================== #
class TestMoexCoverageBuilder:

    def test_builds_auto_coverage(self):
        from types import SimpleNamespace
        from gex.application.moex_service import MOEXGEXService, MOEX_PRIMARY_SOURCE

        svc = MOEXGEXService(repo=object(), runner=object())
        snap = _snapshot(symbol="RTS", T=(7.0 / 365.0, 30.0 / 365.0, 60.0 / 365.0))
        strikes = [
            SimpleNamespace(strike=float(k), oi_call=100.0, oi_put=100.0, gex_net=1.0)
            for k in np.linspace(80.0, 120.0, 11)
        ]
        out = SimpleNamespace(
            symbol="RTS", spot=100.0,
            profile=SimpleNamespace(per_strike=strikes, call_wall=120.0, put_wall=80.0),
        )
        cov = svc._auto_coverage(snap, out, days=90.0, max_expiries=0, elapsed_ms=12)
        assert cov.mode == "auto"
        assert cov.resolved_days == 90.0
        assert cov.resolved_expiries == 0            # 0 = ALL на MOEX
        assert cov.primary_source == MOEX_PRIMARY_SOURCE
        assert cov.sources_used == [MOEX_PRIMARY_SOURCE]
        assert cov.escalated is False
        assert cov.fallback_used is False
        assert cov.strike_count == 11
        assert cov.expirations_merged == 3
        assert cov.sparse is False
        assert cov.total_oi == 2200.0

    def test_six_bucket_chain_not_flagged_low_expiries(self):
        """MOEX без fallback: 6 бакетов (как реальный RTS) — причина не появляется."""
        from types import SimpleNamespace
        from gex.application.moex_service import MOEXGEXService, MOEX_PRIMARY_SOURCE

        svc = MOEXGEXService(repo=object(), runner=object())
        six = (7.0 / 365.0, 14.0 / 365.0, 30.0 / 365.0, 60.0 / 365.0, 90.0 / 365.0, 180.0 / 365.0)
        snap = _snapshot(symbol="RTS", T=six)
        strikes = [
            SimpleNamespace(strike=float(k), oi_call=100.0, oi_put=100.0, gex_net=1.0)
            for k in np.linspace(80.0, 120.0, 11)
        ]
        out = SimpleNamespace(
            symbol="RTS", spot=100.0,
            profile=SimpleNamespace(per_strike=strikes, call_wall=120.0, put_wall=80.0),
        )
        cov = svc._auto_coverage(snap, out, days=90.0, max_expiries=0, elapsed_ms=1)
        assert cov.expirations_merged == 6
        assert cov.sparse is False
        assert "low_expiries" not in cov.sparse_reasons
        assert cov.primary_source == MOEX_PRIMARY_SOURCE
        assert cov.escalated is False

    def test_sparse_flag_when_thin(self):
        from types import SimpleNamespace
        from gex.application.moex_service import MOEXGEXService

        svc = MOEXGEXService(repo=object(), runner=object())
        snap = _snapshot(symbol="RTS")
        out = SimpleNamespace(symbol="RTS", spot=100.0,
                              profile=SimpleNamespace(per_strike=[], call_wall=0.0, put_wall=0.0))
        cov = svc._auto_coverage(snap, out, days=90.0, max_expiries=0, elapsed_ms=1)
        assert cov.sparse is True
        assert "no_data" in cov.sparse_reasons
