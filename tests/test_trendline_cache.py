"""Кэш `/trendlines` (result_cache): проводка роутера, ``fresh``, ``notify``.

Проверяется именно маршрутный слой (сам ResultCache покрыт test_result_cache.py):

  * повторный запрос отдаётся из общего кэша — расчёт не повторяется (это и есть
    «подтянуть состояние от другого пользователя»: одна запись на всех, Redis);
  * ключ кэша включает все параметры анализа (разные параметры — разные записи);
  * ``fresh=true`` инвалидирует запись и пересчитывает (ручное «Обновить»);
  * ``notify=true`` идёт мимо кэша — персональная доставка в Telegram не должна
    «подтягиваться» из чужого запроса.

``result_cache`` подменяется записывающей заглушкой: Redis в тестах не нужен,
проверяем проводку и ключ/TTL.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gex.auth.dependencies as auth_deps
from gex.auth.dependencies import get_current_user
from gex.deps import provide_trendline_service
from gex.schemas import FractalStructureOut, TrendlineAnalysisOut, TrendlineTimeframeOut
from gex.routers import trendline_router as tr_mod


# ====================================================================== #
#  Фикстуры
# ====================================================================== #
def _payload(symbol: str, tf: str) -> TrendlineAnalysisOut:
    """Минимальный валидный ответ по одному таймфрейму."""
    to = TrendlineTimeframeOut(
        timeframe=tf, last_close=100.0, n_bars=300, atr=1.5,
        support_lines=[], resistance_lines=[],
        strongest_support=None, strongest_resistance=None,
        trend_direction="BULLISH", trend_strength=55.0, line_angle_deg=12.0,
        fractals=FractalStructureOut(
            n_swing_highs=10, n_swing_lows=9,
            higher_highs=5, higher_lows=5, lower_highs=2, lower_lows=2,
        ),
        fractal_trend="BULLISH", fractal_strength=50.0,
        combined_trend="BULLISH", combined_strength=52.0,
    )
    return TrendlineAnalysisOut(
        symbol=symbol, asset_type="stock", spot=100.0,
        generated_at=datetime.now(timezone.utc),
        timeframes=[to], consensus_trend="BULLISH", weights={tf: 1.0},
    )


class FakeTrendlineSvc:
    """Фейковый TrendlineService: пишет вызовы, отдаёт валидную схему."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def analyze(self, ticker, **kw):
        self.calls.append(("all", ticker))
        return _payload(ticker, "1d")

    def analyze_timeframe(self, ticker, timeframe, **kw):
        self.calls.append((timeframe, ticker))
        return _payload(ticker, timeframe)


class _RecordingCache:
    """Заглушка result_cache: store + запись ключей/TTL/инвалидаций."""

    def __init__(self):
        self.store: dict[str, object] = {}
        self.keys: list[str] = []
        self.ttls: list[int] = []
        self.invalidated: list[str] = []

    def get(self, key, ttl, compute):
        self.keys.append(key)
        self.ttls.append(ttl)
        if key in self.store:
            return self.store[key]
        value = compute()
        self.store[key] = value
        return value

    def invalidate(self, key):
        self.invalidated.append(key)
        self.store.pop(key, None)


@pytest.fixture
def tl_client(monkeypatch):
    # Пропускаем auth-барьер: get_current_user подменён, bypass всегда True.
    monkeypatch.setattr(auth_deps, "can_bypass_barriers", lambda user: True)
    # Telegram-доставка — no-op (сети в тестах нет).
    monkeypatch.setattr(tr_mod, "notify_trendline_analysis", lambda *a, **k: None)

    svc = FakeTrendlineSvc()
    cache = _RecordingCache()
    monkeypatch.setattr(tr_mod, "result_cache", cache)

    app = FastAPI()
    app.include_router(tr_mod.router)
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[provide_trendline_service] = lambda: svc
    client = TestClient(app)
    return client, svc, cache


# ====================================================================== #
#  Проводка кэша
# ====================================================================== #
class TestTrendlineCacheWiring:

    def test_second_request_hits_shared_cache(self, tl_client):
        client, svc, cache = tl_client
        r1 = client.get("/trendlines/AAPL?timeframe=1d")
        assert r1.status_code == 200
        assert len(svc.calls) == 1

        r2 = client.get("/trendlines/AAPL?timeframe=1d")
        assert r2.status_code == 200
        # Повторный запрос расчёт не повторяет — состояние берётся из кэша.
        assert len(svc.calls) == 1
        assert r1.json() == r2.json()

        assert "TRENDLINES" in cache.keys[-1]
        assert "AAPL" in cache.keys[-1]
        assert cache.ttls[-1] == tr_mod._TRENDLINES_TTL_S

    def test_key_contains_all_params(self, tl_client):
        client, svc, cache = tl_client
        client.get("/trendlines/SPY?timeframe=1d&max_lines=4")
        client.get("/trendlines/SPY?timeframe=1d&max_lines=6")
        client.get("/trendlines/SPY?timeframe=4h&max_lines=4")
        # Три разных набора параметров — три разных ключа и три расчёта.
        assert len(svc.calls) == 3
        assert len(set(cache.keys)) == 3

    def test_fresh_forces_recompute(self, tl_client):
        client, svc, cache = tl_client
        client.get("/trendlines/AAPL?timeframe=1d")
        assert len(svc.calls) == 1

        r = client.get("/trendlines/AAPL?timeframe=1d&fresh=true")
        assert r.status_code == 200
        # fresh=true инвалидирует запись и считает заново.
        assert len(svc.calls) == 2
        assert cache.invalidated and "TRENDLINES" in cache.invalidated[-1]
        # Свежий результат занял место прежнего — следующий обычный запрос снова из кэша.
        client.get("/trendlines/AAPL?timeframe=1d")
        assert len(svc.calls) == 2

    def test_notify_bypasses_cache(self, tl_client):
        client, svc, cache = tl_client
        client.get("/trendlines/AAPL?timeframe=1d&notify=true")
        client.get("/trendlines/AAPL?timeframe=1d&notify=true")
        # notify=true — персональная доставка: каждый запрос считается «живьём».
        assert len(svc.calls) == 2
        assert not cache.keys  # в кэш-слой не заходили вовсе

    def test_all_timeframe_uses_analyze_and_caches(self, tl_client):
        client, svc, _ = tl_client
        r1 = client.get("/trendlines/SPY?timeframe=all")
        assert r1.status_code == 200
        assert svc.calls[-1] == ("all", "SPY")
        r2 = client.get("/trendlines/SPY?timeframe=all")
        assert r2.status_code == 200
        assert len(svc.calls) == 1  # второй — из кэша
