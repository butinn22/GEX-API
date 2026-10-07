"""QA Round-3 — INDEPENDENT httpx client-leak verification (C13).

Proves the loop-scoped fetcher registry reuses one HTTP client pool per exchange
and is closed on shutdown / at the end of a Celery task, i.e. no per-request
client leak (``data.py`` and ``tasks._run_isolated``).

Run explicitly::

    venv/Scripts/python.exe -m pytest trading/tests/qa_round3_clients.py -q
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from trading.adapters.fetchers import aclose_loop_registry, loop_registry
from trading.api.routers import data as data_router
from trading.domain import Bar, Exchange
from trading.main import app


async def test_loop_registry_is_per_loop_singleton_and_closes():
    r1 = loop_registry()
    r2 = loop_registry()
    assert r1 is r2, "registry not reused within one event loop"
    moex = r1.get(Exchange.MOEX)
    client = moex._client
    assert not client.is_closed
    # a further lookup returns the identical registry + client (pool reused)
    assert loop_registry().get(Exchange.MOEX)._client is client

    await aclose_loop_registry()
    assert client.is_closed, "aclose_loop_registry did not close the httpx client"
    # after close the entry is gone → a fresh registry is built next time
    r3 = loop_registry()
    assert r3 is not r1
    await aclose_loop_registry()  # tidy up


def test_celery_run_isolated_closes_clients():
    from trading.tasks import _run_isolated

    captured: dict = {}

    async def _coro() -> int:
        reg = loop_registry()
        captured["client"] = reg.get(Exchange.BYBIT)._client
        return 1

    _run_isolated(_coro())
    assert captured["client"].is_closed, "Celery task leaked its httpx client pool"


async def test_data_endpoint_reuses_registry(monkeypatch):
    calls = {"n": 0}

    class _FakeReg:
        async def get_ohlcv(self, exchanges, symbol, timeframe, *, limit=500):
            calls["n"] += 1
            return [
                Bar(
                    timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                    open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0,
                )
            ]

    shared = _FakeReg()
    monkeypatch.setattr(data_router, "loop_registry", lambda: shared)
    with TestClient(app) as c:
        r1 = c.get("/api/v1/data/ohlcv/BTC")
        r2 = c.get("/api/v1/data/ohlcv/ETH")
    assert r1.status_code == 200 and r2.status_code == 200
    assert calls["n"] == 2, "endpoint did not consult the shared registry per request"
