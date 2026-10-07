"""Round-2 metrics wiring: the counters are actually incremented at their sites.

``test_observability.py`` proves the exporter works and the counters move when
touched directly; these tests prove the *call sites* exist (an idle counter that
nothing increments is the A6 finding this closes).
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from trading.adapters.fetchers.http_util import get_json
from trading.api.routers import portfolio as portfolio_router
from trading.domain import DataFetchError, Order, OrderStatus, OrderType, Side
from trading.main import app
from trading.observability import BACKTEST_DURATION, DATA_FETCH_ERRORS, ORDERS_TOTAL


def _hist_count(histogram) -> float:
    for sample in histogram.collect()[0].samples:
        if sample.name.endswith("_count"):
            return sample.value
    return 0.0


# ── BACKTEST_DURATION.observe around run_backtest ─────────────────────
def test_single_symbol_backtest_observes_duration():
    with TestClient(app) as client:
        before = _hist_count(BACKTEST_DURATION)
        r = client.post("/api/v1/backtest", json={
            "strategy": "sma_crossover", "symbol": "SYNTH",
            "fast": 5, "slow": 20, "source": "synthetic",
        })
        assert r.status_code == 200, r.text
        assert _hist_count(BACKTEST_DURATION) == before + 1


# ── ORDERS_TOTAL.inc after a successful place_order ───────────────────
def test_place_order_increments_orders_total(monkeypatch):
    class _FakeBroker:
        async def place_order(self, intent):
            return Order(
                id="fake-order-1", symbol=intent.symbol, side=intent.side,
                quantity=float(intent.quantity.value), order_type=intent.order_type,
                status=OrderStatus.OPEN,
            )

    async def _fake_broker(_exchange, _session, _svc):
        return _FakeBroker()

    monkeypatch.setattr(portfolio_router, "_broker", _fake_broker)
    label = ORDERS_TOTAL.labels(exchange="bingx", side="buy", status="open")
    with TestClient(app) as client:
        before = label._value.get()
        r = client.post("/api/v1/orders", json={
            "exchange": "bingx", "symbol": "BTC-USDT", "side": "buy", "quantity": 0.1,
        })
        assert r.status_code == 202, r.text
        assert label._value.get() == before + 1


# ── DATA_FETCH_ERRORS.inc in http_util.get_json non-200 ───────────────
class _Resp500:
    status_code = 500


class _Client500:
    async def get(self, *_a, **_k):
        return _Resp500()


def test_http_util_counts_non_200():
    label = DATA_FETCH_ERRORS.labels(source="example.com")
    before = label._value.get()
    with pytest.raises(DataFetchError):
        asyncio.run(get_json(_Client500(), "https://example.com/path"))
    assert label._value.get() == before + 1


# ── broker_health is scheduled (else TradingBrokerDown can never fire) ─
def test_broker_health_is_scheduled():
    from trading.tasks import broker_health_task, celery_app

    schedule = celery_app.conf.beat_schedule
    assert any(v["task"] == "trading.tasks.broker_health_task" for v in schedule.values())
    assert broker_health_task.name == "trading.tasks.broker_health_task"
