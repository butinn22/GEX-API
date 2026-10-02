"""Tests for Prometheus metrics and the /metrics endpoint."""
from __future__ import annotations

from fastapi.testclient import TestClient

from trading.main import app
from trading.observability import ORDERS_TOTAL, STRATEGY_SIGNALS


def test_metrics_endpoint():
    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "trading_orders_total" in r.text
        assert "trading_backtest_duration_seconds" in r.text


def test_counters_increment():
    before = ORDERS_TOTAL.labels(exchange="bingx", side="buy", status="open")._value.get()
    ORDERS_TOTAL.labels(exchange="bingx", side="buy", status="open").inc()
    after = ORDERS_TOTAL.labels(exchange="bingx", side="buy", status="open")._value.get()
    assert after == before + 1
    STRATEGY_SIGNALS.labels(strategy="sma_crossover", side="buy").inc(2)
    assert STRATEGY_SIGNALS.labels(strategy="sma_crossover", side="buy")._value.get() >= 2


def test_correlation_id_header():
    with TestClient(app) as client:
        r = client.get("/health")
        assert "X-Correlation-Id" in r.headers
