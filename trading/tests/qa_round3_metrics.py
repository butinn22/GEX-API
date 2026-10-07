"""QA Round-3 — INDEPENDENT metrics/alerting verification (C16).

Proves all five instrumented counters move at their call-sites, that the
``broker_health`` task is beat-scheduled, and that the Prometheus/Alertmanager
compose wiring is valid YAML with the expected mount/alerting config.

Run explicitly::

    venv/Scripts/python.exe -m pytest trading/tests/qa_round3_metrics.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from trading.adapters.fetchers.registry import FetcherRegistry
from trading.api.routers import portfolio as portfolio_router
from trading.domain import DataFetchError, Exchange, Order, OrderStatus
from trading.main import app
from trading.observability import (
    BACKTEST_DURATION,
    DATA_FETCH_ERRORS,
    ORDERS_TOTAL,
    RATE_LIMIT_HITS,
    STRATEGY_SIGNALS,
)

REPO = Path(__file__).resolve().parents[2]


def _hist_count(h) -> float:
    for s in h.collect()[0].samples:
        if s.name.endswith("_count"):
            return s.value
    return 0.0


# ── ORDERS_TOTAL ──────────────────────────────────────────────────────
def test_orders_total_increments(monkeypatch):
    class _Fake:
        async def place_order(self, intent):
            return Order(id="qa-o1", symbol=intent.symbol, side=intent.side,
                         quantity=float(intent.quantity.value),
                         order_type=intent.order_type, status=OrderStatus.OPEN)

    async def _fake_broker(_e, _s, _svc):
        return _Fake()

    monkeypatch.setattr(portfolio_router, "_broker", _fake_broker)
    label = ORDERS_TOTAL.labels(exchange="bingx", side="buy", status="open")
    before = label._value.get()
    with TestClient(app) as c:
        r = c.post("/api/v1/orders", json={
            "exchange": "bingx", "symbol": "BTC-USDT", "side": "buy", "quantity": 0.1})
    assert r.status_code == 202
    assert label._value.get() == before + 1


# ── STRATEGY_SIGNALS (local-client signal path) ───────────────────────
def test_strategy_signals_increments_on_signal():
    label = STRATEGY_SIGNALS.labels(strategy="qa-strat", side="buy")
    before = label._value.get()
    with TestClient(app) as client:
        with client.websocket_connect("/ws/client") as ws:
            ws.send_json({"type": "handshake", "token": "test-client-token",
                          "client_id": "qa"})
            assert ws.receive_json()["type"] == "handshake_ack"
            ws.send_json({"type": "signal", "payload": {
                "symbol": "BTC-USDT", "side": "buy", "strength": 0.5,
                "quantity": 1, "strategy": "qa-strat", "reason": "qa"}})
            assert ws.receive_json()["type"] == "signal_ack"
    assert label._value.get() == before + 1


# ── DATA_FETCH_ERRORS (registry fallback) ─────────────────────────────
def test_data_fetch_errors_increments_on_fallback():
    class _Broken:
        exchange = Exchange.MOEX

        async def get_ohlcv(self, *a, **k):
            raise DataFetchError("boom")

    reg = FetcherRegistry()
    reg.register(_Broken())
    label = DATA_FETCH_ERRORS.labels(source="moex")
    before = label._value.get()
    import asyncio

    with pytest.raises(DataFetchError):
        asyncio.run(reg.get_ohlcv([Exchange.MOEX], "BTC", "1d"))
    assert label._value.get() == before + 1


# ── RATE_LIMIT_HITS (middleware reject) ───────────────────────────────
def test_rate_limit_hits_increments_on_reject():
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from trading.api.middleware import RateLimitMiddleware

    async def _ok(_r):
        return JSONResponse({"ok": True})

    a = Starlette(routes=[Route("/", _ok)])
    a.add_middleware(RateLimitMiddleware, requests=1, per_seconds=60.0, login_requests=1000)
    label = RATE_LIMIT_HITS.labels(scope="global")
    before = label._value.get()
    with TestClient(a) as c:
        c.get("/")
        assert c.get("/").status_code == 429
    assert label._value.get() == before + 1


# ── BACKTEST_DURATION ─────────────────────────────────────────────────
def test_backtest_duration_observed():
    with TestClient(app) as client:
        before = _hist_count(BACKTEST_DURATION)
        r = client.post("/api/v1/backtest", json={
            "strategy": "sma_crossover", "symbol": "SYNTH",
            "fast": 5, "slow": 20, "source": "synthetic"})
    assert r.status_code == 200, r.text
    assert _hist_count(BACKTEST_DURATION) == before + 1


# ── broker_health scheduled ───────────────────────────────────────────
def test_broker_health_is_beat_scheduled():
    from trading.tasks import celery_app

    tasks = [v["task"] for v in celery_app.conf.beat_schedule.values()]
    assert "trading.tasks.broker_health_task" in tasks


# ── compose / prometheus YAML validity ────────────────────────────────
def test_prometheus_yaml_has_rules_and_alerting():
    cfg = yaml.safe_load((REPO / "docker" / "prometheus.yml").read_text())
    assert cfg.get("rule_files"), "prometheus.yml has no rule_files"
    ams = cfg.get("alerting", {}).get("alertmanagers", [])
    targets = [t for am in ams for sc in am.get("static_configs", []) for t in sc.get("targets", [])]
    assert "alertmanager:9093" in targets


def test_prometheus_rules_and_alertmanager_parse():
    rules = yaml.safe_load((REPO / "docker" / "prometheus-rules.yml").read_text())
    assert rules.get("groups")
    am = yaml.safe_load((REPO / "docker" / "alertmanager.yml").read_text())
    assert isinstance(am, dict) and ami_has_route(am)


def ami_has_route(am: dict) -> bool:
    return "route" in am or "receivers" in am


def test_compose_mounts_rules_and_alertmanager_service():
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    services = compose["services"]
    assert "alertmanager" in services, "no alertmanager service"
    prom_vols = services["prometheus"].get("volumes", [])
    assert any("prometheus-rules.yml" in v and "/etc/prometheus/rules/" in v for v in prom_vols), \
        "prometheus does not mount the rules file into /etc/prometheus/rules/"
    am_vols = services["alertmanager"].get("volumes", [])
    assert any("alertmanager.yml" in v for v in am_vols)
