"""Prometheus metrics for the trading context.

Exposed at ``/metrics``. Labels are bounded and low-cardinality on purpose
(exchange, side, status, strategy, scope) so the cardinality stays sane.
"""
from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

__all__ = [
    "ORDERS_TOTAL",
    "STRATEGY_SIGNALS",
    "DATA_FETCH_ERRORS",
    "RATE_LIMIT_HITS",
    "BACKTEST_DURATION",
    "BROKER_STATUS",
    "metrics_response",
]

ORDERS_TOTAL = Counter(
    "trading_orders_total", "Orders placed", ["exchange", "side", "status"]
)
STRATEGY_SIGNALS = Counter(
    "trading_strategy_signals_total", "Signals generated", ["strategy", "side"]
)
DATA_FETCH_ERRORS = Counter(
    "trading_data_fetch_errors_total", "Market-data fetch errors", ["source"]
)
RATE_LIMIT_HITS = Counter(
    "trading_rate_limit_hits_total", "Rate-limit rejections", ["scope"]
)
BACKTEST_DURATION = Histogram(
    "trading_backtest_duration_seconds", "Backtest wall-clock duration",
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 60.0),
)
BROKER_STATUS = Gauge(
    "trading_broker_status", "Broker reachable (1) / unreachable (0)", ["exchange"]
)


def metrics_response():
    """Return a (content, media_type) pair ready for a FastAPI Response."""
    from fastapi.responses import Response

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
