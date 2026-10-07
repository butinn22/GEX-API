"""Tests for vectorized backtest, walk-forward, PnL methods, smart orders, reporter."""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trading.application.backtest.engine import BacktestConfig, run_backtest
from trading.application.backtest.pnl import realized_pnl_avg, realized_pnl_fifo, realized_pnl_lifo
from trading.application.backtest.reporter import BacktestReporter
from trading.application.backtest.vectorized import run_backtest_vectorized
from trading.application.backtest.walk_forward import parameter_sensitivity, walk_forward
from trading.application.smart_orders import SmartOrderPlanner
from trading.application.strategies.buy_and_hold import BuyAndHold
from trading.application.strategies.sma_crossover import SmaCrossover
from trading.domain import Bar, Fill, Side


def _bar(i: int, close: float) -> Bar:
    ts = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)
    return Bar(timestamp=ts, open=close, high=close, low=close, close=close, volume=1.0)


def _rising(n: int) -> list[Bar]:
    return [_bar(i, 100.0 + i) for i in range(n)]


def test_vectorized_buy_and_hold_positive():
    bars = _rising(50)
    r = asyncio.run(run_backtest_vectorized(
        BuyAndHold("X"), bars,
        BacktestConfig(initial_cash=1000, fee_rate=0.0, slippage=0.0, position_fraction=1.0),
    ))
    assert r.metrics.total_return > 0
    assert r.metrics.n_trades == 0  # vectorized path has no trade ledger
    assert np.all(np.isfinite(r.equity_curve))


def test_walk_forward_windows():
    bars = _rising(80)
    results = asyncio.run(walk_forward(lambda: BuyAndHold("X"), bars, n_windows=4))
    assert 2 <= len(results) <= 3
    assert all(r.metrics.n_periods > 0 for r in results)


def test_parameter_sensitivity_grid():
    bars = _rising(60)
    out = asyncio.run(parameter_sensitivity(
        lambda s: SmaCrossover("X", fast=5, slow=int(s)), bars, "slow",
        [10, 20, 30], metric="sharpe",
    ))
    assert len(out) == 3
    assert all(isinstance(v, float) for v, _ in out)


def test_pnl_methods_differ():
    fills = [
        Fill("o", "X", Side.BUY, 100, 10),
        Fill("o", "X", Side.BUY, 110, 10),
        Fill("o", "X", Side.SELL, 120, 10),
    ]
    assert realized_pnl_fifo(fills) == pytest.approx(200)  # closes the 100 lot
    assert realized_pnl_lifo(fills) == pytest.approx(100)  # closes the 110 lot
    assert realized_pnl_avg(fills) == pytest.approx(150)  # avg cost 105


def test_twap_equal_slices():
    slices = SmartOrderPlanner().twap(100, duration_seconds=60, interval_seconds=10)
    assert len(slices) == 6
    assert all(s == pytest.approx(100 / 6) for s in slices)


def test_vwap_proportional_to_profile():
    slices = SmartOrderPlanner().vwap(100, volume_profile=[1, 1, 2])
    assert sum(slices) == pytest.approx(100)
    assert slices[2] > slices[0]  # more volume → bigger slice


def test_iceberg_chunks():
    assert SmartOrderPlanner().iceberg(10, display_qty=3) == [3, 3, 3, 1]


def test_reporter_json_and_html():
    r = asyncio.run(run_backtest(BuyAndHold("X"), _rising(50), BacktestConfig()))
    rep = BacktestReporter(r, strategy="buy_and_hold", symbol="X")
    data = json.loads(rep.to_json())
    assert data["metrics"]["sharpe"] is not None
    assert "calmar" in data["metrics"]
    html = rep.to_html()
    assert "buy_and_hold" in html and "max_drawdown" in html


def test_reporter_pdf():
    import os
    import tempfile

    r = asyncio.run(run_backtest(BuyAndHold("X"), _rising(50), BacktestConfig()))
    rep = BacktestReporter(r, strategy="buy_and_hold", symbol="X")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "report.pdf")
        rep.to_pdf(path)
        with open(path, "rb") as f:
            assert f.read(5) == b"%PDF-"
