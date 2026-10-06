"""Tests for the event-driven backtest engine (no-lookahead, fees, sizing)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from trading.application.backtest.engine import BacktestConfig, run_backtest
from trading.application.strategies.buy_and_hold import BuyAndHold
from trading.application.strategies.sma_crossover import SmaCrossover
from trading.domain import Bar, Side, Signal


def bar(i: int, open_: float, close: float) -> Bar:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
    return Bar(
        timestamp=ts,
        open=open_,
        high=max(open_, close),
        low=min(open_, close),
        close=close,
        volume=1000.0,
    )


def test_buy_and_hold_no_lookahead_and_equity():
    cfg = BacktestConfig(initial_cash=1000.0, fee_rate=0.0, slippage=0.0,
                         position_fraction=1.0)
    bars = [bar(0, 100, 100), bar(1, 100, 110), bar(2, 110, 120)]
    result = asyncio_run(BuyAndHold("AAPL"), bars, cfg)
    # signal at bar0 → filled at bar1 open (equity[0] still pure cash = no lookahead)
    assert result.equity_curve[0] == pytest.approx(1000.0)
    assert result.equity_curve[1] == pytest.approx(1100.0)  # 10 units @ 110
    assert result.equity_curve[2] == pytest.approx(1200.0)
    assert result.metrics.total_return == pytest.approx(0.2)
    assert result.metrics.n_trades == 0  # held to the end, nothing closed


def test_fees_reduce_final_equity():
    zero = asyncio_run(
        BuyAndHold("AAPL"),
        [bar(0, 100, 100), bar(1, 100, 110)],
        BacktestConfig(initial_cash=1000, fee_rate=0.0, slippage=0.0, position_fraction=1.0),
    )
    with_fees = asyncio_run(
        BuyAndHold("AAPL"),
        [bar(0, 100, 100), bar(1, 100, 110)],
        BacktestConfig(initial_cash=1000, fee_rate=0.01, slippage=0.0, position_fraction=1.0),
    )
    assert with_fees.equity_curve[-1] < zero.equity_curve[-1]
    # fees on an open are costs, NOT a trade — nothing was closed
    assert with_fees.metrics.n_trades == 0
    assert zero.metrics.n_trades == 0


def test_trade_pnl_charges_both_legs_of_fees():
    """A closed trade must pay the entry AND the exit fee.

    Charging only the closing fill's fee used to make ``realized_pnl`` — and
    therefore profit factor and win rate — optimistic by roughly one side of
    costs, which is exactly the metric this platform optimises on.
    """
    class OneRoundTrip:
        name = "one_round_trip"

        def __init__(self):
            self._i = 0

        async def prepare(self, bars):
            return None

        async def on_bar(self, bar):
            self._i += 1
            if self._i == 1:                      # BUY on the first close
                return [Signal(symbol="AAPL", side=Side.BUY, strategy=self.name,
                               reason="enter")]
            if self._i == 3:                      # SELL two bars later
                return [Signal(symbol="AAPL", side=Side.SELL, strategy=self.name,
                               reason="exit")]
            return []

        async def on_tick(self, tick):
            return []

        async def generate_signals(self, bars):
            return []

        async def shutdown(self):
            return None

    fee = 0.01
    bars = [bar(0, 100, 100), bar(1, 100, 110), bar(2, 110, 120), bar(3, 120, 130)]
    cfg = BacktestConfig(initial_cash=1000.0, fee_rate=fee, slippage=0.0,
                         position_fraction=1.0)
    res = asyncio_run(OneRoundTrip(), bars, cfg)
    assert len(res.trades) == 1
    t = res.trades[0]
    # filled at the next bar's open: 100 in, 120 out
    assert t.entry_price == pytest.approx(100.0)
    assert t.exit_price == pytest.approx(120.0)
    gross = (t.exit_price - t.entry_price) * t.quantity
    entry_fee = t.entry_price * t.quantity * fee
    exit_fee = t.exit_price * t.quantity * fee
    assert t.realized_pnl == pytest.approx(gross - entry_fee - exit_fee, abs=1e-9)

    # the same run without fees must earn strictly more (and buy more: the fee
    # no longer eats into the equity that sizes the position)
    free = asyncio_run(OneRoundTrip(), bars,
                       BacktestConfig(initial_cash=1000.0, fee_rate=0.0, slippage=0.0,
                                      position_fraction=1.0))
    f = free.trades[0]
    assert f.realized_pnl == pytest.approx((f.exit_price - f.entry_price) * f.quantity,
                                           abs=1e-9)
    assert f.realized_pnl > t.realized_pnl
    # both runs win their single trade, so profit factor is inf either way —
    # the difference is in the money, not the ratio.
    assert free.metrics.total_return > res.metrics.total_return


def test_sma_crossover_produces_trades():
    # Down-trend → up-trend → down-trend: fast SMA crosses above (BUY) then below (SELL).
    prices = np.concatenate([
        np.linspace(100, 80, 30),
        np.linspace(80, 130, 30),
        np.linspace(130, 90, 30),
    ])
    bars = [bar(i, p, p) for i, p in enumerate(prices)]
    result = asyncio_run(SmaCrossover("X", fast=5, slow=20), bars,
                         BacktestConfig(initial_cash=10000, fee_rate=0.001, slippage=0.0005))
    assert result.metrics.n_trades >= 1
    assert np.all(np.isfinite(result.equity_curve))
    assert result.metrics.max_drawdown >= 0.0
    assert len(result.trades) == result.metrics.n_trades


def test_empty_bars_raises():
    with pytest.raises(ValueError):
        asyncio_run(BuyAndHold("AAPL"), [], BacktestConfig())


def asyncio_run(strategy, bars, cfg):
    import asyncio
    return asyncio.run(run_backtest(strategy, bars, cfg))
