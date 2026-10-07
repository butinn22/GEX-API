"""Tests for the match engine (commission + slippage models)."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading.application.backtest.match_engine import (
    MatchEngine,
    PercentCommissionModel,
    PercentSlippageModel,
)
from trading.domain import Bar, OrderIntent, OrderType, Quantity, Side


def _bar(open_: float) -> Bar:
    return Bar(datetime(2024, 1, 1, tzinfo=UTC), open_, open_, open_, open_, 0.0)


def test_buy_fill_with_commission_and_slippage():
    me = MatchEngine(PercentCommissionModel(0.001), PercentSlippageModel(0.01))
    intent = OrderIntent("X", Side.BUY, Quantity(10), OrderType.MARKET)
    fill = me.execute(intent, _bar(100.0), order_id="o1")
    assert fill.price == pytest.approx(101.0)  # 100 * 1.01
    assert fill.fee == pytest.approx(101.0 * 10 * 0.001)


def test_sell_slippage_reduces_price():
    me = MatchEngine(PercentCommissionModel(0.0), PercentSlippageModel(0.01))
    intent = OrderIntent("X", Side.SELL, Quantity(10), OrderType.MARKET)
    fill = me.execute(intent, _bar(100.0), order_id="o1")
    assert fill.price == pytest.approx(99.0)


def test_models_reject_bad_rates():
    with pytest.raises(ValueError):
        PercentCommissionModel(-0.01)
    with pytest.raises(ValueError):
        PercentSlippageModel(-0.01)
