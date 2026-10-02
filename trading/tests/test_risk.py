"""Tests for position sizing and risk management."""
from __future__ import annotations

import pytest

from trading.application.risk import (
    FixedFractionSizer,
    KellySizer,
    PercentRiskSizer,
    RiskManager,
)
from trading.domain import OrderIntent, OrderType, Portfolio, Quantity, Side


def test_fixed_fraction_sizer():
    s = FixedFractionSizer(0.5)
    assert s.size(equity=1000, price=100) == pytest.approx(5.0)


def test_percent_risk_sizer():
    s = PercentRiskSizer(0.01)
    qty = s.size(equity=1000, price=100, stop_price=90)
    assert qty == pytest.approx(1.0)  # risk 10 at distance 10 → 1 unit
    assert s.size(equity=1000, price=100, stop_price=None) == 0.0


def test_kelly_sizer():
    s = KellySizer(win_rate=0.6, win_loss_ratio=2.0, fraction=0.5, max_fraction=0.25)
    # f* = 0.6 - 0.4/2 = 0.4 → half-Kelly 0.2
    assert s.size(equity=1000, price=100) == pytest.approx(2.0)


def test_risk_manager_approves_within_limits():
    rm = RiskManager(max_position_pct=0.5, max_drawdown=0.25)
    intent = OrderIntent("AAPL", Side.BUY, Quantity(1), OrderType.MARKET)
    pf = Portfolio(cash=1000)
    ok, _ = rm.approve(intent, pf, mark=100, current_equity=1000)
    assert ok is True  # notional 100 < 500 limit


def test_risk_manager_rejects_drawdown():
    rm = RiskManager(max_position_pct=0.5, max_drawdown=0.2)
    rm.update_equity(1000)
    intent = OrderIntent("AAPL", Side.BUY, Quantity(1), OrderType.MARKET)
    ok, reason = rm.approve(intent, Portfolio(cash=1000), mark=100, current_equity=790)
    assert ok is False and "drawdown" in reason


def test_risk_manager_rejects_oversized_position():
    rm = RiskManager(max_position_pct=0.1, max_drawdown=0.5)
    intent = OrderIntent("AAPL", Side.BUY, Quantity(10), OrderType.MARKET)
    ok, reason = rm.approve(intent, Portfolio(cash=1000), mark=100, current_equity=1000)
    assert ok is False and "notional" in reason


def test_sizers_reject_bad_params():
    with pytest.raises(ValueError):
        FixedFractionSizer(1.5)
    with pytest.raises(ValueError):
        KellySizer(win_rate=1.5, win_loss_ratio=1.0)
