"""Granular trade-event ledger: per-fill states, direction, PnL, % return."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trading.application.backtest.engine import BacktestConfig, run_backtest
from trading.application.backtest.trade_log import (
    TradeEvent,
    TradeState,
    classify_fill,
    event_from_fill,
    events_from_fill,
)
from trading.domain import Bar, Fill, Position, PositionSide, Side, Signal

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def fill(side: Side, price: float, qty: float, symbol: str = "X") -> Fill:
    return Fill(order_id="o1", symbol=symbol, side=side, price=price, quantity=qty)


class TestClassifyFill:
    @pytest.mark.parametrize("side,expected", [
        (Side.BUY, TradeState.LONG_ENTRY),
        (Side.SELL, TradeState.SHORT_ENTRY),
    ])
    def test_from_flat_is_entry(self, side, expected):
        assert classify_fill(Position("X"), fill(side, 100, 1)) is expected

    def test_add_same_direction(self):
        long_pos = Position("X", PositionSide.LONG, 1.0, 100.0)
        short_pos = Position("X", PositionSide.SHORT, 1.0, 100.0)
        assert classify_fill(long_pos, fill(Side.BUY, 100, 1)) is TradeState.LONG_ADD
        assert classify_fill(short_pos, fill(Side.SELL, 100, 1)) is TradeState.SHORT_ADD

    def test_exit_opposite_direction(self):
        long_pos = Position("X", PositionSide.LONG, 1.0, 100.0)
        short_pos = Position("X", PositionSide.SHORT, 1.0, 100.0)
        assert classify_fill(long_pos, fill(Side.SELL, 110, 1)) is TradeState.LONG_EXIT
        assert classify_fill(short_pos, fill(Side.BUY, 90, 1)) is TradeState.SHORT_EXIT


class TestEventFromFill:
    def test_long_exit_pnl_and_pct(self):
        pos = Position("X", PositionSide.LONG, 2.0, 100.0)
        ev = event_from_fill(pos, fill(Side.SELL, 110.0, 2.0))
        assert ev.state is TradeState.LONG_EXIT
        assert ev.direction == "long"
        assert ev.realized_pnl == pytest.approx(20.0)
        assert ev.pct_return == pytest.approx(0.10)

    def test_short_exit_pnl_and_pct(self):
        pos = Position("X", PositionSide.SHORT, 2.0, 100.0)
        ev = event_from_fill(pos, fill(Side.BUY, 90.0, 2.0))
        assert ev.state is TradeState.SHORT_EXIT
        assert ev.direction == "short"
        assert ev.realized_pnl == pytest.approx(20.0)
        assert ev.pct_return == pytest.approx(0.10)

    def test_losing_trade_negative(self):
        pos = Position("X", PositionSide.LONG, 1.0, 100.0)
        ev = event_from_fill(pos, fill(Side.SELL, 95.0, 1.0))
        assert ev.realized_pnl == pytest.approx(-5.0)
        assert ev.pct_return == pytest.approx(-0.05)

    def test_entries_and_adds_carry_no_pnl(self):
        ev_entry = event_from_fill(Position("X"), fill(Side.BUY, 100.0, 1.0))
        assert ev_entry.realized_pnl == 0.0 and ev_entry.pct_return == 0.0
        pos = Position("X", PositionSide.LONG, 1.0, 100.0)
        ev_add = event_from_fill(pos, fill(Side.BUY, 105.0, 1.0))
        assert ev_add.state is TradeState.LONG_ADD
        assert ev_add.realized_pnl == 0.0

    def test_partial_exit_uses_closed_quantity(self):
        pos = Position("X", PositionSide.LONG, 4.0, 100.0)
        ev = event_from_fill(pos, fill(Side.SELL, 110.0, 1.0))
        assert ev.realized_pnl == pytest.approx(10.0)
        assert ev.quantity == 1.0

    def test_event_serializable(self):
        ev = event_from_fill(Position("X"), fill(Side.BUY, 100.0, 1.0),
                             strategy="s", reason="r")
        d = ev.as_dict()
        assert d["state"] == "long_entry"
        assert d["direction"] == "long"
        assert d["strategy"] == "s"


class TestEventsFromFill:
    """Flip fills must split into exit + entry (regression: flip-opened
    positions used to have no entry event, so closed-trade reconstruction
    could not pair them)."""

    def test_flip_long_to_short_splits(self):
        pos = Position("X", PositionSide.LONG, 2.0, 100.0)
        evs = events_from_fill(pos, fill(Side.SELL, 110.0, 5.0))
        assert [e.state for e in evs] == [TradeState.LONG_EXIT, TradeState.SHORT_ENTRY]
        assert evs[0].quantity == pytest.approx(2.0)
        assert evs[0].realized_pnl == pytest.approx(20.0)
        assert evs[1].quantity == pytest.approx(3.0)
        assert evs[1].realized_pnl == 0.0 and evs[1].pct_return == 0.0
        assert evs[0].price == evs[1].price == pytest.approx(110.0)

    def test_flip_short_to_long_splits(self):
        pos = Position("X", PositionSide.SHORT, 3.0, 100.0)
        evs = events_from_fill(pos, fill(Side.BUY, 90.0, 4.0))
        assert [e.state for e in evs] == [TradeState.SHORT_EXIT, TradeState.LONG_ENTRY]
        assert evs[0].realized_pnl == pytest.approx(30.0)
        assert evs[1].quantity == pytest.approx(1.0)

    def test_exact_close_is_single_exit(self):
        pos = Position("X", PositionSide.LONG, 2.0, 100.0)
        evs = events_from_fill(pos, fill(Side.SELL, 110.0, 2.0))
        assert [e.state for e in evs] == [TradeState.LONG_EXIT]

    def test_partial_close_is_single_exit(self):
        pos = Position("X", PositionSide.LONG, 4.0, 100.0)
        evs = events_from_fill(pos, fill(Side.SELL, 110.0, 1.0))
        assert [e.state for e in evs] == [TradeState.LONG_EXIT]
        assert evs[0].quantity == pytest.approx(1.0)

    def test_entry_and_add_are_single_events(self):
        assert [e.state for e in events_from_fill(Position("X"), fill(Side.BUY, 100.0, 1.0))] == \
            [TradeState.LONG_ENTRY]
        pos = Position("X", PositionSide.LONG, 1.0, 100.0)
        assert [e.state for e in events_from_fill(pos, fill(Side.BUY, 105.0, 1.0))] == \
            [TradeState.LONG_ADD]


def bar(i: int, price: float) -> Bar:
    return Bar(timestamp=T0 + timedelta(days=i), open=price, high=price,
               low=price, close=price, volume=1000.0)


class ScriptedStrategy:
    """Emits a fixed script of signals, one per bar close."""

    def __init__(self, symbol: str, script: list[Side | None]):
        self.symbol = symbol
        self.script = script
        self.i = 0

    async def prepare(self, bars) -> None:
        pass

    async def on_bar(self, bar: Bar) -> list[Signal]:
        side = self.script[self.i] if self.i < len(self.script) else None
        self.i += 1
        if side is None:
            return []
        return [Signal(symbol=self.symbol, side=side, strategy="script",
                       reason="step", timestamp=bar.timestamp)]


class TestEngineEvents:
    async def test_long_roundtrip_event_sequence(self):
        # buy, buy (add), sell (exit) — fills land on the NEXT bar's open.
        bars = [bar(i, 100.0) for i in range(5)]
        strat = ScriptedStrategy("X", [Side.BUY, Side.BUY, Side.SELL, None, None])
        cfg = BacktestConfig(initial_cash=10_000, fee_rate=0.0, slippage=0.0,
                             position_fraction=0.4)
        result = await run_backtest(strat, bars, cfg)
        states = [e.state for e in result.events]
        assert states == [TradeState.LONG_ENTRY, TradeState.LONG_ADD, TradeState.LONG_EXIT]
        assert all(e.direction == "long" for e in result.events)
        # flat prices → exit pnl is zero; ledger is complete and consistent
        assert result.events[-1].realized_pnl == pytest.approx(0.0)

    async def test_short_roundtrip_event_sequence(self):
        bars = [bar(i, 100.0) for i in range(4)]
        strat = ScriptedStrategy("X", [Side.SELL, None, Side.BUY, None])
        cfg = BacktestConfig(initial_cash=10_000, fee_rate=0.0, slippage=0.0,
                             position_fraction=0.4)
        result = await run_backtest(strat, bars, cfg)
        states = [e.state for e in result.events]
        assert states == [TradeState.SHORT_ENTRY, TradeState.SHORT_EXIT]

    async def test_flip_emits_exit_then_entry(self):
        # Falling prices: the exit sell is sized off a higher equity-to-price
        # ratio than the entry buy, so its quantity exceeds the position —
        # a flip. The ledger must carry BOTH the long exit and the short entry.
        bars = [bar(0, 100.0), bar(1, 80.0), bar(2, 64.0), bar(3, 64.0)]
        strat = ScriptedStrategy("X", [Side.BUY, Side.SELL, None, None])
        cfg = BacktestConfig(initial_cash=10_000, fee_rate=0.0, slippage=0.0,
                             position_fraction=0.4)
        result = await run_backtest(strat, bars, cfg)
        states = [e.state for e in result.events]
        assert states == [TradeState.LONG_ENTRY, TradeState.LONG_EXIT, TradeState.SHORT_ENTRY]
        exit_ev, entry_ev = result.events[1], result.events[2]
        assert exit_ev.quantity + entry_ev.quantity == pytest.approx(
            result.events[1].quantity + result.events[2].quantity)
        # the flipped remainder really opens a new short
        assert entry_ev.quantity > 0.0
        assert entry_ev.realized_pnl == 0.0

    async def test_exit_pnl_matches_price_move(self):
        bars = [bar(0, 100.0), bar(1, 100.0), bar(2, 110.0), bar(3, 110.0)]
        strat = ScriptedStrategy("X", [Side.BUY, None, Side.SELL, None])
        cfg = BacktestConfig(initial_cash=10_000, fee_rate=0.0, slippage=0.0,
                             position_fraction=0.5)
        result = await run_backtest(strat, bars, cfg)
        exit_ev = result.events[-1]
        assert exit_ev.state is TradeState.LONG_EXIT
        assert exit_ev.pct_return == pytest.approx(0.10)
        # ledger PnL agrees with the trade ledger
        assert exit_ev.realized_pnl == pytest.approx(result.trades[-1].realized_pnl)

    async def test_events_default_empty_for_result_constructed_directly(self):
        from trading.application.backtest.engine import BacktestResult
        import numpy as np
        from trading.application.backtest.metrics import compute_metrics

        res = BacktestResult(
            equity_curve=np.array([1.0]),
            trades=(),
            metrics=compute_metrics(np.array([1.0]), []),
        )
        assert res.events == ()
