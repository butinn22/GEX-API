"""Tests for the trading domain ring (pure, no external deps)."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading.domain import (
    Account,
    Bar,
    BookLevel,
    BrokerError,
    DataFetchError,
    Fill,
    InsufficientFundsError,
    InvalidStateError,
    Money,
    Order,
    OrderBook,
    OrderIntent,
    OrderRejectedError,
    OrderStatus,
    OrderType,
    Portfolio,
    Position,
    PositionSide,
    Price,
    Quantity,
    RateLimitExceededError,
    Side,
    Signal,
    StrategyError,
    TradingError,
)

TS = datetime(2024, 1, 1, tzinfo=UTC)


# ── Enums ─────────────────────────────────────────────────────────────


def test_side_sign():
    assert Side.BUY.sign == 1
    assert Side.SELL.sign == -1


def test_position_side_sign():
    assert PositionSide.LONG.sign == 1
    assert PositionSide.SHORT.sign == -1
    assert PositionSide.FLAT.sign == 0


def test_order_status_terminal_and_active():
    assert OrderStatus.FILLED.is_terminal
    assert OrderStatus.CANCELLED.is_terminal
    assert OrderStatus.REJECTED.is_terminal
    assert OrderStatus.EXPIRED.is_terminal
    assert not OrderStatus.OPEN.is_terminal
    assert not OrderStatus.PARTIAL.is_terminal
    assert OrderStatus.PENDING.is_active
    assert OrderStatus.OPEN.is_active
    assert OrderStatus.PARTIAL.is_active
    assert not OrderStatus.FILLED.is_active


# ── Value objects ─────────────────────────────────────────────────────


def test_price_rounds_to_tick():
    assert Price(10.123, 0.01).value == pytest.approx(10.12)
    assert Price(10.127, 0.01).value == pytest.approx(10.13)
    assert Price(10.123).value == pytest.approx(10.123)  # no tick → untouched


def test_price_rounds_half_up():
    # 10.125 / 0.01 = 1012.5 → ties round away from zero → 10.13
    assert Price(10.125, 0.01).value == pytest.approx(10.13)


def test_price_rejects_negative():
    with pytest.raises(ValueError):
        Price(-1.0, 0.01)


def test_quantity_rounds_to_lot():
    assert Quantity(0.07, 0.1).value == pytest.approx(0.1)
    assert Quantity(0.25, 0.1).value == pytest.approx(0.3)


def test_money_arithmetic_and_currency_check():
    assert Money(10, "USD") + Money(5, "USD") == Money(15, "USD")
    assert Money(10, "USD") - Money(3, "USD") == Money(7, "USD")
    assert (Money(10, "USD") * 2).amount == pytest.approx(20)
    assert (-Money(5, "USD")).amount == pytest.approx(-5)
    with pytest.raises(ValueError):
        Money(10, "USD") + Money(5, "EUR")


# ── Market data ───────────────────────────────────────────────────────


def test_bar_validation():
    b = Bar(TS, open=1, high=3, low=0.5, close=2, volume=100)
    assert b.high == 3 and b.low == 0.5
    with pytest.raises(ValueError):
        Bar(TS, open=1, high=0.5, low=0.5, close=1)  # high < open


def test_bar_ts_ms():
    b = Bar(datetime(1970, 1, 1, tzinfo=UTC), 1, 1, 1, 1)
    assert b.ts_ms == 0


def test_orderbook_levels():
    book = OrderBook(
        bids=(BookLevel(100, 2), BookLevel(99, 3)),
        asks=(BookLevel(101, 1), BookLevel(102, 2)),
    )
    assert book.best_bid == 100
    assert book.best_ask == 101
    assert book.mid_price == pytest.approx(100.5)
    assert book.spread == pytest.approx(1.0)


def test_empty_orderbook():
    book = OrderBook()
    assert book.best_bid is None
    assert book.mid_price is None
    assert book.spread is None


# ── Signal / OrderIntent ──────────────────────────────────────────────


def test_signal_validation():
    s = Signal("AAPL", Side.BUY, "sma_cross", "golden cross", strength=0.8)
    assert s.strength == 0.8
    with pytest.raises(ValueError):
        Signal("AAPL", Side.BUY, "s", "r", strength=1.5)


def test_order_intent_requires_prices():
    with pytest.raises(ValueError):
        OrderIntent("AAPL", Side.BUY, Quantity(10), OrderType.LIMIT)
    with pytest.raises(ValueError):
        OrderIntent("AAPL", Side.BUY, Quantity(10), OrderType.STOP_LIMIT)
    with pytest.raises(ValueError):
        OrderIntent("AAPL", Side.BUY, Quantity(0), OrderType.MARKET)


def test_order_intent_to_order():
    intent = OrderIntent(
        "AAPL", Side.BUY, Quantity(10), OrderType.LIMIT,
        limit_price=Price(100, 0.01), strategy="sma_cross", reason="entry",
    )
    o = intent.to_order("o1")
    assert o.id == "o1"
    assert o.status is OrderStatus.PENDING
    assert o.limit_price == pytest.approx(100)
    assert o.strategy == "sma_cross"


# ── Order state machine ───────────────────────────────────────────────


def _mk_order() -> Order:
    return OrderIntent(
        "AAPL", Side.BUY, Quantity(10), OrderType.MARKET, strategy="s", reason="r"
    ).to_order("o1")


def test_order_lifecycle():
    o = _mk_order()
    assert o.status is OrderStatus.PENDING
    o.mark_open()
    assert o.status is OrderStatus.OPEN
    o.apply_fill(Fill("o1", "AAPL", Side.BUY, price=100, quantity=4))
    assert o.status is OrderStatus.PARTIAL
    assert o.filled_quantity == pytest.approx(4)
    o.apply_fill(Fill("o1", "AAPL", Side.BUY, price=101, quantity=6))
    assert o.status is OrderStatus.FILLED
    assert o.average_fill_price == pytest.approx((100 * 4 + 101 * 6) / 10)


def test_order_illegal_transition():
    o = _mk_order()
    o.mark_open()
    with pytest.raises(InvalidStateError):
        o.mark_open()  # OPEN → OPEN is not a legal transition


def test_order_cannot_fill_when_terminal():
    o = _mk_order()
    o.mark_open()
    o.mark_cancelled()
    with pytest.raises(InvalidStateError):
        o.apply_fill(Fill("o1", "AAPL", Side.BUY, 100, 1))


def test_order_fills_cannot_exceed_quantity():
    o = _mk_order()
    o.mark_open()
    with pytest.raises(InvalidStateError):
        o.apply_fill(Fill("o1", "AAPL", Side.BUY, 100, 11))


def test_order_overfill_does_not_corrupt_filled_quantity():
    """A rejected over-fill must leave the order exactly as it was."""
    o = _mk_order()
    o.mark_open()
    o.apply_fill(Fill("o1", "AAPL", Side.BUY, 100, 4))
    with pytest.raises(InvalidStateError):
        o.apply_fill(Fill("o1", "AAPL", Side.BUY, 100, 11))
    assert o.filled_quantity == pytest.approx(4)
    assert o.status is OrderStatus.PARTIAL


def test_order_fill_requires_matching_symbol_and_side():
    o = _mk_order()
    o.mark_open()
    with pytest.raises(InvalidStateError):
        o.apply_fill(Fill("o1", "MSFT", Side.BUY, 100, 1))
    with pytest.raises(InvalidStateError):
        o.apply_fill(Fill("o1", "AAPL", Side.SELL, 100, 1))


def test_order_fill_from_pending_advances_through_open():
    o = _mk_order()  # PENDING, never marked open
    o.apply_fill(Fill("o1", "AAPL", Side.BUY, 100, 10))
    assert o.status is OrderStatus.FILLED


# ── Position ──────────────────────────────────────────────────────────


def _buy(order_id, symbol, price, qty, fee=0.0) -> Fill:
    return Fill(order_id, symbol, Side.BUY, price, qty, fee=fee)


def _sell(order_id, symbol, price, qty, fee=0.0) -> Fill:
    return Fill(order_id, symbol, Side.SELL, price, qty, fee=fee)


def test_open_long():
    p = Position("AAPL").apply_fill(_buy("o", "AAPL", 100, 10))
    assert p.side is PositionSide.LONG
    assert p.quantity == pytest.approx(10)
    assert p.average_entry_price == pytest.approx(100)


def test_add_to_long_averages_cost():
    p = Position("AAPL", PositionSide.LONG, 10, 100)
    p2 = p.apply_fill(_buy("o", "AAPL", 110, 5))
    assert p2.quantity == pytest.approx(15)
    assert p2.average_entry_price == pytest.approx((100 * 10 + 110 * 5) / 15)


def test_partial_reduce_long():
    p = Position("AAPL", PositionSide.LONG, 10, 100)
    p2 = p.apply_fill(_sell("o", "AAPL", 120, 5))
    assert p2.side is PositionSide.LONG
    assert p2.quantity == pytest.approx(5)
    assert p2.average_entry_price == pytest.approx(100)
    assert p2.realized_pnl == pytest.approx(100)


def test_full_close_long():
    p = Position("AAPL", PositionSide.LONG, 10, 100)
    p2 = p.apply_fill(_sell("o", "AAPL", 120, 10))
    assert p2.side is PositionSide.FLAT
    assert p2.quantity == 0
    assert p2.realized_pnl == pytest.approx(200)


def test_flip_long_to_short():
    p = Position("AAPL", PositionSide.LONG, 10, 100)
    p2 = p.apply_fill(_sell("o", "AAPL", 110, 15))
    assert p2.side is PositionSide.SHORT
    assert p2.quantity == pytest.approx(5)
    assert p2.average_entry_price == pytest.approx(110)
    assert p2.realized_pnl == pytest.approx(100)  # closed 10 @ (110-100)


def test_short_cover_profit():
    p = Position("AAPL", PositionSide.SHORT, 10, 100)
    p2 = p.apply_fill(_buy("o", "AAPL", 90, 10))
    assert p2.side is PositionSide.FLAT
    assert p2.realized_pnl == pytest.approx(100)  # (100-90)*10


def test_fee_reduces_realized():
    p = Position("AAPL", PositionSide.LONG, 10, 100)
    p2 = p.apply_fill(_sell("o", "AAPL", 120, 5, fee=1.5))
    assert p2.realized_pnl == pytest.approx(100 - 1.5)


def test_position_unrealized():
    p = Position("AAPL", PositionSide.LONG, 10, 100)
    assert p.unrealized_pnl(105) == pytest.approx(50)
    short = Position("AAPL", PositionSide.SHORT, 10, 100)
    assert short.unrealized_pnl(95) == pytest.approx(50)


def test_position_rejects_mismatched_symbol():
    p = Position("AAPL", PositionSide.LONG, 10, 100)
    with pytest.raises(ValueError):
        p.apply_fill(_buy("o", "MSFT", 100, 1))


# ── Portfolio ─────────────────────────────────────────────────────────


def test_portfolio_cash_on_buy_and_sell():
    pf = Portfolio(cash=10000).apply_fill(_buy("o", "AAPL", 100, 10))
    assert pf.cash == pytest.approx(9000)
    pf2 = pf.apply_fill(_sell("o", "AAPL", 110, 10))
    assert pf2.cash == pytest.approx(9000 + 1100)
    assert pf2.realized_pnl == pytest.approx(100)
    assert pf2.position_for("AAPL").side is PositionSide.FLAT


def test_portfolio_equity_and_unrealized():
    pf = Portfolio(cash=10000).apply_fill(_buy("o", "AAPL", 100, 10))
    # cash 9000, LONG 10 @ 100; mark 105 → unrealized 50, market value 1050, equity 10050
    assert pf.unrealized_pnl({"AAPL": 105}) == pytest.approx(50)
    assert pf.equity({"AAPL": 105}) == pytest.approx(10050)


def test_portfolio_short_equity():
    pf = Portfolio(cash=10000).apply_fill(_sell("o", "AAPL", 100, 10))
    # cash 11000, SHORT 10 @ 100; mark 90 → market value -900, equity 10100
    assert pf.cash == pytest.approx(11000)
    assert pf.equity({"AAPL": 90}) == pytest.approx(10100)
    assert pf.unrealized_pnl({"AAPL": 90}) == pytest.approx(100)


def test_portfolio_realized_survives_reopen():
    pf = Portfolio(cash=10000)
    pf = pf.apply_fill(_buy("o", "AAPL", 100, 10))
    pf = pf.apply_fill(_sell("o", "AAPL", 110, 10))  # close → realized 100
    assert pf.realized_pnl == pytest.approx(100)
    pf = pf.apply_fill(_buy("o", "AAPL", 105, 5))  # reopen
    assert pf.realized_pnl == pytest.approx(100)  # unchanged
    assert pf.position_for("AAPL").side is PositionSide.LONG


def test_portfolio_fee_reduces_cash():
    pf = Portfolio(cash=10000).apply_fill(_buy("o", "AAPL", 100, 10, fee=2.0))
    assert pf.cash == pytest.approx(9000 - 2.0)


# ── Errors hierarchy ──────────────────────────────────────────────────


def test_error_hierarchy():
    assert issubclass(DataFetchError, TradingError)
    assert issubclass(BrokerError, TradingError)
    assert issubclass(OrderRejectedError, BrokerError)
    assert issubclass(InsufficientFundsError, BrokerError)
    assert issubclass(RateLimitExceededError, TradingError)
    assert issubclass(StrategyError, TradingError)
    assert issubclass(InvalidStateError, TradingError)


# ── Account ───────────────────────────────────────────────────────────


def test_account():
    a = Account("acct1", cash=1000, buying_power=2000, margin_used=100)
    assert a.cash == 1000
    assert a.currency == "USD"
