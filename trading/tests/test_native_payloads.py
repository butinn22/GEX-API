"""Native exchange order payload mapping (local signal client API)."""
from __future__ import annotations

import json
import uuid

import pytest

from trading.application.native_payloads import (
    SignalParseError,
    native_payloads_for,
    signal_from_raw,
    to_bingx_order_payload,
    to_tbank_order_payload,
)
from trading.domain import OrderType, Price, Quantity, Side, Signal


def make_signal(**kw) -> Signal:
    kw.setdefault("symbol", "BTC-USDT")
    kw.setdefault("side", Side.BUY)
    kw.setdefault("strategy", "emf")
    kw.setdefault("reason", "unit-test")
    return Signal(**kw)


class TestSignalFromRaw:
    def test_parses_generic_payload(self):
        sig = signal_from_raw({"symbol": "SBER", "side": "buy", "strength": 0.7})
        assert sig.symbol == "SBER"
        assert sig.side is Side.BUY
        assert sig.strength == 0.7

    def test_accepts_ticker_action_aliases(self):
        sig = signal_from_raw({"ticker": "ETH-USDT", "action": "short"})
        assert sig.symbol == "ETH-USDT"
        assert sig.side is Side.SELL

    def test_parses_optional_price_and_quantity(self):
        sig = signal_from_raw(
            {"symbol": "BTC-USDT", "side": "sell", "price": 100.5, "quantity": 2.5}
        )
        assert sig.price is not None and sig.price.value == 100.5
        assert sig.quantity is not None and sig.quantity.value == 2.5

    def test_strength_zero_is_preserved(self):
        sig = signal_from_raw({"symbol": "X", "side": "buy", "strength": 0.0})
        assert sig.strength == 0.0

    def test_rejects_non_mapping(self):
        with pytest.raises(SignalParseError):
            signal_from_raw(["not", "a", "dict"])

    def test_rejects_missing_symbol(self):
        with pytest.raises(SignalParseError):
            signal_from_raw({"side": "buy"})

    def test_rejects_unknown_side(self):
        with pytest.raises(SignalParseError):
            signal_from_raw({"symbol": "X", "side": "hold"})


class TestBingxPayload:
    def test_market_buy_maps_to_long(self):
        sig = make_signal(quantity=Quantity(0.001))
        p = to_bingx_order_payload(sig)
        assert p["symbol"] == "BTC-USDT"
        assert p["side"] == "BUY"
        assert p["positionSide"] == "LONG"
        assert p["type"] == "MARKET"
        assert p["quantity"] == 0.001
        assert "price" not in p

    def test_sell_maps_to_short(self):
        sig = make_signal(side=Side.SELL, quantity=Quantity(1.0))
        assert to_bingx_order_payload(sig)["positionSide"] == "SHORT"

    def test_explicit_position_side_wins(self):
        sig = make_signal(side=Side.SELL, quantity=Quantity(1.0))
        p = to_bingx_order_payload(sig, position_side="BOTH")
        assert p["positionSide"] == "BOTH"

    def test_limit_order_includes_price_and_tif(self):
        sig = make_signal(quantity=Quantity(1.0), price=Price(64000.0))
        p = to_bingx_order_payload(sig, order_type=OrderType.LIMIT)
        assert p["type"] == "LIMIT"
        assert p["price"] == 64000.0
        assert p["timeInForce"] == "GTC"

    def test_limit_without_price_raises(self):
        sig = make_signal(quantity=Quantity(1.0))
        with pytest.raises(SignalParseError):
            to_bingx_order_payload(sig, order_type=OrderType.LIMIT)

    def test_tp_sl_are_json_strings(self):
        sig = make_signal(quantity=Quantity(1.0))
        p = to_bingx_order_payload(sig, take_profit=68000.0, stop_loss=61000.0)
        tp = json.loads(p["takeProfit"])
        sl = json.loads(p["stopLoss"])
        assert tp == {"type": "TAKE_PROFIT_MARKET", "stopPrice": 68000.0,
                      "workingType": "MARK_PRICE"}
        assert sl == {"type": "STOP_MARKET", "stopPrice": 61000.0,
                      "workingType": "MARK_PRICE"}

    def test_missing_quantity_raises(self):
        with pytest.raises(SignalParseError):
            to_bingx_order_payload(make_signal())

    def test_client_order_id_sanitized(self):
        sig = make_signal(quantity=Quantity(1.0))
        p = to_bingx_order_payload(sig, client_order_id="my bot#1/" + "x" * 100)
        assert len(p["clientOrderID"]) <= 40
        assert all(c.isalnum() or c == "_" for c in p["clientOrderID"])


class TestTbankPayload:
    def test_market_buy(self):
        sig = make_signal(symbol="SBER", quantity=Quantity(10))
        p = to_tbank_order_payload(sig, account_id="acc-1", figi="BBG004730N88")
        assert p["figi"] == "BBG004730N88"
        assert p["account_id"] == "acc-1"
        assert p["quantity"] == 10
        assert p["direction"] == "ORDER_DIRECTION_BUY"
        assert p["order_type"] == "ORDER_TYPE_MARKET"
        uuid.UUID(p["order_id"])  # valid uuid
        assert "price" not in p

    def test_sell_direction(self):
        sig = make_signal(side=Side.SELL, quantity=Quantity(3))
        assert to_tbank_order_payload(sig)["direction"] == "ORDER_DIRECTION_SELL"

    def test_limit_price_as_quotation(self):
        sig = make_signal(quantity=Quantity(1), price=Price(250.5))
        p = to_tbank_order_payload(sig, order_type=OrderType.LIMIT)
        assert p["order_type"] == "ORDER_TYPE_LIMIT"
        assert p["price"] == {"units": "250", "nano": 500000000}

    def test_fractional_quantity_floors_to_lots(self):
        sig = make_signal(quantity=Quantity(10.9))
        assert to_tbank_order_payload(sig)["quantity"] == 10

    def test_zero_lots_raises(self):
        sig = make_signal(quantity=Quantity(0.5))
        with pytest.raises(SignalParseError):
            to_tbank_order_payload(sig)

    def test_explicit_lots_override(self):
        sig = make_signal(quantity=Quantity(1))
        p = to_tbank_order_payload(sig, lots=42)
        assert p["quantity"] == 42


class TestNativePayloadsFor:
    def test_returns_both_exchanges(self):
        sig = make_signal(quantity=Quantity(0.001), price=Price(64000.0))
        out = native_payloads_for(sig, tbank_figi="FIGI-X", tbank_lots=10)
        assert set(out) == {"bingx", "tbank"}
        assert out["bingx"]["symbol"] == "BTC-USDT"
        assert out["tbank"]["figi"] == "FIGI-X"
