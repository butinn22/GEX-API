"""Tests for BingX WebSocket message parsing and subscribe envelopes."""
from __future__ import annotations

import json

from trading.adapters.brokers.bingx_ws import build_subscribe, parse_depth, parse_kline, parse_trade


def test_parse_trade():
    msg = {"e": "trade", "E": 1700000000000, "s": "BTC-USDT", "t": "1",
           "p": "45000.5", "q": "0.1", "T": 1700000000000, "m": True}
    tick = parse_trade(msg)
    assert tick.price == 45000.5 and tick.volume == 0.1 and tick.side == "buy"
    assert tick.timestamp.tzinfo is not None


def test_parse_trade_ignores_other_events():
    assert parse_trade({"e": "kline"}) is None


def test_parse_kline():
    msg = {"e": "kline", "E": 1700000000000, "s": "BTC-USDT",
           "k": {"t": 1700000000000, "o": "100", "h": "110", "l": "90",
                 "c": "105", "v": "1000", "T": 1700000100000, "i": "1d"}}
    bar = parse_kline(msg)
    assert bar.open == 100.0 and bar.high == 110.0 and bar.close == 105.0


def test_parse_depth():
    msg = {"e": "depth", "E": 1700000000000, "s": "BTC-USDT",
           "b": [["100", "2"], ["99", "3"]], "a": [["101", "1"]]}
    book = parse_depth(msg)
    assert book.best_bid == 100.0 and book.best_ask == 101.0


def test_build_subscribe_envelope():
    d = json.loads(build_subscribe("sub-0", "BTC-USDT@trade", symbol="BTC-USDT"))
    assert d["id"] == "sub-0"
    assert d["dataType"] == "BTC-USDT@trade"
    assert d["data"]["pair"] == "BTC-USDT"
