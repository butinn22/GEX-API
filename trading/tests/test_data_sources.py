"""Data source auto-detection + synthesize toggle."""
from __future__ import annotations

from trading.application.data_sources import detect_data_source, synthetic_seed


class TestDetectDataSource:
    def test_ru_ticker_routes_to_moex(self):
        info = detect_data_source("SBER")
        assert info["source"] == "moex"
        assert info["category"] == "ru"
        assert info["fetch_symbol"] == "SBER"

    def test_crypto_universe_ticker_routes_to_bybit_pair(self):
        info = detect_data_source("BTC")
        assert info["source"] == "bybit"
        assert info["fetch_symbol"] == "BTCUSDT"

    def test_unknown_ticker_falls_back_to_yfinance(self):
        info = detect_data_source("NOSUCHTICKER123")
        assert info["source"] == "yfinance"
        assert info["category"] == "us"

    def test_every_ticker_can_be_synthesized(self):
        for symbol in ("SBER", "BTC", "AAPL", "NOSUCHTICKER123"):
            assert detect_data_source(symbol)["synthetic_available"] is True

    def test_symbol_is_normalized(self):
        assert detect_data_source("sber")["symbol"] == "SBER"


class TestSyntheticSeed:
    def test_deterministic_per_symbol(self):
        assert synthetic_seed("BTC-USDT") == synthetic_seed("btc-usdt")
        assert synthetic_seed("BTC-USDT") != synthetic_seed("ETH-USDT")

    def test_synth_uses_zero_seed(self):
        assert synthetic_seed("SYNTH") == 0
