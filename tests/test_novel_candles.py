"""Smoke-тесты Novel Candles: алгоритм, EMA, схемы, edge-cases."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gex.application.novel_candles import NovelCandlesService, DEFAULT_EMA_PERIODS
from gex.schemas.novel_candles import NovelCandlesResponse, NovelCandleBar, TrendlineBlock


# ══════════════════════════════════════════════════════════════════════
#  Фикстуры
# ══════════════════════════════════════════════════════════════════════
def _make_ohlcv_df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """OHLCV DataFrame: списки (Open, High, Low, Close)."""
    dates = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="UTC")
    records = []
    for i, (o, h, l, c) in enumerate(rows):
        records.append({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1e6})
    return pd.DataFrame(records, index=dates)

def _uptrend_5bars() -> pd.DataFrame:
    """5 растущих баров: 100→101→102→103→104."""
    return _make_ohlcv_df([
        (100.0, 100.5, 99.5, 100.0),
        (101.0, 102.0, 100.5, 101.5),
        (102.0, 103.0, 101.5, 102.5),
        (103.0, 104.0, 102.5, 103.5),
        (104.0, 105.0, 103.5, 104.5),
    ])


# ══════════════════════════════════════════════════════════════════════
#  compute_novel_candles
# ══════════════════════════════════════════════════════════════════════
class TestComputeNovelCandles:
    def test_basic_5_bars(self):
        df = _uptrend_5bars()
        result = NovelCandlesService.compute_novel_candles(df)
        assert len(result) == 5
        assert list(result.columns) == ["Open", "High", "Low", "Close"]
        # Все цены должны быть конечными
        for col in ["Open", "High", "Low", "Close"]:
            assert result[col].notna().all(), f"NaN in {col}"
        # High >= Low для каждой свечи
        assert (result["High"] >= result["Low"]).all()
        # High >= Open, High >= Close
        assert (result["High"] >= result["Open"]).all()
        assert (result["High"] >= result["Close"]).all()
        # Low <= Open, Low <= Close
        assert (result["Low"] <= result["Open"]).all()
        assert (result["Low"] <= result["Close"]).all()

    def test_empty_df_raises(self):
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
        with pytest.raises(ValueError, match="пуст"):
            NovelCandlesService.compute_novel_candles(df)

    def test_single_bar(self):
        df = _make_ohlcv_df([(100.0, 102.0, 99.0, 101.0)])
        result = NovelCandlesService.compute_novel_candles(df)
        assert len(result) == 1
        # Для одного бара: novelOpen = novelClose = novelsrc
        assert result["Open"].iloc[0] == result["Close"].iloc[0]

    def test_output_index_preserved(self):
        df = _uptrend_5bars()
        result = NovelCandlesService.compute_novel_candles(df)
        pd.testing.assert_index_equal(result.index, df.index)


# ══════════════════════════════════════════════════════════════════════
#  compute_emas
# ══════════════════════════════════════════════════════════════════════
class TestComputeEmas:
    def test_ema20_on_5_bars(self):
        df = _uptrend_5bars()
        novel = NovelCandlesService.compute_novel_candles(df)
        emas = NovelCandlesService.compute_emas(novel, [20])

        assert "ema20" in emas
        # 5 баров — недостаточно для сходимости EMA20, но значения должны быть
        # (первые None из-за недостаточности данных — проверяем что массив есть)
        assert len(emas["ema20"]) == 5

    def test_ema_periods_return_keyed(self):
        novel = NovelCandlesService.compute_novel_candles(_uptrend_5bars())
        emas = NovelCandlesService.compute_emas(novel, [10, 20, 50, 100, 200])
        for p in [10, 20, 50, 100, 200]:
            assert f"ema{p}" in emas

    def test_ema_on_empty(self):
        result = NovelCandlesService.compute_emas(pd.DataFrame(), [10, 20])
        assert result == {}


# ══════════════════════════════════════════════════════════════════════
#  compute_heikin_ashi
# ══════════════════════════════════════════════════════════════════════
class TestHeikinAshi:
    def test_ha_ohlc_valid(self):
        df = _uptrend_5bars()
        ha = NovelCandlesService.compute_heikin_ashi(df)
        assert len(ha) == 5
        assert list(ha.columns) == ["Open", "High", "Low", "Close"]
        assert (ha["High"] >= ha["Low"]).all()
        assert (ha["High"] >= ha["Open"]).all()
        assert (ha["High"] >= ha["Close"]).all()

    def test_ha_close_is_avg(self):
        """haClose = (O+H+L+C)/4."""
        df = _make_ohlcv_df([(10.0, 14.0, 8.0, 12.0)])
        ha = NovelCandlesService.compute_heikin_ashi(df)
        expected_ha_close = (10 + 14 + 8 + 12) / 4.0  # = 11.0
        assert abs(ha["Close"].iloc[0] - expected_ha_close) < 1e-6


# ══════════════════════════════════════════════════════════════════════
#  Asset type detection
# ══════════════════════════════════════════════════════════════════════
class TestDetectAssetType:
    def test_us_stock(self):
        assert NovelCandlesService._detect_asset_type("SPY") == "stock"
        assert NovelCandlesService._detect_asset_type("AAPL") == "stock"
        assert NovelCandlesService._detect_asset_type("NVDA") == "stock"

    def test_crypto(self):
        assert NovelCandlesService._detect_asset_type("BTC") == "crypto"
        assert NovelCandlesService._detect_asset_type("ETH") == "crypto"
        assert NovelCandlesService._detect_asset_type("SOL") == "crypto"

    def test_moex(self):
        assert NovelCandlesService._detect_asset_type("RTS") == "moex"
        assert NovelCandlesService._detect_asset_type("MIX") == "moex"


# ══════════════════════════════════════════════════════════════════════
#  Pydantic schema validation
# ══════════════════════════════════════════════════════════════════════
class TestSchemas:
    def test_novel_candle_bar_valid(self):
        bar = NovelCandleBar(time="2026-01-01T00:00:00+00:00", open=100.0, high=102.0, low=99.0, close=101.0)
        assert bar.open == 100.0
        assert bar.high == 102.0
        assert bar.low == 99.0
        assert bar.close == 101.0

    def test_novel_candle_bar_extra_field_forbidden(self):
        with pytest.raises(Exception):
            NovelCandleBar(time="2026-01-01", open=1, high=2, low=0.5, close=1.5, extra="nope")

    def test_response_minimal(self):
        resp = NovelCandlesResponse(
            ticker="SPY",
            timeframe="1d",
            asset_type="stock",
            n_bars=0,
        )
        assert resp.ticker == "SPY"
        assert resp.bars == []
        assert resp.emas == {}
        assert resp.trendlines is None

    def test_response_with_data(self):
        bars = [
            NovelCandleBar(time="2026-01-01T00:00:00Z", open=100., high=102., low=99., close=101.),
            NovelCandleBar(time="2026-01-02T00:00:00Z", open=101., high=103., low=100., close=102.),
        ]
        resp = NovelCandlesResponse(
            ticker="SPY",
            timeframe="1d",
            asset_type="stock",
            n_bars=2,
            bars=bars,
            emas={"ema10": [None, None, 100.5]},
        )
        assert len(resp.bars) == 2
        assert "ema10" in resp.emas


# ══════════════════════════════════════════════════════════════════════
#  compute_trendlines smoke
# ══════════════════════════════════════════════════════════════════════
class TestTrendlinesOnNovel:
    def test_trendlines_on_novel_bars(self):
        """Трендовые линии на novel-барах: проверка, что возвращает
        валидный TrendlineAnalysis без ошибок."""
        # Генерируем 100 баров с трендом (нужно много для trendlines)
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 2 + 0.5)  # восходящий тренд
        rows = []
        for i in range(100):
            o = prices[i]
            c = prices[i] + np.random.randn() * 0.5
            h = max(o, c) + abs(np.random.randn()) * 0.5
            l = min(o, c) - abs(np.random.randn()) * 0.5
            rows.append((o, h, l, c))
        df = _make_ohlcv_df(rows)
        novel = NovelCandlesService.compute_novel_candles(df)

        tl = NovelCandlesService.compute_trendlines(novel, timeframe="1d")
        assert tl.timeframe == "1d"
        assert tl.combined_trend in ("BULLISH", "BEARISH", "RANGE")
        assert 0 <= tl.combined_strength <= 100
