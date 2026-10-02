"""Tests for gex.volatility_cone — quarterly volatility cone + VWAP Price Channel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gex.domain.volatility_cone import (
    compute_heikin_ashi,
    compute_daily_volatility,
    compute_rsi_wilder,
    detect_quarters,
    compute_hybrid_source,
    compute_volatility_cone,
    compute_vpc,
    compute_all,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════
@pytest.fixture
def simple_ohlcv() -> pd.DataFrame:
    """5-bar OHLCV with a new quarter on bar 0."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [102.0, 103.0, 104.0, 105.0, 106.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [101.0, 102.0, 103.0, 104.0, 105.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        },
        index=dates,
    )


@pytest.fixture
def multi_quarter_ohlcv() -> pd.DataFrame:
    """Daily data spanning 2 quarters (Jan–Jun 2024)."""
    dates = pd.date_range("2024-01-02", periods=120, freq="B")
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, 120))
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 0.3, 120),
            "high": close + rng.uniform(0, 0.8, 120),
            "low": close - rng.uniform(0, 0.8, 120),
            "close": close,
            "volume": rng.integers(1000, 10000, 120).astype(float),
        },
        index=dates,
    )


# ═══════════════════════════════════════════════════════════════════════
# Heikin Ashi
# ═══════════════════════════════════════════════════════════════════════
class TestHeikinAshi:
    def test_length_matches_input(self, simple_ohlcv):
        ha_o, ha_c = compute_heikin_ashi(
            simple_ohlcv["open"].values,
            simple_ohlcv["high"].values,
            simple_ohlcv["low"].values,
            simple_ohlcv["close"].values,
        )
        assert len(ha_o) == 5
        assert len(ha_c) == 5

    def test_first_bar_ha_open_equals_open(self, simple_ohlcv):
        ha_o, _ = compute_heikin_ashi(
            simple_ohlcv["open"].values,
            simple_ohlcv["high"].values,
            simple_ohlcv["low"].values,
            simple_ohlcv["close"].values,
        )
        assert ha_o[0] == pytest.approx(100.0)

    def test_ha_values_are_finite(self, multi_quarter_ohlcv):
        ha_o, ha_c = compute_heikin_ashi(
            multi_quarter_ohlcv["open"].values,
            multi_quarter_ohlcv["high"].values,
            multi_quarter_ohlcv["low"].values,
            multi_quarter_ohlcv["close"].values,
        )
        assert np.all(np.isfinite(ha_o))
        assert np.all(np.isfinite(ha_c))


# ═══════════════════════════════════════════════════════════════════════
# Daily Volatility
# ═══════════════════════════════════════════════════════════════════════
class TestDailyVolatility:
    def test_returns_array_of_same_length(self, simple_ohlcv):
        dv = compute_daily_volatility(
            simple_ohlcv["high"].values,
            simple_ohlcv["low"].values,
            simple_ohlcv["close"].values,
            lookback_days=3,
        )
        assert len(dv) == 5

    def test_non_negative(self, multi_quarter_ohlcv):
        dv = compute_daily_volatility(
            multi_quarter_ohlcv["high"].values,
            multi_quarter_ohlcv["low"].values,
            multi_quarter_ohlcv["close"].values,
        )
        assert np.all(dv >= 0)


# ═══════════════════════════════════════════════════════════════════════
# RSI
# ═══════════════════════════════════════════════════════════════════════
class TestRSI:
    def test_range_0_to_100(self, multi_quarter_ohlcv):
        rsi = compute_rsi_wilder(multi_quarter_ohlcv["close"].values, period=14)
        valid = rsi[~np.isnan(rsi)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_constant_price_gives_100(self):
        """При постоянном росте (нет падений) RSI = 100."""
        close = np.linspace(100, 200, 50)
        rsi = compute_rsi_wilder(close, period=14)
        valid = rsi[~np.isnan(rsi)]
        # Все gains, нет losses → RSI стремится к 100
        assert np.all(valid[-10:] > 90)


# ═══════════════════════════════════════════════════════════════════════
# Quarter Detection
# ═══════════════════════════════════════════════════════════════════════
class TestDetectQuarters:
    def test_january_starts_quarter(self):
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
        result = detect_quarters(idx)
        assert bool(result[0]) is True

    def test_february_not_quarter(self):
        idx = pd.DatetimeIndex([pd.Timestamp("2024-02-01")])
        result = detect_quarters(idx)
        assert bool(result[0]) is False

    def test_april_new_quarter(self):
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2024-03-28"), pd.Timestamp("2024-04-01")]
        )
        result = detect_quarters(idx)
        assert bool(result[0]) is False  # March
        assert bool(result[1]) is True   # April (new month)

    def test_consecutive_january_bars_only_first_triggers(self):
        """Два бара в январе — только первый триггерит."""
        idx = pd.DatetimeIndex(
            [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        )
        result = detect_quarters(idx)
        assert bool(result[0]) is True
        assert bool(result[1]) is False  # month == prev_month → no trigger


# ═══════════════════════════════════════════════════════════════════════
# Hybrid Source
# ═══════════════════════════════════════════════════════════════════════
class TestHybridSource:
    def test_returns_finite_array(self, simple_ohlcv):
        ha_o, ha_c = compute_heikin_ashi(
            simple_ohlcv["open"].values,
            simple_ohlcv["high"].values,
            simple_ohlcv["low"].values,
            simple_ohlcv["close"].values,
        )
        ns = compute_hybrid_source(
            simple_ohlcv["open"].values,
            simple_ohlcv["high"].values,
            simple_ohlcv["low"].values,
            simple_ohlcv["close"].values,
            ha_o,
            ha_c,
        )
        assert len(ns) == 5
        assert np.all(np.isfinite(ns))


# ═══════════════════════════════════════════════════════════════════════
# Full Volatility Cone
# ═══════════════════════════════════════════════════════════════════════
class TestVolatilityCone:
    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_volatility_cone(df)
        assert len(result) == 0

    def test_single_bar_new_quarter(self, simple_ohlcv):
        """Первый бар нового квартала: days_passed=0, границы ≈ start_price."""
        single = simple_ohlcv.iloc[:1].copy()
        result = compute_volatility_cone(single)
        assert bool(result["is_new_quarter"].iloc[0]) is True
        # При days=0: width=0, vwap_offset=0, rsi_mom=0 → границы = start_price
        sp = single["open"].iloc[0]
        assert result["upper_1sd"].iloc[0] == pytest.approx(sp, rel=0.01)
        assert result["lower_1sd"].iloc[0] == pytest.approx(sp, rel=0.01)

    def test_cone_expands_with_time(self, simple_ohlcv):
        """Границы расширяются как sqrt(days)."""
        result = compute_volatility_cone(simple_ohlcv)
        u2 = result["upper_2sd"].values
        # Границы должны расширяться (upper растёт, lower падает)
        assert u2[-1] > u2[1]  # последний бар дальше первого
        assert result["lower_2sd"].values[-1] < result["lower_2sd"].values[1]

    def test_quarter_transition_preserves_prev(self):
        """Переход квартала: prev_* значения сохраняются."""
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-03-28"),  # Q1 конец
                pd.Timestamp("2024-04-01"),  # Q2 начало
                pd.Timestamp("2024-04-02"),
            ]
        )
        df = pd.DataFrame(
            {
                "open": [100.0, 200.0, 201.0],
                "high": [102.0, 202.0, 203.0],
                "low": [99.0, 199.0, 200.0],
                "close": [101.0, 201.0, 202.0],
                "volume": [1000.0, 1000.0, 1000.0],
            },
            index=idx,
        )
        result = compute_volatility_cone(df, use_correction=True)
        # Q2 starts at bar 1
        assert bool(result["is_new_quarter"].iloc[0]) is False
        assert bool(result["is_new_quarter"].iloc[1]) is True
        # Correction should be active on bar 2 (days_passed > 0 in Q2)
        assert not np.isnan(result["upper_2sd_corr"].iloc[2])

    def test_zero_volume_gives_nan_vwap(self):
        """Нулевой объём → VWAP = NaN."""
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [0.0, 0.0],
            },
            index=idx,
        )
        result = compute_volatility_cone(df)
        assert np.isnan(result["vwap"].iloc[0])
        assert np.isnan(result["vwap"].iloc[1])

    def test_all_required_columns_present(self, multi_quarter_ohlcv):
        result = compute_volatility_cone(multi_quarter_ohlcv)
        required = [
            "upper_1sd", "lower_1sd",
            "upper_2sd", "lower_2sd",
            "upper_2sd_mr", "lower_2sd_mr",
            "upper_1sd_mr", "lower_1sd_mr",
            "upper_2sd_corr", "lower_2sd_corr",
            "upper_2sd_mr_corr", "lower_2sd_mr_corr",
            "bb_upper", "bb_lower",
            "median_price", "vwap", "qema21",
            "corr_deviation_pct",
            "novelsrc", "daily_volatility", "current_rsi",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_does_not_mutate_input(self, simple_ohlcv):
        original_cols = list(simple_ohlcv.columns)
        _ = compute_volatility_cone(simple_ohlcv)
        assert list(simple_ohlcv.columns) == original_cols


# ═══════════════════════════════════════════════════════════════════════
# VWAP Price Channel
# ═══════════════════════════════════════════════════════════════════════
class TestVPC:
    def test_returns_required_columns(self, simple_ohlcv):
        result = compute_vpc(simple_ohlcv)
        for col in ["vpc_upper", "vpc_lower", "vpc_mid", "vpc_hst", "vpc_lst"]:
            assert col in result.columns

    def test_vpc_dir_values_are_valid(self, multi_quarter_ohlcv):
        result = compute_vpc(multi_quarter_ohlcv)
        assert set(np.unique(result["vpc_dir"])) <= {-1, 0, 1}
        assert set(np.unique(result["vpc_dir2"])) <= {-1, 0, 1}


# ═══════════════════════════════════════════════════════════════════════
# compute_all
# ═══════════════════════════════════════════════════════════════════════
class TestComputeAll:
    def test_combines_cone_and_vpc(self, multi_quarter_ohlcv):
        result = compute_all(multi_quarter_ohlcv)
        assert "upper_2sd" in result.columns
        assert "vpc_upper" in result.columns
        assert "bb_upper" in result.columns

    def test_custom_params(self, multi_quarter_ohlcv):
        result = compute_all(
            multi_quarter_ohlcv,
            cone_params={"sd2_mult": 3.0, "vwap_influence": 0.5},
            vpc_length=30,
        )
        # Должен отработать без ошибок
        assert len(result) == len(multi_quarter_ohlcv)
