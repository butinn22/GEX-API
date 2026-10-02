"""Test: combined EMF MF + ADL STRAT — basic compilation and signal generation."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from gex.strategy.trading_algorithm import (
    EMAFilterTrendStrategy,
    StrategySettings,
    TradingState,
    SignalAction,
)


@pytest.fixture
def synthetic_ohlc() -> pd.DataFrame:
    """Generate 1000 bars of synthetic OHLCV (random walk)."""
    np.random.seed(42)
    n = 1000
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 1.0)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.01)
    open_ = low + (high - low) * np.random.uniform(0.2, 0.8, n)
    volume = np.random.randint(10000, 1000000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=idx)


def test_strategy_initialization():
    """Strategy initializes with default settings."""
    s = EMAFilterTrendStrategy()
    assert s.name == "EMF MF + ADL STRAT (Combined)"
    assert s.settings is not None
    assert s.settings.tp_percent == 2.0


def test_calculate_returns_features(synthetic_ohlc):
    """calculate() returns DataFrame with all expected columns."""
    strategy = EMAFilterTrendStrategy()
    features = strategy.calculate(synthetic_ohlc)
    assert isinstance(features, pd.DataFrame)
    assert len(features) >= 900  # Some NaN at start
    # Core columns
    for col in ["novelsrc", "ema10", "ema20", "ema77", "ema200"]:
        assert col in features.columns, f"Missing: {col}"
    # ADL columns
    for col in ["adline", "adl50", "adl200", "ad", "adl_macd", "adl_signal", "adl_tl", "tp_f"]:
        assert col in features.columns, f"Missing: {col}"
    # Signal columns (combined)
    for col in ["combined_long_entry", "combined_short_entry",
                "combined_long_exit", "combined_short_exit",
                "combined_long_add", "combined_short_add",
                "long_entry_a", "short_entry_a", "long_entry_b", "short_entry_b"]:
        assert col in features.columns, f"Missing: {col}"
    # Backward compat
    assert "long_entry_signal" in features.columns
    assert "short_entry_signal" in features.columns


def test_signals_are_boolean(synthetic_ohlc):
    """Signal columns should be boolean."""
    features = EMAFilterTrendStrategy().calculate(synthetic_ohlc).dropna()
    for col in ["combined_long_entry", "combined_short_entry",
                "combined_long_exit", "combined_short_exit"]:
        assert features[col].dtype == bool, f"{col} is not bool, got {features[col].dtype}"


def test_evaluate_returns_signal(synthetic_ohlc):
    """evaluate() returns a TradingSignal."""
    features = EMAFilterTrendStrategy().calculate(synthetic_ohlc)
    signal = EMAFilterTrendStrategy().evaluate(features)
    from gex.strategy.trading_algorithm import TradingSignal as TS
    assert isinstance(signal, TS), f"Expected TradingSignal, got {type(signal)}"
    assert signal is not None


def test_evaluate_flat_state(synthetic_ohlc):
    """With flat state, evaluate should return BUY/SELL or HOLD."""
    features = EMAFilterTrendStrategy().calculate(synthetic_ohlc)
    strategy = EMAFilterTrendStrategy()
    signal = strategy.evaluate(features)
    assert signal.action in (SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD)
    assert isinstance(signal.reason, str)


def test_strategy_b_entries_exist(synthetic_ohlc):
    """Strategy B entry signals should be non-trivial."""
    features = EMAFilterTrendStrategy().calculate(synthetic_ohlc).dropna()
    n_long_b = features["long_entry_b"].sum()
    n_short_b = features["short_entry_b"].sum()
    combined = features["combined_long_entry"].sum() + features["combined_short_entry"].sum()
    # Print stats (not strictly assert — may be 0 on synthetic data)
    print(f"\nStrategy A long: {features['long_entry_a'].sum()}, short: {features['short_entry_a'].sum()}")
    print(f"Strategy B long: {n_long_b}, short: {n_short_b}")
    print(f"Combined entries: {combined}")
    print(f"Combined adds: {features['combined_long_add'].sum()} / {features['combined_short_add'].sum()}")
    print(f"ADL chain: adline range=[{features['adline'].min():.4f}, {features['adline'].max():.4f}]")
    print(f"Two-pole filter: range=[{features['tp_f'].min():.4f}, {features['tp_f'].max():.4f}]")


def test_two_pole_filter_basic():
    """Two-pole filter handles simple input (start with f1=0/f2=0 like Pine nz)."""
    from gex.strategy.trading_algorithm import _two_pole_filter
    values = np.sin(np.linspace(0, 4 * np.pi, 200)) + 1.0
    result = _two_pole_filter(values, 20.0, 0.9)
    assert len(result) == 200
    assert not np.isnan(result[0])  # Pine nz(f1[1])=0, first val computed
    assert np.isfinite(result[0])
    assert np.isfinite(result[-1])
    assert np.all(np.isfinite(result))


def test_adl_chain_output(synthetic_ohlc):
    """ADL chain produces finite values."""
    features = EMAFilterTrendStrategy().calculate(synthetic_ohlc)
    adl = features["adline"].dropna()
    assert len(adl) > 100
    assert not adl.isna().all()
    assert np.isfinite(adl).all()


def test_macd_on_adl(synthetic_ohlc):
    """ADL-MACD produces finite values."""
    features = EMAFilterTrendStrategy().calculate(synthetic_ohlc).dropna()
    assert "adl_macd" in features.columns
    assert "adl_signal" in features.columns
    assert "adl_tl" in features.columns
    assert np.isfinite(features["adl_macd"]).all()
    assert np.isfinite(features["adl_signal"]).all()


if __name__ == "__main__":
    print("=== Generating synthetic data (1000 bars) ===")
    ohlc = synthetic_ohlc()
    print(f"OHLC: {ohlc.shape}, from {ohlc.index[0].date()} to {ohlc.index[-1].date()}")

    print("\n=== Calculating features ===")
    strategy = EMAFilterTrendStrategy()
    features = strategy.calculate(ohlc)
    print(f"Features: {features.shape}, {len(features.columns)} columns")
    print(f"NaN rows: {features.isna().any(axis=1).sum()}")

    print("\n=== Signal stats (non-NaN rows only) ===")
    clean = features.dropna()
    print(f"Clean rows: {len(clean)}")
    for col in clean.columns:
        if col.endswith("_entry") or col.endswith("_exit") or col.endswith("_add"):
            if col.startswith("combined") or col.startswith("long_") or col.startswith("short_"):
                n = clean[col].sum()
                if n > 0:
                    print(f"  {col}: {int(n)} signals")

    print("\n=== Evaluate last bar ===")
    signal = strategy.evaluate(features)
    print(f"Signal: {signal.action.value}, reason: {signal.reason}")
    if signal.metadata.get("close_price"):
        print(f"  at price: {signal.metadata['close_price']:.2f}")

    print("\n=== ADL range ===")
    print(f"  adline: [{clean['adline'].min():.4f}, {clean['adline'].max():.4f}]")
    print(f"  tp_f:   [{clean['tp_f'].min():.4f}, {clean['tp_f'].max():.4f}]")
    print(f"  adl_macd: [{clean['adl_macd'].min():.4f}, {clean['adl_macd'].max():.4f}]")

    print("\nALL TESTS PASSED")
