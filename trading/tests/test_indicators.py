"""Tests for the indicator library (known values + shapes)."""
from __future__ import annotations

import numpy as np
import pytest

from trading.application.indicators import (
    INDICATORS,
    atr,
    bollinger,
    ema,
    macd,
    rsi,
    sma,
)


def test_sma_known_values():
    out = sma([1, 2, 3, 4], 2)
    assert np.isnan(out[0])
    assert np.allclose(out[1:], [1.5, 2.5, 3.5])


def test_ema_known_values():
    out = ema([1, 2, 4, 4], 2)
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(1.5)
    assert out[2] == pytest.approx(3.166666, rel=1e-5)
    assert out[3] == pytest.approx(3.722222, rel=1e-5)


def test_rsi_up_and_down_trend():
    up = rsi([1, 2, 3, 4, 5, 6, 7], 3)
    assert np.allclose(up[3:], 100.0)
    down = rsi([7, 6, 5, 4, 3, 2, 1], 3)
    assert np.allclose(down[3:], 0.0)


def test_macd_histogram_is_difference():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    line, signal, hist = macd(values, fast=3, slow=6, signal=3)
    assert np.allclose(hist[~np.isnan(hist)], (line - signal)[~np.isnan(hist)])


def test_bollinger_middle_is_sma_and_bands_ordered():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    middle, upper, lower = bollinger(values, period=3, num_std=2.0)
    assert np.allclose(middle[~np.isnan(middle)], sma(values, 3)[~np.isnan(sma(values, 3))])
    valid = ~np.isnan(middle)
    assert np.all(upper[valid] >= middle[valid])
    assert np.all(middle[valid] >= lower[valid])


def test_atr_constant_range():
    high = [10.0, 10.0, 10.0, 10.0]
    low = [9.0, 9.0, 9.0, 9.0]
    close = [9.5, 9.5, 9.5, 9.5]
    out = atr(high, low, close, period=2)
    assert np.isnan(out[0])
    assert np.allclose(out[1:], 1.0)


def test_indicator_registry():
    assert "sma" in INDICATORS.names()
    assert INDICATORS.get("rsi") is rsi
    with pytest.raises(KeyError):
        INDICATORS.get("nonexistent")
