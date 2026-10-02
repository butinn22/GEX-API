"""Tests for the TA-Lib backend (fallback parity; talib parity when installed)."""
from __future__ import annotations

import numpy as np
import pytest

from trading.application import indicators, ta_lib


def test_fallback_matches_numpy_indicators():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert np.allclose(ta_lib.sma(values, 3), indicators.sma(values, 3), equal_nan=True)
    assert np.allclose(ta_lib.ema(values, 3), indicators.ema(values, 3), equal_nan=True)
    assert np.allclose(ta_lib.rsi(values, 3), indicators.rsi(values, 3), equal_nan=True)
    line_a, sig_a, hist_a = ta_lib.macd(values, 3, 6, 3)
    line_b, sig_b, hist_b = indicators.macd(values, 3, 6, 3)
    assert np.allclose(line_a, line_b, equal_nan=True)


@pytest.mark.skipif(not ta_lib.HAS_TALIB, reason="talib not installed")
def test_talib_parity_beyond_warmup():
    values = np.random.default_rng(0).normal(0, 1, 200).cumsum() + 100.0
    assert np.allclose(ta_lib.sma(values, 20)[19:], indicators.sma(values, 20)[19:], rtol=1e-6)
