"""Optional TA-Lib backend for the indicator library.

TA-Lib (the C library) is not installable on this host (no cp312 wheels and it
requires the native lib). When the ``talib`` package is importable, these
functions use it for C-speed computation; otherwise they fall back to the
pure-numpy implementations in :mod:`trading.application.indicators`.

Warm-up convention note: TA-Lib seeds some indicators differently than the
numpy versions, so parity is guaranteed only beyond the warm-up window. The
numpy implementations remain the reference for exact test values.
"""
from __future__ import annotations

import numpy as np

from . import indicators as _numpy

__all__ = ["HAS_TALIB", "sma", "ema", "rsi", "macd", "bollinger", "atr"]

try:
    import talib as _talib  # type: ignore
    HAS_TALIB = True
except Exception:  # pragma: no cover - environment-dependent
    _talib = None
    HAS_TALIB = False


def sma(values, period: int) -> np.ndarray:
    if HAS_TALIB:
        return _talib.SMA(np.asarray(values, float), timeperiod=period)
    return _numpy.sma(values, period)


def ema(values, period: int) -> np.ndarray:
    if HAS_TALIB:
        return _talib.EMA(np.asarray(values, float), timeperiod=period)
    return _numpy.ema(values, period)


def rsi(values, period: int) -> np.ndarray:
    if HAS_TALIB:
        return _talib.RSI(np.asarray(values, float), timeperiod=period)
    return _numpy.rsi(values, period)


def macd(values, fast: int, slow: int, signal: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if HAS_TALIB:
        line, sig, hist = _talib.MACD(
            np.asarray(values, float), fastperiod=fast, slowperiod=slow, signalperiod=signal
        )
        return line, sig, hist
    return _numpy.macd(values, fast, slow, signal)


def bollinger(values, period: int, num_std: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if HAS_TALIB:
        upper, middle, lower = _talib.BBANDS(
            np.asarray(values, float), timeperiod=period, nbdevup=num_std, nbdevdn=num_std
        )
        return middle, upper, lower
    return _numpy.bollinger(values, period, num_std)


def atr(high, low, close, period: int) -> np.ndarray:
    if HAS_TALIB:
        return _talib.ATR(
            np.asarray(high, float), np.asarray(low, float), np.asarray(close, float),
            timeperiod=period,
        )
    return _numpy.atr(high, low, close, period)
