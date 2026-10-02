"""Technical indicators (pure numpy) + a pluggable indicator registry.

All indicators return NaN-padded arrays so they align with an input series and
the warm-up period is explicit. Conventions:

* ``sma`` / ``ema`` / ``rsi`` / ``atr`` use Wilder's smoothing where applicable;
* ``ema`` is seeded with the SMA of the first ``period`` bars (no look-ahead);
* ``bollinger`` uses population std (ddof=0), matching the classic definition.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "sma", "ema", "rsi", "macd", "bollinger", "atr",
    "IndicatorLibrary", "INDICATORS",
]


def _to_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def sma(values, period: int) -> np.ndarray:
    """Simple moving average. First ``period-1`` positions are NaN."""
    v = _to_array(values)
    if period <= 0:
        raise ValueError("period must be > 0")
    out = np.full(v.shape, np.nan)
    if v.size >= period:
        csum = np.cumsum(np.insert(v, 0, 0.0))
        out[period - 1 :] = (csum[period:] - csum[:-period]) / period
    return out


def ema(values, period: int) -> np.ndarray:
    """Exponential moving average, seeded with the SMA of the first ``period`` bars."""
    v = _to_array(values)
    if period <= 0:
        raise ValueError("period must be > 0")
    out = np.full(v.shape, np.nan)
    if v.size < period:
        return out
    k = 2.0 / (period + 1.0)
    out[period - 1] = np.mean(v[:period])
    for i in range(period, v.size):
        out[i] = v[i] * k + out[i - 1] * (1.0 - k)
    return out


def rsi(values, period: int) -> np.ndarray:
    """Relative Strength Index (Wilder's smoothing)."""
    v = _to_array(values)
    if period <= 0:
        raise ValueError("period must be > 0")
    out = np.full(v.shape, np.nan)
    if v.size <= period:
        return out
    delta = np.diff(v)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, v.size):
        avg_gain = (avg_gain * (period - 1) + gain[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i - 1]) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd(values, fast: int, slow: int, signal: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, histogram."""
    if not (fast < slow):
        raise ValueError("require fast < slow")
    macd_line = ema(values, fast) - ema(values, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def bollinger(values, period: int, num_std: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Middle (SMA), upper, lower Bollinger bands."""
    v = _to_array(values)
    middle = sma(v, period)
    std = np.full(v.shape, np.nan)
    if v.size >= period:
        # rolling population std
        csum = np.cumsum(np.insert(v, 0, 0.0))
        csum2 = np.cumsum(np.insert(v * v, 0, 0.0))
        n = period
        mean = (csum[n:] - csum[:-n]) / n
        mean_sq = (csum2[n:] - csum2[:-n]) / n
        var = np.maximum(mean_sq - mean * mean, 0.0)
        std[period - 1 :] = np.sqrt(var)
    return middle, middle + num_std * std, middle - num_std * std


def atr(high, low, close, period: int) -> np.ndarray:
    """Average True Range (Wilder's smoothing)."""
    h = _to_array(high)
    l = _to_array(low)
    c = _to_array(close)
    n = c.size
    if period <= 0:
        raise ValueError("period must be > 0")
    out = np.full(n, np.nan)
    if n < period:
        return out
    prev_close = np.roll(c, 1)
    prev_close[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    out[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


# ── Registry ───────────────────────────────────────────────────────────


class IndicatorLibrary:
    """Name → indicator callable, so strategies can load indicators by name."""

    def __init__(self) -> None:
        self._registry: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._registry[name] = fn

    def get(self, name: str) -> Callable:
        if name not in self._registry:
            raise KeyError(f"indicator '{name}' not registered")
        return self._registry[name]

    def names(self) -> list[str]:
        return sorted(self._registry)


INDICATORS = IndicatorLibrary()
for _name, _fn in {
    "sma": sma, "ema": ema, "rsi": rsi, "macd": macd,
    "bollinger": bollinger, "atr": atr,
}.items():
    INDICATORS.register(_name, _fn)
