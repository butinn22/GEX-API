"""Свечные трансформации — канон (ring: domain, только numpy).

Heikin-Ashi: три реализации, две из них расходятся на первом баре
-----------------------------------------------------------------
| Место | Сид ``ha_open[0]`` | Что возвращает |
|---|---|---|
| ``novel_candles.compute_heikin_ashi:267`` | ``(O+C)/2`` | полный OHLC |
| ``hybrid_trend.compute_heikin_ashi:146`` | ``(O+C)/2`` | пара ``(ha_open, ha_close)`` |
| ``volatility_cone.compute_heikin_ashi`` (``:62-83``) | **``O``** | пара ``(ha_open, ha_close)`` |

Разница в сиде затухает вдвое на каждом баре, поэтому на длинных сериях она невидима, но на
коротких (первые ~10 баров, скользящие окна, «хвост» после обрезки данных) даёт другие числа.
Канон сохраняет **обе** семантики явным параметром ``seed``, а не «выбирает правильную молча».

Ряд рекуррентностей (``ha_close``, ``ha_open[i]``) во всех трёх реализациях идентичен и повторён
здесь без изменений.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._kernels import as_float_array

__all__ = ["heikin_ashi", "SEEDS"]

SEEDS = ("midpoint", "open")


def heikin_ashi(
    open_: Sequence[float] | np.ndarray,
    high: Sequence[float] | np.ndarray,
    low: Sequence[float] | np.ndarray,
    close: Sequence[float] | np.ndarray,
    *,
    seed: str = "midpoint",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Heikin-Ashi OHLC.

    Returns
    -------
    (ha_open, ha_high, ha_low, ha_close) — четыре массива той же длины.

    Parameters
    ----------
    seed : {"midpoint", "open"}
        ``midpoint`` — ``ha_open[0] = (O₀ + C₀)/2`` (``novel_candles``, ``hybrid_trend``);
        ``open`` — ``ha_open[0] = O₀`` (``volatility_cone``).
    """
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed!r}")

    o = as_float_array(open_)
    h = as_float_array(high)
    l = as_float_array(low)
    c = as_float_array(close)
    n = len(o)
    ha_open = np.full(n, np.nan, dtype=float)
    ha_close = np.full(n, np.nan, dtype=float)
    ha_high = np.full(n, np.nan, dtype=float)
    ha_low = np.full(n, np.nan, dtype=float)
    if n == 0:
        return ha_open, ha_high, ha_low, ha_close

    ha_close[0] = (o[0] + h[0] + l[0] + c[0]) / 4.0
    ha_open[0] = (o[0] + c[0]) / 2.0 if seed == "midpoint" else o[0]
    ha_high[0] = max(h[0], ha_open[0], ha_close[0])
    ha_low[0] = min(l[0], ha_open[0], ha_close[0])

    for i in range(1, n):
        ha_close[i] = (o[i] + h[i] + l[i] + c[i]) / 4.0
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high[i] = max(h[i], ha_open[i], ha_close[i])
        ha_low[i] = min(l[i], ha_open[i], ha_close[i])

    return ha_open, ha_high, ha_low, ha_close
