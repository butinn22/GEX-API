"""ATR и True Range — канон (ring: domain, только numpy).

Зачем канон
-----------
До рефакторинга в проекте было **восемь** реализаций ATR, и они распадаются на **две разные
математики** — отсюда расхождение чисел на разных страницах для одного тикера:

| Место                            | TR                     | Сглаживание                     | Прогрев |
|----------------------------------|------------------------|---------------------------------|---------|
| ``hybrid_trend.compute_atr`` (14)| loop                   | рекурсия Wilder, ``atr[0]=tr[0]``| нет     |
| ``novel_candles.compute_atr`` (200)| loop (байт-в-байт)    | то же                           | нет     |
| ``trendlines._wilder_atr`` (14)  | pandas max             | ``ewm(alpha=1/p, min_periods=p)``| NaN до p |
| ``trend_regime`` (колонка atr)   | pandas max             | то же                           | NaN до p |
| ``breadth_imoex_service._atr_pct``| pandas max            | то же                           | NaN до p |
| ``trading_algorithm._atr``       | pandas max             | ``ewm(alpha=1/w, min_periods=1)``| нет     |
| ``macd_trend._atr``              | numpy (shift)          | Wilder                          | None при нехватке |
| **``gexcone.atr_14``**           | pandas max, dropna     | **простое среднее TR за period**| n >= period+2 |

Первые семь — одна и та же рекурсия (значения совпадают; различаются только период по умолчанию,
экспозиция прогрева и политика фоллбэка). Восьмая (``gexcone``) — принципиально другая: среднее TR.
Канон ниже покрывает **обе** и требует выбирать метод явно, чтобы расхождение больше не возникало
случайно.

Терминология
------------
* ``tr[0] = high[0] - low[0]`` (как и в pandas-вариантах: ``|h-prev_c|`` на первом баре = NaN и
  отбрасывается ``max(axis=1)``);
* ``method="wilder"`` — рекурсия ``alpha = 1/period`` с сидом ``atr[0] = tr[0]`` (Pine ``ta.atr``);
* ``method="mean"`` — простое среднее последних ``period`` значений TR (то, что делал ``gexcone``);
* ``warmup="period"`` — первые ``period-1`` значений = NaN (как ``ewm(..., min_periods=period)``);
  ``warmup="none"`` — значения со первого бара (как ``min_periods=1`` и как loop-версии).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._kernels import as_float_array

__all__ = ["true_range", "wilder_smooth", "atr", "atr_last", "METHODS", "WARMUPS"]

METHODS = ("wilder", "mean")
WARMUPS = ("none", "period")


def true_range(
    high: Sequence[float] | np.ndarray,
    low: Sequence[float] | np.ndarray,
    close: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """True Range: ``max(H-L, |H-C_prev|, |L-C_prev|)``; на первом баре — ``H-L``.

    Реализация совпадает и с loop-версией (``hybrid_trend``/``novel_candles``), и с pandas-версией
    ``pd.concat([...]).max(axis=1)``: ``np.fmax`` отбрасывает NaN-слагаемые так же, как ``max(skipna=True)``.
    """
    h = as_float_array(high)
    l = as_float_array(low)
    c = as_float_array(close)
    n = len(h)
    if n == 0:
        return np.empty(0, dtype=float)
    prev_close = np.concatenate(([np.nan], c[:-1]))
    hl = h - l
    hc = np.abs(h - prev_close)
    lc = np.abs(l - prev_close)
    return np.fmax(hl, np.fmax(hc, lc))


def wilder_smooth(tr: Sequence[float] | np.ndarray, period: int) -> np.ndarray:
    """Рекурсивное сглаживание Wilder: ``atr[0] = tr[0]``, ``a_i = alpha*tr_i + (1-alpha)*a_{i-1}``.

    NaN во входе переносится как NaN (расширение по отношению к loop-версиям, которые такого входа
    не получают; на «чистом» входе значения совпадают до последнего бита).
    """
    tr_arr = as_float_array(tr)
    n = len(tr_arr)
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out
    if period <= 0:
        return out
    alpha = 1.0 / float(period)
    out[0] = tr_arr[0]
    for i in range(1, n):
        value = tr_arr[i]
        if np.isnan(value) or np.isnan(out[i - 1]):
            out[i] = np.nan
        else:
            out[i] = alpha * value + (1.0 - alpha) * out[i - 1]
    return out


def atr(
    high: Sequence[float] | np.ndarray,
    low: Sequence[float] | np.ndarray,
    close: Sequence[float] | np.ndarray,
    period: int = 14,
    *,
    method: str = "wilder",
    warmup: str = "none",
) -> np.ndarray:
    """ATR как массив (см. таблицу в докстринге модуля).

    Parameters
    ----------
    method : {"wilder", "mean"}
        ``wilder`` — рекурсия (Pine ``ta.atr``); ``mean`` — простое среднее TR за ``period``.
    warmup : {"none", "period"}
        ``none`` — значения с первого бара; ``period`` — первые ``period-1`` = NaN.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if warmup not in WARMUPS:
        raise ValueError(f"warmup must be one of {WARMUPS}, got {warmup!r}")

    tr = true_range(high, low, close)
    n = len(tr)
    if n == 0 or period <= 0:
        return np.full(n, np.nan, dtype=float)

    if method == "wilder":
        out = wilder_smooth(tr, period)
    else:
        out = _rolling_mean_tr(tr, period)

    if warmup == "period" and period > 1:
        head = min(period - 1, n)
        out[:head] = np.nan
    return out


def _rolling_mean_tr(tr: np.ndarray, period: int) -> np.ndarray:
    """Простое среднее TR за окно ``period`` (семантика ``gexcone.atr_14``)."""
    n = len(tr)
    out = np.full(n, np.nan, dtype=float)
    for i in range(period - 1, n):
        window = tr[i - period + 1 : i + 1]
        if not np.isnan(window).any():
            out[i] = float(np.mean(window))
    return out


def atr_last(
    high: Sequence[float] | np.ndarray,
    low: Sequence[float] | np.ndarray,
    close: Sequence[float] | np.ndarray,
    period: int = 14,
    *,
    method: str = "wilder",
    fallback: str = "none",
) -> float | None:
    """Скалярный ATR последнего бара — с политикой фоллбэка, как у прежних скалярных реализаций.

    Parameters
    ----------
    fallback : {"none", "hl_mean_then_one", "require_history", "none_if_insufficient"}
        * ``none`` — вернуть ``None``, если значение не посчиталось (`macd_trend`-стиль);
        * ``hl_mean_then_one`` — при NaN/<=0 подставить ``mean(H-L)`` за последние ``period`` баров,
          а если и оно <= 0 — ``1.0`` (``trendlines._wilder_atr``);
        * ``none_if_insufficient`` — требовать ``n >= period + 2`` и вернуть ``None`` иначе
          (``gexcone.atr_14``).
    """
    h = as_float_array(high)
    l = as_float_array(low)
    n = min(len(h), len(l), len(as_float_array(close)))
    if n == 0:
        return None
    if fallback == "none_if_insufficient" and n < period + 2:
        return None

    prev_close = np.concatenate(([np.nan], as_float_array(close)[:n][:-1]))
    tr = true_range(h[:n], l[:n], as_float_array(close)[:n])
    tr = tr[~np.isnan(tr)]
    if len(tr) < period:
        value = float("nan")
    elif method == "mean":
        value = float(np.mean(tr[-period:]))
    else:
        smoothed = wilder_smooth(tr, period)
        value = float(smoothed[-1])
        if fallback == "none" and n < period + 1:
            # macd_trend-стиль: при нехватке истории значение не считается
            value = float("nan")

    if np.isfinite(value) and value > 0:
        return value
    if fallback == "hl_mean_then_one":
        hl = (h - l)[-period:] if period > 0 else (h - l)
        candidate = float(np.mean(hl)) if len(hl) else 1.0
        return candidate if candidate > 0 else 1.0
    return None
