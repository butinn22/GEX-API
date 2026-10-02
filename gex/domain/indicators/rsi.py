"""RSI — канон (ring: domain, только numpy).

До рефакторинга в проекте было три «RSI», и две из них — один и тот же Wilder-RSI с **разной
обработкой краевых случаев**, а третья — вообще другое определение:

| Место | Определение | Прогрев | Плоский ряд |
|---|---|---|---|
| ``ta._wilder_rsi`` (вызывается из ``ta.py:366,619,932``, ``vol_fetcher.py:42``) | Wilder-RSI, сид = SMA | ``50`` (нейтрально) | ``50`` |
| ``volatility_cone.compute_rsi_wilder:127`` | тот же Wilder-RSI | ``NaN`` | ``100`` |
| ``rsi_novel._rsi:363`` | **Pine-нормированный**: ``50 + 50·rma(Δ/norm)/rma(|Δ|/norm)`` по 4 ценам O/H/L/C | — | — |

Следствие первых двух: для одного и того же тикера страница, считающая RSI через ``ta``, и страница
``/cone`` показывают **разные значения на одном и том же баре** (на разогреве — 50 против NaN, на
плоском рынке — 50 против 100). Канон делает выбор **явным** и сохраняет обе семантики, чтобы
миграция вызывающих была поведение-сохраняющей.

Третий вариант — это отдельный индикатор (используется в `/rsi-novel`), поэтому у него честное имя
:func:`normalized_rsi`, а не «RSI», чтобы его нельзя было спутать с RSI(14) (находка аудита 02 D-6).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._kernels import as_float_array, rma

__all__ = ["wilder_rsi", "normalized_rsi", "WARMUPS", "FLAT_POLICIES"]

WARMUPS = ("neutral_50", "nan")
FLAT_POLICIES = ("neutral_50", "hundred")


def wilder_rsi(
    close: Sequence[float] | np.ndarray,
    period: int = 14,
    *,
    warmup: str = "neutral_50",
    flat: str = "neutral_50",
) -> np.ndarray:
    """Wilder RSI (0..100), массив той же длины, что вход.

    Parameters
    ----------
    warmup : {"neutral_50", "nan"}
        Чем заполнять разогрев (первые ``period`` баров):
        ``neutral_50`` — как ``ta._wilder_rsi`` (серия короче ``period`` вернёт сплошные 50);
        ``nan`` — как ``volatility_cone.compute_rsi_wilder``.
    flat : {"neutral_50", "hundred"}
        Что отдавать на плоском ряде (нет ни роста, ни падения):
        ``50`` (как ``ta``) или ``100`` (как ``volatility_cone``).
    """
    if warmup not in WARMUPS:
        raise ValueError(f"warmup must be one of {WARMUPS}, got {warmup!r}")
    if flat not in FLAT_POLICIES:
        raise ValueError(f"flat must be one of {FLAT_POLICIES}, got {flat!r}")

    c = as_float_array(close)
    n = len(c)
    warmup_value = 50.0 if warmup == "neutral_50" else np.nan
    if n == 0:
        return np.empty(0, dtype=float)
    if period <= 0:
        return np.full(n, warmup_value, dtype=float)
    if n <= period:
        return np.full(n, warmup_value, dtype=float)

    # delta как в pandas .diff(): первый элемент NaN
    delta = np.empty(n, dtype=float)
    delta[0] = np.nan
    delta[1:] = np.diff(c)
    gain = np.where(np.isnan(delta), np.nan, np.clip(delta, 0.0, None))
    loss = np.where(np.isnan(delta), np.nan, np.clip(-delta, 0.0, None))

    avg_gain = np.full(n, np.nan, dtype=float)
    avg_loss = np.full(n, np.nan, dtype=float)
    avg_gain[period] = float(np.mean(gain[1 : period + 1]))
    avg_loss[period] = float(np.mean(loss[1 : period + 1]))

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0.0, np.nan, avg_gain / avg_loss)
        rsi = 100.0 - (100.0 / (1.0 + rs))

    warmup_mask = np.isnan(avg_gain) & np.isnan(avg_loss)
    rsi = np.where(warmup_mask, warmup_value, rsi)

    flat_mask = (avg_gain == 0.0) & (avg_loss == 0.0)
    rsi = np.where(flat_mask, 50.0 if flat == "neutral_50" else 100.0, rsi)

    # avg_loss == 0 при avg_gain > 0 (только рост) → 100; разогрев не трогаем
    post_warmup_nan = (~warmup_mask) & np.isnan(rsi)
    rsi = np.where(post_warmup_nan, 100.0, rsi)
    return np.clip(rsi, 0.0, 100.0)


def normalized_rsi(
    src: Sequence[float] | np.ndarray,
    length: int,
    *,
    norm_src: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Pine-нормированный RSI (``rsi_novel._rsi``): ``50 + 50·rma(g)/rma(|g|)``, ``g = Δsrc / norm``.

    Parameters
    ----------
    length : int
        Период RMA (в ``rsi_novel`` это ``lenn``).
    norm_src : array-like, optional
        Источник нормировки. Если не задан — нормировка строится по ``src``
        (как в ``rsi_novel`` для close-цены). Значения ``±inf`` не фильтруются до сглаживания —
        это сохраняет поведение оригинала, где ``replace([inf, -inf], nan)`` применяется в конце.
    """
    s = as_float_array(src)
    norm = as_float_array(norm_src) if norm_src is not None else s
    n = len(s)
    if n == 0 or length <= 0:
        return np.full(n, np.nan, dtype=float)

    change = s - np.concatenate(([np.nan], s[:-1]))
    norm_avg = (norm + np.concatenate(([np.nan], norm[:-1]))) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_loss = change / norm_avg
        numerator = rma(gain_loss, length)
        denominator = rma(np.abs(gain_loss), length)
        rsi = 50.0 + 50.0 * (numerator / denominator)
    return np.where(np.isfinite(rsi), rsi, np.nan)
