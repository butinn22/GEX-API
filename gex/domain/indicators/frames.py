"""Pandas-фасад индикаторов (ring: domain) — обёртки с сигнатурами прежних реализаций.

Зачем: вызывающие модули передают ``DataFrame`` с колонками ``High/Low/Close``
(``hybrid_trend``, ``novel_candles``) либо ``high/low/close`` (``trend_regime``,
``breadth_imoex_service``). Чтобы миграция была механической и числа не менялись, фасад
повторяет **точные** сигнатуры и политики фоллбэка каждой семьи:

* :func:`compute_atr` — ``hybrid_trend.compute_atr`` / ``novel_candles.compute_atr`` (массив, без прогрева);
* :func:`atr_scalar_wilder` — ``trendlines._wilder_atr`` (скаляр, фоллбэк ``mean(H-L)`` → ``1.0``);
* :func:`atr_scalar_wilder_or_none` — ``gexcone.atr_14`` (None при нехватке истории);
* :func:`atr_series_wilder` — ``trend_regime`` / ``breadth_imoex_service`` (Series, прогрев = period).

Выбор метода и прогрева теперь **явный**: раньше он был скрыт в реализации, из-за чего
``/gexcone`` (среднее TR) и ``/ta`` (рекурсия Wilder) показывали разные ATR для одного тикера.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import atr as _atr
from . import rsi as _rsi

__all__ = [
    "compute_atr",
    "atr_scalar_wilder",
    "atr_scalar_wilder_or_none",
    "atr_scalar_mean",
    "atr_series_wilder",
]


def _column(df: pd.DataFrame, *candidates: str) -> np.ndarray:
    """Достать колонку по любому из допустимых имён (``High``/``high`` и т.п.)."""
    for name in candidates:
        if name in df.columns:
            return np.asarray(df[name], dtype=float)
    raise KeyError(f"в DataFrame нет ни одной из колонок {candidates}")


def compute_atr(
    df: pd.DataFrame,
    period: int = 14,
    *,
    method: str = "wilder",
    warmup: str = "none",
) -> np.ndarray:
    """ATR-массив по колонкам OHLC. По умолчанию — поведение ``hybrid_trend.compute_atr``."""
    return _atr.atr(
        _column(df, "High", "high"),
        _column(df, "Low", "low"),
        _column(df, "Close", "close"),
        period,
        method=method,
        warmup=warmup,
    )


def atr_series_wilder(
    df: pd.DataFrame,
    period: int = 14,
    *,
    warmup: str = "period",
    method: str = "wilder",
) -> pd.Series:
    """ATR как ``pd.Series`` с индексом ``df`` (поведение ``trend_regime``/``breadth_imoex_service``)."""
    values = compute_atr(df, period, method=method, warmup=warmup)
    return pd.Series(values, index=df.index, name="atr")


def atr_scalar_wilder(df: pd.DataFrame, period: int = 14) -> float:
    """ATR последнего бара с фоллбэком ``mean(H-L)`` → ``1.0`` (поведение ``trendlines._wilder_atr``)."""
    value = _atr.atr_last(
        _column(df, "High", "high"),
        _column(df, "Low", "low"),
        _column(df, "Close", "close"),
        period,
        method="wilder",
        fallback="hl_mean_then_one",
    )
    return float(value) if value is not None else 1.0


def atr_scalar_wilder_or_none(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """ATR последнего бара как у ``macd_trend._atr``: ``None`` при нехватке истории/NaN."""
    return _atr.atr_last(highs, lows, closes, period, method="wilder", fallback="none")


def atr_scalar_mean(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """ATR как среднее TR за ``period`` — поведение ``gexcone.atr_14`` ('''None''' при нехватке)."""
    return _atr.atr_last(
        highs,
        lows,
        closes,
        period,
        method="mean",
        fallback="none_if_insufficient",
    )


# ── RSI (поведение прежних реализаций с pandas-обёрткой) ─────────────────────

def wilder_rsi_series(
    close: pd.Series,
    period: int = 14,
    *,
    warmup: str = "neutral_50",
    flat: str = "neutral_50",
) -> pd.Series:
    """RSI как ``pd.Series`` с индексом входа.

    * ``warmup="neutral_50", flat="neutral_50"`` — поведение ``ta._wilder_rsi``;
    * ``warmup="nan", flat="hundred"`` — поведение ``volatility_cone.compute_rsi_wilder``.
    """
    values = _rsi.wilder_rsi(close, period, warmup=warmup, flat=flat)
    return pd.Series(values, index=close.index, name="rsi")


def normalized_rsi_series(
    src: pd.Series,
    length: int,
    *,
    norm_src: pd.Series | None = None,
) -> pd.Series:
    """Pine-нормированный RSI как ``pd.Series`` (поведение ``rsi_novel._rsi``)."""
    values = _rsi.normalized_rsi(src, length, norm_src=norm_src)
    return pd.Series(values, index=src.index, name="rsi_normalized")
