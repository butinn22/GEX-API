"""Примитивы индикаторов с pandas-обёртками (ring: domain).

Тонкий слой над :mod:`gex.domain.indicators._kernels`: сохраняет **точные сигнатуры** прежних
реализаций (`rsi_novel.pine_rma(series, length) -> pd.Series` и т.д.), чтобы миграция вызывающих
кода была механической и поведение не менялось.

Правило кольца: только numpy/pandas, никакого I/O. Всё, что нужно от внешнего мира, приходит
аргументами.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _kernels as k

__all__ = [
    "pine_rma",
    "pine_ema",
    "pine_sma",
    "pine_stdev",
    "pine_bb",
    "ema",
    "sma",
    "rma",
]


def _wrap(values: np.ndarray, like: pd.Series | None) -> pd.Series:
    """Вернуть результат как Series с индексом исходного ряда (или без индекса)."""
    if like is not None and getattr(like, "index", None) is not None:
        return pd.Series(values, index=like.index)
    return pd.Series(values)


def pine_rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's Moving Average (``ta.rma``) — каноническая реализация."""
    return _wrap(k.rma(series, length), series if isinstance(series, pd.Series) else None)


def pine_ema(series: pd.Series, length: int) -> pd.Series:
    """EMA со SMA-сидом (``ta.ema``) — каноническая реализация."""
    return _wrap(k.ema(series, length), series if isinstance(series, pd.Series) else None)


def pine_sma(series: pd.Series, length: int) -> pd.Series:
    """SMA (``ta.sma``) — каноническая реализация."""
    return _wrap(k.sma(series, length), series if isinstance(series, pd.Series) else None)


def pine_stdev(series: pd.Series, length: int) -> pd.Series:
    """Стандартное отклонение (``ta.stdev``, biased) — каноническая реализация."""
    return _wrap(k.stdev(series, length, ddof=0), series if isinstance(series, pd.Series) else None)


def pine_bb(series: pd.Series, length: int, mult: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Полосы Боллинджера (``ta.bb``): (basis, upper, lower)."""
    basis, upper, lower = k.bollinger(series, length, mult)
    like = series if isinstance(series, pd.Series) else None
    return _wrap(basis, like), _wrap(upper, like), _wrap(lower, like)


def ema(series: pd.Series, length: int) -> pd.Series:
    """Короткий алиас: канонический EMA (см. :func:`pine_ema`)."""
    return pine_ema(series, length)


def sma(series: pd.Series, length: int) -> pd.Series:
    """Короткий алиас: канонический SMA (см. :func:`pine_sma`)."""
    return pine_sma(series, length)


def rma(series: pd.Series, length: int) -> pd.Series:
    """Короткий алиас: канонический RMA (см. :func:`pine_rma`)."""
    return pine_rma(series, length)
