"""Модель конуса: константы, структуры и общие примитивы (ring: domain).

Вынесено из ``gex/gexcone.py`` (итерация 40). Здесь то, что нужно **всем** частям конуса:
константы расчёта, четыре структуры данных и примитивы, на которые опираются и
экспирации, и путь, и вероятности (гамма страйка, сила уровня, множитель волатильности).

Отдельный модуль, а не часть фасада: ``gexcone`` импортирует три части конуса, поэтому
общее для них не может лежать в нём — был бы цикл импортов. Раскладка следует графу:
``model`` ← ``probabilities`` ← ``expiries``; ``model`` ← ``path``.

Константы перенесены дословно, включая знаки (``_POSITIVE_GAMMA_MULT`` отрицательный:
это знак дилерской позиции, а не опечатка) — числа закреплены эталоном конуса.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

_ATM_BAND = 0.10
_IV_FLOOR = 0.02
_IV_CEIL = 3.0
DEFAULT_TOP_OI_PER_EXPIRY = 4
DEFAULT_GLOBAL_TOP = 3
_POSITIVE_GAMMA_MULT = -0.05
_NEGATIVE_GAMMA_MULT = +0.20
_W_OI = 0.5
_FLIP_STRENGTH = 0.8
_MIN_STICK_STRENGTH = 0.1
_WALL_PULL = 0.07
_PULL_AMP = 2.0
DEFAULT_OI_QUANTILE = 0.9
_HV_PERIOD_DAYS = 30
_OTM_SIGMAS = 3.0
from ...greeks import (
    bs_gamma,
)


@dataclass
class ConeExpiryLevel:
    """Топ-уровень одной экспирации (страйк со ступенькой вероятности).

    Attributes
    ----------
    strike : float
        Цена страйка.
    oi : float
        Суммарный открытый интерес (call + put) на страйке.
    side : str
        ``resistance`` (strike > spot) или ``support`` (strike < spot).
    strength : float
        Композитная сила (0..1): 0.5·OI/maxOI + 0.5·AG/maxAG; для GEX-стен —
        |GEX|/max|GEX| экспирации.
    gex_net : float
        Net GEX страйка ($/%spot): суммарная дилерская гамма-экспозиция.
    ag : float
        Aggregate Gamma страйка: |GEX_call| + |GEX_put|.
    kind : str
        ``composite`` — топ по OI+AG; ``call_wall``/``put_wall`` — GEX-стены;
        ``gamma_flip`` — уровень нулевого GEX (глобально).
    probs : list[dict]
        Вероятности по экспирациям (индекс совпадает с ``expirations``):
        ``p_above`` — скорректированная P(выше), ``p_below`` — P(ниже),
        ``p_above_base`` — чистая логнормальная, ``p_touch`` — вероятность
        касания, ``drop_pp`` — ступенька в п.п.
    """

    strike: float
    oi: float
    side: str
    strength: float
    gex_net: float = 0.0
    ag: float = 0.0
    kind: str = "composite"
    probs: list[dict] = field(default_factory=list)


@dataclass
class ConeExpiry:
    """Одна экспирация конуса: IV, GEX-метрики, границы σ, квантили и уровни."""

    date: str
    dte: int
    iv_atm: float
    vol_gex: float
    gex_net: float
    ag: float
    gamma_score: float
    regime: str
    median: float
    upper_1sd: float
    lower_1sd: float
    upper_2sd: float
    lower_2sd: float
    upper_3sd: float
    lower_3sd: float
    p10: float
    p25: float
    p75: float
    p90: float
    expected_move_1sd: float
    ag_weight: float = 1.0
    levels: list[ConeExpiryLevel] = field(default_factory=list)
    # Ступенчатые границы конуса (рассчитываются после сборки уровней):
    # линия «залипает» плато на объёмном страйке / стене, пока логнормальная
    # граница не пробьёт её (см. ``_STICK_LAMBDA``).
    upper_cone: float = 0.0
    lower_cone: float = 0.0
    upper_stick: Optional[dict] = None
    lower_stick: Optional[dict] = None


@dataclass
class GlobalLevel:
    """Глобальный уровень всей цепочки (горизонтальная линия на графике).

    ``kind``: ``composite`` — топ по OI+AG, ``call_wall``/``put_wall`` — GEX-стены
    цепочки, ``gamma_flip`` — уровень нулевого GEX.
    """

    strike: float
    oi: float
    side: str
    strength: float
    kind: str = "composite"
    gex_net: float = 0.0
    ag: float = 0.0
    probs: list[dict] = field(default_factory=list)


@dataclass
class GexConeData:
    """Итоговые данные конуса: метаданные + экспирации + глобальные уровни."""

    ticker: str
    spot: float
    as_of: Optional[str]
    r: float
    q: float
    iv_atm: float
    wall_decay: float
    regime: str
    net_gex: float
    total_ag: float = 0.0
    gamma_score: float = 0.0
    vol_mult: float = 1.0
    call_wall: Optional[float] = None
    put_wall: Optional[float] = None
    gamma_flip: Optional[float] = None
    oi_quantile: float = DEFAULT_OI_QUANTILE
    hv: Optional[float] = None
    atr: Optional[float] = None
    axis_min: Optional[float] = None
    axis_max: Optional[float] = None
    cone_path: list[dict] = field(default_factory=list)
    expirations: list[ConeExpiry] = field(default_factory=list)
    levels: list[GlobalLevel] = field(default_factory=list)


def _finite_or_none(v) -> Optional[float]:
    """np.nan/None → None (JSON-safe)."""
    if v is None:
        return None
    try:
        return float(v) if math.isfinite(float(v)) else None
    except (TypeError, ValueError):
        return None


def _strike_gex(
    sub: pd.DataFrame,
    spot: float,
    r: float,
    q: float,
    per_contract: int,
    call_sign: float,
    put_sign: float,
) -> pd.DataFrame:
    """Per-strike GEX/AG для подмножества цепочки.

    Возвращает DataFrame с колонками ``strike``, ``gex_net`` (суммарный GEX
    в $/%spot) и ``ag`` (Aggregate Gamma = Σ|GEX|). Модель — как в
    :mod:`gex.metrics` (BSM-гамма × знак дилера × OI × 100 × S² × 0.01).
    """
    df = sub.copy()
    gamma = bs_gamma(spot, df["strike"], df["T"], r, df["iv"], q)
    sign = np.where(df["type"].values == "C", call_sign, put_sign)
    gex = (
        sign * gamma * per_contract * (spot ** 2) * 0.01 * df["oi"].values
    )
    g = df.assign(gex=gex).groupby("strike")["gex"]
    out = pd.DataFrame({
        "gex_net": g.sum(),
        "ag": g.apply(lambda s: float(np.abs(s).sum())),
    }).reset_index()
    return out


def _gamma_score(net_gex: float, total_ag: float) -> float:
    """γ-score 0..100: |Net GEX| / AG · 100 (доля направленной экспозиции)."""
    if total_ag <= 1e-12:
        return 0.0
    return float(min(100.0, abs(net_gex) / total_ag * 100.0))


def _vol_mult(regime: str, gamma_score: float) -> float:
    """Множитель волатильности конуса по GEX-режиму экспирации."""
    s = max(0.0, min(100.0, gamma_score)) / 100.0
    if regime == "NEGATIVE":
        return 1.0 + _NEGATIVE_GAMMA_MULT * s
    return 1.0 + _POSITIVE_GAMMA_MULT * s


def _composite_strength(stk: pd.DataFrame) -> pd.DataFrame:
    """Композитная сила уровня: 0.5·OI/maxOI + 0.5·AG/maxAG (0..1)."""
    df = stk.copy()
    max_oi = float(df["oi"].max()) if len(df) and df["oi"].max() > 0 else 0.0
    max_ag = float(df["ag"].max()) if len(df) and df["ag"].max() > 0 else 0.0
    oi_norm = df["oi"] / max_oi if max_oi > 0 else df["oi"] * 0.0
    ag_norm = df["ag"] / max_ag if max_ag > 0 else df["ag"] * 0.0
    df["strength"] = _W_OI * oi_norm + (1.0 - _W_OI) * ag_norm
    return df
