"""Вероятности конуса: логнормальная база, поправка на стены и лестница (ring: domain).

Порядок внутри модуля — это порядок вычисления: ``_lognorm_*`` дают чистую логнормальную
вероятность, ``_wall_factor`` добавляет влияние объёмных страйков, ``_append_prob``
собирает итоговую лестницу для уровня.

Модуль самодостаточен: он зависит только от ``model`` и не знает ни об экспирациях, ни о
пути. Именно поэтому вероятности проверяются отдельно — их числа не зависят от того,
как построена цепочка.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.stats import norm
from .model import (
    ConeExpiry,
    ConeExpiryLevel,
)


def _lognorm_z(spot: float, strike: float, sigma_t: float, mu: float) -> float:
    if strike <= 0 or sigma_t <= 1e-12:
        return 0.0
    return (math.log(strike / spot) - mu) / sigma_t


def _lognorm_quantile(spot: float, sigma_t: float, p: float, mu: float) -> float:
    return spot * math.exp(mu + sigma_t * norm.ppf(p))


def _prob_above(spot: float, strike: float, sigma_t: float, mu: float) -> float:
    return float(1.0 - norm.cdf(_lognorm_z(spot, strike, sigma_t, mu)))


def _prob_below(spot: float, strike: float, sigma_t: float, mu: float) -> float:
    return float(norm.cdf(_lognorm_z(spot, strike, sigma_t, mu)))


def _wall_factor(
    strike: float,
    wall: float,
    strength: float,
    spot: float,
    wall_decay: float,
    side: str,
) -> float:
    """Затухающий фактор стены: ``exp(-λ·w·(dist + S0)/S0)``.

    На самом уровне (dist = 0) фактор = ``exp(-λ·w)`` — дискретная ступенька;
    дальше затухает экспоненциально. ``wall_decay = 0`` → фактор 1.
    """
    if wall_decay <= 0:
        return 1.0
    if side == "resistance":
        dist = max(0.0, strike - wall)
    else:
        dist = max(0.0, wall - strike)
    return math.exp(-wall_decay * strength * (dist + spot) / spot)


def _append_prob(
    lvl,
    walls: list[ConeExpiryLevel],
    exp: ConeExpiry,
    spot: float,
    r: float,
    q: float,
    wall_decay: float,
) -> None:
    """Посчитать вероятности уровня для ОДНОЙ экспирации со ступеньками.

    Каскад: для сопротивления K учитываются все сопротивления ``walls`` этой
    экспирации с R_i ≤ K; для поддержки — все поддержки с P_i ≥ K. Каждая
    стена добавляет ступеньку (пропорционально своей силе), хвост режется
    произведением факторов.
    """
    if lvl.side == "resistance":
        cascade = [w for w in walls
                   if w.side == "resistance" and w.strike <= lvl.strike + 1e-9]
    else:
        cascade = [w for w in walls
                   if w.side == "support" and w.strike >= lvl.strike - 1e-9]

    sigma_t = exp.vol_gex * math.sqrt(max(exp.dte, 1) / 365.0)
    K = lvl.strike
    mu = (r - q - 0.5 * exp.vol_gex * exp.vol_gex) * (max(exp.dte, 1) / 365.0)

    if sigma_t <= 1e-12:
        p_above = 1.0 if K < spot else 0.0
        p_below = 1.0 - p_above
        lvl.probs.append({
            "dte": exp.dte, "date": exp.date,
            "p_above": p_above, "p_below": p_below,
            "p_above_base": p_above, "p_touch": p_above,
            "drop_pp": 0.0,
        })
        return

    p_above_base = _prob_above(spot, K, sigma_t, mu)
    p_below_base = _prob_below(spot, K, sigma_t, mu)

    if lvl.side == "resistance":
        factor = 1.0
        for w in cascade:
            factor *= _wall_factor(K, w.strike, w.strength, spot, wall_decay, "resistance")
        p_above = min(1.0, p_above_base * factor)
        p_below = 1.0 - p_above
        drop_pp = max(0.0, (p_above_base - p_above) * 100.0)
        p_touch = min(1.0, 2.0 * p_above)
    else:
        factor = 1.0
        for w in cascade:
            factor *= _wall_factor(K, w.strike, w.strength, spot, wall_decay, "support")
        p_below = min(1.0, p_below_base * factor)
        p_above = 1.0 - p_below
        drop_pp = max(0.0, (p_below_base - p_below) * 100.0)
        p_touch = min(1.0, 2.0 * p_below)

    lvl.probs.append({
        "dte": exp.dte, "date": exp.date,
        "p_above": round(p_above, 6), "p_below": round(p_below, 6),
        "p_above_base": round(p_above_base, 6),
        "p_touch": round(p_touch, 6),
        "drop_pp": round(drop_pp, 4),
    })
