"""Путь конуса и глобальные уровни (ring: domain).

Путь — это то, что рисуется линией вероятности вдоль горизонта: ``_build_cone_path``
строит его из сглаженного профиля (``_smooth_pull``). ``_global_levels`` сводит уровни
всех экспираций в один список — из него собирается таблица вероятностей.

Сглаживание вынесено отдельной функцией не для красоты: исходный профиль ступенчатый
(уровни дискретны), и без него линия на графике выглядит как разрывы, а не как
вероятностная траектория.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from .model import (
    ConeExpiry,
    GlobalLevel,
    _FLIP_STRENGTH,
    _MIN_STICK_STRENGTH,
    _PULL_AMP,
    _WALL_PULL,
    _composite_strength,
)


def _smooth_pull(
    log_bound: float,
    spot: float,
    walls: list,
    side: str,
    min_strength: float = _MIN_STICK_STRENGTH,
) -> float:
    r"""Граница конуса с плавным учётом объёмов на страйках.

    Логнормальная граница ``log_bound`` «притягивается» к ближайшей стене
    (объёмному страйку / Call·Put Wall) своей стороны. Ближайшая — по
    нормированному расстоянию ``dist/s``, где :math:`s = w \cdot \lambda
    \cdot S_0` (сильнее стена → шире зона). Сила притяжения экспоненциально
    затухает с расстоянием до стены:

    .. math::
        pull = A \, w \, e^{-dist/s}\, dist

    Только одна (ближайшая) стена влияет в каждый момент — суммарное
    притяжение от кластера стен не может «перевернуть» линию. На самой
    стене pull = 0, максимум изгиба — на расстоянии s от неё; слабые
    страйки дают лишь лёгкий изгиб. Кривая остаётся гладкой.
    """
    # Уникальные стены нужной стороны, дедуп по страйку (сильнейшая побеждает)
    dedup: dict[float, object] = {}
    for w in walls:
        if w.side != side or w.strength < min_strength:
            continue
        cur = dedup.get(w.strike)
        if cur is None or w.strength > cur.strength:
            dedup[w.strike] = w
    if not dedup:
        return float(log_bound)

    best_w = None
    best_dist = None
    best_s = None
    for w in dedup.values():
        if side == "resistance":
            if w.strike <= spot or log_bound <= w.strike:
                continue
            dist = log_bound - w.strike
        else:
            if w.strike >= spot or log_bound >= w.strike:
                continue
            dist = w.strike - log_bound
        s = w.strength * _WALL_PULL * spot
        if s <= 1e-9:
            continue
        if best_w is None or (dist / s) < (best_dist / best_s):
            best_w, best_dist, best_s = w, dist, s
    if best_w is None:
        return float(log_bound)

    kernel = math.exp(-best_dist / best_s)
    pull = _PULL_AMP * best_w.strength * kernel * best_dist
    if side == "resistance":
        return float(log_bound - pull)
    return float(log_bound + pull)


def _build_cone_path(
    expirations: list[ConeExpiry],
    spot: float,
    as_of,
    walls: list,
    iv_atm: float,
) -> list[dict]:
    """Плотная полилиния конуса по дням (для плавной отрисовки).

    Для каждого дня t = 1..T_max: ATM-вола интерполируется между
    экспирациями (по dte), считаются логнормальные границы 2σ (верх) и
    3σ (низ), затем применяется плавное притяжение к стенам
    (:func:`_smooth_pull`). Точки по дням дают гладкую кривую без
    ломаных сегментов.
    """
    if not expirations:
        return []
    dtes = [0] + [e.dte for e in expirations]
    vols = [float(iv_atm)] + [float(e.vol_gex) for e in expirations]
    t_max = max(e.dte for e in expirations)
    base = pd.Timestamp(as_of)
    path: list[dict] = []
    for t in range(1, t_max + 1):
        v = float(np.interp(t, dtes, vols))
        sigma_t = v * math.sqrt(t / 365.0)
        u_log = spot * math.exp(2.0 * sigma_t)
        l_log = spot * math.exp(-3.0 * sigma_t)
        path.append({
            "t": t,
            "date": (base + pd.Timedelta(days=t)).date().isoformat(),
            "upper": round(_smooth_pull(u_log, spot, walls, "resistance"), 4),
            "lower": round(_smooth_pull(l_log, spot, walls, "support"), 4),
        })
    return path


def _global_levels(
    chain: pd.DataFrame,
    spot: float,
    top_n: int,
    profile,
    stk_all: pd.DataFrame,
) -> list[GlobalLevel]:
    """Глобальные уровни цепочки: топ по OI+AG + Call/Put Wall + Gamma Flip."""
    oi = chain.groupby("strike")["oi"].sum()
    oi = oi[oi > 0].reset_index()
    if oi.empty:
        return []
    merged = oi.merge(stk_all, on="strike", how="left")
    merged["ag"] = merged["ag"].fillna(0.0)
    merged["gex_net"] = merged["gex_net"].fillna(0.0)
    merged = _composite_strength(merged)

    def _mk(row, kind, strength=None) -> GlobalLevel:
        side = "resistance" if row["strike"] > spot else "support"
        return GlobalLevel(
            strike=float(row["strike"]),
            oi=float(row["oi"]),
            side=side,
            strength=float(strength if strength is not None else row["strength"]),
            kind=kind,
            gex_net=float(row["gex_net"]),
            ag=float(row["ag"]),
        )

    selected: list[GlobalLevel] = []
    ranked = merged.sort_values("strength", ascending=False)
    for _, row in ranked.iterrows():
        if abs(row["strike"] - spot) / spot < 0.002:
            continue
        selected.append(_mk(row, "composite"))
        n_res = sum(1 for l in selected if l.side == "resistance")
        n_sup = sum(1 for l in selected if l.side == "support")
        if n_res >= top_n and n_sup >= top_n:
            break

    def _add_wall(strike, kind, side):
        if strike is None:
            return
        try:
            f = float(strike)
        except (TypeError, ValueError):
            return
        if not math.isfinite(f) or abs(f - spot) / spot < 0.002:
            return
        row = None
        hit = merged.iloc[merged["strike"].sub(f).abs().idxmin()] if len(merged) else None
        if hit is not None and abs(hit["strike"] - f) / max(f, 1e-9) < 0.01:
            row = hit
        if row is None:
            row = {"strike": f, "oi": 0.0, "gex_net": 0.0, "ag": 0.0, "strength": _FLIP_STRENGTH if kind == "gamma_flip" else 0.6}
        max_abs = max(float(stk_all["gex_net"].abs().max()), 1e-12) if not stk_all.empty else 1e-12
        strength = (
            _FLIP_STRENGTH if kind == "gamma_flip"
            else max(0.6, float(abs(row["gex_net"])) / max_abs)
        )
        dup = next((l for l in selected if abs(l.strike - f) < 1e-9), None)
        if dup is not None:
            dup.kind = kind
            dup.strength = strength
        else:
            selected.append(_mk(row, kind, strength))

    _add_wall(profile.call_wall, "call_wall", "resistance")
    _add_wall(profile.put_wall, "put_wall", "support")
    _add_wall(profile.gamma_flip, "gamma_flip",
              "resistance" if (profile.gamma_flip is not None and float(profile.gamma_flip) > spot) else "support")

    resist = sorted((l for l in selected if l.side == "resistance"), key=lambda l: -l.strike)
    support = sorted((l for l in selected if l.side == "support"), key=lambda l: l.strike)
    return resist + support
