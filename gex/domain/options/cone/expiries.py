"""Экспирации конуса: отбор цепочки и построение уровней (ring: domain).

Две задачи в одном модуле, потому что вторая без первой невозможна:

* отбор — что попадает в расчёт: квантиль по OI (``_filter_oi_quantile``) и отсечение
  дальних OTM (``_filter_extreme_otm``);
* построение — экспирации и их топ-уровни (``_build_expirations``, ``_expiry_levels``)
  плюс ATM-волатильность конуса (``_atm_iv``).

Отсечение по квантилю важно смыслом, а не оптимизацией: хвост OI — это дальние страйки,
которые в горизонт конуса не входят, но при подсчёте ATM-волатильности сдвигают её.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd
from ...data_loader import (
    OptionSnapshot,
)
from .model import (
    ConeExpiry,
    ConeExpiryLevel,
    _ATM_BAND,
    _IV_CEIL,
    _IV_FLOOR,
    _OTM_SIGMAS,
    _composite_strength,
    _gamma_score,
    _vol_mult,
)
from .probabilities import (
    _lognorm_quantile,
)


def _filter_oi_quantile(chain: pd.DataFrame, oi_quantile: float) -> pd.DataFrame:
    """Оставить страйки, покрывающие ``oi_quantile`` суммарного OI.

    Страйки сортируются по суммарному OI (call+put) по убыванию и
    накапливаются, пока не наберётся ``oi_quantile`` (0.9 = 90%) всего
    открытого интереса цепочки. Остальные (обычно дальние, малоликвидные)
    отбрасываются. Если квант ≥ 1 — фильтр выключен.
    """
    if oi_quantile >= 1.0 or len(chain) == 0:
        return chain
    oi_by_strike = chain.groupby("strike")["oi"].sum()
    total = float(oi_by_strike.sum())
    if total <= 0:
        return chain
    target = total * max(0.0, min(1.0, oi_quantile))
    keep: list[float] = []
    acc = 0.0
    for strike, o in oi_by_strike.sort_values(ascending=False).items():
        keep.append(float(strike))
        acc += float(o)
        if acc >= target:
            break
    keep_set = set(keep)
    return chain[chain["strike"].isin(keep_set)].copy()


def _filter_extreme_otm(
    chain: pd.DataFrame,
    spot: float,
    hv: Optional[float],
    iv_atm: Optional[float],
    atr: Optional[float],
    horizon_years: Optional[float] = None,
) -> pd.DataFrame:
    r"""Отбросить «явно OTM» страйки: дальше 3σ HV / 3σ IV / 3×ATR от спота.

    Страйк исключается из построения конуса, если его относительное
    расстояние от спота больше самого широкого из порогов:

    * :math:`3\,\sigma_{HV}\sqrt{T}` — 3σ исторической волатильности;
    * :math:`3\,\sigma_{IV}\sqrt{T}` — 3σ имплайд-волатильности;
    * :math:`3\,ATR / S_0` — три дневных средних истинных диапазона.

    Такие страйки (дальний мусор, LEAPS с мизерной гаммой) не должны
    влиять на конус, уровни и стены. Пороги, для которых нет данных
    (hv/iv/atr отсутствуют), пропускаются; если нет ни одного — фильтр
    выключен.
    """
    if len(chain) == 0 or spot <= 0:
        return chain
    if horizon_years is None:
        horizon_years = float(chain["T"].max()) if "T" in chain.columns else 14.0 / 365.0
    sqrt_t = math.sqrt(max(horizon_years, 1.0 / 365.0))
    limits: list[float] = []
    if hv and hv > 0:
        limits.append(_OTM_SIGMAS * hv * sqrt_t)
    if iv_atm and iv_atm > 0:
        limits.append(_OTM_SIGMAS * iv_atm * sqrt_t)
    if atr and atr > 0:
        limits.append(_OTM_SIGMAS * atr / spot)
    if not limits:
        return chain
    limit = max(limits)
    dist = (chain["strike"] / spot - 1.0).abs()
    return chain[dist <= limit].copy()


def expirations_levels_iter(expirations: list[ConeExpiry]):
    """Итератор по всем per-expiry уровням (для обрезки probs)."""
    for exp in expirations:
        yield from exp.levels


def _atm_iv(chain: pd.DataFrame, spot: float) -> float:
    """OI-взвешенная ATM-волатильность: страйки в ±10% от спота.

    Защита от глюков yfinance: IV ниже ``_IV_FLOOR`` или выше ``_IV_CEIL``
    отбрасываются. Fallback — медиана IV по валидным контрактам.
    """
    sub = chain[(chain["oi"] > 0) & (chain["iv"] >= _IV_FLOOR) & (chain["iv"] <= _IV_CEIL)]
    if sub.empty:
        return 0.0
    band = sub[(sub["strike"] >= spot * (1 - _ATM_BAND)) & (sub["strike"] <= spot * (1 + _ATM_BAND))]
    if band.empty:
        band = sub
    w = band["oi"].values
    if w.sum() <= 0:
        return float(band["iv"].median())
    return float(np.average(band["iv"].values, weights=w))


def _build_expirations(
    chain: pd.DataFrame,
    snapshot: OptionSnapshot,
    spot: float,
    r: float,
    q: float,
    top_oi: int,
    stk_all: pd.DataFrame,
) -> list[ConeExpiry]:
    """Построить экспирации, группируя цепочку по календарной дате.

    yfinance может отдавать несколько разных T для одной даты экспирации
    (разное время суток при кэшировании) — объединяем строки одной даты.
    Даты с IV ниже ``_IV_FLOOR`` (глюки данных) пропускаются.

    Для каждой даты: OI-взвешенная ATM-IV → GEX-режим/γ-score экспирации →
    ``vol_gex`` (скорректированная вола) → границы σ/квантили по ``vol_gex``.

    GEX/AG даты НЕ пересчитываются: берутся из канонического ``stk_all``
    (адаптер per-strike профиля единого движка GEX) — подмножеством строк,
    чьи страйки входят в цепочку даты.
    """
    chain = chain.copy()
    base = pd.Timestamp(snapshot.as_of)

    chain["_dte"] = chain["T"].apply(lambda t: max(1, int(round(float(t) * 365.0))))
    chain["_date"] = chain["_dte"].apply(lambda d: (base + pd.Timedelta(days=d)).date().isoformat())

    # Предварительный проход: AG по каждой дате — чтобы коррекция волатильности
    # не раздувалась на «пустых» экспирациях с мизерным OI (там γ-score может
    # быть 100% при почти нулевом GEX). Взвешиваем score по AG/maxAG дат.
    ag_by_date: dict[str, float] = {}
    for date, sub in chain.groupby("_date"):
        stk = stk_all[stk_all["strike"].isin(sub["strike"].unique())]
        ag_by_date[date] = float(stk["ag"].sum()) if not stk.empty else 0.0
    max_ag_date = max(ag_by_date.values()) if ag_by_date else 0.0

    out: list[ConeExpiry] = []
    for date, sub in chain.groupby("_date"):
        T = float(sub["T"].max())
        if T <= 0:
            continue
        iv = _atm_iv(sub, spot)
        if iv < _IV_FLOOR:
            continue
        dte = int(sub["_dte"].iloc[0])

        # GEX/AG экспирации из канонического stk_all → режим → множитель → вола
        stk = stk_all[stk_all["strike"].isin(sub["strike"].unique())]
        gex_net = float(stk["gex_net"].sum()) if not stk.empty else 0.0
        ag = float(stk["ag"].sum()) if not stk.empty else 0.0
        score = _gamma_score(gex_net, ag)
        # Вес AG: пустые даты почти не двигают конус (защита от мусорных IV)
        ag_weight = min(1.0, ag / max_ag_date) if max_ag_date > 0 else 0.0
        score_eff = score * ag_weight
        regime = "POSITIVE" if gex_net >= 0 else "NEGATIVE"
        vol_gex = iv * _vol_mult(regime, score_eff)

        sigma_t = vol_gex * math.sqrt(T)
        drift = (r - q) * T
        mu = (r - q - 0.5 * vol_gex * vol_gex) * T

        def _band(k: float) -> float:
            return spot * math.exp(drift + k * sigma_t)

        # Уровни этой даты: топ по OI+AG + GEX-стены экспирации
        lvls = _expiry_levels(sub, spot, top_oi, stk)

        out.append(ConeExpiry(
            date=date,
            dte=dte,
            iv_atm=iv,
            vol_gex=vol_gex,
            gex_net=gex_net,
            ag=ag,
            gamma_score=score,
            ag_weight=ag_weight,
            regime=regime,
            median=spot * math.exp(drift),
            upper_1sd=_band(1), lower_1sd=_band(-1),
            upper_2sd=_band(2), lower_2sd=_band(-2),
            upper_3sd=_band(3), lower_3sd=_band(-3),
            p10=_lognorm_quantile(spot, sigma_t, 0.10, mu),
            p25=_lognorm_quantile(spot, sigma_t, 0.25, mu),
            p75=_lognorm_quantile(spot, sigma_t, 0.75, mu),
            p90=_lognorm_quantile(spot, sigma_t, 0.90, mu),
            expected_move_1sd=spot * sigma_t,
            levels=lvls,
        ))

    out = sorted(out, key=lambda e: e.dte)
    if len(out) < 1:
        raise ValueError("В цепочке нет экспираций с T > 0 и корректной IV.")
    return out


def _expiry_levels(
    sub: pd.DataFrame,
    spot: float,
    top_n: int,
    stk: pd.DataFrame,
) -> list[ConeExpiryLevel]:
    """Уровни одной экспирации: топ-N по OI+AG с каждой стороны + GEX-стены.

    Сначала считаем OI по страйкам, мёржим с per-strike GEX/AG, ранжируем по
    композитной силе (``_composite_strength``). Затем добавляем GEX-стены даты:
    страйк с максимальным положительным net GEX (call wall, сопротивление) и
    минимальным отрицательным (put wall, поддержка) — если ещё не выбраны.
    """
    oi = sub.groupby("strike")["oi"].sum()
    oi = oi[oi > 0].reset_index()
    if oi.empty:
        return []
    oi_map = dict(zip(oi["strike"], oi["oi"]))
    merged = oi.merge(stk, on="strike", how="left")
    merged["ag"] = merged["ag"].fillna(0.0)
    merged["gex_net"] = merged["gex_net"].fillna(0.0)
    merged = _composite_strength(merged)

    def _mk(row, kind, strength=None, oi_val=None) -> ConeExpiryLevel:
        side = "resistance" if row["strike"] > spot else "support"
        return ConeExpiryLevel(
            strike=float(row["strike"]),
            oi=float(oi_val if oi_val is not None else row["oi"]),
            side=side,
            strength=float(strength if strength is not None else row["strength"]),
            gex_net=float(row["gex_net"]),
            ag=float(row["ag"]),
            kind=kind,
        )

    selected: list[ConeExpiryLevel] = []
    # Топ-N по композитной силе с каждой стороны (спот пропускаем)
    ranked = merged.sort_values("strength", ascending=False)
    for _, row in ranked.iterrows():
        if abs(row["strike"] - spot) / spot < 0.002:
            continue
        lvl = _mk(row, "composite")
        selected.append(lvl)
        n_res = sum(1 for l in selected if l.side == "resistance")
        n_sup = sum(1 for l in selected if l.side == "support")
        if n_res >= top_n and n_sup >= top_n:
            break

    # GEX-стены экспирации (если есть знак в net GEX)
    stk_sorted = stk.sort_values("gex_net", ascending=False)
    pos = stk_sorted[stk_sorted["gex_net"] > 0]
    neg = stk_sorted[stk_sorted["gex_net"] < 0]
    for wall_df, kind, side in (
        (pos, "call_wall", "resistance"),
        (neg, "put_wall", "support"),
    ):
        if wall_df.empty:
            continue
        row = wall_df.iloc[0]
        if abs(row["strike"] - spot) / spot < 0.002:
            continue
        max_abs = max(float(stk["gex_net"].abs().max()), 1e-12)
        strength = float(abs(row["gex_net"])) / max_abs
        # Если страйк уже выбран как composite — повышаем kind до стены
        dup = next((l for l in selected if abs(l.strike - row["strike"]) < 1e-9), None)
        if dup is not None:
            dup.kind = kind
            dup.strength = strength
        else:
            selected.append(_mk(row, kind, strength, oi_map.get(float(row["strike"]), 0.0)))

    # Сортировка: сопротивления сверху вниз, поддержки снизу вверх
    resist = sorted((l for l in selected if l.side == "resistance"), key=lambda l: -l.strike)
    support = sorted((l for l in selected if l.side == "support"), key=lambda l: l.strike)
    return resist + support
