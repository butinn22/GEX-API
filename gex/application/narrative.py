"""Аналитическая сводка GEX-профиля: структура рынка + уровни + вероятности разворота + доверительные диапазоны по срокам экспирации.

Модуль преобразует расширенный GEX-отчёт в кристально ясную сводку:
  1. **Рыночная структура**: направление (HH/HL/LH/LL), глобальный режим (EMA100, GammaFlip);
  2. **Ключевые уровни**: Call Wall, Put Wall, Gamma Flip, Max Pain с расстоянием от spot;
  3. **Вероятность разворота на уровнях**: на основе плотности гаммы (AG) и рыночного режима;
  4. **Доверительные диапазоны по срокам экспирации** (10 / 30 / 60 дней): поддержка/сопротивление
     по ключевым объёмам в окне, свойственным данному сроку;
  5. **Итоговый доверительный интервал** на основе GEX и ключевых страйков.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from gex.application.extended import ExtendedGEXReport, ExtendedStrike

logger = logging.getLogger(__name__)

_EMA100_SPAN = 100
_WINDOW_BY_DAYS = {10: 0.05, 30: 0.10, 60: 0.15}
_NARRATIVE_TF = "1d"


@dataclass
class LevelProbability:
    strike: float
    label: str          # "Call Wall" | "Put Wall" | "Gamma Flip"
    distance_pct: float # (strike - spot) / spot * 100
    reversal_pct: float # 0..100 — оценка вероятности разворота на уровне
    reasoning: str      # коротко: почему такая вероятность


@dataclass
class ExpiryBand:
    days: int           # 10 | 30 | 60
    support: float
    resistance: float
    band_width_pct: float
    n_strikes: int


@dataclass
class GEXNarrative:
    # --- Сводка ---
    symbol: str
    spot: float
    net_gex: float
    regime: str          # POSITIVE | NEGATIVE
    summary: str         # 1-2 предложения: куда движется рынок

    # --- Рыночная структура ---
    direction: str       # BULLISH | BEARISH | NEUTRAL
    structure: str       # "HH=12 HL=10 LH=8 LL=6 — BULLISH (восходящий)"
    ema_100: Optional[float]
    ema_100_distance_pct: Optional[float]   # (spot - EMA100)/spot*100
    global_regime: str   # "бычий (цена > EMA100 и > GammaFlip)" | "медвежий" | "смешанный"

    # --- Ключевые уровни ---
    call_wall: Optional[float] = None
    call_wall_dist_pct: Optional[float] = None
    put_wall: Optional[float] = None
    put_wall_dist_pct: Optional[float] = None
    gamma_flip: Optional[float] = None
    gamma_flip_dist_pct: Optional[float] = None
    max_pain: Optional[float] = None
    max_pain_dist_pct: Optional[float] = None

    # --- Вероятность разворота на уровнях ---
    level_probabilities: list[LevelProbability] = field(default_factory=list)

    # --- Доверительный диапазон по срокам экспирации ---
    expiry_bands: list[ExpiryBand] = field(default_factory=list)

    # --- Итоговый доверительный интервал ---
    confidence_low: Optional[float] = None
    confidence_high: Optional[float] = None
    confidence_center: Optional[float] = None

    # --- EMA-статус ---
    ema_cluster: bool = False
    ema_cluster_note: str = ""

    # --- Структура HH/HL (сырые данные) ---
    fractal_structure: str = ""


def build_narrative(
    report: ExtendedGEXReport,
    ticker: str,
    df_1d: Optional[pd.DataFrame] = None,
) -> Optional[GEXNarrative]:
    if not report.per_strike:
        return None
    spot = report.spot
    if not np.isfinite(spot) or spot <= 0:
        return None

    # --- Загрузка OHLCV ---
    if df_1d is None:
        df_1d = _try_load_ohlcv(ticker)

    # --- EMA100 ---
    ema_100 = None
    ema_100_dist = None
    if df_1d is not None and not df_1d.empty and len(df_1d) >= _EMA100_SPAN:
        ema_100 = float(df_1d["Close"].ewm(span=_EMA100_SPAN, adjust=False).mean().iloc[-1])
        ema_100_dist = (spot - ema_100) / spot * 100.0 if ema_100 else None

    # --- Рыночная структура ---
    fractal_str = _compute_fractal(df_1d) if df_1d is not None else "нет данных"
    direction = _direction_from_fractal(fractal_str)
    global_regime = _global_regime_label(spot, ema_100, report.zero_gamma)

    # --- Ключевые уровни ---
    cw = float(report.call_wall) if np.isfinite(report.call_wall) else None
    pw = float(report.put_wall) if np.isfinite(report.put_wall) else None
    gf = float(report.zero_gamma) if np.isfinite(report.zero_gamma) else None
    mp = float(report.max_pain) if report.max_pain is not None and np.isfinite(report.max_pain) else None

    # --- Вероятность разворота на уровнях ---
    level_probs = _level_reversal_probs(report, spot, cw, pw, gf)

    # --- Доверительные диапазоны по срокам экспирации ---
    expiry_bands = _expiry_bands(report, spot)

    # --- Итоговый доверительный интервал ---
    conf_low, conf_high = _confidence_interval(report, spot)

    # --- Сводка ---
    summary = _summary_line(ticker, spot, report.net_gex, report.regime, direction,
                            fractal_str, level_probs, expiry_bands)

    # --- EMA кластер ---
    ema_cluster, ema_cluster_note = _ema_cluster_info(df_1d, report, spot)

    return GEXNarrative(
        symbol=ticker,
        spot=float(spot),
        net_gex=float(report.net_gex),
        regime=report.regime,
        summary=summary,
        direction=direction,
        structure=fractal_str,
        ema_100=ema_100,
        ema_100_distance_pct=ema_100_dist,
        global_regime=global_regime,
        call_wall=cw,
        call_wall_dist_pct=((cw - spot) / spot * 100.0) if cw else None,
        put_wall=pw,
        put_wall_dist_pct=((pw - spot) / spot * 100.0) if pw else None,
        gamma_flip=gf,
        gamma_flip_dist_pct=((gf - spot) / spot * 100.0) if gf else None,
        max_pain=mp,
        max_pain_dist_pct=((mp - spot) / spot * 100.0) if mp else None,
        level_probabilities=level_probs,
        expiry_bands=expiry_bands,
        confidence_low=conf_low,
        confidence_high=conf_high,
        confidence_center=((conf_low + conf_high) / 2.0) if conf_low and conf_high else None,
        ema_cluster=ema_cluster,
        ema_cluster_note=ema_cluster_note,
        fractal_structure=fractal_str,
    )


# ======================================================================
#  Рыночная структура
# ======================================================================
def _compute_fractal(df: Optional[pd.DataFrame]) -> str:
    if df is None:
        return "нет данных (OHLCV не загружен)"
    try:
        from gex.domain.trendlines import analyze_trendlines
        tl = analyze_trendlines(df, timeframe=_NARRATIVE_TF)
        fr = tl.fractals
        return (
            f"{tl.fractal_trend}: "
            f"HH={fr.higher_highs} HL={fr.higher_lows} "
            f"LH={fr.lower_highs} LL={fr.lower_lows}"
        )
    except Exception:
        return "нет данных"


def _direction_from_fractal(fractal_str: str) -> str:
    if fractal_str.startswith("BULLISH"):
        return "BULLISH"
    if fractal_str.startswith("BEARISH"):
        return "BEARISH"
    return "NEUTRAL"


def _global_regime_label(spot: float, ema_100: Optional[float],
                         zero_gamma: Optional[float]) -> str:
    above_ema = ema_100 is not None and spot > ema_100
    above_flip = zero_gamma is not None and spot > zero_gamma
    if above_ema and above_flip:
        return "глобально-бычий (цена > EMA100 и > Gamma Flip)"
    if (not above_ema) and (zero_gamma is not None and spot < zero_gamma):
        return "глобально-медвежий (цена < EMA100 и < Gamma Flip)"
    if ema_100 is not None and zero_gamma is not None:
        return "смешанный (EMA100 и Gamma Flip дают противоречивые сигналы)"
    return "смешанный (недостаточно данных)"


# ======================================================================
#  Вероятность разворота на уровнях
# ======================================================================
def _level_reversal_probs(
    report: ExtendedGEXReport, spot: float,
    cw: Optional[float], pw: Optional[float], gf: Optional[float],
) -> list[LevelProbability]:
    rows = sorted(report.per_strike, key=lambda s: s.ag, reverse=True)
    max_ag = max((s.ag for s in rows), default=0.0)
    probs: list[LevelProbability] = []

    for strike, label in ((cw, "Call Wall"), (pw, "Put Wall"), (gf, "Gamma Flip")):
        if strike is None or not np.isfinite(strike):
            continue
        dist = (strike - spot) / spot * 100.0
        # Плотность гаммы на страйке: берём ближайший из per_strike по ag.
        nearest = min(rows, key=lambda s: abs(s.strike - strike), default=None)
        ag_density = (nearest.ag / max_ag) if nearest and max_ag > 0 else 0.0

        # Базовая вероятность: POSITIVE regime = выше вероятность разворота
        # (ММ гасят движение), NEGATIVE = ниже (ММ усиливают тренд).
        base = 0.50
        if report.regime == "POSITIVE":
            base = 0.55 + ag_density * 0.25
        else:
            base = 0.40 + ag_density * 0.15
        prob = round(min(max(base, 0.10), 0.95) * 100, 1)

        if label == "Gamma Flip":
            reasoning = "граница режимов: переход через неё меняет поведение ММ"
        elif label == "Call Wall":
            reasoning = f"сопротивление (плотность AG: {ag_density*100:.0f}%)"
        else:
            reasoning = f"поддержка (плотность AG: {ag_density*100:.0f}%)"

        probs.append(LevelProbability(
            strike=float(strike), label=label,
            distance_pct=round(dist, 2),
            reversal_pct=prob, reasoning=reasoning,
        ))
    return probs


# ======================================================================
#  Доверительные диапазоны по срокам экспирации (10 / 30 / 60 дней)
# ======================================================================
def _expiry_bands(report: ExtendedGEXReport, spot: float) -> list[ExpiryBand]:
    # Временные группы: (дней, лет)
    groups = {10: 10.0 / 365.0, 30: 30.0 / 365.0, 60: 60.0 / 365.0}
    bands: list[ExpiryBand] = []
    for days, max_t in groups.items():
        strikes_in_group = [s for s in report.per_strike if s.weight > 0]
        if not strikes_in_group:
            continue
        # Окно в %: для ближних экспираций уже, для дальних шире.
        window_pct = _WINDOW_BY_DAYS.get(days, 0.10)
        lo_price = spot * (1 - window_pct)
        hi_price = spot * (1 + window_pct)
        in_window = [s for s in strikes_in_group
                     if lo_price <= s.strike <= hi_price]
        if len(in_window) < 3:
            continue

        # Поддержка: сильнейший отрицательный gex_net ниже spot.
        below = [s for s in in_window if s.strike <= spot and s.gex_net < 0]
        above = [s for s in in_window if s.strike >= spot and s.gex_net > 0]
        support = min(below, key=lambda s: s.gex_net).strike if below else min(s.strike for s in in_window)
        resistance = max(above, key=lambda s: s.gex_net).strike if above else max(s.strike for s in in_window)

        bands.append(ExpiryBand(
            days=days,
            support=float(support),
            resistance=float(resistance),
            band_width_pct=round((resistance - support) / spot * 100, 2),
            n_strikes=len(in_window),
        ))
    return bands


# ======================================================================
#  Итоговый доверительный интервал
# ======================================================================
def _confidence_interval(report: ExtendedGEXReport, spot: float) -> tuple[float | None, float | None]:
    from gex.domain.price_band import compute_price_band
    pb = compute_price_band(report)
    if pb is not None:
        return pb.low, pb.high
    # Fallback
    if np.isfinite(report.put_wall) and np.isfinite(report.call_wall):
        return float(report.put_wall), float(report.call_wall)
    return None, None


# ======================================================================
#  Сводка
# ======================================================================
def _summary_line(ticker: str, spot: float, net_gex: float, regime: str,
                  direction: str, fractal: str,
                  level_probs: list[LevelProbability],
                  expiry_bands: list[ExpiryBand]) -> str:
    """1-2 предложения, кристально ясных."""
    dir_ru = {"BULLISH": "восходящий", "BEARISH": "нисходящий", "NEUTRAL": "неопределённый"}
    d = dir_ru.get(direction, direction)

    parts = [f"{ticker}: рынок {d}. Net GEX: {net_gex:+,.0f} (режим {regime})."]

    # Структура
    if fractal and not fractal.startswith("нет данных"):
        parts.append(f"Структура: {fractal}.")

    # Диапазон по ближайшей экспирации
    if expiry_bands:
        b = expiry_bands[0]
        parts.append(
            f"Ближайший диапазон ({b.days}д): [{b.support:,.0f}, {b.resistance:,.0f}] "
            f"(ширина {b.band_width_pct}%)."
        )

    # Уровни с наибольшей вероятностью разворота
    top_prob = max(level_probs, key=lambda lp: lp.reversal_pct, default=None)
    if top_prob:
        parts.append(
            f"Наибольшая вероятность разворота на {top_prob.label}: "
            f"{top_prob.reversal_pct:.0f}%."
        )

    return " ".join(parts)


# ======================================================================
#  EMA-кластер
# ======================================================================
def _ema_cluster_info(df: Optional[pd.DataFrame], report: ExtendedGEXReport,
                      spot: float) -> tuple[bool, str]:
    if df is None or df.empty:
        return False, ""
    try:
        from gex.domain.ta import compute_indicators
        ind = compute_indicators(df)
        ema20, ema50 = float(ind.ema20), float(ind.ema50)
        ema100 = float(df["Close"].ewm(span=_EMA100_SPAN, adjust=False).mean().iloc[-1])
        vals = [ema20, ema50, ema100]
        lo, hi = min(vals), max(vals)
        cluster = lo > 0 and (hi - lo) / lo < 0.01
        note = ""
        if cluster:
            note = f"EMA20/50/100 в кластере ({ema20:.0f}/{ema50:.0f}/{ema100:.0f}) — накопление перед движением"
            for lvl, name in ((report.call_wall, "Call Wall"),
                              (report.put_wall, "Put Wall"),
                              (report.zero_gamma, "Gamma Flip")):
                if lvl and np.isfinite(lvl) and abs(lvl - (ema20 + ema50 + ema100) / 3) / spot < 0.02:
                    note = f"Кластер EMA20/50/100 ({ema20:.0f}/{ema50:.0f}/{ema100:.0f}) СОВПАДАЕТ с {name} — очень сильный уровень"
                    break
        return cluster, note
    except Exception:
        return False, ""


# ======================================================================
#  Загрузка OHLCV
# ======================================================================
def _try_load_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    try:
        from gex.application.ohlcv_service import detect_asset_type, fetch_all_timeframes
        asset_type = detect_asset_type(ticker)
        tfs, _ = fetch_all_timeframes(ticker, asset_type)
        return tfs.get(_NARRATIVE_TF)
    except Exception as exc:
        logger.debug("narrative: OHLCV для %s недоступен: %s", ticker, exc)
        return None
