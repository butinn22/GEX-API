"""Многофакторная логит-модель направления GEX-анализа.

Заменяет прежнюю GBM-аппроксимацию в :meth:`GEXService._direction`, которая
давала почти всегда ``p_up ≈ 0.50–0.53`` и ``confidence < 15%``. Причина
деградации: дрейф дополнительно умножался на ``σ√T`` (≈0.14 для 30 дней),
поэтому сигнал схлопывался к нулю.

Новая модель
------------
Каждый фактор рыночной микроструктуры переводится в **стандартизированный
z-вклад** (безразмерная шкала отклонения, ограниченная по модулю). Вклады
комбинируются как взвешенная сумма логитов:

.. math::
    L = \\sum_i w_i\\, z_i, \\qquad p_{up} = \\sigma(L) = \\frac{1}{1+e^{-L}}

Поскольку z-вклады уже нормированы в шкале «сигма», повторного умножения на
``σ√T`` нет — сигнал не прижимается к 0.5. Сигмоида плавно растягивает центр:
``L=+1.0 → p_up≈0.73``, ``L=+1.5 → 0.82``, ``L=+2.0 → 0.88``.

Факторы
~~~~~~~
1. **Асимметрия стен** (GEX-микроструктура): перевес совокупной гаммы выше
   спота над гаммой ниже → давление вверх/вниз.
2. **Магнит ближайшей стены**: сильная стена (по |gex_net| и OI), близкая к
   споту, притягивает цену к себе.
3. **Моментум свечей**: нивелирующая выборка + EMA5(Close) vs EMA10(нейтр.)
   + подтверждение готовой :func:`gex.ta.compute_momentum_strength`.
4. **Давление Flip**: regime-зависимый знак (POSITIVE→reversion,
   NEGATIVE→продолжение), масштаб через существующий ``profile.z_score``.

Все факторы опциональны: при отсутствии OHLCV (MOEX, VIX) модель работает
только на GEX-факторах, gracefully деградируя.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Константы модели (калибровка)
# ====================================================================== #
# Веса факторов в финальном логите. Сумма = 1.0 (нормировка не обязательна
# для сигмоиды, но удобна для интерпретации «доля вклада»).
_WEIGHTS: dict[str, float] = {
    "level_asymmetry": 0.40,   # главный GEX-структурный сигнал
    "momentum":        0.35,   # динамика свечей
    "wall_magnet":     0.15,   # притяжение ближайшей стены
    "flip_pressure":   0.10,   # позиция к Gamma Flip
}

# Пороги направления: расширены от старых 0.60/0.40, чтобы NEUTRAL не съедал
# весь рабочий диапазон.
_BULL_THRESHOLD = 0.58
_BEAR_THRESHOLD = 0.42

# Минимальное число баров OHLCV для расчёта EMA-моментума.
_MIN_BARS_MOMENTUM = 12


# ====================================================================== #
#  Результат
# ====================================================================== #
@dataclass
class DirectionResult:
    """Итог многофакторного расчёта направления.

    Attributes
    ----------
    p_up, p_down : float
        Вероятности роста/снижения за горизонт, [0.01, 0.99].
    direction : str
        'BULLISH' | 'BEARISH' | 'NEUTRAL'.
    confidence : float
        Уверенность в направлении, 0..100. Зависит и от силы итогового p_up,
        и от согласованности факторов.
    trend_strength : float
        Сила тренда 0..100 (для контекста направления; не поле схемы, но
        может использоваться в summarize-блоках).
    logit : float
        Итоговый логит L (диагностика).
    factors : dict[str, float]
        z-вклад каждого фактора (диагностика).
    """

    p_up: float
    p_down: float
    direction: str
    confidence: float
    trend_strength: float
    logit: float
    factors: dict[str, float] = field(default_factory=dict)


# ====================================================================== #
#  Утилиты
# ====================================================================== #
def _clip_z(value: float, lo: float = -3.0, hi: float = 3.0) -> float:
    """Ограничить z-вклад в разумный диапазон (защита от выбросов)."""
    if not np.isfinite(value):
        return 0.0
    return float(max(lo, min(hi, value)))


def _sigmoid(x: float) -> float:
    """Численно-стабильная сигмоида."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# ====================================================================== #
#  Фактор 1: Асимметрия стен (GEX-микроструктура)
# ====================================================================== #
def level_asymmetry_signal_simple(spot: float, profile) -> float:
    r"""z-вклад асимметрии совокупной гаммы относительно спота.

    Считаем суммарный |gex_net| по страйкам выше и ниже спота:

    .. math::
        A = \\frac{\\sum_{K > S} |gex_{net}(K)| - \\sum_{K < S} |gex_{net}(K)|}
                 {\\sum_{\\forall K} |gex_{net}(K)|}

    ``A ∈ [-1, +1]``. Положительная гамма = стена сопротивления (дилеры
    подавляют движение вверх), поэтому **перевес гаммы сверху → давление
    вниз** (отрицательный вклад в p_up), и наоборот → знак инвертируется.

    Насыщение через ``tanh``: при ``|A|=1`` даёт ``|z|≈2.5``, при ``A=0.5``
    → ``≈1.15``. Плавно насыщается, без резких крыльев.
    """
    df = getattr(profile, "per_strike", None)
    if df is None or len(df) == 0:
        return 0.0
    strikes = df["strike"].to_numpy(dtype=float)
    gex_abs = df["gex_abs"].to_numpy(dtype=float)
    total = float(gex_abs.sum())
    if total <= 0:
        return 0.0
    above = float(gex_abs[strikes > spot].sum())
    below = float(gex_abs[strikes < spot].sum())
    asym = (above - below) / total
    # Насыщение: при больших |asym| z не уходит за ~±2.18.
    z = -math.tanh(asym * 2.0) * 2.5
    return _clip_z(z)


# ====================================================================== #
#  Фактор 2: Магнит ближайшей стены
# ====================================================================== #
def _wall_pull(strike: float, strength: float, spot: float, max_abs: float) -> tuple[float, float]:
    """Магнитный «подтягивающий» вклад одной стены.

    Возвращает ``(signed_magnitude, sign)``: величина притяжения (с учётом
    близости) и направление (``+1`` если стена выше спота, ``−1`` если ниже).
    """
    if not np.isfinite(strike) or strike <= 0 or max_abs <= 0:
        return 0.0, 0.0
    dist_pct = abs(strike - spot) / spot
    if dist_pct <= 0.001:  # стена в точке спота — не магнит
        return 0.0, 0.0
    norm_strength = max(0.0, min(1.0, abs(strength) / max_abs))
    d_scale = 0.02  # 2% — типичная ширина магнитной зоны
    proximity = 1.0 / (1.0 + dist_pct / d_scale)
    magnitude = norm_strength * proximity
    sign = 1.0 if strike > spot else -1.0
    return magnitude, sign


def wall_magnet_signal(spot: float, profile) -> float:
    r"""z-вклад магнитного притяжения ближайших сильных стен ∈ [-3, +3].

    Использует **настоящие стены** из GEX-профиля (``call_wall`` сверху —
    сопротивление, ``put_wall`` снизу — поддержка) плюс их secondary. Стена
    притягивает цену тем сильнее, чем мощнее её гамма (|gex_net|) и чем ближе
    она к споту.

    Считается **конкуренция** двух сторон: верхняя стена тянет вверх
    (signed +), нижняя — вниз (signed −). Итоговый вклад:

    .. math::
        z = (pull_{up} - pull_{down}) \\cdot k

    где ``pull`` — magnitude·tanh-насыщение, ``k`` масштабирует в шкалу z.
    Так корректно учитывается, что ближняя слабая стена может перевесить
    дальнюю сильную, и наоборот.

    Сила каждой стены берётся из ``per_strike`` по её страйку (поиск |gex_net|).
    """
    df = getattr(profile, "per_strike", None)
    if df is None or len(df) == 0 or spot <= 0:
        return 0.0

    # Таблица strike → |gex_net| для определения силы настоящих стен.
    strike_to_abs = dict(zip(df["strike"].to_numpy(dtype=float),
                             df["gex_abs"].to_numpy(dtype=float)))
    max_abs = float(df["gex_abs"].max())
    if max_abs <= 0:
        return 0.0

    # Собираем реальные стены: primary + secondary, с каждой стороны.
    call_walls = []
    cw = getattr(profile, "call_wall", None)
    if cw is not None and np.isfinite(cw):
        call_walls.append(float(cw))
    call_walls.extend(s for s in getattr(profile, "secondary_call_walls", []) if np.isfinite(s))

    put_walls = []
    pw = getattr(profile, "put_wall", None)
    if pw is not None and np.isfinite(pw):
        put_walls.append(float(pw))
    put_walls.extend(s for s in getattr(profile, "secondary_put_walls", []) if np.isfinite(s))

    pull_up = 0.0   # суммарное притяжение вверх (от стен выше spot)
    pull_down = 0.0  # суммарное притяжение вниз (от стен ниже spot)
    for k in call_walls:
        if k <= spot:
            continue  # call wall ниже spot — не сопротивление, пропускаем
        strength = strike_to_abs.get(k, 0.0)
        mag, _ = _wall_pull(k, strength, spot, max_abs)
        pull_up = max(pull_up, mag)  # берём сильнейшую стену сверху
    for k in put_walls:
        if k >= spot:
            continue  # put wall выше spot — не поддержка
        strength = strike_to_abs.get(k, 0.0)
        mag, _ = _wall_pull(k, strength, spot, max_abs)
        pull_down = max(pull_down, mag)

    if pull_up == 0.0 and pull_down == 0.0:
        return 0.0

    # Конкуренция сторон: разница, насыщенная через tanh, масштаб в z.
    diff = pull_up - pull_down  # ∈ [-1, +1]
    z = math.tanh(diff * 2.0) * 2.5
    return _clip_z(z)


# ====================================================================== #
#  Фактор 3: Моментум свечей (нивелирующая выборка + EMA)
# ====================================================================== #
def neutralizing_price_series(df: pd.DataFrame) -> pd.Series:
    r"""Нивелирующая цена свечи — давит выбросы хвостов.

    Принцип ТЗ: если свеча падающая (``open > close``) — берём среднее между
    открытием и лоу (тело+низ, игнорируя верхнюю тень-выброс); если растущая
    (``close >= open``) — среднее между закрытием и хаем (тело+верх, игнорируя
    нижнюю тень-выброс):

    .. math::
        p_i = \\begin{cases}
            (Open_i + Low_i) / 2, & Open_i > Close_i \\\\
            (Close_i + High_i) / 2, & Open_i \\le Close_i
        \\end{cases}

    Тем самым нивелируются пин-бары и теневые проколы, ряд становится
    более гладким и лучше отражает «истинную» динамику.
    """
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    low = df["Low"].astype(float)
    c = df["Close"].astype(float)
    bearish = o > c
    return pd.Series(
        np.where(bearish, (o + low) / 2.0, (c + h) / 2.0),
        index=df.index,
    )


def momentum_signal(df: Optional[pd.DataFrame]) -> float:
    r"""z-вклад моментума свечей ∈ [-2, +2].

    Два компонента комбинируются:

    1. **EMA-перекрытие**: ``EMA5(Close)`` против ``EMA10(нейтрализ.)``.
       Если быстрая EMA5 close выше медленной EMA10 нейтрализующей —
       короткий тренд растущий (знак +). Величина вклада пропорциональна
       относительному расстоянию между ними, насыщается через ``tanh``.

    2. **Объёмно-ценовой бустер** от готовой
       :func:`gex.ta.compute_momentum_strength`: её ``strength`` (0..100)
       и ``side`` (BULLISH/BEARISH) усиливают или ослабляют EMA-сигнал.

    При нехватке баров возвращает 0 (фактор выключен).
    """
    if df is None or len(df) < _MIN_BARS_MOMENTUM:
        return 0.0
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        return 0.0

    close = df["Close"].astype(float)
    neut = neutralizing_price_series(df)

    ema5_close = close.ewm(span=5, adjust=False).mean().iloc[-1]
    ema10_neut = neut.ewm(span=10, adjust=False).mean().iloc[-1]
    if not np.isfinite(ema5_close) or not np.isfinite(ema10_neut) or ema10_neut <= 0:
        return 0.0

    # Относительное расстояние EMA: ~0.001 (0.1%) — заметное перекрытие.
    rel_gap = float((ema5_close - ema10_neut) / ema10_neut)
    # tanh-насыщение: rel_gap=0.005 → ~1.0, rel_gap=0.01 → ~1.4.
    ema_z = math.tanh(rel_gap / 0.003) * 1.5

    # Объёмно-ценовой бустер (опционально, не падает при отсутствии ta).
    booster = 0.0
    try:
        from gex.domain.ta import compute_momentum_strength
        ms = compute_momentum_strength(df)
        if ms is not None and ms.side != "NEUTRAL":
            # strength 0..100 → 0..0.5 дополнительного z-вклада.
            sign = 1.0 if ms.side == "BULLISH" else -1.0
            booster = sign * (ms.strength / 100.0) * 0.5
    except Exception as exc:  # noqa: BLE001 — бустер опционален
        logger.debug("compute_momentum_strength unavailable: %s", exc)

    return _clip_z(ema_z + booster, lo=-2.0, hi=2.0)


# ====================================================================== #
#  Фактор 4: Давление Gamma Flip
# ====================================================================== #
def flip_pressure_signal(spot: float, profile, horizon_years: float) -> float:
    r"""z-вклад давления Gamma Flip ∈ [-2, +2], regime-зависимый.

    * **POSITIVE gamma** — дилеры гасят волу, mean-reversion к Flip:
      спот ниже Flip → давление вверх (+), выше → вниз (−).
    * **NEGATIVE gamma** — дилеры усиливают тренд, продолжение от Flip:
      спот ниже Flip → давление вниз (−), выше → вверх (+).

    Масштаб берётся из существующего ``profile.z_score`` (отклонение спота
    от Flip в единицах σ√T) — это уже стандартизированная величина.
    """
    gamma_flip = getattr(profile, "gamma_flip", None)
    if gamma_flip is None or gamma_flip <= 0 or spot <= 0:
        return 0.0

    z_score = getattr(profile, "z_score", None)
    regime = getattr(profile, "regime", "POSITIVE")

    # Если z_score посчитан — используем его (он уже в шкале σ).
    if z_score is not None and np.isfinite(z_score):
        z = float(z_score)
    else:
        # Фолбэк: лог-расстояние, грубо в шкале σ.
        log_dist = float(np.log(spot / gamma_flip))
        z = _clip_z(log_dist / 0.01)  # 1% ≈ 1σ (грубая эвристика)

    if regime == "POSITIVE":
        # Mean-reversion: спот выше Flip (z>0) → давление вниз (−).
        return _clip_z(-z * 0.7, lo=-2.0, hi=2.0)
    else:
        # Trend-continuation: спот выше Flip (z>0) → давление вверх (+).
        return _clip_z(z * 0.7, lo=-2.0, hi=2.0)


# ====================================================================== #
#  Комбайн: logit → p_up, confidence, direction
# ====================================================================== #
def combine_logit(factors: dict[str, float], weights: Optional[dict] = None) -> float:
    """Взвешенная сумма z-вкладов → итоговый логит L.

    Факторы со значением ``0.0`` (выключены/недоступны) исключаются из
    суммы, а их вес **перераспределяется** на активные факторы. Это
    гарантирует, что при отсутствии OHLCV (MOEX, VIX) оставшиеся GEX-факторы
    сохраняют полный масштаб сигнала, а не ослабляются вдвое.
    """
    weights = weights or _WEIGHTS
    active = {k: v for k, v in factors.items() if abs(v) > 1e-9 and k in weights}
    if not active:
        return 0.0
    w_sum = sum(weights[k] for k in active)
    if w_sum <= 0:
        return 0.0
    # Перенормировка весов на активные.
    L = sum(weights[k] / w_sum * active[k] for k in active)
    return float(L)


def direction_from_p(p_up: float) -> str:
    """BULLISH/BEARISH/NEUTRAL по итоговой вероятности (пороги 0.58/0.42)."""
    if p_up >= _BULL_THRESHOLD:
        return "BULLISH"
    if p_up <= _BEAR_THRESHOLD:
        return "BEARISH"
    return "NEUTRAL"


def confidence_from_factors(
    p_up: float, factors: dict[str, float]
) -> float:
    r"""Уверенность в направлении, 0..100.

    База — отклонение p_up от нейтрали::

        base = |2·p_up − 1| · 100   ∈ [0, 98]

    Умножается на **коэффициент согласованности** ``agreement`` — долю
    активных факторов, знак которых совпадает со знаком итогового p_up−0.5.
    Если все факторы смотрят в одну сторону → agreement=1.0 (полная сила);
    если мнения разделились → confidence снижается даже при сильном p_up.

    Минимальный порог: даже при идеальном согласии p_up≈0.99 даёт
    ``base≈98``, без обрезки на 100 (реалистично «почти уверены»).
    """
    base = abs(2.0 * p_up - 1.0) * 100.0
    target_sign = 1.0 if p_up > 0.5 else (-1.0 if p_up < 0.5 else 0.0)
    if target_sign == 0.0:
        return 0.0
    active = [v for v in factors.values() if abs(v) > 1e-9]
    if not active:
        return float(np.clip(base, 0.0, 100.0))
    agree = sum(1 for v in active if (v > 0) == (target_sign > 0)) / len(active)
    # agreement ∈ [0, 1]; даже при 0 согласии оставляем 30% base (не ноль,
    # т.к. p_up всё же ненулевой).
    mult = 0.3 + 0.7 * agree
    return float(np.clip(base * mult, 0.0, 100.0))


def trend_strength_from_factors(
    p_up: float, factors: dict[str, float], momentum_strength: float = 0.0
) -> float:
    r"""Сила тренда 0..100 (для контекста направления).

    Компоненты:
      * ``|2·p_up − 1| · 100`` — сила направленного смещения вероятности;
      * средний |z| активных факторов — насколько факторная картина
        выражена (вне зависимости от знака);
      * ``momentum_strength`` — прямой вклад моментума свечей (0..100),
        если посчитан.

    Каждый компонент нормируется в [0, 100] и берётся взвешенная сумма.
    """
    prob_component = abs(2.0 * p_up - 1.0) * 100.0
    active = [abs(v) for v in factors.values() if abs(v) > 1e-9]
    z_component = (float(np.mean(active)) / 3.0 * 100.0) if active else 0.0
    mom_component = float(np.clip(momentum_strength, 0.0, 100.0))
    # Веса компонентов силы тренда.
    strength = (
        0.40 * prob_component
        + 0.25 * z_component
        + 0.35 * mom_component
    )
    return float(np.clip(strength, 0.0, 100.0))


# ====================================================================== #
#  OHLCV-фетчер с fallback
# ====================================================================== #
def fetch_momentum_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    """Получить OHLCV для расчёта моментума (TF=4h с fallback на 1h).

    Обёртка над :func:`gex.ohlcv_service.fetch_all_timeframes` с
    ``try/except`` → ``None`` при любых сбоях (нет сети, неподдерживаемый
    тикер, MOEX/VIX). Паттерн повторяет ``signal_service`` / ``ohlcv_service``.

    Возвращает DataFrame с колонками ``Open/High/Low/Close/Volume`` и
    tz-aware DatetimeIndex, отсортированный по времени, либо ``None``.
    """
    if not ticker:
        return None
    try:
        from gex.application.ohlcv_service import fetch_all_timeframes, detect_asset_type
        asset_type = detect_asset_type(ticker)
        tfs, _spot = fetch_all_timeframes(ticker.strip().upper(), asset_type)
    except Exception as exc:  # noqa: BLE001 — graceful degradation
        logger.warning("OHLCV-фетчер для моментума %s упал: %s", ticker, exc)
        return None

    # Предпочитаем 4h (баланс реактивности и устойчивости), fallback на 1h.
    for tf in ("4h", "1h"):
        df = tfs.get(tf)
        if df is not None and len(df) >= _MIN_BARS_MOMENTUM:
            return df
    # Любой доступный TF как последний ресурс.
    for df in tfs.values():
        if df is not None and len(df) >= _MIN_BARS_MOMENTUM:
            return df
    return None


# ====================================================================== #
#  Оркестратор: главная точка входа
# ====================================================================== #
def compute_direction(
    spot: float,
    profile,
    horizon_years: float,
    atm_vol: float,
    ticker: Optional[str] = None,
    ohlcv_df: Optional[pd.DataFrame] = None,
) -> DirectionResult:
    """Многофакторный расчёт направления и вероятностей роста/снижения.

    Parameters
    ----------
    spot : float
        Текущая цена базиса.
    profile : GEXProfile
        Профиль GEX (стены, Flip, regime, z_score, per_strike).
    horizon_years, atm_vol : float
        Горизонт (годы) и ATM-вола. В базовой модели не участвуют в схлопывании
        сигнала (в отличие от прежней GBM-аппроксимации); ``horizon_years``
        используется только в :func:`flip_pressure_signal` при отсутствии
        ``profile.z_score``.
    ticker : str, optional
        Тикер для OHLCV-фетчера (если ``ohlcv_df`` не передан).
    ohlcv_df : pd.DataFrame, optional
        Готовые OHLCV-свечи (для тестов или если caller уже их загрузил).
        Приоритет над ``ticker``.
    """
    factors: dict[str, float] = {}

    # --- GEX-факторы (всегда доступны) ---
    factors["level_asymmetry"] = level_asymmetry_signal_simple(spot, profile)
    factors["wall_magnet"] = wall_magnet_signal(spot, profile)
    factors["flip_pressure"] = flip_pressure_signal(spot, profile, horizon_years)

    # --- Моментум свечей (опционально) ---
    mom_strength = 0.0
    df = ohlcv_df
    if df is None and ticker:
        df = fetch_momentum_ohlcv(ticker)
    if df is not None and len(df) >= _MIN_BARS_MOMENTUM:
        factors["momentum"] = momentum_signal(df)
        try:
            from gex.domain.ta import compute_momentum_strength
            ms = compute_momentum_strength(df)
            if ms is not None:
                mom_strength = ms.strength
        except Exception:  # noqa: BLE001
            mom_strength = 0.0
    else:
        factors["momentum"] = 0.0

    # --- Комбинация в логит ---
    L = combine_logit(factors, _WEIGHTS)
    p_up = _sigmoid(L)
    p_up = float(np.clip(p_up, 0.01, 0.99))
    p_down = 1.0 - p_up

    direction = direction_from_p(p_up)
    confidence = confidence_from_factors(p_up, factors)
    trend_strength = trend_strength_from_factors(p_up, factors, mom_strength)

    return DirectionResult(
        p_up=p_up,
        p_down=p_down,
        direction=direction,
        confidence=confidence,
        trend_strength=trend_strength,
        logit=L,
        factors=factors,
    )


__all__ = [
    "DirectionResult",
    "compute_direction",
    "fetch_momentum_ohlcv",
    "momentum_signal",
    "neutralizing_price_series",
    "wall_magnet_signal",
    "flip_pressure_signal",
    "combine_logit",
    "confidence_from_factors",
    "direction_from_p",
    "trend_strength_from_factors",
]
