"""Структура рынка HH/HL/LH/LL по гибридным свечам (фракталы Билла Уильямса).

Модуль реализует определение направления тренда по принципу:
- Uptrend: Higher High (HH) + Higher Low (HL);
- Downtrend: Lower High (LH) + Lower Low (LL).

Пайплайн (все расчёты — без использования будущих данных):
1. Heikin-Ashi open/close;
2. гибридные свечные серии (median/candle/novelsrc);
3. фракталы Билла Уильямса по выбранному источнику, подтверждаемые
   только после ``fractal_right`` баров справа;
4. пороговая разметка HH/LH/HL/LL по ATR в момент подтверждения;
5. строгий тренд + подтверждение импульсом novelsrc (EMA);
6. альтернативный скоринг с экспоненциальным затуханием событий;
7. зигзаг по чередующимся подтверждённым пивотам;
8. стадия рынка: UPTREND / DOWNTREND / REVERSAL / RANGE.

Конвенция колонок входа — ``Open/High/Low/Close`` (как у всех фетчеров
репозитория; эквивалент ``open/high/low/close`` из ТЗ).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from gex.application.novel_candles import NovelCandlesService
from gex.adapters.fetchers.ta_fetcher import TIMEFRAMES
from gex.domain.trendlines import analyze_trendlines

logger = logging.getLogger(__name__)

# Допустимые источники фракталов.
FRACTAL_SOURCES = ("standard", "candle", "median", "median_candle")

# Стадии рынка.
STAGE_UPTREND = "UPTREND"
STAGE_DOWNTREND = "DOWNTREND"
STAGE_REVERSAL = "REVERSAL"
STAGE_RANGE = "RANGE"


@dataclass
class HybridTrendParams:
    """Параметры определения тренда по гибридным свечам и фракталам.

    Parameters
    ----------
    fractal_left : int
        Баров слева от центра фрактала.
    fractal_right : int
        Баров справа от центра; это же число задаёт лаг подтверждения.
    fractal_source : str
        Источник фракталов: "standard" (high/low, классика Bill Williams),
        "candle", "median" или "median_candle" (композит).
    alpha : float
        Вес median-серии в композите "median_candle": 1.0 → только median,
        0.0 → только candle.
    strict_fractals : bool
        True — строгое сравнение с соседями; False — с защитой от плато.
    atr_period : int
        Период ATR (Wilder) по standard high/low/close.
    atr_mult : float
        Множитель ATR для порога значимости HH/HL/LH/LL.
    min_change : float
        Минимальный абсолютный порог изменения цены.
    max_event_age : int
        Максимальный возраст события (в барах после подтверждения),
        при котором оно считается активным.
    decay_tau : float
        Постоянная экспоненциального затухания силы события для score.
    w_high : float
        Вес last_high_label в score.
    w_low : float
        Вес last_low_label в score.
    w_novel : float
        Вес импульса novelsrc в score.
    score_threshold : float
        Порог для trend_score_dir.
    use_novel_filter : bool
        True — строгий тренд подтверждается только при совпадении
        направления структуры и импульса novelsrc.
    novel_ema : int
        Период EMA для novelsrc.
    novel_atr_mult : float
        Множитель ATR для нормировки импульса novelsrc.
    allow_score_fallback : bool
        True — trend_final может взять score-направление, когда
        строгий тренд равен 0.
    """

    fractal_left: int = 2
    fractal_right: int = 2
    fractal_source: str = "median_candle"
    alpha: float = 0.5
    strict_fractals: bool = True

    atr_period: int = 14
    atr_mult: float = 0.25
    min_change: float = 0.0

    max_event_age: int = 120
    decay_tau: float = 60.0

    w_high: float = 0.45
    w_low: float = 0.45
    w_novel: float = 0.10
    score_threshold: float = 0.45

    use_novel_filter: bool = True
    novel_ema: int = 20
    novel_atr_mult: float = 0.5

    allow_score_fallback: bool = False

    min_fractal_distance: int = 0
    """Минимальное расстояние между принятыми пивотами (в барах), 0 = выкл.

    Применяется к соседним пивотам любого типа (хай→лоу→хай…): фрактал,
    подтверждённый раньше чем через ``min_fractal_distance`` баров после
    последнего принятого пивота, отбрасывается и не участвует в событиях,
    тренде, стадии и зигзаге.
    """

    def __post_init__(self) -> None:
        if self.fractal_left < 0 or self.fractal_right < 0:
            raise ValueError("fractal_left и fractal_right должны быть >= 0.")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha должна быть в диапазоне [0, 1].")
        if self.fractal_source.lower() not in FRACTAL_SOURCES:
            raise ValueError(
                f"fractal_source должен быть одним из: {', '.join(FRACTAL_SOURCES)}."
            )
        if self.atr_period <= 0:
            raise ValueError("atr_period должен быть > 0.")
        if self.min_fractal_distance < 0:
            raise ValueError("min_fractal_distance должен быть >= 0.")


# ---------------------------------------------------------------------- #
#  Шаг 1: Heikin-Ashi
# ---------------------------------------------------------------------- #
def compute_heikin_ashi(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Heikin-Ashi open/close.

    HA_close[t] = (O + H + L + C) / 4
    HA_open[0]  = (O + C) / 2
    HA_open[t]  = (HA_open[t-1] + HA_close[t-1]) / 2
    """
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)

    n = len(df)
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty(n, dtype=float)
    if n > 0:
        ha_open[0] = (o[0] + c[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    return ha_open, ha_close


# ---------------------------------------------------------------------- #
#  Шаг 2: гибридные свечные серии
# ---------------------------------------------------------------------- #
def compute_hybrid_features(df: pd.DataFrame) -> pd.DataFrame:
    """Гибридные серии: median/candle/hybrid + hlcc4/sourceformas/novelsrc.

    Добавляет колонки: ha_open, ha_close, median_top, median_bottom,
    hybrid_open, hybrid_close, candle_top, candle_bottom, avg_candle,
    hlcc4, sourceformas, novelsrc.
    """
    out = df.copy()

    ha_open, ha_close = compute_heikin_ashi(df)

    o = out["Open"].to_numpy(dtype=float)
    h = out["High"].to_numpy(dtype=float)
    l = out["Low"].to_numpy(dtype=float)
    c = out["Close"].to_numpy(dtype=float)

    out["ha_open"] = ha_open
    out["ha_close"] = ha_close

    std_body_top = np.maximum(o, c)
    std_body_bottom = np.minimum(o, c)
    ha_body_top = np.maximum(ha_open, ha_close)
    ha_body_bottom = np.minimum(ha_open, ha_close)

    out["median_top"] = 0.5 * (std_body_top + ha_body_top)
    out["median_bottom"] = 0.5 * (std_body_bottom + ha_body_bottom)

    hybrid_open = 0.5 * (o + ha_open)
    hybrid_close = 0.5 * (c + ha_close)
    out["hybrid_open"] = hybrid_open
    out["hybrid_close"] = hybrid_close

    candle_top = np.maximum(hybrid_open, hybrid_close)
    candle_bottom = np.minimum(hybrid_open, hybrid_close)
    out["candle_top"] = candle_top
    out["candle_bottom"] = candle_bottom
    out["avg_candle"] = 0.5 * (candle_top + candle_bottom)

    hlcc4 = (h + l + c + c) / 4.0
    out["hlcc4"] = hlcc4

    sourceformas = np.where(o > c, 0.5 * (o + l), 0.5 * (c + h))
    out["sourceformas"] = sourceformas

    out["novelsrc"] = (hlcc4 + out["avg_candle"].to_numpy(dtype=float) + sourceformas) / 3.0
    return out


# ---------------------------------------------------------------------- #
#  ATR (Wilder) по standard high/low/close
# ---------------------------------------------------------------------- #
def compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """ATR (Wilder RMA) по standard High/Low/Close. Массив без NaN."""
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    n = len(df)

    tr = np.empty(n, dtype=float)
    for i in range(n):
        if i == 0:
            tr[i] = h[i] - l[i]
        else:
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    atr = np.empty(n, dtype=float)
    if n > 0:
        alpha = 1.0 / float(period)
        atr[0] = tr[0]
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr


# ---------------------------------------------------------------------- #
#  Шаг 4: выбор серий для фракталов
# ---------------------------------------------------------------------- #
def select_fractal_series(
    df: pd.DataFrame,
    params: HybridTrendParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Верхняя и нижняя серии для поиска фракталов по fractal_source."""
    source = params.fractal_source.lower()

    if source == "standard":
        top = df["High"].to_numpy(dtype=float)
        bottom = df["Low"].to_numpy(dtype=float)
    elif source == "candle":
        top = df["candle_top"].to_numpy(dtype=float)
        bottom = df["candle_bottom"].to_numpy(dtype=float)
    elif source == "median":
        top = df["median_top"].to_numpy(dtype=float)
        bottom = df["median_bottom"].to_numpy(dtype=float)
    else:  # median_candle
        top = params.alpha * df["median_top"].to_numpy(dtype=float) \
            + (1.0 - params.alpha) * df["candle_top"].to_numpy(dtype=float)
        bottom = params.alpha * df["median_bottom"].to_numpy(dtype=float) \
            + (1.0 - params.alpha) * df["candle_bottom"].to_numpy(dtype=float)
    return top, bottom


# ---------------------------------------------------------------------- #
#  Шаг 5: фракталы Билла Уильямса
# ---------------------------------------------------------------------- #
def detect_fractals(
    top: np.ndarray,
    bottom: np.ndarray,
    left: int,
    right: int,
    strict: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Фракталы по верхней/нижней серии.

    Возвращает булевы массивы на барах **возникновения**: фрактал на баре
    ``i`` подтверждается только на баре ``i + right``.
    """
    n = len(top)
    fractal_high = np.zeros(n, dtype=bool)
    fractal_low = np.zeros(n, dtype=bool)

    if n < left + right + 1 or (left == 0 and right == 0):
        return fractal_high, fractal_low

    for i in range(left, n - right):
        top_window = top[i - left: i + right + 1]
        center = left
        center_top = top_window[center]
        others_top = np.concatenate([top_window[:center], top_window[center + 1:]])
        if len(others_top) > 0:
            mx = float(np.max(others_top))
            mn = float(np.min(others_top))
            if strict:
                fractal_high[i] = center_top > mx
            else:
                # Нестрогий режим с защитой от полностью плоского плато.
                fractal_high[i] = (center_top >= mx) and (center_top > mn)

        bottom_window = bottom[i - left: i + right + 1]
        center_bottom = bottom_window[center]
        others_bottom = np.concatenate([bottom_window[:center], bottom_window[center + 1:]])
        if len(others_bottom) > 0:
            mx = float(np.max(others_bottom))
            mn = float(np.min(others_bottom))
            if strict:
                fractal_low[i] = center_bottom < mn
            else:
                fractal_low[i] = (center_bottom <= mn) and (center_bottom < mx)

    return fractal_high, fractal_low


# ---------------------------------------------------------------------- #
#  Зигзаг: чередующиеся подтверждённые пивоты
# ---------------------------------------------------------------------- #
def build_zigzag(
    high_confirmed: np.ndarray,
    low_confirmed: np.ndarray,
    right: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Зигзаг по чередующимся подтверждённым пивотам.

    Пивоты обрабатываются в порядке подтверждения; подряд идущие пивоты
    одного типа пропускаются (берётся первый). Метки ставятся на баре
    возникновения пивота (``i = t - right``) — исторический факт, который
    становится известен с момента подтверждения ``t``.
    """
    n = len(high_confirmed)
    zh = np.zeros(n, dtype=bool)
    zl = np.zeros(n, dtype=bool)
    last_kind: Optional[str] = None

    for t in range(right, n):
        occ = t - right
        if high_confirmed[t] and last_kind != "high":
            zh[occ] = True
            last_kind = "high"
        elif low_confirmed[t] and last_kind != "low":
            zl[occ] = True
            last_kind = "low"
    return zh, zl


# ---------------------------------------------------------------------- #
#  Главная функция
# ---------------------------------------------------------------------- #
def build_trend(
    df: pd.DataFrame,
    params: Optional[HybridTrendParams] = None,
) -> pd.DataFrame:
    """Гибридные свечи → фракталы → HH/HL/LH/LL → тренд + score + зигзаг.

    Возвращает входной DataFrame со всеми расчётными колонками
    (см. ТЗ п.10 + ``stage``, ``zigzag_high``, ``zigzag_low``).
    Пустой вход → пустой DataFrame с теми же колонками.
    """
    if params is None:
        params = HybridTrendParams()

    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Во входном DataFrame отсутствуют колонки: {missing}")

    if df.empty:
        out = df.copy()
        for col in (
            "ha_open", "ha_close", "median_top", "median_bottom",
            "hybrid_open", "hybrid_close", "candle_top", "candle_bottom",
            "avg_candle", "hlcc4", "sourceformas", "novelsrc",
            "atr", "fractal_top", "fractal_bottom", "trend_score",
            "novel_momentum",
        ):
            out[col] = pd.Series(dtype=float)
        for col in (
            "fractal_high", "fractal_low",
            "fractal_high_confirmed", "fractal_low_confirmed",
            "fractal_high_accepted", "fractal_low_accepted",
            "hh_event", "lh_event", "hl_event", "ll_event",
            "zigzag_high", "zigzag_low",
        ):
            out[col] = pd.Series(dtype=bool)
        for col in (
            "last_high_label", "last_low_label",
            "trend_strict", "trend_score_dir", "trend_final",
        ):
            out[col] = pd.Series(dtype=int)
        out["stage"] = pd.Series(dtype=object)
        return out

    # --- 1-3. Признаки + ATR + серии фракталов ---
    out = compute_hybrid_features(df)
    atr = compute_atr(df, period=params.atr_period)
    out["atr"] = atr

    top, bottom = select_fractal_series(out, params)
    out["fractal_top"] = top
    out["fractal_bottom"] = bottom

    # --- 4. Фракталы (возникновение) и подтверждение ---
    fractal_high_occ, fractal_low_occ = detect_fractals(
        top, bottom, params.fractal_left, params.fractal_right,
        params.strict_fractals,
    )
    n = len(out)
    right = params.fractal_right

    fractal_high_confirmed = np.zeros(n, dtype=bool)
    fractal_low_confirmed = np.zeros(n, dtype=bool)
    if right < n:
        fractal_high_confirmed[right:] = fractal_high_occ[: n - right]
        fractal_low_confirmed[right:] = fractal_low_occ[: n - right]

    # Принятые пивоты (после фильтра минимального расстояния).
    fractal_high_accepted = np.zeros(n, dtype=bool)
    fractal_low_accepted = np.zeros(n, dtype=bool)

    # --- 5-9. Разметка структуры, тренд, score ---
    hh_event = np.zeros(n, dtype=bool)
    lh_event = np.zeros(n, dtype=bool)
    hl_event = np.zeros(n, dtype=bool)
    ll_event = np.zeros(n, dtype=bool)

    trend_strict = np.zeros(n, dtype=int)
    trend_score_dir = np.zeros(n, dtype=int)
    trend_final = np.zeros(n, dtype=int)
    score_raw = np.zeros(n, dtype=float)
    momentum_raw = np.zeros(n, dtype=float)

    state_high_label = np.zeros(n, dtype=int)
    state_low_label = np.zeros(n, dtype=int)
    stage = np.full(n, STAGE_RANGE, dtype=object)

    last_high_price = np.nan
    last_low_price = np.nan
    last_high_time = -1
    last_low_time = -1
    last_high_label = 0
    last_low_label = 0
    last_accepted_occ: Optional[int] = None

    threshold = atr * params.atr_mult
    if params.min_change > 0.0:
        threshold = np.maximum(threshold, params.min_change)

    novelsrc = out["novelsrc"].to_numpy(dtype=float)
    ema_novel = (
        pd.Series(novelsrc)
        .ewm(span=params.novel_ema, adjust=False, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )

    weight_sum = params.w_high + params.w_low + params.w_novel
    if weight_sum > 0.0:
        w_high = params.w_high / weight_sum
        w_low = params.w_low / weight_sum
        w_novel = params.w_novel / weight_sum
    else:
        w_high = w_low = w_novel = 0.0

    for t in range(n):
        occ = t - right
        if occ >= 0:
            # Фильтр минимального расстояния между соседними пивотами любого типа.
            accepted = (
                params.min_fractal_distance <= 0
                or last_accepted_occ is None
                or (occ - last_accepted_occ) >= params.min_fractal_distance
            )
            # --- Подтверждение фрактального максимума ---
            if fractal_high_occ[occ] and accepted:
                if np.isfinite(last_high_price):
                    diff = top[occ] - last_high_price
                    th = threshold[t] if np.isfinite(threshold[t]) else 0.0
                    label = 1 if diff > th else (-1 if diff < -th else 0)
                else:
                    label = 0
                last_high_price = float(top[occ])
                last_high_time = t
                last_high_label = label
                fractal_high_accepted[t] = True
                last_accepted_occ = occ
                if label == 1:
                    hh_event[t] = True
                elif label == -1:
                    lh_event[t] = True

            # --- Подтверждение фрактального минимума ---
            if fractal_low_occ[occ] and accepted:
                if np.isfinite(last_low_price):
                    diff = bottom[occ] - last_low_price
                    th = threshold[t] if np.isfinite(threshold[t]) else 0.0
                    label = 1 if diff > th else (-1 if diff < -th else 0)
                else:
                    label = 0
                last_low_price = float(bottom[occ])
                last_low_time = t
                last_low_label = label
                fractal_low_accepted[t] = True
                last_accepted_occ = occ
                if label == 1:
                    hl_event[t] = True
                elif label == -1:
                    ll_event[t] = True

        # --- Возраст и валидность событий ---
        age_high = t - last_high_time if last_high_time >= 0 else 10**9
        age_low = t - last_low_time if last_low_time >= 0 else 10**9
        valid_high = age_high <= params.max_event_age
        valid_low = age_low <= params.max_event_age

        state_high_label[t] = last_high_label if valid_high else 0
        state_low_label[t] = last_low_label if valid_low else 0

        # --- Строгая структура и стадия ---
        if valid_high and valid_low:
            if last_high_label == 1 and last_low_label == 1:
                strict_signal = 1
                stage[t] = STAGE_UPTREND
            elif last_high_label == -1 and last_low_label == -1:
                strict_signal = -1
                stage[t] = STAGE_DOWNTREND
            elif last_high_label != 0 and last_low_label != 0:
                strict_signal = 0
                stage[t] = STAGE_REVERSAL
            else:
                strict_signal = 0
                stage[t] = STAGE_RANGE
        else:
            strict_signal = 0
            stage[t] = STAGE_RANGE

        # --- Импульс novelsrc ---
        if (
            atr[t] > 0.0
            and params.novel_atr_mult > 0.0
            and np.isfinite(novelsrc[t])
            and np.isfinite(ema_novel[t])
        ):
            denom = atr[t] * params.novel_atr_mult
            momentum = float(np.clip((novelsrc[t] - ema_novel[t]) / denom, -1.0, 1.0))
        else:
            momentum = 0.0
        momentum_raw[t] = momentum

        # --- Score с decay ---
        if valid_high:
            decay_high = float(np.exp(-age_high / params.decay_tau)) if params.decay_tau > 0.0 else 1.0
        else:
            decay_high = 0.0
        if valid_low:
            decay_low = float(np.exp(-age_low / params.decay_tau)) if params.decay_tau > 0.0 else 1.0
        else:
            decay_low = 0.0

        score = (
            w_high * last_high_label * decay_high
            + w_low * last_low_label * decay_low
            + w_novel * momentum
        )
        score_raw[t] = score

        if score >= params.score_threshold:
            score_signal = 1
        elif score <= -params.score_threshold:
            score_signal = -1
        else:
            score_signal = 0
        trend_score_dir[t] = score_signal

        # --- Novel-фильтр строгого тренда ---
        if params.use_novel_filter:
            if strict_signal == 1 and momentum < 0.0:
                strict_confirmed = 0
            elif strict_signal == -1 and momentum > 0.0:
                strict_confirmed = 0
            else:
                strict_confirmed = strict_signal
        else:
            strict_confirmed = strict_signal
        trend_strict[t] = strict_confirmed

        # --- Финальный тренд ---
        if params.allow_score_fallback and strict_confirmed == 0:
            trend_final[t] = score_signal
        else:
            trend_final[t] = strict_confirmed

    # --- Зигзаг по принятым (отфильтрованным) пивотам ---
    zigzag_high, zigzag_low = build_zigzag(fractal_high_accepted, fractal_low_accepted, right)

    # --- Выходные колонки ---
    out["fractal_high"] = fractal_high_occ
    out["fractal_low"] = fractal_low_occ
    out["fractal_high_confirmed"] = fractal_high_confirmed
    out["fractal_low_confirmed"] = fractal_low_confirmed
    out["fractal_high_accepted"] = fractal_high_accepted
    out["fractal_low_accepted"] = fractal_low_accepted

    out["hh_event"] = hh_event
    out["lh_event"] = lh_event
    out["hl_event"] = hl_event
    out["ll_event"] = ll_event

    out["last_high_label"] = state_high_label
    out["last_low_label"] = state_low_label

    out["trend_strict"] = trend_strict
    out["trend_score"] = score_raw
    out["trend_score_dir"] = trend_score_dir
    out["trend_final"] = trend_final
    out["novel_momentum"] = momentum_raw

    out["stage"] = stage
    out["zigzag_high"] = zigzag_high
    out["zigzag_low"] = zigzag_low

    return out


# ---------------------------------------------------------------------- #
#  Сервис: фетч → расчёт → сериализация для API
# ---------------------------------------------------------------------- #
def _serialize_bars(df: pd.DataFrame) -> list[dict]:
    """OHLC-DataFrame → список баров {time, open, high, low, close}."""
    bars = []
    for ts, row in df.iterrows():
        bars.append({
            "time": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
        })
    return bars


class HybridTrendService:
    """Структура рынка: OHLCV → build_trend → трендовые линии → dict.

    Фетч OHLCV переиспользует роутинг :class:`NovelCandlesService`
    (stock/crypto/moex/commodity) без дублирования логики.
    """

    def __init__(self, redis_client=None):
        self._nc = NovelCandlesService(redis_client=redis_client)

    def fetch_and_analyze(
        self,
        ticker: str,
        timeframe: str = "1d",
        limit: int = 300,
        params: Optional[HybridTrendParams] = None,
        with_trendlines: bool = True,
        trendline_resolution: int = 6,
        max_trendlines: int = 5,
        pivot_left: int = 5,
        pivot_right: int = 5,
    ) -> dict:
        """Полный ответ для графика структуры рынка.

        Returns
        -------
        dict
            Ключи: ticker, timeframe, asset_type, n_bars, bars, novel_bars,
            fractals, events, zigzag, trendlines, trend_strict, trend_final,
            stage, last_close, atr.
        """
        if params is None:
            params = HybridTrendParams()

        ticker = ticker.strip().upper()
        tf = timeframe.strip().lower()
        if tf not in TIMEFRAMES:
            raise ValueError(
                f"Неподдерживаемый таймфрейм '{timeframe}'. Доступно: {', '.join(TIMEFRAMES)}"
            )

        asset_type = NovelCandlesService._detect_asset_type(ticker)
        df_raw = self._nc._fetch_ohlcv(ticker, tf, asset_type, limit)
        if df_raw is None or len(df_raw) == 0:
            raise ValueError(f"Нет OHLCV данных для '{ticker}' [{tf}]")

        out = build_trend(df_raw, params)
        right = params.fractal_right
        n = len(out)

        # --- Фракталы (принятые после фильтра расстояния) ---
        fractals: list[dict] = []
        for i in range(n):
            if bool(out["fractal_high_accepted"].iloc[i]):
                fractals.append({
                    "index": int(i),
                    "pivot_index": int(i - right),
                    "price": round(float(out["fractal_top"].iloc[i - right]), 4),
                    "kind": "high",
                })
            if bool(out["fractal_low_accepted"].iloc[i]):
                fractals.append({
                    "index": int(i),
                    "pivot_index": int(i - right),
                    "price": round(float(out["fractal_bottom"].iloc[i - right]), 4),
                    "kind": "low",
                })

        # --- События HH/LH/HL/LL ---
        events: list[dict] = []
        for col, label, series_name in (
            ("hh_event", "HH", "fractal_top"),
            ("lh_event", "LH", "fractal_top"),
            ("hl_event", "HL", "fractal_bottom"),
            ("ll_event", "LL", "fractal_bottom"),
        ):
            for i in range(n):
                if bool(out[col].iloc[i]):
                    occ = i - right
                    events.append({
                        "index": int(i),
                        "pivot_index": int(occ),
                        "price": round(float(out[series_name].iloc[occ]), 4),
                        "label": label,
                    })
        events.sort(key=lambda e: e["index"])

        # --- Зигзаг ---
        zigzag: list[dict] = []
        for i in range(n):
            if bool(out["zigzag_high"].iloc[i]):
                zigzag.append({
                    "index": int(i),
                    "price": round(float(out["fractal_top"].iloc[i]), 4),
                    "kind": "high",
                })
            if bool(out["zigzag_low"].iloc[i]):
                zigzag.append({
                    "index": int(i),
                    "price": round(float(out["fractal_bottom"].iloc[i]), 4),
                    "kind": "low",
                })
        zigzag.sort(key=lambda p: p["index"])

        # --- Трендовые линии по стандартным барам ---
        trendlines = None
        if with_trendlines:
            try:
                tl = analyze_trendlines(
                    df_raw,
                    timeframe=tf,
                    resolution=trendline_resolution,
                    max_support_lines=max_trendlines,
                    max_resistance_lines=max_trendlines,
                    pivot_left=pivot_left,
                    pivot_right=pivot_right,
                )
                trendlines = {
                    "support": [ln.as_dict() for ln in tl.support_lines],
                    "resistance": [ln.as_dict() for ln in tl.resistance_lines],
                    "combined_trend": tl.combined_trend,
                    "combined_strength": round(tl.combined_strength, 1),
                }
            except Exception as exc:
                logger.warning("HybridTrend trendlines failed for %s: %s", ticker, exc)

        # --- Novel-бары для переключателя отображения ---
        try:
            novel_df = NovelCandlesService.compute_novel_candles(df_raw)
            novel_bars = _serialize_bars(novel_df)
        except Exception as exc:
            logger.warning("HybridTrend novel bars failed for %s: %s", ticker, exc)
            novel_bars = []

        return {
            "ticker": ticker,
            "timeframe": tf,
            "asset_type": asset_type,
            "n_bars": n,
            "bars": _serialize_bars(df_raw),
            "novel_bars": novel_bars,
            "fractals": fractals,
            "events": events,
            "zigzag": zigzag,
            "trendlines": trendlines,
            "trend_strict": out["trend_strict"].astype(int).tolist(),
            "trend_final": out["trend_final"].astype(int).tolist(),
            "stage": str(out["stage"].iloc[-1]),
            "last_close": round(float(df_raw["Close"].iloc[-1]), 4),
            "atr": round(float(out["atr"].iloc[-1]), 4),
        }
