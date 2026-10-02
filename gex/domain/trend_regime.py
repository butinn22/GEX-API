"""Детектор тренда / флэта на окне 200 баров (ATR + Bollinger BandWidth + z-цена).

Модуль **чистый**: numpy/pandas и ничего из проекта — его можно тестировать
и переиспользовать где угодно (авто-сканер, сигнальный сканер, бэктест).

Идея
----
На баре ``i`` смотрим назад на окно ``W = 200`` баров, отдельно оцениваем
последние ``N = 50`` баров и по трём группам признаков решаем, есть ли тренд:

1. **Динамика ATR** — растёт ли средний диапазон (``atr_change_n``) и высоко ли
   он стоит внутри окна (``atr_rank_w``).
2. **Динамика Bollinger BandWidth** — расширяется ли канал (``bbw_change_n``,
   ``bbw_rank_w``); сжатие канала — признак боковика.
3. **Изменение цены**, нормированное на волатильность. Сырые проценты не
   переносятся между инструментами, поэтому берём лог-приращение и делим на
   ожидаемое волатильное движение ``mean(ATR%) · sqrt(bars)``.

Ключевое архитектурное решение — **раскол «метрики ↔ вердикт»**:

============================  ==================  =================
стадия                        зависит от данных   зависит от слайдера
============================  ==================  =================
``compute_regime_metrics``    да (OHLCV)          нет
``evaluate_regime``           нет (только dict)   да
============================  ==================  =================

Метрики считаются один раз при сканировании тикера (дорого — нужен OHLCV),
вердикт — на каждый запрос пользователя с его личным слайдером (микросекунды).
Благодаря этому персональный порог флэта не требует пересканирования.

Слайдер
-------
``slider_multiplier(s) = 2 ** (2s - 1)``: ``0 → ×0.5`` (строгий флэт, режем
только совсем мёртвый рынок), ``0.5 → ×1.0`` (умеренно), ``1 → ×2.0``
(широкий флэт, «боковиком» помечается больше состояний).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

__all__ = [
    "RegimeParams",
    "RegimeSliders",
    "DEFAULT_PARAMS",
    "DEFAULT_SLIDERS",
    "sigmoid",
    "below_score",
    "slider_multiplier",
    "add_indicators",
    "percentile_rank",
    "compute_regime_metrics",
    "evaluate_regime",
    "analyze_regime",
    "signal_allowed",
    "ENTRY_ORDER_TYPES",
    "EXIT_ORDER_TYPES",
]

_EPS = 1e-12

#: Мягкость сигмоидных переходов (насколько плавно порог превращается в 0/1).
TAU_ATR = 0.05
TAU_BBW = 0.05
TAU_RANK = 0.10
TAU_PRICE = 0.25
TAU_EXPANSION = 0.05


# ====================================================================== #
#  Параметры
# ====================================================================== #
@dataclass(frozen=True)
class RegimeParams:
    """Слайдер-независимые параметры расчёта метрик.

    Attributes
    ----------
    window : int
        Основное окно анализа ``W`` (по умолчанию 200 баров).
    recent : int
        Недавнее окно ``N`` для оценки свежего движения (по умолчанию 50).
    atr_period : int
        Период ATR (Wilder RMA), по умолчанию 14.
    bb_period : int
        Период Bollinger Bands, по умолчанию 20.
    bb_mult : float
        Множитель стандартного отклонения BB, по умолчанию 2.0.
    w_price_recent : float
        Вес свежего окна в итоговом ``z`` (0.6 — свежее движение весомее).
    """

    window: int = 200
    recent: int = 50
    atr_period: int = 14
    bb_period: int = 20
    bb_mult: float = 2.0
    w_price_recent: float = 0.6

    def __post_init__(self) -> None:
        for name in ("window", "recent", "atr_period", "bb_period"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.recent > self.window:
            raise ValueError("recent (N) must be <= window (W)")
        if self.bb_mult <= 0:
            raise ValueError("bb_mult must be positive")
        if not 0.0 <= self.w_price_recent <= 1.0:
            raise ValueError("w_price_recent must be in [0, 1]")

    @property
    def min_bars(self) -> int:
        """Минимальная длина истории для расчёта метрик."""
        return max(self.window, self.recent) + 5


@dataclass(frozen=True)
class RegimeSliders:
    """Персональные слайдеры «что считаем флэтом» (0..1 каждый).

    ``flat`` — общий слайдер; ``atr``/``bbw``/``pct`` — индивидуальные
    (``None`` → берётся общий). ``0`` — очень строгое определение флэта,
    ``1`` — широкое.
    """

    flat: float = 0.5
    atr: Optional[float] = None
    bbw: Optional[float] = None
    pct: Optional[float] = None
    flat_score_threshold: float = 60.0
    trend_strength_low: float = 30.0

    def resolved(self) -> tuple[float, float, float]:
        """Вернуть (s_atr, s_bbw, s_pct) с подстановкой общего слайдера."""
        base = _clamp01(self.flat)
        return (
            base if self.atr is None else _clamp01(self.atr),
            base if self.bbw is None else _clamp01(self.bbw),
            base if self.pct is None else _clamp01(self.pct),
        )

    def multipliers(self) -> tuple[float, float, float]:
        """Мультипликаторы порогов (m_atr, m_bbw, m_pct)."""
        s_atr, s_bbw, s_pct = self.resolved()
        return (
            slider_multiplier(s_atr),
            slider_multiplier(s_bbw),
            slider_multiplier(s_pct),
        )

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "RegimeSliders":
        """Собрать слайдеры из «сырого» словаря (настройки пользователя)."""
        if not isinstance(data, dict):
            return cls()
        kwargs: dict[str, Any] = {}
        for key in ("flat", "atr", "bbw", "pct"):
            raw = data.get(key, data.get(f"slider_{key}"))
            val = _to_float(raw)
            if val is None:
                continue
            kwargs[key] = _clamp01(val)
        for key, lo, hi in (
            ("flat_score_threshold", 0.0, 100.0),
            ("trend_strength_low", 0.0, 100.0),
        ):
            val = _to_float(data.get(key))
            if val is not None:
                kwargs[key] = float(min(max(val, lo), hi))
        return cls(**kwargs)


DEFAULT_PARAMS = RegimeParams()
DEFAULT_SLIDERS = RegimeSliders()

#: Типы ордеров, требующие подтверждённого направления тренда.
ENTRY_ORDER_TYPES: dict[str, int] = {
    "entry_long": +1,
    "add_long": +1,
    "entry_short": -1,
    "add_short": -1,
}
#: Выходы: закрытие позиции снижает риск — флэт-фильтром не режем.
EXIT_ORDER_TYPES: frozenset[str] = frozenset({"exit_long", "exit_short"})


# ====================================================================== #
#  Скалярные хелперы
# ====================================================================== #
def _clamp01(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(v):
        return 0.5
    return float(min(max(v, 0.0), 1.0))


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def sigmoid(x: float) -> float:
    """Сигмоида с клипом аргумента (защита от переполнения ``exp``)."""
    xv = float(np.clip(float(x), -20.0, 20.0))
    return float(1.0 / (1.0 + math.exp(-xv)))


def below_score(x: float, threshold: float, tau: float) -> float:
    """Насколько ``x`` «заметно ниже» порога: 1.0 — да, 0.0 — нет.

    При ``tau <= 0`` превращается в жёсткое сравнение.
    """
    if tau <= 0:
        return 1.0 if x <= threshold else 0.0
    return sigmoid((threshold - x) / tau)


def slider_multiplier(s: float) -> float:
    """Мультипликатор порогов флэта: ``0 → 0.5``, ``0.5 → 1.0``, ``1 → 2.0``."""
    return float(2.0 ** (2.0 * _clamp01(s) - 1.0))


def percentile_rank(value: float, window: Any) -> float:
    """Доля значений окна, которые меньше ``value`` (0..1); NaN при пустом окне."""
    series = pd.Series(window, dtype="float64").dropna()
    if len(series) == 0:
        return float("nan")
    return float((series < value).mean())


# ====================================================================== #
#  Индикаторы
# ====================================================================== #
def _normalize_ohlc(df: Any) -> pd.DataFrame:
    """Привести входной OHLCV к нижнему регистру колонок; проверить обязательные."""
    if df is None:
        raise ValueError("OHLCV не передан")
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    missing = [c for c in ("high", "low", "close") if c not in out.columns]
    if missing:
        raise ValueError(f"В OHLCV нет колонок: {', '.join(missing)}")
    if "open" not in out.columns:
        out["open"] = out["close"]
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_indicators(df: Any, params: RegimeParams | None = None) -> pd.DataFrame:
    """Добавить ``atr`` (Wilder RMA), ``atr_pct`` и ``bbw`` к копии OHLCV.

    ``ATR`` — RMA(TrueRange, atr_period); ``BBW = 2·mult·stdev(close) / SMA(close)``
    (``ddof=0``, как в Pine/``ta``); ``atr_pct = ATR / close``.
    """
    p = params or DEFAULT_PARAMS
    out = _normalize_ohlc(df)

    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["atr"] = tr.ewm(
        alpha=1.0 / float(p.atr_period), adjust=False, min_periods=p.atr_period
    ).mean()

    ma = out["close"].rolling(p.bb_period).mean()
    std = out["close"].rolling(p.bb_period).std(ddof=0)
    ma_safe = ma.replace(0.0, np.nan)

    out["bbw"] = (2.0 * float(p.bb_mult) * std) / ma_safe
    out["atr_pct"] = out["atr"] / out["close"].replace(0.0, np.nan)
    return out


def _slope_norm(series: pd.Series, mean_value: float) -> Optional[float]:
    """Нормированный наклон: ``slope · n / mean`` (диагностика «режима slope»)."""
    vals = pd.Series(series, dtype="float64").dropna()
    n = len(vals)
    if n < 3 or not math.isfinite(mean_value) or abs(mean_value) <= _EPS:
        return None
    x = np.arange(n, dtype="float64")
    try:
        slope = float(np.polyfit(x, vals.to_numpy(dtype="float64"), 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not math.isfinite(slope):
        return None
    return float(slope * n / mean_value)


# ====================================================================== #
#  Стадия 1: метрики (не зависят от слайдера)
# ====================================================================== #
def compute_regime_metrics(
    df: Any,
    i: int = -1,
    params: RegimeParams | None = None,
    *,
    with_indicators: bool = True,
) -> Optional[dict[str, Any]]:
    """Слайдер-независимые метрики режима на баре ``i``.

    Parameters
    ----------
    df : DataFrame
        OHLCV (колонки ``open/high/low/close``, регистр не важен). Если
        ``with_indicators=False``, ожидаются готовые ``atr``/``bbw``/``atr_pct``.
    i : int
        Индекс бара (по позиции). ``-1`` — последний бар.
    params : RegimeParams
        Окна и периоды индикаторов.

    Returns
    -------
    dict | None
        ``None``, если данных недостаточно или в нужных точках NaN — вызывающая
        сторона обязана трактовать это как «верифицировать нельзя» и НЕ резать
        сигналы молча.
    """
    p = params or DEFAULT_PARAMS
    try:
        data = add_indicators(df, p) if with_indicators else _normalize_ohlc(df)
    except (ValueError, TypeError):
        return None
    if not {"atr", "bbw", "atr_pct"} <= set(data.columns):
        return None

    n_rows = len(data)
    idx = int(i)
    if idx < 0:
        idx += n_rows
    if idx < 0 or idx >= n_rows:
        return None
    if n_rows < p.min_bars or idx < p.min_bars - 1:
        return None
    if idx - p.recent < 0 or idx - p.window + 1 < 0:
        return None

    row = data.iloc[idx]
    row_prev_n = data.iloc[idx - p.recent]
    row_start_w = data.iloc[idx - p.window + 1]

    required = (
        row["close"], row["atr"], row["bbw"], row["atr_pct"],
        row_prev_n["close"], row_prev_n["atr"], row_prev_n["bbw"],
        row_start_w["close"],
    )
    if any(pd.isna(v) for v in required):
        return None

    close = float(row["close"])
    close_start_w = float(row_start_w["close"])
    close_prev_n = float(row_prev_n["close"])
    if close <= 0 or close_start_w <= 0 or close_prev_n <= 0:
        return None

    atr_now = float(row["atr"])
    atr_prev_n = float(row_prev_n["atr"])
    bbw_now = float(row["bbw"])
    bbw_prev_n = float(row_prev_n["bbw"])
    atr_pct_now = float(row["atr_pct"])

    window_w = data.iloc[idx - p.window + 1: idx + 1]
    window_n = data.iloc[idx - p.recent + 1: idx + 1]

    # --- 1. Нормированное на волатильность движение цены -------------
    ret_w = math.log(close / max(close_start_w, _EPS))
    ret_n = math.log(close / max(close_prev_n, _EPS))

    atr_pct_mean_w = _safe_mean(window_w["atr_pct"])
    atr_pct_mean_n = _safe_mean(window_n["atr_pct"])

    expected_move_w = atr_pct_mean_w * math.sqrt(p.window)
    expected_move_n = atr_pct_mean_n * math.sqrt(p.recent)

    z_w = ret_w / max(expected_move_w, _EPS)
    z_n = ret_n / max(expected_move_n, _EPS)
    w_recent = float(p.w_price_recent)
    z = (1.0 - w_recent) * z_w + w_recent * z_n

    # --- 2. Изменение ATR и BBW за последние N баров ------------------
    atr_change_n = atr_now / max(atr_prev_n, _EPS) - 1.0
    bbw_change_n = bbw_now / max(bbw_prev_n, _EPS) - 1.0

    # --- 3. Ранги внутри окна W --------------------------------------
    atr_rank_w = percentile_rank(atr_pct_now, window_w["atr_pct"])
    bbw_rank_w = percentile_rank(bbw_now, window_w["bbw"])
    if not math.isfinite(atr_rank_w):
        atr_rank_w = 0.5
    if not math.isfinite(bbw_rank_w):
        bbw_rank_w = 0.5

    return {
        "bar_index": idx,
        "bars_available": n_rows,
        "window": int(p.window),
        "recent": int(p.recent),
        "close": close,
        "atr": atr_now,
        "bbw": bbw_now,
        "atr_pct": atr_pct_now,
        "ret_w_log": float(ret_w),
        "ret_n_log": float(ret_n),
        "pct_change_w": float((close / close_start_w - 1.0) * 100.0),
        "pct_change_n": float((close / close_prev_n - 1.0) * 100.0),
        "z_w": float(z_w),
        "z_n": float(z_n),
        "z": float(z),
        "atr_change_n": float(atr_change_n),
        "bbw_change_n": float(bbw_change_n),
        "atr_slope_n": _slope_norm(window_n["atr"], _safe_mean(window_n["atr"])),
        "bbw_slope_n": _slope_norm(window_n["bbw"], _safe_mean(window_n["bbw"])),
        "atr_rank_w": float(atr_rank_w),
        "bbw_rank_w": float(bbw_rank_w),
        "atr_pct_mean_w": float(atr_pct_mean_w),
        "atr_pct_mean_n": float(atr_pct_mean_n),
    }


def _safe_mean(series: Any) -> float:
    """Среднее с защитой: NaN/0/отрицательное → ``_EPS``."""
    try:
        value = float(pd.Series(series, dtype="float64").mean())
    except (TypeError, ValueError):
        return _EPS
    if not math.isfinite(value) or value <= _EPS:
        return _EPS
    return value


# ====================================================================== #
#  Стадия 2: вердикт (зависит только от слайдера)
# ====================================================================== #
def evaluate_regime(
    metrics: Optional[dict[str, Any]],
    sliders: RegimeSliders | None = None,
) -> Optional[dict[str, Any]]:
    """Вердикт по метрикам с учётом персональных слайдеров.

    Чистая арифметика над результатом :func:`compute_regime_metrics` — без
    обращения к данным, поэтому вызывается на каждый запрос пользователя.

    Returns
    -------
    dict | None
        ``state`` (``UP``/``DOWN``/``FLAT``), ``direction`` (+1/0/-1),
        ``trend_strength`` 0..100, ``flat_score`` 0..100, ``is_flat``,
        ``is_chop``, плюс компоненты/веса/пороги для диагностики.
        ``None`` — если метрик нет (верифицировать нечем).
    """
    if not isinstance(metrics, dict):
        return None
    s = sliders or DEFAULT_SLIDERS

    z = _to_float(metrics.get("z"))
    atr_change_n = _to_float(metrics.get("atr_change_n"))
    bbw_change_n = _to_float(metrics.get("bbw_change_n"))
    atr_rank_w = _to_float(metrics.get("atr_rank_w"))
    bbw_rank_w = _to_float(metrics.get("bbw_rank_w"))
    if z is None or atr_change_n is None or bbw_change_n is None:
        return None
    if atr_rank_w is None:
        atr_rank_w = 0.5
    if bbw_rank_w is None:
        bbw_rank_w = 0.5

    m_atr, m_bbw, m_pct = s.multipliers()
    abs_z = abs(z)

    # --- Пороги флэта с учётом слайдера ------------------------------
    atr_flat_change_thr = 0.05 * m_atr
    bbw_flat_change_thr = 0.05 * m_bbw
    rank_flat_thr = float(np.clip(0.35 * ((m_atr + m_bbw) / 2.0), 0.05, 0.90))
    norm_return_flat_thr = 0.60 * m_pct

    # --- Компоненты флэта --------------------------------------------
    f_atr_change = below_score(atr_change_n, atr_flat_change_thr, TAU_ATR)
    f_atr_rank = below_score(atr_rank_w, rank_flat_thr, TAU_RANK)
    f_atr = f_atr_change * f_atr_rank

    f_bbw_change = below_score(bbw_change_n, bbw_flat_change_thr, TAU_BBW)
    f_bbw_rank = below_score(bbw_rank_w, rank_flat_thr, TAU_RANK)
    f_bbw = f_bbw_change * f_bbw_rank

    f_price = below_score(abs_z, norm_return_flat_thr, TAU_PRICE)

    w_atr_raw = 0.35 * m_atr
    w_bbw_raw = 0.30 * m_bbw
    w_price_raw = 0.35 * m_pct
    w_sum = w_atr_raw + w_bbw_raw + w_price_raw
    if w_sum <= _EPS:
        w_atr, w_bbw, w_price = 0.35, 0.30, 0.35
    else:
        w_atr = w_atr_raw / w_sum
        w_bbw = w_bbw_raw / w_sum
        w_price = w_price_raw / w_sum

    flat_score = 100.0 * (w_atr * f_atr + w_bbw * f_bbw + w_price * f_price)

    # --- Сила тренда --------------------------------------------------
    z_trend_target = 1.50 * m_pct
    dir_score = min(1.0, abs_z / max(z_trend_target, _EPS))

    e_atr = sigmoid((atr_change_n - 0.10 * m_atr) / TAU_EXPANSION)
    e_bbw = sigmoid((bbw_change_n - 0.10 * m_bbw) / TAU_EXPANSION)
    expansion_score = 0.5 * e_atr + 0.5 * e_bbw
    level_score = 0.5 * atr_rank_w + 0.5 * bbw_rank_w

    trend_strength_raw = 100.0 * (
        0.55 * dir_score + 0.30 * expansion_score + 0.15 * level_score
    )
    if flat_score > 60.0:
        penalty = (flat_score - 60.0) / 60.0
        trend_strength = trend_strength_raw * max(0.0, 1.0 - penalty)
    else:
        trend_strength = trend_strength_raw
    trend_strength = float(np.clip(trend_strength, 0.0, 100.0))

    # --- Направление и итоговое состояние ----------------------------
    direction_threshold = 0.30 * m_pct
    strong_direction = abs_z > (1.20 * m_pct)

    is_flat_candidate = (
        flat_score >= float(s.flat_score_threshold)
        or trend_strength < float(s.trend_strength_low)
        or abs_z < direction_threshold
    )
    is_flat = bool(is_flat_candidate and not strong_direction)
    is_chop = bool(
        abs_z < direction_threshold and (atr_rank_w > 0.65 or bbw_rank_w > 0.65)
    )

    if is_flat:
        state, direction = "FLAT", 0
    elif z > 0:
        state, direction = "UP", 1
    else:
        state, direction = "DOWN", -1

    atr_pct_mean_n = _to_float(metrics.get("atr_pct_mean_n")) or 0.0
    recent = int(_to_float(metrics.get("recent")) or DEFAULT_PARAMS.recent)
    allowed_flat_pct_n = (
        norm_return_flat_thr * atr_pct_mean_n * math.sqrt(max(recent, 1)) * 100.0
    )

    s_atr, s_bbw, s_pct = s.resolved()
    return {
        "state": state,
        "direction": direction,
        "trend_strength": trend_strength,
        "flat_score": float(flat_score),
        "is_flat": is_flat,
        "is_chop": is_chop,
        "allowed_flat_pct_n": float(allowed_flat_pct_n),
        "sliders": {
            "flat": float(_clamp01(s.flat)),
            "atr": float(s_atr),
            "bbw": float(s_bbw),
            "pct": float(s_pct),
            "flat_score_threshold": float(s.flat_score_threshold),
            "trend_strength_low": float(s.trend_strength_low),
        },
        "flat_components": {
            "f_atr": float(f_atr),
            "f_bbw": float(f_bbw),
            "f_price": float(f_price),
        },
        "weights": {
            "w_atr": float(w_atr),
            "w_bbw": float(w_bbw),
            "w_price": float(w_price),
        },
        "thresholds": {
            "atr_flat_change_thr": float(atr_flat_change_thr),
            "bbw_flat_change_thr": float(bbw_flat_change_thr),
            "rank_flat_thr": float(rank_flat_thr),
            "norm_return_flat_thr": float(norm_return_flat_thr),
            "direction_threshold": float(direction_threshold),
            "z_trend_target": float(z_trend_target),
        },
        "multipliers": {
            "m_atr": float(m_atr),
            "m_bbw": float(m_bbw),
            "m_pct": float(m_pct),
        },
    }


def analyze_regime(
    df: Any,
    i: int = -1,
    params: RegimeParams | None = None,
    sliders: RegimeSliders | None = None,
) -> Optional[dict[str, Any]]:
    """Удобная обёртка: метрики + вердикт одним вызовом (``metrics`` внутри)."""
    metrics = compute_regime_metrics(df, i=i, params=params)
    verdict = evaluate_regime(metrics, sliders)
    if verdict is None:
        return None
    return {**verdict, "metrics": metrics}


# ====================================================================== #
#  Гейт сигналов
# ====================================================================== #
def signal_allowed(
    order_type: Optional[str],
    verdict: Optional[dict[str, Any]],
    *,
    gate_exits: bool = False,
) -> tuple[bool, Optional[str]]:
    """Пропускать ли сигнал при данном вердикте режима.

    Правила:

    * ``verdict is None`` → **пропускаем** (верифицировать нечем — молча резать
      сигналы нельзя);
    * выходы (``exit_long``/``exit_short``) не блокируются, пока
      ``gate_exits=False`` — закрытие позиции снижает риск;
    * ``is_flat`` → блок с причиной ``flat``;
    * вход в лонг требует ``state == UP``, в шорт — ``DOWN``; иначе блок с
      причиной ``direction_mismatch``.

    Returns
    -------
    (bool, str | None)
        Разрешён ли сигнал и причина блокировки (``None`` если разрешён).
    """
    if not isinstance(verdict, dict):
        return True, None

    ot = str(order_type or "").strip().lower()
    if ot in EXIT_ORDER_TYPES and not gate_exits:
        return True, None

    if verdict.get("is_flat"):
        return False, "flat"

    want = ENTRY_ORDER_TYPES.get(ot)
    if want is None:
        return True, None

    state = str(verdict.get("state") or "").upper()
    if want > 0 and state != "UP":
        return False, "direction_mismatch"
    if want < 0 and state != "DOWN":
        return False, "direction_mismatch"
    return True, None
