"""Примитивы расчёта: скользящие, ATR, линрег, two-pole filter, полосы Боллинджера.

Почему отдельным модулем: это единственная часть стратегии **без состояния** — результат
зависит только от входа. Такие функции проверяются напрямую, без сборки класса и без кадра
на 110 колонок.

``_two_pole_filter`` — рекурсивный фильтр (каждый бар зависит от предыдущего), поэтому
векторизации нет: есть numba-версия и numpy-фолбэк, выбор происходит при импорте
``_HAVE_NUMBA``. Имена сохранены с подчёркиванием: их импортируют тесты, и переименование
здесь не дало бы ничего, кроме правок в чужих файлах.

Ниже — миксин класса: методы-обёртки над этими же функциями. Оставлены методами, а не
переписаны на вызовы модульных функций сознательно: переписать все места вызова
(``self._ema(...)`` в десятках методов) — отдельная правка со своим риском, а разбиение
god-класса обязано быть проверяемым шагом без изменения чисел.
"""
from __future__ import annotations

import logging
from math import pi
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame

from .settings import TradingSignal, TradingState

logger = logging.getLogger(__name__)


def _two_pole_filter_vectorized(
    values: np.ndarray,
    length: float,
    damping: float,
) -> np.ndarray:
    """Двухполюсный фильтр Баттерворта.

    Pine Script v5 equivalent::

        two_pole_filter(src, len, damp) =>
            omega = 2.0 * pi / len
            alpha = damp * omega
            beta = omega ** 2
            f1[1] += alpha * (src - f1[1])
            f2[1] += beta * (f1 - f2[1])
            f2

    Реализация в Python — прямой цикл (numba при импорте с numba=True).

    Parameters
    ----------
    values : np.ndarray
        1D-массив входных значений (например adline или novelsrc).
    length : float
        Период фильтра (Pine ``length_tp``).
    damping : float
        Коэффициент демпфирования (0.1..1.0).

    Returns
    -------
    np.ndarray
        Отфильтрованный сигнал (f2), той же длины, что и values.
        Первые 2 элемента — NaN (нет предыдущего состояния).
    """
    out = np.full(len(values), np.nan)
    if len(values) < 3:
        return out

    omega = 2.0 * pi / max(length, 1.0)
    alpha = float(damping) * omega
    beta = omega ** 2

    # Pine: var float f1=na → nz(f1[1]) = 0 на первом баре
    f1 = 0.0
    f2 = 0.0
    for i in range(len(values)):
        src = float(values[i])
        f1 = f1 + alpha * (src - f1)
        f2 = f2 + beta * (f1 - f2)
        out[i] = f2
    return out


# Попытка импорта numba для JIT-ускорения; если нет — падаем на чистый numpy.
try:
    from numba import njit

    @njit(cache=True)
    def _two_pole_filter_numba(values: np.ndarray, length: float, damping: float) -> np.ndarray:
        """numba-JIT версия ``_two_pole_filter_vectorized``.

        Инициализация как в Pine: nz(f1[1]) = 0 на первом баре.
        """
        out = np.full(len(values), np.nan)
        if len(values) < 3:
            return out
        omega = 2.0 * pi / max(length, 1.0)
        alpha = damping * omega
        beta = omega ** 2
        # Pine: var float f1 = na → nz(f1[1]) = 0 → f1[0] = alpha * src[0]
        f1 = 0.0
        f2 = 0.0
        for i in range(len(values)):
            src = values[i]
            f1 = f1 + alpha * (src - f1)
            f2 = f2 + beta * (f1 - f2)
            out[i] = f2
        return out

    _two_pole_filter = _two_pole_filter_numba
    _HAVE_NUMBA = True

except ImportError:
    _two_pole_filter = _two_pole_filter_vectorized
    _HAVE_NUMBA = False


def _bb(series: pd.Series, window: int, mult: float) -> tuple:
    """Bollinger Bands: (middle, upper, lower)."""
    mid = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std()
    return mid, mid + mult * std, mid - mult * std


def _cumsum_reset(values: np.ndarray, groups: np.ndarray) -> pd.Series:
    """Cumulative sum с периодическим сбросом по группам.

    Эквивалент Pine: ``if bar_index % N == 0: var := 0 else var += value``.
    """
    result = np.zeros_like(values, dtype=float)
    for g in np.unique(groups):
        mask = groups == g
        seq = np.arange(mask.sum())
        result[mask] = np.cumsum(values[mask]) * (seq >= 0)
    return pd.Series(result)


def _linreg_endpoint(values: np.ndarray) -> float:
    """Правый край МНК-прямой по окну (Pine ``ta.linreg(src, len, offset=0)``).

    Точная копия того, что делал ``np.linalg.lstsq`` на каждом окне — включая
    поведение при NaN (lstsq возвращает NaN, не бросает) и при окне короче
    двух точек (берётся последнее значение). Оставлена для «головы» ряда, где
    окно ещё короче ``length``.
    """
    n = len(values)
    if n < 2:
        return float(values[-1]) if n else np.nan
    x = np.arange(n, dtype=float)
    a = np.vstack([x, np.ones(n)]).T
    slope, intercept = np.linalg.lstsq(a, values, rcond=None)[0]
    return float(slope * (n - 1) + intercept)


def _linreg_weights(length: int) -> np.ndarray:
    """Веса ``c_k`` такие, что ``y_hat = Σ c_k y_k`` (k=0 — самый старый бар).

    Значение МНК-прямой на правом краю окна линейно по значениям окна, поэтому
    веса **постоянны** для данного ``length``. Вывод: наклон
    ``slope = (L*Σk·y_k - Sx*Σy_k) / denom``, значение на правом краю
    ``y_hat = ȳ + slope*(L-1)/2``, откуда
    ``c_k = 1/L + (L-1)*(L*k - Sx) / (2*denom)``, где ``denom = L*Sxx - Sx²``.
    """
    k = np.arange(length, dtype=float)
    sx = float(k.sum())
    sxx = float((k * k).sum())
    denom = length * sxx - sx * sx
    return 1.0 / length + (length - 1.0) * (length * k - sx) / (2.0 * denom)


def _linreg_series(y: "pd.Series", length: int) -> "pd.Series":
    """``rolling(length, min_periods=2).apply(linreg_endpoint)``, но одной свёрткой.

    Замер: расхождение с поштучным lstsq ~1e-13 (шум float64), ускорение
    340–1300x — на 2 979 окнах это была заметная доля ``calculate``.
    """
    l = int(length)
    vals = y.to_numpy(dtype=float)
    n = vals.size
    if n == 0:
        return pd.Series(vals, index=y.index)
    if l < 2:  # вырожденный период: поведение как у rolling(min_periods=1)
        return y.rolling(max(l, 1), min_periods=1).apply(
            lambda w: float(w.iloc[-1]) if len(w) else np.nan, raw=False
        )

    out = np.full(n, np.nan)
    start = l - 1

    # «Голова»: окна короче length. pandas с ``min_periods=2`` не вызывает ядро,
    # пока не набрано двух наблюдений, поэтому t=0 остаётся NaN (раньше здесь
    # случайно бралось ``values[-1]`` — расхождение ровно на первом баре).
    for t in range(1, min(start, n)):
        out[t] = _linreg_endpoint(vals[: t + 1])

    if n > start:
        weights = _linreg_weights(l)[::-1]  # r[j] умножает y[t-j]
        finite = np.nan_to_num(vals, nan=0.0)
        conv = np.convolve(finite, weights)[start:n]
        # lstsq на окне с NaN даёт NaN → окно, содержащее NaN, обнуляем.
        nan_csum = np.concatenate(([0], np.cumsum(np.isnan(vals).astype(np.int64))))
        t_idx = np.arange(start, n)
        win_nan = nan_csum[t_idx + 1] - nan_csum[t_idx - start]
        out[start:] = np.where(win_nan > 0, np.nan, conv)

    return pd.Series(out, index=y.index)


class _IndicatorMixin:
    @staticmethod
    def _ema(series: Any, window: int) -> Any:
        return series.ewm(span=window, adjust=False, min_periods=1).mean()


    @staticmethod
    def _sma(series: Any, window: int) -> Any:
        return series.rolling(window, min_periods=1).mean()


    @staticmethod
    def _rma(series: Any, window: int) -> Any:
        """Wilder's RMA (RMA = EMA with alpha=1/window, adjust=False)."""
        return series.ewm(alpha=1.0 / window, adjust=False, min_periods=1).mean()


    @staticmethod
    def _linreg(series: Any, length: int) -> Any:
        """Линейная регрессия (наклон + intercept) за length баров.

        Эквивалент Pine ``ta.linreg(source, length, offset=0)``.

        Раньше это был ``rolling(length).apply(np.linalg.lstsq)`` — по одному
        решателю на каждое окно (2 979 вызовов за прогон). Значение МНК-прямой на
        правом краю окна — фиксированная линейная комбинация значений окна, то
        есть один FIR-фильтр (свёртка). Числа те же (расхождение ~1e-13).
        """
        return _linreg_series(series, int(length))


    @staticmethod
    def _atr(high, low, close, window):
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=1).mean()


    @staticmethod
    def _crossover(left, right):
        return (left.shift(1) <= right.shift(1)) & (left > right)


    @staticmethod
    def _crossunder(left, right):
        return (left.shift(1) >= right.shift(1)) & (left < right)


    @staticmethod
    def _can_add(state: TradingState, bar_idx: int) -> bool:
        return state.last_add_bar is None or bar_idx - state.last_add_bar >= 10


    @staticmethod
    def _prepare_ohlc(ohlc: Any, pd_module: Any) -> Any:
        if ohlc is None:
            raise ValueError("ohlc DataFrame is required")
        data = ohlc.copy()
        data.columns = [str(c).strip().lower() for c in data.columns]
        aliases = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        data.rename(columns=aliases, inplace=True)
        for req in ("open", "high", "low", "close"):
            if req not in data.columns:
                raise ValueError(f"Missing OHLC column: {req}")
            data[req] = pd_module.to_numeric(data[req], errors="raise")
        if data.index.has_duplicates:
            raise ValueError("OHLC index contains duplicate timestamps")
        return data.sort_index()


    @staticmethod
    def _signal(action, reason, quantity_fraction=0.0, order_type="", **metadata):
        metadata["order_type"] = order_type
        return TradingSignal(action=action, reason=reason,
                             quantity_fraction=quantity_fraction, metadata=metadata)


    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            f = float(value)
            return f if f == f else None
        except (TypeError, ValueError):
            return None


    @staticmethod
    def _import_pandas():
        try:
            import pandas as pd
            return pd
        except ModuleNotFoundError as e:
            raise RuntimeError("pandas required") from e
