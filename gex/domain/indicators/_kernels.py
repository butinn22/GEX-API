"""Индикаторные ядра (ring: domain) — только numpy, без pandas и без I/O.

Зачем разделение `_kernels` ↔ `primitives`
------------------------------------------
* **ядро** принимает/возвращает ``numpy.ndarray`` и не тянет pandas — его можно запускать и
  проверять на любой машине, где есть только numpy (см. `tests/test_indicator_kernels.py`);
* **обёртки** (`primitives.py`) добавляют работу с ``pd.Series`` (индекс, dtype) для вызывающих.

Что здесь канон, а что было копиями
-----------------------------------
До рефакторинга одни и те же примитивы жили в нескольких местах:
``rsi_novel.pine_rma/pine_ema/pine_sma/pine_stdev``, ``ta._wilder_rsi`` (свой RMA),
``volatility_cone.compute_rsi_wilder``, ещё один RMA-цикл в ``trendlines._wilder_atr``.
Семантика ниже **дословно повторяет** реализацию из ``rsi_novel.py:53-151`` (Pine-совместимую):
поэтому вынос в канон не меняет числа ни на одной странице.

Семантика (совпадает с оригиналом, включая краевые случаи)
----------------------------------------------------------
* ``rma`` — Wilder's MA: сид = SMA первого сплошного окна длины ``length``, далее
  ``alpha*x + (1-alpha)*prev``, ``alpha = 1/length``; на NaN входе — NaN, на первом валидном
  значении после «дырки» — само значение;
* ``ema`` — то же, но ``alpha = 2/(length+1)``;
* ``sma``/``stdev`` — окно длины ``length``, ``min_periods=length`` (любой NaN в окне → NaN),
  ``stdev`` считается с ``ddof=0`` (biased), как ``ta.stdev()``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "as_float_array",
    "first_solid_window",
    "rma",
    "ema",
    "sma",
    "stdev",
    "bollinger",
]


def as_float_array(values: object) -> np.ndarray:
    """Привести вход к ``float64``-массиву без pandas (list/tuple/ndarray/Series-подобное)."""
    if isinstance(values, np.ndarray):
        return values.astype(float, copy=False)
    # pd.Series и любой объект с to_numpy() — принимаем через duck-typing, не импортируя pandas
    to_numpy = getattr(values, "to_numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy(), dtype=float)
    return np.asarray(values, dtype=float)


def first_solid_window(values: np.ndarray, length: int) -> tuple[int, np.ndarray] | None:
    """Индекс конца и содержимое первого окна длины ``length`` без NaN.

    Повторяет двухступенчатый поиск оригинала: сначала берём окно, заканчивающееся на
    ``length``-м валидном значении, и только если в нём есть NaN — ищем первое сплошное окно.
    """
    if length <= 0 or len(values) < length:
        return None
    valid_positions = np.flatnonzero(~np.isnan(values))
    if len(valid_positions) < length:
        return None

    start = int(valid_positions[length - 1])
    window = values[start - length + 1 : start + 1]
    if not np.isnan(window).any():
        return start, window

    for i in range(length - 1, len(values)):
        candidate = values[i - length + 1 : i + 1]
        if not np.isnan(candidate).any():
            return i, candidate
    return None


def _recursive_smoothed(values: np.ndarray, length: int, alpha: float) -> np.ndarray:
    """Общий цикл для RMA/EMA: сид из SMA, дальше рекурсия с заданным ``alpha``."""
    result = np.full(len(values), np.nan, dtype=float)
    seeded = first_solid_window(values, length)
    if seeded is None:
        return result
    start, window = seeded
    result[start] = float(np.mean(window))

    for i in range(start + 1, len(values)):
        value = values[i]
        if np.isnan(value):
            result[i] = np.nan
        elif np.isnan(result[i - 1]):
            result[i] = value
        else:
            result[i] = alpha * value + (1.0 - alpha) * result[i - 1]
    return result


def rma(values: Sequence[float] | np.ndarray, length: int) -> np.ndarray:
    """Wilder's Moving Average (``ta.rma``): ``alpha = 1/length``."""
    arr = as_float_array(values)
    if length <= 0:
        return np.full(len(arr), np.nan, dtype=float)
    return _recursive_smoothed(arr, length, 1.0 / length)


def ema(values: Sequence[float] | np.ndarray, length: int) -> np.ndarray:
    """EMA с SMA-сидом (``ta.ema``): ``alpha = 2/(length+1)``."""
    arr = as_float_array(values)
    if length <= 0:
        return np.full(len(arr), np.nan, dtype=float)
    return _recursive_smoothed(arr, length, 2.0 / (length + 1.0))


def _rolling_windows(values: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Матрица окон без NaN-строк + маска валидности (аналог rolling(min_periods=length))."""
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    valid = np.zeros(n, dtype=bool)
    if length <= 0 or n < length:
        return out, valid
    solid = ~np.isnan(values)
    for i in range(length - 1, n):
        window = values[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            out[i] = float(np.mean(window))
            valid[i] = True
    return out, valid


def sma(values: Sequence[float] | np.ndarray, length: int) -> np.ndarray:
    """SMA с ``min_periods=length``: любое NaN в окне → NaN (как ``pine_sma``)."""
    arr = as_float_array(values)
    out, _ = _rolling_windows(arr, length)
    return out


def stdev(values: Sequence[float] | np.ndarray, length: int, *, ddof: int = 0) -> np.ndarray:
    """Скользящее стандартное отклонение (по умолчанию biased, ``ddof=0``, как ``ta.stdev``)."""
    arr = as_float_array(values)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if length <= 0 or n < length:
        return out
    for i in range(length - 1, n):
        window = arr[i - length + 1 : i + 1]
        if not np.isnan(window).any():
            out[i] = float(np.std(window, ddof=ddof))
    return out


def bollinger(
    values: Sequence[float] | np.ndarray, length: int, mult: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Полосы Боллинджера ``ta.bb``: (basis, upper, lower)."""
    arr = as_float_array(values)
    basis = sma(arr, length)
    deviation = mult * stdev(arr, length)
    return basis, basis + deviation, basis - deviation


def linreg(values: Sequence[float] | np.ndarray, length: int, offset: int = 0) -> np.ndarray:
    """Линейная регрессия ``ta.linreg(source, length, offset)`` — канон.

    Значение берётся в точке окна ``x = length - 1 - offset`` (offset=0 → последний бар окна).
    Окно с любым NaN пропускается (остаётся NaN) — как в оригинале ``rsi_novel.pine_linreg``.
    """
    arr = as_float_array(values)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if length <= 0 or n < length:
        return out

    x = np.arange(length, dtype=float)
    sum_x = float(np.sum(x))
    sum_x2 = float(np.sum(x * x))
    denominator = length * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return out

    for i in range(length - 1, n):
        window = arr[i - length + 1 : i + 1]
        if np.isnan(window).any():
            continue
        sum_y = float(np.sum(window))
        sum_xy = float(np.sum(x * window))
        slope = (length * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / length
        out[i] = intercept + slope * (length - 1 - offset)
    return out


def linear_fit(
    x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray
) -> tuple[float, float]:
    """МНК-регрессия ``y = slope·x + intercept`` → ``(slope, intercept)``.

    Канон для ``rsi_novel.calc_linreg_custom``. Внимание: оригинал в одном месте вызывался с
    **переставленными** x/y (`rsi_novel.py:403`), из-за чего знак наклона зеркалился — при миграции
    это исправляется явным порядком аргументов, а не «наследованием» ошибки.
    """
    xs = as_float_array(x)
    ys = as_float_array(y)
    n = len(xs)
    if n == 0 or len(ys) != n:
        return float("nan"), float("nan")
    sum_x = float(np.sum(xs))
    sum_y = float(np.sum(ys))
    sum_xy = float(np.sum(xs * ys))
    sum_x2 = float(np.sum(xs * xs))
    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return float("nan"), float("nan")
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept
