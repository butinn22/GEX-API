"""Пивоты (свинг-экстремумы) — канон (ring: domain, только numpy).

До рефакторинга «пивот» вычислялся четырьмя фрагментами кода с **тремя разными предикатами** —
отсюда расхождение свингов между ``/ta``, ``/trendlines`` и ``/ta-structure`` для одного тикера:

| Место | Окно | Предикат |
|---|---|---|
| ``ta.detect_trend`` (``ta.py:749-755``) | ``[i-k, i+k]``, ``k = SWING_K`` | ``center == max(окно)`` и центр уникален |
| ``ta.detect_divergences`` (``ta.py:631-640``) | ``[i-2, i+2]`` (k=2 захардкожен) | тот же предикат (копия) |
| ``trendlines.find_pivot_highs/lows`` (``:494-526``) | ``[i-left, i+right]`` (вызовы 5/5) | ``center >= max(окно)`` и центр уникален |
| ``hybrid_trend._fractals`` (``:294-319``) | ``[i-left, i+right]`` | ``strict``: ``center > max(остальных)``; ``strict=False``: ``center >= max`` **и** ``center > min`` |

Первые три предиката **математически эквивалентны** (уникальный максимум окна ⟺ строго больше всех
остальных) — различаются только окном. Четвёртый в нестрогом режиме — **другой**: он допускает плато
на вершине (равные соседи), но отбрасывает полностью плоское окно. Канон сохраняет оба режима явно,
поэтому миграция вызывающих не меняет числа:

* ``mode="strict"`` — как ``ta``/``trendlines``;
* ``mode="plateau"`` — как ``hybrid_trend`` при ``strict=False``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..indicators._kernels import as_float_array

__all__ = ["MODES", "pivot_mask", "pivot_indices", "pivot_prices"]

MODES = ("strict", "plateau")


def _validate(left: int, right: int, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if left < 0 or right < 0:
        raise ValueError("left/right must be non-negative")


def pivot_mask(
    values: Sequence[float] | np.ndarray,
    left: int,
    right: int,
    *,
    mode: str = "strict",
    kind: str = "high",
) -> np.ndarray:
    """Булев массив «бар является пивотом» (форма ответа ``hybrid_trend._fractals``).

    Границы совпадают со всеми прежними реализациями: рассматриваются индексы
    ``left .. n-right-1`` (нужно ``right`` баров справа для подтверждения).

    Parameters
    ----------
    kind : {"high", "low"}
        ``high`` — ищем локальный максимум (свинг-хай), ``low`` — минимум (свинг-лой).
        Раньше это были две разные функции (``find_pivot_highs``/``find_pivot_lows``), а в
        ``ta.detect_trend`` — два инлайн-блока с зеркальным предикатом.
    """
    _validate(left, right, mode)
    if kind not in ("high", "low"):
        raise ValueError(f"kind must be 'high' or 'low', got {kind!r}")
    arr = as_float_array(values)
    n = len(arr)
    mask = np.zeros(n, dtype=bool)
    if n < left + right + 1 or (left == 0 and right == 0):
        return mask

    for i in range(left, n - right):
        center = arr[i]
        if np.isnan(center):
            continue
        window = arr[i - left : i + right + 1]
        others = np.concatenate([window[:left], window[left + 1 :]])
        if others.size == 0 or np.isnan(others).any():
            continue
        if kind == "high":
            if mode == "strict":
                mask[i] = bool(center > np.max(others))
            else:  # plateau: допускаем равных соседей, но не полностью плоское окно
                mask[i] = bool(center >= np.max(others) and center > np.min(others))
        else:
            if mode == "strict":
                mask[i] = bool(center < np.min(others))
            else:
                mask[i] = bool(center <= np.min(others) and center < np.max(others))
    return mask


def pivot_indices(
    values: Sequence[float] | np.ndarray,
    left: int,
    right: int,
    *,
    mode: str = "strict",
    kind: str = "high",
) -> np.ndarray:
    """Индексы пивотов (форма ответа ``ta.detect_trend`` и ``trendlines``)."""
    return np.flatnonzero(pivot_mask(values, left, right, mode=mode, kind=kind))


def pivot_prices(
    values: Sequence[float] | np.ndarray,
    left: int,
    right: int,
    *,
    mode: str = "strict",
    kind: str = "high",
) -> np.ndarray:
    """Цены пивотов (в порядке появления)."""
    arr = as_float_array(values)
    return arr[pivot_indices(arr, left, right, mode=mode, kind=kind)]
