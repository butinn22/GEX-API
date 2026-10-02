"""Числовые хелперы (ring: domain, только numpy/math) — канон.

Что дублировалось
-----------------
Сигмоида, клиппинг z-вкладов и «насыщение через tanh» были переписаны в нескольких модулях с
**разными константами**: ``direction._sigmoid`` / ``_clip_z(lo=-3, hi=3)``, ``ta`` (своя сигмоида
для силы моментума, ``k=8``), ``hybrid_trend`` (tanh-насыщение), ``novel_candles`` (своя сигмоида).
Из-за этого одинаковые по смыслу факторы давали разные числа в зависимости от модуля.

Здесь собраны канонические примитивы: они параметризованы **явно**, поэтому вызывающий обязан
назвать свою семантику, а не наследовать чужую константу.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = ["sigmoid", "tanh_saturate", "clip_z", "z_combine", "safe_ratio"]


def sigmoid(x: float, *, gain: float = 1.0, center: float = 0.0) -> float:
    """Численно-стабильная логистическая функция ``1/(1+e^{-gain·(x-center)})``.

    Совпадает с ``direction._sigmoid`` при ``gain=1, center=0`` и с сигмоидой из
    ``ta.compute_momentum_strength`` (там ``gain=8`` и сдвиг ``center=0.15``) без переписывания.

    Ветвление по знаку аргумента (а не по его отрицанию!) — иначе функция зеркалится:
    ``sigmoid(-50)`` обязан быть ≈0, ``sigmoid(+50)`` ≈1.
    """
    if not np.isfinite(x):
        return 0.0
    a = gain * (x - center)
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a)) if a < 700 else 1.0
    if a < -700:
        return 0.0
    ea = math.exp(a)
    return ea / (1.0 + ea)


def tanh_saturate(value: float, *, scale: float = 1.0, gain: float = 1.0) -> float:
    """Насыщение через ``tanh``: ``gain · tanh(value / scale)``.

    Так работал EMA-вклад в ``direction.momentum_signal`` (``scale=0.003``, ``gain=1.5``)
    и асимметрия стен (``gain≈2.5``).
    """
    if not np.isfinite(value) or scale == 0:
        return 0.0
    return float(gain * math.tanh(value / scale))


def clip_z(value: float, lo: float = -3.0, hi: float = 3.0) -> float:
    """Ограничить z-вклад; нечисловое значение → 0.0 (как ``direction._clip_z``)."""
    if not np.isfinite(value):
        return 0.0
    return float(max(lo, min(hi, value)))


def z_combine(components: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Взвешенная сумма z-вкладов с отбрасыванием нечисловых (``NaN/Inf`` → пропуск)."""
    if weights is None:
        return float(sum(v for v in components if np.isfinite(v)))
    if len(weights) != len(components):
        raise ValueError("weights и components должны быть одной длины")
    total = 0.0
    for value, weight in zip(components, weights):
        if np.isfinite(value):
            total += float(value) * float(weight)
    return total


def safe_ratio(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    """Деление с защитой от нуля/нечисла (частый паттерн ``x / y if y > 0 else 1.0``)."""
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return default
    return float(numerator) / float(denominator)
