"""Масштаб GEX — канон (ring: domain, только numpy).

Критическое расхождение чисел
-----------------------------
В проекте GEX считался **двумя разными масштабами**, и они совпадают только при цене 100:

| Место | Формула на контракт | Комментарий |
|---|---|---|
| ``metrics.GEXMetrics.compute`` (``:194-197``) | ``sign · Γ · per_contract · S² · pct_move`` | «долларовая гамма на 1% движения» — индустриальный стандарт |
| ``gexcone._strike_gex`` (``:588-590``) | ``sign · Γ · per_contract · S² · 0.01`` | то же самое (``pct_move``=0.01 зашит) |
| ``extended._build_per_strike`` (``:677``) | ``sign · Γ · oi · per_contract · 100`` | «per_contract × dollar-scale=100» — **без S²**, другой масштаб |

Отношение ``extended`` к ``metrics`` равно ``S² · pct_move / 100``: при ``S=100`` это ровно 1
(отсюда впечатление «одинаково»), при ``S=600`` — **36×**, при ``S=10`` — **0.01×**. То есть
``/ext/gex`` и ``/gex`` показывают несопоставимые числа для одного тикера.

Канон сохраняет **оба** масштаба явным параметром ``method`` и даёт функцию, которая считает их
отношение — чтобы расхождение было видно в тесте, а не обнаруживалось пользователем.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["METHODS", "DEFAULT_PCT_MOVE", "per_contract_gex", "gex_values", "scale_ratio"]

METHODS = ("dollar_gamma_1pct", "contract_scale_100")
DEFAULT_PCT_MOVE = 0.01


def per_contract_gex(
    gamma: Sequence[float] | np.ndarray,
    spot: float,
    per_contract: int = 100,
    *,
    method: str = "dollar_gamma_1pct",
    pct_move: float = DEFAULT_PCT_MOVE,
) -> np.ndarray:
    """GEX на один контракт (без знака дилера и без OI).

    * ``dollar_gamma_1pct`` — ``Γ · per_contract · S² · pct_move`` (``metrics``/``gexcone``);
    * ``contract_scale_100`` — ``Γ · per_contract · 100`` (``extended``, ТЗ 1.2).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    g = np.asarray(gamma, dtype=float)
    if method == "dollar_gamma_1pct":
        return g * float(per_contract) * (float(spot) ** 2) * float(pct_move)
    return g * float(per_contract) * 100.0


def gex_values(
    gamma: Sequence[float] | np.ndarray,
    oi: Sequence[float] | np.ndarray,
    sign: Sequence[float] | np.ndarray,
    spot: float,
    per_contract: int = 100,
    *,
    method: str = "dollar_gamma_1pct",
    pct_move: float = DEFAULT_PCT_MOVE,
) -> np.ndarray:
    """Полный GEX опциона: ``sign · per_contract_gex(gamma, S) · OI``."""
    return (
        np.asarray(sign, dtype=float)
        * per_contract_gex(gamma, spot, per_contract, method=method, pct_move=pct_move)
        * np.asarray(oi, dtype=float)
    )


def scale_ratio(spot: float, *, pct_move: float = DEFAULT_PCT_MOVE) -> float:
    """Во сколько раз ``dollar_gamma_1pct`` больше ``contract_scale_100`` при данной цене.

    ``ratio = S² · pct_move / 100``. Значение 1.0 только при ``S = 100`` (или ``pct_move=0.01`` и
    ``S=100``) — именно поэтому расхождение долго не замечали.
    """
    return float(spot) ** 2 * float(pct_move) / 100.0
