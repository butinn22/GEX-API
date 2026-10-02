"""Ключевые уровни GEX: gamma flip (zero-gamma) и стены — канон (ring: domain, только numpy).

Что дублировалось
-----------------
1. **Gamma flip / zero-gamma** — один и тот же алгоритм (кумулятив по страйкам → смена знака →
   линейная интерполяция) жил дважды: ``metrics._find_gamma_flip_cumulative`` (``:432-463``) и
   ``extended._zero_gamma_level`` (``:739-764``). Различие — только защитный ``if i == 0`` у
   ``extended`` (недостижим: ``diff()`` первой строки всегда 0 → индекс 0 не бывает сменой знака).
   Оставляем guard как документированную защиту.

2. **Стены** — четыре разных отбора: ``metrics._walls_by_gex`` (макс. положительный / мин.
   отрицательный net GEX), ``metrics._walls_by_oi`` (макс. OI по коллам/путам),
   ``metrics.ranked_walls`` (топ-N по ``|GEX|``), ``extended._wall`` (то же + сила ``|net|/Σ|net|``),
   плюс отбор в ``gexcone``. Канон ниже повторяет все четыре **точно** и добавляет силу к ранжированию.

Соглашение по знаку: положительный net GEX → call-стена (сопротивление сверху), отрицательный →
put-стена (поддержка снизу), как в оригинале.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "gamma_flip_cumulative",
    "primary_walls_by_gex",
    "walls_by_oi",
    "ranked_walls",
    "wall_with_strength",
]


def _sorted_by_strike(
    strikes: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(strikes, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(s) != len(v):
        raise ValueError("strikes и значения должны быть одной длины")
    order = np.argsort(s, kind="stable")
    return s[order], v[order]


def gamma_flip_cumulative(
    strikes: Sequence[float] | np.ndarray,
    gex_net: Sequence[float] | np.ndarray,
) -> float | None:
    """Уровень, где кумулятивный net GEX переходит через ноль (он же «zero gamma level»).

    Точный перенос ``metrics._find_gamma_flip_cumulative`` / ``extended._zero_gamma_level``:
    сортировка по страйку, кумулятивная сумма, первая смена знака, линейная интерполяция между
    соседними страйками. ``None``, если перехода нет (рынок глубоко в одном режиме).
    """
    s, v = _sorted_by_strike(strikes, gex_net)
    if s.size == 0:
        return None
    cum = np.cumsum(v)
    signs = np.sign(cum)
    changes = np.flatnonzero(np.diff(signs) != 0)
    if changes.size == 0:
        return None
    i = int(changes[0]) + 1  # индекс первой строки нового знака
    if i == 0:  # недостижимо (diff первой строки = 0), защита как в extended
        return float(s[0])
    s0, s1 = float(s[i - 1]), float(s[i])
    g0, g1 = float(cum[i - 1]), float(cum[i])
    if np.isclose(g1 - g0, 0.0):
        return float(0.5 * (s0 + s1))
    return float(s0 - g0 * (s1 - s0) / (g1 - g0))


def primary_walls_by_gex(
    strikes: Sequence[float] | np.ndarray,
    gex_net: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Call/Put стены по экстремумам net GEX (``metrics._walls_by_gex``).

    ``(call_wall, put_wall)``; ``nan``, если на соответствующей стороне нет страйков.
    """
    s = np.asarray(strikes, dtype=float)
    v = np.asarray(gex_net, dtype=float)
    if s.size == 0:
        return float("nan"), float("nan")
    pos = np.flatnonzero(v > 0)
    neg = np.flatnonzero(v < 0)
    call_wall = float(s[pos[np.argmax(v[pos])]]) if pos.size else float("nan")
    put_wall = float(s[neg[np.argmin(v[neg])]]) if neg.size else float("nan")
    return call_wall, put_wall


def walls_by_oi(
    strikes: Sequence[float] | np.ndarray,
    oi_call: Sequence[float] | np.ndarray,
    oi_put: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Стены по открытому интересу (``metrics._walls_by_oi``): max OI по коллам / по путам."""
    s = np.asarray(strikes, dtype=float)
    call = np.asarray(oi_call, dtype=float)
    put = np.asarray(oi_put, dtype=float)
    if s.size == 0:
        return float("nan"), float("nan")
    return float(s[int(np.argmax(call))]), float(s[int(np.argmax(put))])


def _wall_rows(
    strikes: Sequence[float] | np.ndarray,
    gex_net: Sequence[float] | np.ndarray,
    *,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Отобрать страйки нужной стороны, отсортированные по ``|GEX|`` по убыванию."""
    s = np.asarray(strikes, dtype=float)
    v = np.asarray(gex_net, dtype=float)
    if side not in ("call", "put"):
        raise ValueError("side must be 'call' or 'put'")
    mask = v > 0 if side == "call" else v < 0
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.empty(0), np.empty(0)
    order = np.argsort(-np.abs(v[idx]), kind="stable")
    return s[idx][order], v[idx][order]


def ranked_walls(
    strikes: Sequence[float] | np.ndarray,
    gex_net: Sequence[float] | np.ndarray,
    *,
    top_n: int = 3,
) -> tuple[list[float], list[float]]:
    """Топ-N стен по ``|GEX|`` с каждой стороны (``metrics.ranked_walls``)."""
    call_s, _ = _wall_rows(strikes, gex_net, side="call")
    put_s, _ = _wall_rows(strikes, gex_net, side="put")
    return call_s[:top_n].tolist(), put_s[:top_n].tolist()


def wall_with_strength(
    strikes: Sequence[float] | np.ndarray,
    gex_net: Sequence[float] | np.ndarray,
    *,
    side: str,
) -> tuple[float, float]:
    """Главная стена и её сила (``extended._wall``): ``(strike, |net|/Σ|net|)``.

    ``(nan, 0.0)``, если на стороне нет страйков; сила 0.0, если суммарный ``|GEX|`` нулевой.
    """
    s_all = np.asarray(strikes, dtype=float)
    v_all = np.asarray(gex_net, dtype=float)
    s, v = _wall_rows(strikes, gex_net, side=side)
    if s.size == 0:
        return float("nan"), 0.0
    total_abs = float(np.abs(v_all).sum()) if s_all.size else 0.0
    strength = abs(float(v[0])) / total_abs if total_abs > 0 else 0.0
    return float(s[0]), strength
