"""Моментум — канон (ring: domain, только numpy).

Пять разных «моментумов» в проекте
-----------------------------------
| Место | Определение | Шкала |
|---|---|---|
| ``ta.compute_momentum_strength`` (`:456-577`) | форма свечей: close-to-close %, расширение диапазона, тело, подтверждение объёмом → сигмоидная сила | 0..100 + сторона |
| ``direction.momentum_signal`` (`:278-325`) | EMA5(close) против EMA10(нейтрализованной цены), ``tanh(rel_gap/0.003)·1.5`` **плюс бустер** от ``ta``-версии (вес 0.5) | z ∈ [-2, +2] |
| ``hybrid_trend`` (`:544-554`) | ``(novelsrc − EMA) / (ATR·mult)`` по бару | [-1, +1] |
| ``novel_candles`` (two-pole) | иной фильтровый сигнал (отдельный продукт) | — |
| ``direction.py:316/535`` | **тот же** ``ta``-моментум как бустер — то есть на одной странице вход учитывается дважды | — |

Канон повторяет первые три **точно** (каждая — своим именем и со своей шкалой) и делает бустер
``direction``-варианта **явным параметром** ``booster_weight`` (0.0 = выключить), чтобы двойной учёт
был видимым решением, а не следствием кода.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..indicators._kernels import as_float_array
from .numeric import clip_z, sigmoid, tanh_saturate

__all__ = [
    "MomentumReading",
    "momentum_strength",
    "atr_distance_momentum",
    "neutralizing_price",
    "ema_gap_z",
    "direction_momentum_signal",
]

# Константы ровно те, что в оригиналах (менять нельзя без решения владельца)
_C2C_EPS = 1e-9
_VOL_BASE = 1.3              # объёмное подтверждение, когда все бары в сторону тренда
_VOL_GAIN = 2.5              # крутизна сигмоиды по volume_trend
_STRENGTH_GAIN = 8.0         # ta: strength = 100·sigmoid(8·(raw−0.15))
_STRENGTH_CENTER = 0.15
_EMA_GAP_SCALE = 0.003       # direction: tanh(rel_gap / 0.003)
_EMA_GAP_GAIN = 1.5
_DEFAULT_BOOSTER_WEIGHT = 0.5


@dataclass(frozen=True)
class MomentumReading:
    """Результат ``ta.compute_momentum_strength`` (поля сохранены один-в-один)."""

    side: str
    strength: float
    close_to_close_pct: float
    range_pct: float
    avg_body_pct: float
    volume_trend: float
    net_move_pct: float
    n_bars: int


def momentum_strength(
    open_: Sequence[float] | np.ndarray,
    high: Sequence[float] | np.ndarray,
    low: Sequence[float] | np.ndarray,
    close: Sequence[float] | np.ndarray,
    volume: Sequence[float] | np.ndarray | None = None,
    *,
    lookback: int = 20,
) -> MomentumReading | None:
    """Сила тренда 0..100 и сторона — точный перенос ``ta.compute_momentum_strength``.

    Вход — уже обрезанное окно (в оригинале ``df.tail(lookback)``). ``None``, если меньше 2 баров.
    """
    o = as_float_array(open_)
    h = as_float_array(high)
    low_arr = as_float_array(low)
    c = as_float_array(close)
    n = len(c)
    if n < 2:
        return None
    if len(o) != n or len(h) != n or len(low_arr) != n:
        raise ValueError("OHLC должны быть одной длины")

    vol = as_float_array(volume) if volume is not None else None
    if vol is not None and len(vol) != n:
        vol = None

    # 1. close-to-close %
    c2c = np.diff(c) / c[:-1] * 100.0
    avg_c2c = float(np.mean(c2c)) if n > 1 else 0.0

    # 2. range expansion: (High_i − Low_{i-1}) / Low_{i-1}
    rng = np.zeros(max(n - 1, 0), dtype=float)
    prev_low = low_arr[:-1]
    cur_high = h[1:]
    nz = prev_low > 0
    rng[nz] = (cur_high[nz] - prev_low[nz]) / prev_low[nz] * 100.0
    avg_range = float(np.mean(rng)) if len(rng) else 0.0

    # 3. body size |Close−Open|/Open
    body = np.abs(c - o) / np.where(o > 0, o, np.nan) * 100.0
    body = body[np.isfinite(body)]
    avg_body = float(np.mean(body)) if body.size else 0.0

    # сторона движения
    if avg_c2c > _C2C_EPS:
        side = "BULLISH"
    elif avg_c2c < -_C2C_EPS:
        side = "BEARISH"
    else:
        side = "NEUTRAL"

    # 4. подтверждение объёмом
    volume_trend = 1.0
    if vol is not None and side != "NEUTRAL" and n > 1:
        bar_dir = np.sign(np.diff(c))
        if side == "BULLISH":
            trend_mask = np.append(bar_dir > 0, c[-1] >= o[-1])
        else:
            trend_mask = np.append(bar_dir < 0, c[-1] < o[-1])
        trend_mask = trend_mask.astype(bool)
        if trend_mask.any() and (~trend_mask).any():
            avg_vol_trend = float(np.mean(vol[trend_mask]))
            avg_vol_other = float(np.mean(vol[~trend_mask]))
            volume_trend = avg_vol_trend / avg_vol_other if avg_vol_other > 0 else 1.0
        elif trend_mask.all():
            volume_trend = _VOL_BASE

    # 5. итоговая сила
    body_factor = float(np.tanh(avg_body / 1.0))
    vol_factor = sigmoid(volume_trend, gain=_VOL_GAIN, center=1.0)
    raw = abs(avg_c2c) * vol_factor * (1.0 + body_factor)
    strength = 100.0 * sigmoid(raw, gain=_STRENGTH_GAIN, center=_STRENGTH_CENTER)

    return MomentumReading(
        side=side,
        strength=float(np.clip(strength, 0.0, 100.0)),
        close_to_close_pct=avg_c2c,
        range_pct=avg_range,
        avg_body_pct=avg_body,
        volume_trend=float(volume_trend),
        net_move_pct=float((c[-1] - c[0]) / c[0] * 100.0) if c[0] > 0 else 0.0,
        n_bars=int(n),
    )


def atr_distance_momentum(
    series: Sequence[float] | np.ndarray,
    ema_series: Sequence[float] | np.ndarray,
    atr_series: Sequence[float] | np.ndarray,
    *,
    atr_mult: float = 1.0,
) -> np.ndarray:
    """Поштучный моментум ``hybrid_trend`` (``:544-554``): ``clip((src − EMA)/(ATR·mult), −1, 1)``.

    Где знаменатель или вход нечисловые — 0.0 (как в оригинале).
    """
    src = as_float_array(series)
    ema = as_float_array(ema_series)
    atr = as_float_array(atr_series)
    n = min(len(src), len(ema), len(atr))
    out = np.zeros(len(src), dtype=float)
    for t in range(n):
        if atr[t] > 0.0 and atr_mult > 0.0 and np.isfinite(src[t]) and np.isfinite(ema[t]):
            out[t] = float(np.clip((src[t] - ema[t]) / (atr[t] * atr_mult), -1.0, 1.0))
    return out


def neutralizing_price(
    open_: Sequence[float] | np.ndarray,
    high: Sequence[float] | np.ndarray,
    low: Sequence[float] | np.ndarray,
    close: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Нейтрализующая цена ``direction.neutralizing_price_series`` (``:255-275``).

    ``(Open+Low)/2`` для медвежьей свечи, ``(Close+High)/2`` — иначе (гасит пин-бары и проколы тенью).
    """
    o = as_float_array(open_)
    h = as_float_array(high)
    l = as_float_array(low)
    c = as_float_array(close)
    bearish = o > c
    return np.where(bearish, (o + l) / 2.0, (c + h) / 2.0)


def ema_gap_z(
    ema_fast_last: float,
    ema_slow_last: float,
    *,
    scale: float = _EMA_GAP_SCALE,
    gain: float = _EMA_GAP_GAIN,
) -> float:
    """EMA-вклад ``direction.momentum_signal`` (``:303-311``) без бустера."""
    if not np.isfinite(ema_fast_last) or not np.isfinite(ema_slow_last) or ema_slow_last <= 0:
        return 0.0
    rel_gap = float((ema_fast_last - ema_slow_last) / ema_slow_last)
    return tanh_saturate(rel_gap, scale=scale, gain=gain)


def direction_momentum_signal(
    ema_fast_last: float,
    ema_slow_last: float,
    *,
    ta_strength: float | None = None,
    ta_side: str | None = None,
    booster_weight: float = _DEFAULT_BOOSTER_WEIGHT,
) -> float:
    """Полный z-вклад фактора моментума (``direction.momentum_signal``, ``:278-325``).

    ``booster_weight`` — вес объёмно-ценового бустера от ``ta.compute_momentum_strength``
    (в оригинале жёстко 0.5). Передайте ``0.0``, чтобы **убрать двойной учёт** входа:
    без бустера фактор считается только по EMA-перекрытию.
    """
    ema_z = ema_gap_z(ema_fast_last, ema_slow_last)
    booster = 0.0
    if booster_weight and ta_strength is not None and ta_side in ("BULLISH", "BEARISH"):
        sign = 1.0 if ta_side == "BULLISH" else -1.0
        booster = sign * (float(ta_strength) / 100.0) * float(booster_weight)
    return clip_z(ema_z + booster, lo=-2.0, hi=2.0)
