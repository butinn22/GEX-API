"""VWAP Price Channel (VPC): канал по экстремумам и VWAP (ring: domain).

Вынесено из ``gex/volatility_cone.py`` (итерация 41). Канал строится по highest/lowest
за окно и VWAP на экстремальных барах, с направлением (``vpc_dir``/``vpc_dir2``).

Отдельным модулем, потому что это независимый расчёт: ``compute_all`` просто складывает
конус и канал, и раньше оба жили в одном файле только потому, что так исторически вышло.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_VPC_LENGTH = 20


def compute_vpc(
    df: pd.DataFrame,
    length: int = DEFAULT_VPC_LENGTH,
) -> pd.DataFrame:
    """VWAP Price Channel (порт get_vpc из Pine Script).

    Канал строится на основе скользящих highest/lowest за ``length`` баров
    и VWAP на экстремальных барах.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        length: Window for highest/lowest.

    Returns:
        Новый DataFrame с колонками:
        - ``vpc_upper`` — верхняя граница канала
        - ``vpc_lower`` — нижняя граница канала
        - ``vpc_mid`` — средняя линия
        - ``vpc_hst`` — highest за период
        - ``vpc_lst`` — lowest за период
        - ``vpc_dir`` — направление (1=up, -1=down, 0=neutral)
        - ``vpc_dir2`` — персистентное направление
    """
    df = df.copy()
    n = len(df)
    if n == 0:
        for col in ['vpc_upper', 'vpc_lower', 'vpc_mid', 'vpc_hst', 'vpc_lst']:
            df[col] = np.nan
        df['vpc_dir'] = 0
        df['vpc_dir2'] = 0
        return df

    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    volume = df['volume'].fillna(0.0).values.astype(np.float64)

    # Pre-allocate
    out_upper = np.full(n, np.nan)
    out_lower = np.full(n, np.nan)
    out_mid = np.full(n, np.nan)
    out_hst = np.full(n, np.nan)
    out_lst = np.full(n, np.nan)
    out_dir = np.zeros(n, dtype=np.int32)
    out_dir2 = np.zeros(n, dtype=np.int32)

    # VWAP accumulators for high/low extremes
    h_vwap: Optional[float] = None
    l_vwap: Optional[float] = None
    h_vwap_prev: Optional[float] = None
    l_vwap_prev: Optional[float] = None

    prev_upper: Optional[float] = None
    prev_lower: Optional[float] = None
    prev_hst: Optional[float] = None
    prev_lst: Optional[float] = None

    dir2_persist: int = 0

    for i in range(n):
        h = high[i]
        l = low[i]
        c = close[i]

        # Rolling highest/lowest
        start = max(0, i - length + 1)
        hst = np.max(high[start:i + 1])
        lst = np.min(low[start:i + 1])

        new_high = (h == hst)
        new_low = (l == lst)

        # VWAP on extreme bars (cumulative, reset on new extreme)
        if h_vwap is None:
            h_vwap = h
            l_vwap = l

        # Simplified VWAP for extremes:
        # In Pine Script, ta.vwap(high, new_high) starts accumulating from first bar
        # where new_high is true. We approximate with cumulative VWAP resets.
        # Full emulation would require tracking separate PV/V accumulators.
        if i == 0:
            h_vwap = h
            l_vwap = l
        else:
            if new_high:
                h_vwap = h  # reset VWAP high
            if new_low:
                l_vwap = l  # reset VWAP low

        # Change
        h_change = 0.0 if h_vwap_prev is None else h_vwap - h_vwap_prev
        l_change = 0.0 if l_vwap_prev is None else l_vwap - l_vwap_prev

        # Upper bound
        if new_high:
            upper = hst
        elif prev_hst is not None and hst == prev_hst:
            upper = (prev_upper or 0.0) + h_change
        else:
            upper = min(hst, (prev_upper or hst) + h_change)

        # Lower bound
        if new_low:
            lower = lst
        elif prev_lst is not None and lst == prev_lst:
            lower = (prev_lower or 0.0) + l_change
        else:
            lower = max(lst, (prev_lower or lst) + l_change)

        mid = (upper + lower) / 2.0

        # Trend direction
        if new_high:
            direction = 1
        elif new_low:
            direction = -1
        else:
            direction = 0

        if new_high:
            dir2_persist = 1
        elif new_low:
            dir2_persist = -1
        # else: keep previous dir2

        out_upper[i] = upper
        out_lower[i] = lower
        out_mid[i] = mid
        out_hst[i] = hst
        out_lst[i] = lst
        out_dir[i] = direction
        out_dir2[i] = dir2_persist

        # Save previous for next bar
        h_vwap_prev = h_vwap
        l_vwap_prev = l_vwap
        prev_upper = upper
        prev_lower = lower
        prev_hst = hst
        prev_lst = lst

    df['vpc_upper'] = out_upper
    df['vpc_lower'] = out_lower
    df['vpc_mid'] = out_mid
    df['vpc_hst'] = out_hst
    df['vpc_lst'] = out_lst
    df['vpc_dir'] = out_dir
    df['vpc_dir2'] = out_dir2

    return df
