"""Квартальный конус волатильности: базис, полосы σ и коррекции (ring: domain).

Вынесено из ``gex/volatility_cone.py`` (итерация 41). Здесь расчёт конуса: дневная
волатильность, определение границ кварталов, гибридный источник и сам конус с полосами
σ1/σ2 и VWAP/RSI-коррекциями.

Кварталы, а не скользящее окно, потому что так устроена исходная логика (Pine): базис
пересчитывается по кварталам, а внутри квартала переносится. ``detect_quarters`` возвращает
маску границ, и её неверная разметка сдвинула бы весь конус — поэтому вход эталона имеет
дневной DatetimeIndex длиной больше года.

Числа конуса закреплены эталоном ``tests/test_volatility_cone_golden.py`` (34 колонки
полного кадра, плюс отдельные прогоны с не-дефолтными параметрами и выключенной коррекцией).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .rsi import wilder_rsi as _rsi_kernel


DEFAULT_LOOKBACK_DAYS = 500


DEFAULT_RSI_LENGTH = 14


DEFAULT_SD1_MULT = 1.0


DEFAULT_SD2_MULT = 2.0


DEFAULT_VWAP_INFLUENCE = 0.2


DEFAULT_RSI_INFLUENCE = 0.15


DEFAULT_EMA_LEN = 21


DEFAULT_BB_MULT = 2.0


DEFAULT_CARRY_WEIGHT = 0.62


DEFAULT_RSI_1SD_BOOST = 1.2


DEFAULT_USE_CORRECTION = True


DEFAULT_CORRECTION_PCT = 33.0


def compute_daily_volatility(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> np.ndarray:
    """SMA(True Range, lookback_days).

    Args:
        high, low, close: Price arrays.
        lookback_days: SMA window for daily volatility.

    Returns:
        Float64 array of same length — SMA of True Range.
    """
    n = len(close)
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ),
    )

    # SMA over lookback_days
    daily_vol = pd.Series(tr).rolling(window=lookback_days, min_periods=1).mean().values
    return daily_vol


def detect_quarters(index: pd.DatetimeIndex) -> np.ndarray:
    """Detect new quarter start bars.

    Квартал начинается в январе, апреле, июле или октябре,
    и месяц не равен предыдущему (чтобы не триггерить на каждом баре
    первого месяца).

    Args:
        index: DatetimeIndex from OHLCV DataFrame.

    Returns:
        Boolean array, True where a new quarter starts.
    """
    n = len(index)
    is_new = np.zeros(n, dtype=bool)
    if n == 0:
        return is_new

    months = index.month
    quarter_months = {1, 4, 7, 10}

    is_new[0] = months[0] in quarter_months

    for i in range(1, n):
        m = months[i]
        prev_m = months[i - 1]
        is_new[i] = (m in quarter_months) and (m != prev_m)

    return is_new


def compute_hybrid_source(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ha_open: np.ndarray,
    ha_close: np.ndarray,
) -> np.ndarray:
    """Compute composite source ``novelsrc`` from hybrid candles.

    Формула (Pine Script):
        medianTop = (max(stdO,stdC) + max(haO,haC)) / 2
        medianBottom = (min(stdO,stdC) + min(haO,haC)) / 2
        hybridOpen = (stdO + haO) / 2
        hybridClose = (stdC + haC) / 2
        candleTop = max(hybridOpen, hybridClose)
        candleBottom = min(hybridOpen, hybridClose)
        AvgCandle = avg(candleBottom, candleTop)
        sourceformas = open>close ? avg(open,low) : avg(close,high)
        novelsrc = avg(hlcc4, AvgCandle, sourceformas)

    Args:
        open_, high, low, close: Standard OHLC.
        ha_open, ha_close: Heikin Ashi OHLC.

    Returns:
        Float64 array ``novelsrc``.
    """
    hlcc4 = (high + low + close + close) / 4.0

    median_top = (np.maximum(open_, close) + np.maximum(ha_open, ha_close)) / 2.0
    median_bottom = (np.minimum(open_, close) + np.minimum(ha_open, ha_close)) / 2.0

    hybrid_open = (open_ + ha_open) / 2.0
    hybrid_close = (close + ha_close) / 2.0

    candle_top = np.maximum(hybrid_open, hybrid_close)
    candle_bottom = np.minimum(hybrid_open, hybrid_close)

    avg_candle = (candle_bottom + candle_top) / 2.0

    # sourceformas: ternary Open > Close
    condition = open_ > close
    sourceformas = np.where(
        condition,
        (open_ + low) / 2.0,
        (close + high) / 2.0,
    )

    novelsrc = (hlcc4 + avg_candle + sourceformas) / 3.0
    return novelsrc


def compute_volatility_cone(
    df: pd.DataFrame,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    rsi_length: int = DEFAULT_RSI_LENGTH,
    sd1_mult: float = DEFAULT_SD1_MULT,
    sd2_mult: float = DEFAULT_SD2_MULT,
    vwap_influence: float = DEFAULT_VWAP_INFLUENCE,
    rsi_influence: float = DEFAULT_RSI_INFLUENCE,
    ema_len: int = DEFAULT_EMA_LEN,
    bb_mult: float = DEFAULT_BB_MULT,
    carry_weight: float = DEFAULT_CARRY_WEIGHT,
    rsi_1sd_boost: float = DEFAULT_RSI_1SD_BOOST,
    use_correction: bool = DEFAULT_USE_CORRECTION,
    correction_pct: float = DEFAULT_CORRECTION_PCT,
) -> pd.DataFrame:
    """Quarterly Volatility Cone with VWAP, RSI, BB, Mean Reversion, Correction.

    Принимает DataFrame с колонками ``open, high, low, close, volume`` и
    DateTimeIndex, возвращает новый DataFrame со всеми добавленными колонками.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        lookback_days: История волатильности (дней) для SMA True Range.
        rsi_length: Период RSI.
        sd1_mult: Множитель 1 SD (внутренний конус).
        sd2_mult: Множитель 2 SD (внешний конус).
        vwap_influence: Влияние VWAP (0-1).
        rsi_influence: Влияние RSI (0-1).
        ema_len: Период EMA для QEMA21.
        bb_mult: Множитель отклонения для Bollinger Bands.
        carry_weight: Перенос QEMA между кварталами.
        rsi_1sd_boost: Усилитель RSI для 1 SD границ.
        use_correction: Включить канал коррекции 33%.
        correction_pct: Процент компенсации (0-100).

    Returns:
        Новый DataFrame с колонками:

        - ``is_new_quarter`` — bool, начало нового квартала
        - ``median_price`` — медиана цены с начала квартала
        - ``vwap`` — VWAP с начала квартала
        - ``qema21`` — Quarterly EMA 21
        - ``upper_1sd``, ``lower_1sd`` — границы 1 SD
        - ``upper_2sd``, ``lower_2sd`` — границы 2 SD
        - ``upper_1sd_mr``, ``lower_1sd_mr`` — Mean Reversion 1 SD
        - ``upper_2sd_mr``, ``lower_2sd_mr`` — Mean Reversion 2 SD
        - ``upper_2sd_corr``, ``lower_2sd_corr`` — Корректированные 2 SD
        - ``upper_2sd_mr_corr``, ``lower_2sd_mr_corr`` — Корректированные MR 2 SD
        - ``corr_deviation_pct`` — Процент отклонения текущего конуса от предыдущего
        - ``bb_upper``, ``bb_lower`` — Bollinger Bands (квартальные)
        - ``novelsrc`` — гибридный источник
        - ``daily_volatility`` — SMA(True Range)
        - ``current_rsi`` — RSI значение
    """
    df = df.copy()
    n = len(df)

    if n == 0:
        empty_cols = [
            'is_new_quarter', 'median_price', 'vwap', 'qema21',
            'upper_1sd', 'lower_1sd', 'upper_2sd', 'lower_2sd',
            'upper_1sd_mr', 'lower_1sd_mr', 'upper_2sd_mr', 'lower_2sd_mr',
            'upper_2sd_corr', 'lower_2sd_corr',
            'upper_2sd_mr_corr', 'lower_2sd_mr_corr',
            'corr_deviation_pct', 'bb_upper', 'bb_lower',
            'novelsrc', 'daily_volatility', 'current_rsi',
        ]
        for col in empty_cols:
            df[col] = np.nan
        return df

    # --- Extract numpy arrays for speed ---
    open_ = df['open'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    volume = df['volume'].fillna(0.0).values.astype(np.float64)
    index = df.index

    # --- BLOCK A: Heikin Ashi ---
    ha_open, ha_close = compute_heikin_ashi(open_, high, low, close)

    # --- BLOCK B: Daily Volatility ---
    daily_vol = compute_daily_volatility(high, low, close, lookback_days)

    # --- BLOCK C: RSI ---
    current_rsi = compute_rsi_wilder(close, rsi_length)

    # --- BLOCK D: Quarter detection ---
    is_new_quarter = detect_quarters(index)

    # --- BLOCK E: Hybrid source novelsrc ---
    novelsrc = compute_hybrid_source(open_, high, low, close, ha_open, ha_close)

    # --- Pre-allocate output arrays ---
    out_upper_1sd = np.full(n, np.nan)
    out_lower_1sd = np.full(n, np.nan)
    out_upper_2sd = np.full(n, np.nan)
    out_lower_2sd = np.full(n, np.nan)
    out_upper_1sd_mr = np.full(n, np.nan)
    out_lower_1sd_mr = np.full(n, np.nan)
    out_upper_2sd_mr = np.full(n, np.nan)
    out_lower_2sd_mr = np.full(n, np.nan)
    out_upper_2sd_corr = np.full(n, np.nan)
    out_lower_2sd_corr = np.full(n, np.nan)
    out_upper_2sd_mr_corr = np.full(n, np.nan)
    out_lower_2sd_mr_corr = np.full(n, np.nan)
    out_corr_dev_pct = np.full(n, np.nan)
    out_median = np.full(n, np.nan)
    out_vwap = np.full(n, np.nan)
    out_qema21 = np.full(n, np.nan)
    out_bb_upper = np.full(n, np.nan)
    out_bb_lower = np.full(n, np.nan)

    # --- Quarter state variables (соответствуют var в Pine Script) ---
    sp: Optional[float] = None       # start_price
    dp: int = 0                      # days_passed
    bv: float = 0.0                  # base_volatility
    cv: float = 0.0                  # cum_volume
    cpv: float = 0.0                 # cum_pv
    qh: float = 0.0                  # quarter_high
    ql: float = 0.0                  # quarter_low
    cc: float = 0.0                  # cum_close
    qema: Optional[float] = None     # QEMA21
    bb_cs: float = 0.0               # bb_cum_sum
    bb_csq: float = 0.0              # bb_cum_sq_sum

    # Previous quarter final values
    prev_u2: Optional[float] = None
    prev_l2: Optional[float] = None
    prev_u2mr: Optional[float] = None
    prev_l2mr: Optional[float] = None
    prev_qep: Optional[float] = None  # prev_quarter_end_price
    corr_dev: Optional[float] = None

    # EMA alpha
    alpha = 2.0 / (ema_len + 1.0)
    correction_factor = correction_pct / 100.0

    # --- Rolling stddev of novelsrc (для BB) ---
    # Предрасчитаем скользящее окно для novelsrc stddev
    novelsrc_series = pd.Series(novelsrc)
    novelsrc_std = novelsrc_series.rolling(window=ema_len, min_periods=1).std().values

    # --- MAIN BAR-BY-BAR LOOP ---
    for i in range(n):
        o = open_[i]
        h = high[i]
        l = low[i]
        c = close[i]
        v = volume[i]
        has_vol = v > 0
        iq = is_new_quarter[i]
        rsi_val = current_rsi[i]
        dv = daily_vol[i]
        bb_stdb = novelsrc_std[i] if not np.isnan(novelsrc_std[i]) else 0.0

        if iq:
            # ── SAVE previous quarter final values ──
            if sp is not None and dp > 0:
                prev_u2 = out_upper_2sd[i - 1] if not np.isnan(out_upper_2sd[i - 1]) else prev_u2
                prev_l2 = out_lower_2sd[i - 1] if not np.isnan(out_lower_2sd[i - 1]) else prev_l2
                prev_u2mr = out_upper_2sd_mr[i - 1] if not np.isnan(out_upper_2sd_mr[i - 1]) else prev_u2mr
                prev_l2mr = out_lower_2sd_mr[i - 1] if not np.isnan(out_lower_2sd_mr[i - 1]) else prev_l2mr
                if i > 0:
                    prev_qep = close[i - 1]
                else:
                    prev_qep = c

            # ── RESET for new quarter ──
            sp = o
            dp = 0
            bv = dv if not np.isnan(dv) else 0.0
            cv = v if has_vol else 0.0
            cpv = c * v if has_vol else 0.0
            qh = h
            ql = l
            cc = c
            # carry_weight init
            if qema is None or np.isnan(qema):
                qema = o
            else:
                qema = qema * carry_weight + o * (1.0 - carry_weight)
            bb_cs = c
            bb_csq = c * c
        else:
            # ── ACCUMULATE ──
            if sp is not None:
                dp += 1
                cv += v if has_vol else 0.0
                cpv += c * v if has_vol else 0.0
                qh = max(qh, h)
                ql = min(ql, l)
                cc += c
                bb_cs += c
                bb_csq += c * c

        # ── Update QEMA21 (exponential smoothing) ──
        if qema is not None and not np.isnan(qema):
            qema = alpha * c + (1.0 - alpha) * qema
        out_qema21[i] = qema

        # ── VWAP since quarter start ──
        vwap_val = cpv / cv if (cv != 0 and sp is not None) else np.nan
        out_vwap[i] = vwap_val

        # ── Median price ──
        if sp is not None and dp >= 0:
            med_price = cc / (dp + 1)
        else:
            med_price = np.nan
        out_median[i] = med_price

        # ── Bollinger Bands (quarterly cumulative) ──
        if sp is not None and dp >= 0:
            n_bars = float(dp + 1)
            mean_bb = bb_cs / n_bars
            variance = (bb_csq / n_bars) - (mean_bb * mean_bb)
            bb_std = np.sqrt(max(variance, 0.0)) if variance > 0 else 0.0
            # BB = QEMA ± avg(σ_close * mult, σ_novelsrc * mult)
            avg_dev = (bb_std * bb_mult + bb_stdb * bb_mult) / 2.0
            out_bb_upper[i] = qema + avg_dev if qema is not None else np.nan
            out_bb_lower[i] = qema - avg_dev if qema is not None else np.nan
        else:
            out_bb_upper[i] = np.nan
            out_bb_lower[i] = np.nan

        # ── BOUNDARY CALCULATION ──
        if sp is not None and dp >= 0:
            tf = np.sqrt(float(dp))
            w1 = bv * sd1_mult * tf
            w2 = bv * sd2_mult * tf

            # VWAP offset
            vwap_off = 0.0
            if vwap_val is not None and not np.isnan(vwap_val):
                vwap_off = (vwap_val - sp) * vwap_influence * (1.0 - np.exp(-dp / 30.0))

            # RSI factor
            rsi_f = 0.0
            if not np.isnan(rsi_val):
                rsi_f = (rsi_val - 50.0) / 50.0

            # RSI momentum adjustment
            rsi_mom = w2 * rsi_f * rsi_influence * np.sqrt(max(tf / 10.0, 0.0))

            # Standard boundaries
            u1 = sp + w1 + vwap_off + rsi_mom
            l1 = sp - w1 + vwap_off + rsi_mom
            u2 = sp + w2 + vwap_off + rsi_mom * 1.5
            l2 = sp - w2 + vwap_off - rsi_mom * 1.5

            out_upper_1sd[i] = u1
            out_lower_1sd[i] = l1
            out_upper_2sd[i] = u2
            out_lower_2sd[i] = l2

            # Mean Reversion boundaries
            rsi_ob = max(0.0, (rsi_val - 50.0) / 50.0) if not np.isnan(rsi_val) else 0.0
            rsi_os = max(0.0, (50.0 - rsi_val) / 50.0) if not np.isnan(rsi_val) else 0.0

            compress_2sd = w2 * rsi_influence * np.sqrt(max(tf / 10.0, 0.0))
            u2_mr = sp + w2 + vwap_off - compress_2sd * rsi_ob * 1.5
            l2_mr = sp - w2 + vwap_off + compress_2sd * rsi_os * 1.5

            compress_1sd = w1 * rsi_influence * rsi_1sd_boost * np.sqrt(max(tf / 10.0, 0.0))
            u1_mr = sp + w1 + vwap_off - compress_1sd * rsi_ob
            l1_mr = sp - w1 + vwap_off + compress_1sd * rsi_os

            out_upper_2sd_mr[i] = u2_mr
            out_lower_2sd_mr[i] = l2_mr
            out_upper_1sd_mr[i] = u1_mr
            out_lower_1sd_mr[i] = l1_mr

            # Correction channel
            u2c, l2c, u2mrc, l2mrc = u2, l2, u2_mr, l2_mr
            if use_correction and prev_u2 is not None and prev_u2 != 0.0 and dp > 0:
                dev_upper = (u2 - prev_u2) / abs(prev_u2)
                u2c = u2 - (u2 - prev_u2) * correction_factor
                corr_dev = dev_upper * 100.0

                if prev_l2 is not None and prev_l2 != 0.0:
                    l2c = l2 - (l2 - prev_l2) * correction_factor
                if prev_u2mr is not None and prev_u2mr != 0.0:
                    u2mrc = u2_mr - (u2_mr - prev_u2mr) * correction_factor
                if prev_l2mr is not None and prev_l2mr != 0.0:
                    l2mrc = l2_mr - (l2_mr - prev_l2mr) * correction_factor
            else:
                corr_dev = np.nan

            out_upper_2sd_corr[i] = u2c
            out_lower_2sd_corr[i] = l2c
            out_upper_2sd_mr_corr[i] = u2mrc
            out_lower_2sd_mr_corr[i] = l2mrc
            out_corr_dev_pct[i] = corr_dev
        else:
            # No quarter started yet — leave as NaN
            pass

    # --- Assemble result DataFrame ---
    df['is_new_quarter'] = is_new_quarter
    df['median_price'] = out_median
    df['vwap'] = out_vwap
    df['qema21'] = out_qema21
    df['upper_1sd'] = out_upper_1sd
    df['lower_1sd'] = out_lower_1sd
    df['upper_2sd'] = out_upper_2sd
    df['lower_2sd'] = out_lower_2sd
    df['upper_1sd_mr'] = out_upper_1sd_mr
    df['lower_1sd_mr'] = out_lower_1sd_mr
    df['upper_2sd_mr'] = out_upper_2sd_mr
    df['lower_2sd_mr'] = out_lower_2sd_mr
    df['upper_2sd_corr'] = out_upper_2sd_corr
    df['lower_2sd_corr'] = out_lower_2sd_corr
    df['upper_2sd_mr_corr'] = out_upper_2sd_mr_corr
    df['lower_2sd_mr_corr'] = out_lower_2sd_mr_corr
    df['corr_deviation_pct'] = out_corr_dev_pct
    df['bb_upper'] = out_bb_upper
    df['bb_lower'] = out_bb_lower
    df['novelsrc'] = novelsrc
    df['daily_volatility'] = daily_vol
    df['current_rsi'] = current_rsi

    return df


def compute_rsi_wilder(close, period: int = 14):
    """RSI по Уайлдеру — каноническое ядро домена (``warmup="nan"``, ``flat="hundred"``).

    Здесь была своя реализация RSI. Политики этого модуля отличаются от ``ta``: разогрев
    даёт ``NaN`` (а не 50), а плоский ряд — 100 (а не 50). Ядро принимает политики
    параметрами, и паритет с прежней реализацией измерен: расхождение 0.0 на всей длине,
    число ``NaN`` совпадает. Поэтому делегирование — шаг без изменения чисел.

    Возвращается numpy-массив: прежняя функция возвращала массив, и вызывающий код
    (``compute_hybrid_source``) работает с ним позиционно.
    """
    import numpy as _np

    return _rsi_kernel(_np.asarray(close, dtype=float), period, warmup="nan", flat="hundred")


def compute_heikin_ashi(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Heikin Ashi candles.

    Args:
        open_: Standard open prices.
        high: Standard high prices.
        low: Standard low prices.
        close: Standard close prices.

    Returns:
        (ha_open, ha_close) — both float64 ndarrays of same length as input.
    """
    n = len(open_)
    ha_close = np.empty(n, dtype=np.float64)
    ha_open = np.empty(n, dtype=np.float64)

    ha_close[0] = (open_[0] + high[0] + low[0] + close[0]) / 4.0
    ha_open[0] = open_[0]

    for i in range(1, n):
        ha_close[i] = (open_[i] + high[i] + low[i] + close[i]) / 4.0
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    return ha_open, ha_close