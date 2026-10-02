"""features: Построение колонок кадра: HA-свечи, тренд, VWAP, ADL-цепочка, BB, две колонки тренда.

Вынесено из ``gex/trading_algorithm.py`` (итерация 37). Методы перенесены дословно: разбиение god-класса не должно менять числа, а доказательство — golden-эталон
``tests/test_strategy_golden.py``, сверяющий все колонки кадра, решения ``evaluate``,
режим, оценку входа и риск до и после выноса.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame

logger = logging.getLogger(__name__)
from .indicators import _bb, _cumsum_reset, _two_pole_filter


class _FeaturesMixin:
    @staticmethod
    def _heikin_ashi(data: DataFrame, pd_module: Any) -> DataFrame:
        ha_close = (data["open"] + data["high"] + data["low"] + data["close"]) / 4
        ha_open = pd_module.Series(index=data.index, dtype=float)
        ha_open.iloc[0] = (data["open"].iloc[0] + data["close"].iloc[0]) / 2
        ha_open.iloc[1:] = (ha_open.shift(1).iloc[1:] + ha_close.shift(1).iloc[1:]) / 2
        # Pine: ha_high = max(standard_high, ha_open, ha_close), ha_low = min(standard_low, ha_open, ha_close)
        ha_high_vals = pd.concat([data["high"], ha_open, ha_close], axis=1).max(axis=1)
        ha_low_vals = pd.concat([data["low"], ha_open, ha_close], axis=1).min(axis=1)
        return pd_module.DataFrame({
            "ha_open": ha_open,
            "ha_high": ha_high_vals,
            "ha_low": ha_low_vals,
            "ha_close": ha_close,
        }, index=data.index)


    @staticmethod
    def _add_hybrid_candles(f: DataFrame, pd_module: Any) -> DataFrame:
        std_o, std_c = f["open"], f["close"]
        ha_o, ha_c = f["ha_open"], f["ha_close"]

        f["median_top"] = (pd.concat([std_o, std_c], axis=1).max(axis=1)
                           + pd.concat([ha_o, ha_c], axis=1).max(axis=1)) / 2
        f["median_bottom"] = (pd.concat([std_o, std_c], axis=1).min(axis=1)
                              + pd.concat([ha_o, ha_c], axis=1).min(axis=1)) / 2
        f["hybrid_open"] = (std_o + ha_o) / 2
        f["hybrid_close"] = (std_c + ha_c) / 2
        f["candle_top"] = pd.concat([f["hybrid_open"], f["hybrid_close"]], axis=1).max(axis=1)
        f["candle_bottom"] = pd.concat([f["hybrid_open"], f["hybrid_close"]], axis=1).min(axis=1)
        f["avg_candle"] = (f["candle_bottom"] + f["candle_top"]) / 2

        bearish = (std_o > std_c)
        f["sourceformas"] = np.where(bearish, (std_o + f["low"]) / 2, (std_c + f["high"]) / 2)
        f["hlcc4"] = (f["high"] + f["low"] + 2 * f["close"]) / 4
        avg_candle = f["avg_candle"]
        sourceformas = f["sourceformas"]
        hlcc4 = f["hlcc4"]
        f["novelsrc"] = (hlcc4 + avg_candle + sourceformas) / 3
        return f


    def _add_trend_analysis(self, f: DataFrame, pd_module: Any) -> DataFrame:
        length = self.settings.length_bars
        threshold = self.settings.movement_threshold / 100.0

        g_w_std, r_w_std, _, _ = self._bar_trend_analysis(
            f["open"], f["close"], length, threshold, pd_module)
        g_w_hyb, r_w_hyb, _, _ = self._bar_trend_analysis(
            f["hybrid_open"], f["hybrid_close"], length, threshold, pd_module)

        total_w = g_w_std + r_w_std + g_w_hyb + r_w_hyb
        f["trend_coefficient"] = np.where(
            total_w > 0,
            ((g_w_std + g_w_hyb) - (r_w_std + r_w_hyb)) / total_w,
            0.0,
        )
        f["trend_direction"] = np.sign(f["trend_coefficient"])

        # Flat zone
        change = (f["close"] - f["close"].shift(20)).abs() / f["close"].shift(20) * 100
        avg_range = (f["high"] - f["low"]).rolling(20, min_periods=1).mean() / f["close"] * 100
        f["is_flat_zone"] = (change < self.settings.flat_threshold_d) & (
            avg_range < self.settings.flat_threshold_d)

        bt = self.settings.bull_trend_threshold
        brt = self.settings.bear_trend_threshold
        if self.settings.flat_filter_enabled:
            f["can_enter_long"] = np.where(
                f["is_flat_zone"],
                f["trend_coefficient"] > bt * 1.5,
                f["trend_coefficient"] > bt,
            )
            f["can_enter_short"] = np.where(
                f["is_flat_zone"],
                f["trend_coefficient"] < brt * 1.5,
                f["trend_coefficient"] < brt,
            )
        else:
            f["can_enter_long"] = f["trend_coefficient"] > bt
            f["can_enter_short"] = f["trend_coefficient"] < brt
        return f


    @staticmethod
    def _bar_trend_analysis(
        price_open: DataFrame, price_close: DataFrame, length: int,
        threshold: float, pd_module: Any,
    ) -> tuple:
        """Векторизованный порт Pine ``barTrendAnalysis()``.

        Использует weighted sum: ``weight = length - offset`` для каждого бара в окне.
        """
        movement = (price_close - price_open) / price_open
        abs_move = movement.abs()
        green_raw = abs_move.where(movement > threshold, 0.0)
        red_raw = abs_move.where(movement < -threshold, 0.0)
        green_bar_flag = (movement > threshold).astype(int)
        red_bar_flag = (movement < -threshold).astype(int)

        g_w = pd.Series(0.0, index=price_open.index)
        r_w = pd.Series(0.0, index=price_open.index)
        g_b = pd.Series(0, index=price_open.index, dtype=int)
        r_b = pd.Series(0, index=price_open.index, dtype=int)

        for offset in range(length):
            w = float(length - offset)
            s_g = green_raw.shift(offset).fillna(0.0)
            s_r = red_raw.shift(offset).fillna(0.0)
            g_w += s_g * w
            r_w += s_r * w
            g_b += (s_g > 0).astype(int)
            r_b += (s_r > 0).astype(int)

        return g_w, r_w, g_b, r_b


    def _add_vwap_features(self, f: DataFrame, pd_module: Any) -> DataFrame:
        if "volume" not in f.columns:
            logger.warning("VWAP: нет колонки volume — заполняем 1.0")
            f["volume"] = 1.0

        high, low, close, volume = f["high"], f["low"], f["close"], f["volume"]
        hlc3 = (high + low + close) / 3
        ohlc4 = (f["open"] + high + low + close) / 4
        f["hlc3"] = hlc3
        f["ohlc4"] = ohlc4

        # ATR-based APT
        atr = self._atr(high, low, close, self.settings.atr_length)
        f["atr"] = atr
        atr_avg = atr.ewm(alpha=1.0 / self.settings.atr_length, adjust=False, min_periods=self.settings.atr_length).mean()
        ratio = np.where(atr_avg > 0, atr / atr_avg, 1.0)
        if self.settings.use_adaptive_apt:
            apt = self.settings.base_apt / (ratio ** self.settings.volatility_bias)
        else:
            apt = pd.Series(self.settings.base_apt, index=f.index)
        apt_clamped = apt.clip(lower=5.0, upper=300.0).round().astype(int)
        f["adaptive_apt"] = apt_clamped

        # Exponential alpha from APT: alpha = 1 - 2^(-1/apt)
        alpha = 1.0 - 2.0 ** (-1.0 / apt_clamped.replace(0, 1).astype(float))

        # Pivot detection
        positions = pd.Series(range(len(f)), index=f.index, dtype=float)
        is_ph = high == high.rolling(self.settings.swing_period, min_periods=1, center=False).max()
        is_pl = low == low.rolling(self.settings.swing_period, min_periods=1, center=False).min()
        ph_loc = positions.where(is_ph).ffill()
        pl_loc = positions.where(is_pl).ffill()
        ph_price = high.where(is_ph).ffill()
        pl_price = low.where(is_pl).ffill()

        pivot_dir = pd.Series(-1.0, index=f.index)
        pivot_dir[ph_loc > pl_loc] = 1.0
        f["pivot_direction"] = pivot_dir

        # Single-pass VWAP.
        #
        # Порт сделан через numpy-массивы, а не ``series.iloc[pos]``: раньше в
        # каждой итерации ветки ``else`` заново строился ряд
        # ``((novelsrc + ohlc4) / 2)`` длиной n, чтобы взять из него один элемент —
        # это была основная стоимость шага (замер: 834 мс на 8 000 баров). Порядок
        # и вид арифметики сохранены, поэтому числа не меняются.
        novelsrc = f["novelsrc"]
        direction_arr = pivot_dir.to_numpy(dtype=float)
        ph_loc_arr = ph_loc.to_numpy(dtype=float)
        pl_loc_arr = pl_loc.to_numpy(dtype=float)
        ph_price_arr = ph_price.to_numpy(dtype=float)
        pl_price_arr = pl_price.to_numpy(dtype=float)
        alpha_arr = alpha.to_numpy(dtype=float)
        hlc3_arr = hlc3.to_numpy(dtype=float)
        vol_arr = volume.to_numpy(dtype=float)
        #: «Принуждающее» значение ветки else: avg(novelsrc, ohlc4) * volume.
        #: Считается один раз, а не на каждом баре.
        else_forcing = (novelsrc.to_numpy(dtype=float) + ohlc4.to_numpy(dtype=float)) / 2.0

        n = len(f)
        vwap_out = np.full(n, np.nan)

        prev_dir: float | None = None
        state_p = 0.0
        state_vol = 0.0

        for pos in range(n):
            direction = direction_arr[pos]
            changed = (prev_dir is None or direction != prev_dir)

            if changed:
                if direction > 0:
                    start = int(pl_loc_arr[pos])
                    pv = pl_price_arr[pos]
                else:
                    start = int(ph_loc_arr[pos])
                    pv = ph_price_arr[pos]
                state_p = pv * vol_arr[start]
                state_vol = vol_arr[start]
                for inner in range(start, pos + 1):
                    a = alpha_arr[inner]
                    pv_i = hlc3_arr[inner] * vol_arr[inner]
                    v_i = vol_arr[inner]
                    state_p = (1.0 - a) * state_p + a * pv_i
                    state_vol = (1.0 - a) * state_vol + a * v_i
            else:
                a = alpha_arr[pos]
                pv = else_forcing[pos] * vol_arr[pos]
                v = vol_arr[pos]
                state_p = (1.0 - a) * state_p + a * pv
                state_vol = (1.0 - a) * state_vol + a * v

            vwap_out[pos] = state_p / state_vol if state_vol > 0 else np.nan
            prev_dir = direction

        my_vwap = pd.Series(vwap_out, index=f.index)
        f["my_vwap_state"] = my_vwap
        f["my_vwap_state_1"] = my_vwap.shift(1)
        f["my_vwap_state_5"] = my_vwap.shift(5)
        return f


    def _add_tema_dema(self, f: DataFrame) -> DataFrame:
        ns = f["novelsrc"]
        e4 = self._ema(ns, self.settings.tema_length)
        e5 = self._ema(e4, self.settings.tema_length)
        e6 = self._ema(e5, self.settings.tema_length)
        f["tema820"] = 3 * (e4 - e5) + e6

        e1 = self._ema(ns, self.settings.dema_length)
        e2 = self._ema(e1, self.settings.dema_length)
        f["dema510"] = 2 * e1 - e2
        f["dema_tema_avg_sma3"] = self._sma((f["dema510"] + f["tema820"]) / 2, 3)
        return f


    @staticmethod
    def _add_adl_chain(f: DataFrame) -> DataFrame:
        """ADL: Accumulation/Distribution Line derivative → EMA2.

        Pine::

            sym = avg(novelsrc, AvgCandle, sourceformas, AvgCandle)  # 4-term, AvgCandle ×2
            diff = (sym - sym[1]) / (sym + 1)
            adline = ema(cum(sqrt(diff) if diff>0 else -sqrt(-diff)), 2)
        """
        # Pine использует 4-term average с дублированием AvgCandle
        sym = (f["novelsrc"] + 2 * f["avg_candle"] + f["sourceformas"]) / 4
        f["adl_sym"] = sym

        diff = (sym - sym.shift(1)) / (sym + 1e-12)
        sqrt_diff = np.sqrt(diff.abs()) * np.sign(diff)
        cum_sqrt = sqrt_diff.fillna(0).cumsum()
        f["adline"] = cum_sqrt.ewm(alpha=2.0 / 3.0, adjust=False, min_periods=1).mean()
        return f


    def _add_adl_rsi(self, f: DataFrame) -> DataFrame:
        """ADL-based RSI bands (inverse-engineered from Pine).

        Pine использует нестандартную RSI-формулу на adline для расчёта
        верхней/нижней/средней полос (ubb/lbb/lm).
        """
        src = f["adline"]
        ep = 2 * self.settings.rsi_length - 1
        auc = self._ema(np.maximum(src - src.shift(1), 0), ep)
        adc = self._ema(np.maximum(src.shift(1) - src, 0), ep)
        ob = self.settings.ob_level
        os = self.settings.os_level
        om = self.settings.om_level

        x11 = (self.settings.rsi_length - 1) * (adc * ob / (100 - ob) - auc)
        f["ubb"] = np.where(x11 >= 0, src + x11, src + x11 * (100 - ob) / ob)
        x22 = (self.settings.rsi_length - 1) * (adc * os / (100 - os) - auc)
        f["lbb"] = np.where(x22 >= 0, src + x22, src + x22 * (100 - os) / os)
        x3 = (self.settings.rsi_length - 1) * (adc * om / (100 - om) - auc)
        f["lm"] = np.where(x3 >= 0, src + x3, src + x3 * (100 - om) / om)
        return f


    def _add_adl_bb_linreg(self, f: DataFrame) -> DataFrame:
        """Bollinger Bands + linear regression on adline (shifted by lkbk_bb).

        Pine::

            srcbb = adline[lkbk_bb]
            basisbb = linreg(sma(srcbb, lengthbb), 10, 0)
            devbb = mult_bb * stdev(srcbb, lengthbb)
            upperbb = basisbb + devbb * percent_bb
            lowerbb = basisbb - devbb * percent_bb
            ad = avg(basisbb, lm)
        """
        lkbk = int(round(self.settings.lkbk_bb))
        srcbb = f["adline"].shift(lkbk)
        basisbb = self._linreg(self._sma(srcbb, self.settings.length_bb), 10)
        # Pine: ta.stdev использует ddof=1 (N-1, unbiased estimator)
        devbb = self.settings.mult_bb * srcbb.rolling(self.settings.length_bb, min_periods=1).std()
        pct = self.settings.percent_bb / 100.0
        f["basisbb"] = basisbb
        f["upperbb"] = basisbb + devbb * pct
        f["lowerbb"] = basisbb - devbb * pct
        f["ad"] = (basisbb + f["lm"]) / 2
        return f


    @staticmethod
    def _add_adl_ma(f: DataFrame) -> DataFrame:
        """Ряд усреднённых MA adline (4-term = как в Pine).

        Pine::

            adl50 = avg(
                sma(adline,50),
                ema(adline,50),
                ema(avg(lowerbb, lbb), 50),   # lowerbb из BB, lbb из ADL-RSI
                ema(avg(upperbb, ubb), 50),   # upperbb из BB, ubb из ADL-RSI
            )
        """
        adl = f["adline"]
        lm_rsi = f["lm"]  # ADL-RSI middle
        ubb_rsi = f["ubb"]  # ADL-RSI upper
        lbb_rsi = f["lbb"]  # ADL-RSI lower
        lowerbb = f["lowerbb"]  # ADL-BB lower
        upperbb = f["upperbb"]  # ADL-BB upper

        for period, name in [(50, "adl50"), (100, "adl100"), (200, "adl200"), (1000, "adl1000")]:
            sma = adl.rolling(period, min_periods=1).mean()
            ema_adl = adl.ewm(span=period, adjust=False, min_periods=1).mean()
            ema_lower = ((lowerbb + lbb_rsi) / 2).ewm(span=period, adjust=False, min_periods=1).mean()
            ema_upper = ((upperbb + ubb_rsi) / 2).ewm(span=period, adjust=False, min_periods=1).mean()
            f[name] = (sma + ema_adl + ema_lower + ema_upper) / 4
        return f


    def _add_adl_macd(self, f: DataFrame) -> DataFrame:
        """MACD на ADL::

            fast_ma = ad (avg(basisbb, lm))
            slow_ma = adl50
            macd = fast_ma - slow_ma
            signal = sma(macd, signal_length)
            hist = macd - signal
            tl = avg(linreg(avg(macd,signal), 50, 0), rma(avg(macd,signal), 50))
        """
        fast = f["ad"]
        slow = f["adl50"]
        f["adl_macd"] = fast - slow
        f["adl_signal"] = f["adl_macd"].rolling(self.settings.signal_length, min_periods=1).mean()
        f["adl_hist"] = f["adl_macd"] - f["adl_signal"]

        avg_ms = (f["adl_macd"] + f["adl_signal"]) / 2
        f["adl_tl"] = (self._linreg(avg_ms, 50) + self._rma(avg_ms, 50)) / 2
        return f


    def _add_two_pole_filter(self, f: DataFrame, *, include_decorative: bool = True) -> DataFrame:
        """Two-pole filter на adline и novelsrc.

        Pine::

            tp_f = two_pole_filter(adline, length_adl, damping)
            tp_b = two_pole_filter(novelsrc, length_adl, damping)

            rising/falling counter на tp_f > tp_f[2]:
                up → rising += 1, falling = 0
                dn → rising = 0, falling += 1

        ``tp_f``/``tp_b`` нужны сигналам; счётчики ``tp_rising``/``tp_falling`` —
        нет (их никто не читает), поэтому при ``include_decorative=False`` цикл
        подсчёта пропускается.
        """
        length = float(self.settings.length_adl)
        damp = self.settings.damping

        f["tp_f"] = _two_pole_filter(f["adline"].values, length, damp)
        f["tp_b"] = _two_pole_filter(f["novelsrc"].values, length, damp)

        if not include_decorative:
            return f

        # rising/falling counter
        tp_f = f["tp_f"]
        up = tp_f > tp_f.shift(2)
        dn = tp_f < tp_f.shift(2)

        rising_count = np.zeros(len(f), dtype=int)
        falling_count = np.zeros(len(f), dtype=int)
        r, fl = 0, 0
        for i in range(len(f)):
            if up.iloc[i]:
                r += 1
                fl = 0
            elif dn.iloc[i]:
                r = 0
                fl += 1
            rising_count[i] = r
            falling_count[i] = fl

        f["tp_rising"] = rising_count
        f["tp_falling"] = falling_count
        return f


    @staticmethod
    def _add_bb_bands(f: DataFrame) -> DataFrame:
        """Bollinger Bands (из оригинального Pine, для визуализации)."""
        ohlc4 = (f["open"] + f["high"] + f["low"] + f["close"]) / 4
        bb20 = _bb(ohlc4, 20, 2.618)
        f["bb_mid_20"], f["bb_upper_20"], f["bb_lower_20"] = bb20
        bb30 = _bb(f["close"], 20, 2.0)
        f["bb_mid_close_20"], f["bb_upper_close_20"], f["bb_lower_close_20"] = bb30
        bb50 = _bb(ohlc4, 50, 2.618)
        f["bb_mid_50"], f["bb_upper_50"], f["bb_lower_50"] = bb50
        bb5 = _bb(f["close"], 5, 4.0)
        f["bb_mid_5"], f["bb_upper_5"], f["bb_lower_5"] = bb5
        return f


    @staticmethod
    def _add_small_trend_table(f: DataFrame, pd_module: Any) -> DataFrame:
        """Порт small trend table из Pine (cumulative баров, сброс каждые N)."""
        length = 20  # LengthBarsAmount
        threshold = 0.002  # movementThresholdd

        movement = (f["close"] - f["open"]) / f["open"]
        weight = movement.abs().where(movement.abs() > threshold, 0.0)

        # Cumulative с периодическим сбросом
        bars_idx = np.arange(len(f))
        reset_mask = (bars_idx % length) == 0

        green_weight = np.where(movement > 0, weight, 0.0)
        red_weight = np.where(movement < 0, weight, 0.0)
        green_bars_raw = (movement > 0).astype(int)
        red_bars_raw = (movement < 0).astype(int)

        # group-apply reset каждые length баров
        groups = bars_idx // length
        f["trend_green_bars"] = _cumsum_reset(green_bars_raw, groups)
        f["trend_red_bars"] = _cumsum_reset(red_bars_raw, groups)
        f["trend_green_weight"] = _cumsum_reset(green_weight, groups)
        f["trend_red_weight"] = _cumsum_reset(red_weight, groups)
        return f


    @staticmethod
    def _above_vwap(row: Any) -> bool:
        ns = row.get("novelsrc", float("inf"))
        return bool(
            ns > row.get("my_vwap_state", float("-inf"))
            and ns > row.get("my_vwap_state_1", float("-inf"))
            and ns > row.get("my_vwap_state_5", float("-inf"))
        )


    @staticmethod
    def _below_vwap(row: Any) -> bool:
        ns = row.get("novelsrc", float("-inf"))
        return bool(
            ns < row.get("my_vwap_state", float("inf"))
            and ns < row.get("my_vwap_state_1", float("inf"))
            and ns < row.get("my_vwap_state_5", float("inf"))
        )
