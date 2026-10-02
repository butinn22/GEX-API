"""decision: Решение: конвейер ``calculate`` и ``evaluate`` по состоянию позиции.

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
from .ports import GEXContext
from .settings import SignalAction, StrategySettings, TradingSignal, TradingState


class _DecisionMixin:
    def calculate(self, ohlc: Any, *, include_decorative: bool = True) -> DataFrame:
        """Полный конвейер: OHLCV → все фичи + сигнальные колонки.

        Parameters
        ----------
        ohlc : DataFrame
            OHLCV с колонками ``open, high, low, close, [volume]``.
        include_decorative : bool, default True
            Считать ли колонки, которые **никем не читаются** — только пишутся:
            ``tema820``/``dema510``/``dema_tema_avg_sma3`` (декоративные графики),
            12 колонок ``bb_*``, ``trend_*`` и ``tp_rising``/``tp_falling``.
            Проверено поиском по репозиторию: эти имена не встречаются ни в одном
            потребителе. ``True`` сохраняет полный кадр (golden-эталон и
            дашборды), ``False`` экономит ~40% времени одного прогона — это
            бэктест-путь, которому нужны только сигнальные колонки.

        Returns
        -------
        DataFrame
            Все расчётные колонки + сигналы:
            ``combined_long_entry, combined_short_entry,
             combined_long_exit, combined_short_exit,
             combined_long_add, combined_short_add``.
        """
        pd_module = self._import_pandas()
        f = self._prepare_ohlc(ohlc, pd_module)

        # ---- Preprocess: HA + hybrid candles ----
        ha = self._heikin_ashi(f, pd_module)
        f = pd.concat([f, ha], axis=1)
        f = self._add_hybrid_candles(f, pd_module)

        # ---- Common EMAs (both strategies) ----
        ns = f["novelsrc"]
        f["ema10"] = self._ema(ns, 10)
        f["ema20"] = self._ema(ns, 20)
        f["ema33"] = self._ema(ns, 33)
        # Pine: ta.ema(novelsrc, 77/100) — НЕ sma
        f["ema77"] = self._ema(ns, 77)
        f["ema100"] = self._ema(ns, 100)
        f["ema200"] = self._ema(ns, 200)
        f["novelsrc_ema3"] = self._ema(ns, 3)

        # Close-based EMAs (for Strategy B conditions)
        close = f["close"]
        f["close_ema10"] = self._ema(close, 10)
        f["close_ema20"] = self._ema(close, 20)
        f["close_ema33"] = self._ema(close, 33)
        f["close_ema50"] = self._ema(close, 50)
        f["close_sma77"] = self._sma(close, 77)
        f["close_sma100"] = self._sma(close, 100)
        f["close_ema200"] = self._ema(close, 200)

        # ---- Trend analysis (shared) ----
        f = self._add_trend_analysis(f, pd_module)

        # ---- Adaptive VWAP (Strategy A) ----
        f = self._add_vwap_features(f, pd_module)

        if include_decorative:
            # ---- TEMA / DEMA (decorative plots from original) ----
            f = self._add_tema_dema(f)

        # ---- ADL chain (Strategy B) ----
        f = self._add_adl_chain(f)

        # ---- ADL-based RSI/BB (Strategy B) ----
        f = self._add_adl_rsi(f)
        f = self._add_adl_bb_linreg(f)

        # ---- ADL EMAs/SMAs (adl50/100/200/1000) ----
        f = self._add_adl_ma(f)

        # ---- ADL-based MACD/Signal/TL (Strategy B) ----
        f = self._add_adl_macd(f)

        # ---- Two-pole filter (Strategy B) ----
        f = self._add_two_pole_filter(f, include_decorative=include_decorative)

        if include_decorative:
            # ---- BB/linreg closing bands (from original) ----
            f = self._add_bb_bands(f)

            # ---- Small trend table (from original) ----
            f = self._add_small_trend_table(f, pd_module)

        # ---- Entry/exit/add signals: A, B, Combined ----
        f = self._add_signal_columns(f)

        return f


    def evaluate(
        self, features: Any, state: TradingState | None = None,
        position_qty: float | None = None,
    ) -> TradingSignal:
        if state is None:
            qty = 0.0 if position_qty is None else position_qty
            state = TradingState(
                position_side="flat" if qty == 0 else "long", position_qty=qty)
        if features.empty:
            return TradingSignal(action=SignalAction.HOLD, reason="empty_features")

        latest = features.iloc[-1]
        bar_idx = len(features) - 1

        if state.is_flat:
            if bool(latest.get("combined_long_entry", False)):
                return self._signal(SignalAction.BUY, "combined_long_entry",
                                    quantity_fraction=1.0, order_type="entry_long",
                                    close_price=float(latest["close"]))
            if bool(latest.get("combined_short_entry", False)):
                return self._signal(SignalAction.SELL, "combined_short_entry",
                                    quantity_fraction=1.0, order_type="entry_short",
                                    close_price=float(latest["close"]))
            return TradingSignal(action=SignalAction.HOLD, reason="no_entry")

        if state.is_long:
            if self._can_add(state, bar_idx) and bool(latest.get("combined_long_add", False)):
                return self._signal(SignalAction.BUY, "combined_long_add",
                                    quantity_fraction=0.1, order_type="add_long",
                                    close_price=float(latest["close"]))
            return self._evaluate_long_exit(state, latest)

        if state.is_short:
            if self._can_add(state, bar_idx) and bool(latest.get("combined_short_add", False)):
                return self._signal(SignalAction.SELL, "combined_short_add",
                                    quantity_fraction=0.1, order_type="add_short",
                                    close_price=float(latest["close"]))
            return self._evaluate_short_exit(state, latest)

        return TradingSignal(action=SignalAction.HOLD, reason="unknown_position")


    def _evaluate_long_exit(self, state: TradingState, latest: Any) -> TradingSignal:
        cp = float(latest["close"])
        hp = float(latest["high"])
        lp = float(latest["low"])
        atr = self._safe_float(latest.get("atr"))
        bar_idx = int(latest.get("bar_index", 0))

        # TP с cooldown
        if self.settings.use_take_profit and state.long_entry_price is not None:
            tp = self.take_profit_price(state.long_entry_price, "long", atr_value=atr)
            if hp >= tp:
                cooled = (state.last_tp_bar_long is None
                          or bar_idx - state.last_tp_bar_long >= self.settings.tp_cooldown_bars)
                if cooled:
                    return self._signal(SignalAction.SELL, "take_profit",
                                        quantity_fraction=0.5, order_type="exit_long",
                                        close_price=cp, entry_price=state.long_entry_price,
                                        tp_price=tp)

        # Trailing
        if self.settings.use_trailing:
            ts = self._trailing_stop_long(state, cp, atr)
            if ts is not None and lp <= ts:
                return self._signal(SignalAction.SELL, "trailing_stop",
                                    quantity_fraction=1.0, order_type="exit_long",
                                    close_price=cp, entry_price=state.long_entry_price,
                                    trailing_stop=ts)

        # EMA/ADL exit
        if bool(latest.get("combined_long_exit", False)):
            return self._signal(SignalAction.SELL, "combined_exit",
                                quantity_fraction=1.0, order_type="exit_long",
                                close_price=cp, entry_price=state.long_entry_price)

        return TradingSignal(action=SignalAction.HOLD, reason="long_hold")


    def _evaluate_short_exit(self, state: TradingState, latest: Any) -> TradingSignal:
        cp = float(latest["close"])
        hp = float(latest["high"])
        lp = float(latest["low"])
        atr = self._safe_float(latest.get("atr"))
        bar_idx = int(latest.get("bar_index", 0))

        if self.settings.use_take_profit and state.short_entry_price is not None:
            tp = self.take_profit_price(state.short_entry_price, "short", atr_value=atr)
            if lp <= tp:
                cooled = (state.last_tp_bar_short is None
                          or bar_idx - state.last_tp_bar_short >= self.settings.tp_cooldown_bars)
                if cooled:
                    return self._signal(SignalAction.BUY, "take_profit",
                                        quantity_fraction=0.5, order_type="exit_short",
                                        close_price=cp, entry_price=state.short_entry_price,
                                        tp_price=tp)

        if self.settings.use_trailing:
            ts = self._trailing_stop_short(state, cp, atr)
            if ts is not None and hp >= ts:
                return self._signal(SignalAction.BUY, "trailing_stop",
                                    quantity_fraction=1.0, order_type="exit_short",
                                    close_price=cp, entry_price=state.short_entry_price,
                                    trailing_stop=ts)

        if bool(latest.get("combined_short_exit", False)):
            return self._signal(SignalAction.BUY, "combined_exit",
                                quantity_fraction=1.0, order_type="exit_short",
                                close_price=cp, entry_price=state.short_entry_price)

        return TradingSignal(action=SignalAction.HOLD, reason="short_hold")


    def score_entry_quality(self, row: Any, direction: str) -> float:
        """Оценка силы сетапа входа в [0, 1] (учитывает обе стратегии)."""
        if direction == "long":
            conditions = [
                (0.12, bool(row.get("long_entry_a", False))),
                (0.12, bool(row.get("long_entry_b", False))),
                (0.14, bool(row.get("can_enter_long", False))),
                (0.12, bool(row.get("novelsrc", 0) > row.get("ema10", 0))),
                (0.12, bool(row.get("novelsrc", 0) > row.get("ema200", 0))),
                (0.10, bool(row.get("adline", 0) > row.get("adl50", 0))),
                (0.10, bool(row.get("adl_macd", 0) > 0)),
                (0.08, self._above_vwap(row)),
                (0.05, float(row.get("trend_coefficient", 0.0)) > self.settings.bull_trend_threshold),
                (0.05, bool(row.get("adline", 0) > row.get("tp_f", 0))),
            ]
        elif direction == "short":
            conditions = [
                (0.12, bool(row.get("short_entry_a", False))),
                (0.12, bool(row.get("short_entry_b", False))),
                (0.14, bool(row.get("can_enter_short", False))),
                (0.12, bool(row.get("novelsrc", 0) < row.get("ema10", 0))),
                (0.12, bool(row.get("novelsrc", 0) < row.get("ema200", 0))),
                (0.10, bool(row.get("adline", 0) < row.get("adl50", 0))),
                (0.10, bool(row.get("adl_macd", 0) < 0)),
                (0.08, self._below_vwap(row)),
                (0.05, float(row.get("trend_coefficient", 0.0)) < self.settings.bear_trend_threshold),
                (0.05, bool(row.get("adline", 0) < row.get("tp_f", 0))),
            ]
        else:
            raise ValueError("direction must be 'long' or 'short'")

        tw = sum(w for w, _ in conditions)
        return float(sum(w for w, m in conditions if m) / tw) if tw > 0 else 0.0


    def apply_gex_filter(
        self, signal: TradingSignal, gex_context: GEXContext | None,
    ) -> tuple[TradingSignal, float, str]:
        if gex_context is None or gex_context.gamma_flip is None:
            return signal, 1.0, "gex_unavailable"
        is_long = signal.action == SignalAction.BUY
        is_short = signal.action == SignalAction.SELL
        if not (is_long or is_short):
            return signal, 1.0, "hold_signal"

        flip = gex_context.gamma_flip
        z = gex_context.z_score if gex_context.z_score is not None else 0.0
        z_strength = min(abs(z) / 3.0, 1.0)

        if gex_context.regime == "POSITIVE":
            spot = signal.metadata.get("close_price")
            spot_below = (spot < flip) if (isinstance(spot, (int, float)) and spot > 0) else (z < 0)
            if (is_long and spot_below) or (is_short and not spot_below):
                return signal, 1.15 + 0.15 * z_strength, f"positive_toward_flip(z={z:+.2f})"
            return signal, 0.85 - 0.15 * z_strength, f"positive_against_flip(z={z:+.2f})"

        spot = signal.metadata.get("close_price")
        above = (spot > flip) if (isinstance(spot, (int, float)) and spot > 0) else (z > 0)
        if (is_long and above) or (is_short and not above):
            return signal, 1.10 + 0.15 * z_strength, f"negative_with_trend(z={z:+.2f})"
        return signal, 0.90 - 0.10 * z_strength, f"negative_against_trend(z={z:+.2f})"


    @staticmethod
    def confidence_class(
        entry_score: float, gex_mult: float, v_score: float | None, threshold: float,
    ) -> str:
        v_norm = 0.5 if v_score is None else min(max(v_score / 100.0, 0.0), 1.0)
        gf = min(max(gex_mult, 0.5), 1.4)
        composite = entry_score * 0.40 + v_norm * 0.30 + (gf - 1.0) * 0.30
        composite = min(max(composite + 0.15, 0.0), 1.0)
        confirmed = v_score is not None and v_score >= threshold
        if composite >= 0.75 and confirmed:
            return "high"
        if composite >= 0.55 and confirmed:
            return "medium"
        if composite >= 0.45:
            return "low"
        return "none"
