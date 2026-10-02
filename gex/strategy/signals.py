"""signals: Сигнальные колонки: входы, выходы и добавления по двум подстратегиям и объединению.

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
from .settings import StrategySettings


class _SignalsMixin:
    def _add_signal_columns(self, f: DataFrame) -> DataFrame:
        """Собрать все сигналы: A, B, Combined.

        После вызова DataFrame содержит колонки:
          * ``long_entry_a, short_entry_a, long_exit_a, short_exit_a,
             long_add_a, short_add_a``
          * ``long_entry_b, short_entry_b, long_exit_b, short_exit_b,
             long_add_b, short_add_b``
          * ``combined_long_entry, combined_short_entry,
             combined_long_exit, combined_short_exit,
             combined_long_add, combined_short_add``
        """
        ns = f["novelsrc"]
        close = f["close"]
        high = f["high"]
        low = f["low"]
        ema10 = ns if False else f["ema10"]
        close_ema10 = f["close_ema10"]
        adline = f["adline"]
        ad = f["ad"]

        # VWAP context (общий)
        above_vwap = (
            ns > f["my_vwap_state_5"]) & (ns > f["my_vwap_state_1"]) & (ns > f["my_vwap_state"]
        )
        below_vwap = (
            ns < f["my_vwap_state_5"]) & (ns < f["my_vwap_state_1"]) & (ns < f["my_vwap_state"]
        )

        # ---- Strategy A entries ----
        novelsrc_ema3 = f["novelsrc_ema3"]
        f["long_entry_a"] = (
            (ns > f["ema10"]) & (ns > f["ema20"]) & (ns > f["ema77"])
            & (novelsrc_ema3 > f["ema200"])
            & (ns > f["my_vwap_state_5"]) & (ns > f["my_vwap_state_1"]) & (ns > f["my_vwap_state"])
            & f["can_enter_long"]
        )
        f["short_entry_a"] = (
            (ns < f["ema10"]) & (ns < f["ema20"]) & (ns < f["ema77"])
            & (novelsrc_ema3 < f["ema200"])
            & (ns < f["my_vwap_state_5"]) & (ns < f["my_vwap_state_1"]) & (ns < f["my_vwap_state"])
            & f["can_enter_short"]
        )
        f["long_exit_a"] = novelsrc_ema3 < f["ema77"]
        f["short_exit_a"] = novelsrc_ema3 > f["ema77"]

        f["long_add_a"] = (
            (self._crossunder(low, f["ema20"]) | self._crossunder(low, f["ema33"]))
            & (novelsrc_ema3 > f["ema200"])
            & above_vwap & f["can_enter_long"]
        )
        f["short_add_a"] = (
            (self._crossover(high, f["ema20"]) | self._crossover(high, f["ema33"]))
            & (novelsrc_ema3 < f["ema200"])
            & below_vwap & f["can_enter_short"]
        )

        # ---- Strategy B entries ----
        f["long_entry_b"] = (
            (ns > f["ema10"]) & (ns > f["ema20"]) & (ns > f["close_ema50"])
            & (ns > f["ema77"]) & (novelsrc_ema3 > f["ema200"])
            & (adline > f["adl50"]) & (adline > ad) & (adline > f["adl200"])
            & f["can_enter_long"]
            & (adline > f["tp_f"])
            & (f["adl_macd"] > 0) & (f["adl_signal"] > 0)
            & (f["adl_macd"] > f["adl_tl"]) & (f["adl_signal"] > f["adl_tl"])
        )
        f["short_entry_b"] = (
            (ns < f["ema10"]) & (ns < f["ema20"]) & (ns < f["close_ema50"])
            & (ns < f["ema77"]) & (novelsrc_ema3 < f["ema200"])
            & (adline < f["adl50"]) & (adline < ad)
            & f["can_enter_short"]
            & (adline < f["tp_f"])
            & (f["adl_macd"] < 0) & (f["adl_signal"] < 0)
            & (f["adl_macd"] < f["adl_tl"]) & (f["adl_signal"] < f["adl_tl"])
        )
        f["long_exit_b"] = (novelsrc_ema3 < f["ema77"]) | (adline < f["adl50"])
        f["short_exit_b"] = (novelsrc_ema3 > f["ema77"]) | (adline > f["adl50"])

        f["long_add_b"] = (
            (self._crossunder(low, f["ema20"]) | (self._crossunder(adline, ad)))
            & (novelsrc_ema3 > f["ema200"])
            & (adline > f["adl200"]) & f["can_enter_long"]
            & (adline > f["tp_f"])
        )
        f["short_add_b"] = (
            (self._crossover(high, f["ema20"]) | (self._crossover(adline, ad)))
            & (novelsrc_ema3 < f["ema200"])
            & (adline < f["adl200"]) & f["can_enter_short"]
            & (adline < f["tp_f"])
        )

        # ---- Combined (OR) — построить все новые колонки разом, без фрагментации ----
        combined = {
            "combined_long_entry": f["long_entry_a"] | f["long_entry_b"],
            "combined_short_entry": f["short_entry_a"] | f["short_entry_b"],
            "combined_long_exit": f["long_exit_a"] | f["long_exit_b"],
            "combined_short_exit": f["short_exit_a"] | f["short_exit_b"],
            "combined_long_add": f["long_add_a"] | f["long_add_b"],
            "combined_short_add": f["short_add_a"] | f["short_add_b"],
            # backward-compat
            "long_entry_signal": None,
            "short_entry_signal": None,
            "long_exit_signal": None,
            "short_exit_signal": None,
            "long_add_signal": None,
            "short_add_signal": None,
        }
        # fill backward-compat aliases
        combined["long_entry_signal"] = combined["combined_long_entry"]
        combined["short_entry_signal"] = combined["combined_short_entry"]
        combined["long_exit_signal"] = combined["combined_long_exit"]
        combined["short_exit_signal"] = combined["combined_short_exit"]
        combined["long_add_signal"] = combined["combined_long_add"]
        combined["short_add_signal"] = combined["combined_short_add"]

        # Единый pd.concat вместо 15 f["col"] = ... — убирает PerformanceWarning
        import pandas as _pd
        return _pd.concat([f, _pd.DataFrame(combined, index=f.index)], axis=1)
