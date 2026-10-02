"""risk: Риск: тейк-профит и трейлинг-стоп (в том числе ATR-вариант).

Вынесено из ``gex/trading_algorithm.py`` (итерация 37). Методы перенесены дословно: разбиение god-класса не должно менять числа, а доказательство — golden-эталон
``tests/test_strategy_golden.py``, сверяющий все колонки кадра, решения ``evaluate``,
режим, оценку входа и риск до и после выноса.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from .settings import StrategySettings, TradingState


class _RiskMixin:
    def take_profit_price(self, ep: float, direction: str, atr_value: float | None = None) -> float:
        use_atr = self.settings.use_atr_stops and atr_value is not None and atr_value > 0
        if direction == "long":
            return ep + (self.settings.atr_tp_mult * atr_value) if use_atr else ep * (1 + self.settings.tp_percent / 100)
        return ep - (self.settings.atr_tp_mult * atr_value) if use_atr else ep * (1 - self.settings.tp_percent / 100)


    def trailing_stop_price(self, ep: float, direction: str,
                            highest_price: float | None = None, lowest_price: float | None = None,
                            atr_value: float | None = None) -> float | None:
        use_atr = self.settings.use_atr_stops and atr_value is not None and atr_value > 0
        off = (self.settings.atr_sl_mult * atr_value) if use_atr else ep * (self.settings.trailing_percent / 100)
        if direction == "long":
            base = ep - off
            if highest_price is not None:
                if use_atr:
                    base = max(base, highest_price - off)
                else:
                    base = max(base, highest_price * (1 - self.settings.trailing_percent / 100))
            return base
        base = ep + off
        if lowest_price is not None:
            if use_atr:
                base = min(base, lowest_price + off)
            else:
                base = min(base, lowest_price * (1 + self.settings.trailing_percent / 100))
        return base


    def _trailing_stop_long(self, state: TradingState, cp: float, atr: float | None) -> float | None:
        ep = state.long_entry_price or cp
        ts = self.trailing_stop_price(ep, "long", highest_price=state.trailing_long, atr_value=atr)
        fallback = ep * (1 - self.settings.trailing_percent / 100) if state.long_entry_price else None
        candidates = [v for v in [ts, state.trailing_long, fallback] if v is not None]
        if not candidates:
            return None
        stop = max(candidates)
        extra = cp - (atr or cp * self.settings.trailing_percent / 100)
        return max(stop, extra)


    def _trailing_stop_short(self, state: TradingState, cp: float, atr: float | None) -> float | None:
        ep = state.short_entry_price or cp
        ts = self.trailing_stop_price(ep, "short", lowest_price=state.trailing_short, atr_value=atr)
        fallback = ep * (1 + self.settings.trailing_percent / 100) if state.short_entry_price else None
        candidates = [v for v in [ts, state.trailing_short, fallback] if v is not None]
        if not candidates:
            return None
        stop = min(candidates)
        extra = cp + (atr or cp * self.settings.trailing_percent / 100)
        return min(stop, extra)
