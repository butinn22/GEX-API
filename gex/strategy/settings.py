"""Входные параметры, состояние позиции и результат решения (ring: domain).

Всё, что описывает **данные** стратегии, а не её расчёты. Вынесено первым, потому что от
этих типов зависят все остальные модули — и потому, что ``StrategySettings`` это вход из
Pine Script: 101 строка параметров, которую следует читать отдельно от алгоритма.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SignalAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class StrategySettings:
    # --- TP / Trailing (общие) ---
    tp_percent: float = 2.0
    trailing_percent: float = 1.0
    use_trailing: bool = True
    use_take_profit: bool = True
    tp_cooldown_bars: int = 3

    # --- Trend analysis (общие) ---
    length_bars: int = 15
    movement_threshold: float = 0.314  # в %, делится на 100 в коде
    bull_trend_threshold: float = 0.3
    bear_trend_threshold: float = -0.3
    flat_filter_enabled: bool = True
    flat_threshold_d: float = 1.2
    smooth_len: int = 3

    # --- Swing / VWAP (Strategy A) ---
    swing_period: int = 50
    base_apt: float = 20.0
    use_adaptive_apt: bool = False
    volatility_bias: float = 10.0
    atr_length: int = 50
    cooldown_bars: int = 10

    # --- Extra EMAs / TEMA / DEMA (декоративные) ---
    tema_length: int = 820
    dema_length: int = 510

    # --- RSI/BB (Strategy B, ADL-related) ---
    rsi_length: int = 14
    ob_level: int = 75
    os_level: int = 30
    om_level: int = 50

    # --- ADL Two-Pole Filter (Strategy B) ---
    length_adl: int = 20
    damping: float = 0.9
    rising_falling: int = 5

    # --- ADL BB/linreg (Strategy B) ---
    length_bb: int = 33
    lkbk_bb: float = 4.618
    mult_bb: float = 2.618
    percent_bb: float = 61.8  # в %, делится на 100

    # --- MACD Signal (Strategy B) ---
    signal_length: int = 9

    # --- Small trend helpers ---
    length_bars_amount: int = 20
    movement_threshold_d: float = 0.002  # 0.2%

    # --- ATR-адаптивные TP/SL (кастомная надстройка, не из Pine) ---
    use_atr_stops: bool = True
    atr_tp_mult: float = 2.0
    atr_sl_mult: float = 1.5

    # --- Verification threshold (кастомная надстройка) ---
    verification_threshold: float = 55.0

    # --- Детектор тренда/флэта на 200 барах (ATR + BBW + z-цена) ---
    # Аддитивная надстройка: на расчёты calculate() НЕ влияет, используется
    # только методами trend_regime_* (см. gex/trend_regime.py). Дефолты
    # подобраны так, чтобы поведение всех существующих страниц не менялось:
    # flat_filter_signals=False → гейт сигналов выключен.
    regime_window: int = 200          # W — основное окно анализа
    regime_recent: int = 50           # N — недавнее окно (свежее движение)
    regime_atr_period: int = 14
    regime_bb_period: int = 20
    regime_bb_mult: float = 2.0
    regime_price_recent_weight: float = 0.6  # вес z_n против z_w
    flat_slider: float = 0.5          # 0 — строгий флэт, 1 — широкий
    flat_slider_atr: float | None = None     # None → берётся flat_slider
    flat_slider_bbw: float | None = None
    flat_slider_pct: float | None = None
    flat_score_threshold: float = 60.0
    trend_strength_low: float = 30.0
    flat_filter_signals: bool = False  # резать ли сигналы без подтверждения
    flat_gate_exits: bool = False      # резать ли выходы (по умолчанию нет)

    def __post_init__(self) -> None:
        for name in ("tp_percent", "trailing_percent", "length_bars", "smooth_len",
                     "cooldown_bars", "swing_period", "base_apt", "atr_length",
                     "tema_length", "dema_length", "atr_tp_mult", "atr_sl_mult",
                     "rsi_length", "length_adl", "rising_falling",
                     "regime_window", "regime_recent", "regime_atr_period",
                     "regime_bb_period", "regime_bb_mult"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.verification_threshold <= 100.0:
            raise ValueError("verification_threshold must be in [0, 100]")
        if not 0.0 <= self.flat_slider <= 1.0:
            raise ValueError("flat_slider must be in [0, 1]")
        if self.regime_recent > self.regime_window:
            raise ValueError("regime_recent must be <= regime_window")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "StrategySettings":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in values.items() if k in allowed})


@dataclass(frozen=True)
class TradingSignal:
    action: SignalAction
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    quantity_fraction: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        return self.metadata.get("order_type") in {"entry_long", "entry_short"}

    @property
    def is_exit(self) -> bool:
        return self.reason in {"take_profit", "trailing_stop", "ema_exit", "rsi_oversold"}


@dataclass
class TradingState:
    position_side: str = "flat"
    position_qty: float = 0.0
    long_entry_price: float | None = None
    short_entry_price: float | None = None
    trailing_long: float | None = None
    trailing_short: float | None = None
    last_add_bar: int | None = None
    last_tp_bar_long: int | None = None
    last_tp_bar_short: int | None = None

    @property
    def is_flat(self) -> bool:
        return self.position_side == "flat"

    @property
    def is_long(self) -> bool:
        return self.position_side == "long"

    @property
    def is_short(self) -> bool:
        return self.position_side == "short"
