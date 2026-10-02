"""Pydantic v2 — схемы для RSI Novel Candles API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ._base import _Base


class RsiNovelBar(_Base):
    """Одна RSI-свеча (значения 0..100; первые бары — None до прогрева RSI)."""
    time: str = Field(..., description="ISO-8601 timestamp")
    open: Optional[float] = Field(None, description="RSI Open")
    high: Optional[float] = Field(None, description="RSI High (скорректированный)")
    low: Optional[float] = Field(None, description="RSI Low (скорректированный)")
    close: Optional[float] = Field(None, description="RSI Close")


class RsiNovelLevels(_Base):
    """Настройки динамических уровней запроса."""
    ob: float = Field(..., description="RSI Overbought level")
    os: float = Field(..., description="RSI Oversold level")
    om: float = Field(..., description="RSI Middle level")


class RsiNovelResponse(_Base):
    """Ответ эндпоинта /rsi-novel/{ticker}."""
    ticker: str = Field(...)
    timeframe: str = Field(...)
    asset_type: str = Field(...)
    n_bars: int = Field(..., ge=0)
    bars: list[RsiNovelBar] = Field(default_factory=list)

    rsi_close: list[Optional[float]] = Field(default_factory=list, description="Классическая линия RSI (Close)")
    rsi_open: list[Optional[float]] = Field(default_factory=list, description="RSI Open")
    rsi_avg: list[Optional[float]] = Field(default_factory=list, description="avg(RSI O/H/L/C)")

    rsi_ma_fast: list[Optional[float]] = Field(default_factory=list, description="EMA10 от rsi_avg")
    rsi_ma: list[Optional[float]] = Field(default_factory=list, description="EMA20 от rsi_avg")
    rsi_ma3: list[Optional[float]] = Field(default_factory=list, description="EMA50 от rsi_avg")
    trend_ma: list[Optional[float]] = Field(default_factory=list, description="EMA33 от rsi_avg")

    linear_reg_curve: list[Optional[float]] = Field(default_factory=list, description="Кривая линейной регрессии")

    resistance: list[Optional[float]] = Field(default_factory=list, description="Верхняя линия (offset=1 на фронте)")
    support: list[Optional[float]] = Field(default_factory=list, description="Нижняя линия (offset=1 на фронте)")
    mid: list[Optional[float]] = Field(default_factory=list, description="Средняя линия (offset=1 на фронте)")

    slope: Optional[float] = Field(None, description="Наклон кастомной period100-регрессии (последний бар)")
    mad: Optional[float] = Field(None, description="MAD кастомной регрессии (последний бар)")
    last_rsi: Optional[float] = Field(None, description="Последнее конечное значение RSI Close")
    signal: Optional[str] = Field(None, description="overbought | oversold | neutral | null")

    rsi_length: int = Field(..., ge=1, description="Длина RSI (lenn)")
    levels: RsiNovelLevels = Field(..., description="Уровни динамических линий")
