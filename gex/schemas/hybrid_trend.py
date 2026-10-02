"""Pydantic v2 — схемы структуры рынка HH/HL/LH/LL (GET /ta-structure/{ticker})."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from ._base import _Base
from .novel_candles import NovelCandleBar, TrendlineBlock


class StructurePoint(_Base):
    """Точка фрактала: подтверждённый пивот high/low."""

    index: int = Field(..., description="Бар подтверждения фрактала")
    pivot_index: int = Field(..., description="Бар возникновения пивота")
    price: float = Field(..., description="Цена пивота (по fractal-серии)")
    kind: Literal["high", "low"] = Field(..., description="Тип пивота")


class StructureEvent(_Base):
    """Событие структуры HH/LH/HL/LL в момент подтверждения."""

    index: int = Field(..., description="Бар подтверждения события")
    pivot_index: int = Field(..., description="Бар возникновения пивота")
    price: float = Field(..., description="Цена пивота")
    label: Literal["HH", "LH", "HL", "LL"] = Field(..., description="Метка структуры")


class ZigzagPoint(_Base):
    """Точка зигзага (чередующиеся подтверждённые пивоты)."""

    index: int = Field(..., description="Бар возникновения пивота")
    price: float = Field(..., description="Цена пивота")
    kind: Literal["high", "low"] = Field(..., description="Тип пивота")


class HybridStructureResponse(_Base):
    """Ответ GET /ta-structure/{ticker}: бары + структура + трендовые линии."""

    ticker: str = Field(...)
    timeframe: str = Field(...)
    asset_type: str = Field(...)
    n_bars: int = Field(..., ge=0)

    bars: list[NovelCandleBar] = Field(default_factory=list, description="Стандартные OHLC")
    novel_bars: list[NovelCandleBar] = Field(default_factory=list, description="Novel-свечи для переключателя")

    fractals: list[StructurePoint] = Field(default_factory=list, description="Подтверждённые фракталы")
    events: list[StructureEvent] = Field(default_factory=list, description="События HH/LH/HL/LL")
    zigzag: list[ZigzagPoint] = Field(default_factory=list, description="Точки зигзага")

    trendlines: Optional[TrendlineBlock] = Field(None, description="Линии поддержки/сопротивления")

    trend_strict: list[int] = Field(default_factory=list, description="Строгий тренд с учётом фильтра, -1/0/+1")
    trend_final: list[int] = Field(default_factory=list, description="Финальный тренд, -1/0/+1")

    stage: Literal["UPTREND", "DOWNTREND", "REVERSAL", "RANGE"] = Field(
        "RANGE", description="Стадия рынка по структуре"
    )
    last_close: Optional[float] = Field(None, description="Close последнего бара")
    atr: Optional[float] = Field(None, description="ATR последнего бара")
