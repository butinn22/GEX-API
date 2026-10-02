"""Pydantic v2 — схемы для Novel Candles API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ._base import _Base


class NovelCandleBar(_Base):
    """Одна novel-свеча."""
    time: str = Field(..., description="ISO-8601 timestamp")
    open: float = Field(..., description="Novel Open")
    high: float = Field(..., description="Novel High")
    low: float = Field(..., description="Novel Low")
    close: float = Field(..., description="Novel Close")


class TrendlineItem(_Base):
    """Одна трендовая линия (support/resistance)."""
    kind: str = Field(..., description="support | resistance")
    x1: int = Field(..., description="Бар-индекс старшего экстремума")
    x2: int = Field(..., description="Бар-индекс ближнего экстремума")
    price1: float = Field(..., description="Цена в точке x1")
    price2: float = Field(..., description="Цена в точке x2")
    angle_deg: float = Field(..., description="Угол наклона, градусы")
    slope: float = Field(..., description="Наклон Δprice/Δbar")
    current_price: float = Field(..., description="Экстраполяция на последний бар")


class TrendlineBlock(_Base):
    """Блок трендовых линий."""
    support: list[TrendlineItem] = Field(default_factory=list)
    resistance: list[TrendlineItem] = Field(default_factory=list)
    combined_trend: str = Field("RANGE")
    combined_strength: float = Field(0.0)


class TwoPoleSignalPoint(_Base):
    """Точка сигнала Two-Pole Filter."""
    index: int = Field(..., description="Индекс бара")
    price: float = Field(..., description="Цена сигнала")


class TwoPoleFilterData(_Base):
    """Данные Two-Pole Filter."""
    tp_f: list[float] = Field(default_factory=list, description="Значения фильтра")
    rising: list[int] = Field(default_factory=list, description="Счётчики rising")
    falling: list[int] = Field(default_factory=list, description="Счётчики falling")
    rising_positions: list[TwoPoleSignalPoint] = Field(default_factory=list)
    falling_positions: list[TwoPoleSignalPoint] = Field(default_factory=list)
    signal_up: list[TwoPoleSignalPoint] = Field(default_factory=list)
    signal_dn: list[TwoPoleSignalPoint] = Field(default_factory=list)


class NovelCandlesResponse(_Base):
    """Ответ эндпоинта /novel-candles/{ticker}."""
    ticker: str = Field(...)
    timeframe: str = Field(...)
    asset_type: str = Field(...)
    n_bars: int = Field(..., ge=0)
    bars: list[NovelCandleBar] = Field(default_factory=list)
    emas: dict[str, list[Optional[float]]] = Field(default_factory=dict)
    trendlines: Optional[TrendlineBlock] = Field(None)
    two_pole: Optional[TwoPoleFilterData] = Field(None)
