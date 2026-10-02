"""Pydantic v2 — схемы сериализации для API."""

from datetime import date, datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator

from ._base import _Base


# ====================================================================== #
class OHLCVBarOut(_Base):
    """Одна свеча OHLCV для фронтового графика.

    Поля названы коротко (o/h/l/c/v), чтобы минимизировать вес ответа
    при больших ``limit``. ``t`` — ISO-строка времени открытия бара (UTC).
    """

    t: str = Field(..., description="Время открытия бара, ISO 8601 (UTC)")
    o: float = Field(..., description="Open")
    h: float = Field(..., description="High")
    l: float = Field(..., description="Low")
    c: float = Field(..., description="Close")
    v: float = Field(..., description="Volume")


class OHLCVOut(_Base):
    """Итоговый JSON-ответ GET /ohlcv/{ticker} — свечи для графика.

    Источник OHLCV **идентичен** тому, что использует бэкенд для расчётов
    TA/GEX/trendlines (одни и те же фетчеры), поэтому график на фронте
    консистентен с анализом.
    """

    symbol: str = Field(..., description="Тикер (upper)")
    asset_type: Literal["stock", "crypto", "moex", "commodity"] = Field(
        ..., description="Тип актива (определяет источник OHLCV)"
    )
    timeframe: str = Field(..., description="Таймфрейм (1h, 2h, 4h, 1d)")
    spot: float = Field(..., description="Текущая цена (последний Close)")
    bars: list[OHLCVBarOut] = Field(
        ..., min_length=1, description="Свечи, от старых к новым"
    )

    @model_validator(mode="after")
    def _round(self) -> "OHLCVOut":
        self.spot = round(self.spot, 2)
        return self


