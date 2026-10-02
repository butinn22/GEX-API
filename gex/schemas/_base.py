"""Pydantic v2 — схемы сериализации для API.
Конвенция: ``*In`` — вход (запрос), ``*Out`` — выход (ответ).
"""

from datetime import date, datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator


# ====================================================================== #
#  Базовые настройки
# ====================================================================== #
class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


# ====================================================================== #
#  Вход: одна строка опционной цепи
# ====================================================================== #
class OptionRowIn(_Base):
    """Строка опционной цепочки в запросе."""

    strike: float = Field(..., gt=0, description="Цена страйка")
    type: Literal["C", "P"] = Field(..., description="Тип опциона: 'C' или 'P'")
    oi: float = Field(..., ge=0, description="Открытый интерес, контракты")
    iv: float = Field(..., gt=0, le=5.0, description="Подразумеваемая волатильность, годовых")
    T: Optional[float] = Field(None, gt=0, description="Время до экспирации, лет")
    expiry: Optional[datetime] = Field(None, description="Дата экспирации (альтернатива T)")

    @model_validator(mode="after")
    def _need_time(self) -> "OptionRowIn":
        if self.T is None and self.expiry is None:
            raise ValueError("Нужно задать либо T, либо expiry")
        return self


class ChainIn(_Base):
    """Тело POST /chains/{ticker}: цепочка + параметры базиса."""

    spot: float = Field(..., gt=0, description="Цена базового актива")
    as_of: Optional[datetime] = Field(None, description="Метка времени данных")
    r: float = Field(0.045, description="Безрисковая ставка")
    q: float = Field(0.0, description="Дивидендная доходность")
    per_contract: int = Field(100, ge=1, description="Множитель контракта")
    chain: list[OptionRowIn] = Field(..., min_length=1)

