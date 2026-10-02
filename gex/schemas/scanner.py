"""Pydantic v2 — схемы сериализации для API."""

from datetime import date, datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator

from ._base import _Base
from .gex_analysis import GEXAnalysisOut
from .ta_analysis import TAAnalysisOut


class ScanRecordOut(_Base):
    """Результат одного прогона TA + GEX по тикеру.

    Поля ``ta`` и ``gex`` присутствуют только при успешном расчёте
    соответствующего анализа. ``status == 'error'`` означает, что хотя бы один
    из них упал (см. ``error``).
    """

    ticker: str
    scanned_at: datetime = Field(..., description="Время завершения прогона (UTC)")
    status: Literal["ok", "error"] = Field(
        ..., description="'ok' — оба анализа успешны, 'error' — сбой"
    )
    error: Optional[str] = Field(None, description="Текст ошибки при status='error'")
    ta_ok: bool = Field(False, description="TA-анализ успешен")
    gex_ok: bool = Field(False, description="GEX-анализ успешен")
    ta: Optional[TAAnalysisOut] = Field(None, description="Полный TA-отчёт")
    gex: Optional[GEXAnalysisOut] = Field(None, description="Полный GEX-отчёт")


class ScanReportOut(_Base):
    """Сводный отчёт одного прогона по всем тикерам."""

    scanned_at: datetime = Field(..., description="Время завершения прогона (UTC)")
    interval_hours: float = Field(..., description="Интервал между прогонами, часов")
    running: bool = Field(..., description="Работает ли фоновый сканер")
    total: int = Field(..., ge=0, description="Всего тикеров в списке")
    ok: int = Field(..., ge=0, description="Успешных прогонов")
    failed: int = Field(..., ge=0, description="Провальных прогонов")
    records: list[ScanRecordOut] = Field(
        default_factory=list, description="Результаты по каждому тикеру"
    )

    @model_validator(mode="after")
    def _round(self) -> "ScanReportOut":
        self.interval_hours = round(self.interval_hours, 2)
        return self


# ====================================================================== #
#  Разворот SPY по индикаторам волатильности (VIX/VVIX/MOVE/COR1M)
# ====================================================================== #