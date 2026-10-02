"""Pydantic v2 — схемы сериализации для API."""

from datetime import date, datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator

from ._base import _Base
from .auto_coverage import AutoCoverageOut


# ====================================================================== #
#  Выход: GEX-профиль по страйкам (с округлением и expiry range)
# ====================================================================== #
class StrikeProfileOut(_Base):
    """Aggregated GEX на одном страйке."""

    strike: float
    gex_call: float
    gex_put: float
    gex_net: float
    gex_abs: float
    gamma_call: float
    gamma_put: float
    oi_call: float
    oi_put: float
    oi_total: float
    t_min: float = Field(..., description="Мин. время до экспирации, дней")
    t_max: float = Field(..., description="Макс. время до экспирации, дней")
    expiry_from: Optional[date] = Field(
        None, description="Ближайшая дата экспирации опционов на страйке"
    )
    expiry_to: Optional[date] = Field(
        None, description="Самая дальняя дата экспирации опционов на страйке"
    )

    @model_validator(mode="after")
    def _round(self) -> "StrikeProfileOut":
        self.strike = round(self.strike, 2)
        self.gex_call = round(self.gex_call)
        self.gex_put = round(self.gex_put)
        self.gex_net = round(self.gex_net)
        self.gex_abs = round(self.gex_abs)
        self.gamma_call = round(self.gamma_call, 3)
        self.gamma_put = round(self.gamma_put, 3)
        self.oi_call = round(self.oi_call)
        self.oi_put = round(self.oi_put)
        self.oi_total = round(self.oi_total)
        self.t_min = round(self.t_min, 1)
        self.t_max = round(self.t_max, 1)
        return self


class GEXProfileOut(_Base):
    """GEX-профиль по активу."""

    net_gex: float
    gamma_flip: Optional[float] = None
    call_wall: float
    put_wall: float
    call_wall_oi: float
    put_wall_oi: float
    regime: Literal["POSITIVE", "NEGATIVE"]
    z_score: Optional[float] = None
    per_strike: list[StrikeProfileOut] = Field(default_factory=list)

    @model_validator(mode="after")
    def _round(self) -> "GEXProfileOut":
        self.net_gex = round(self.net_gex)
        if self.gamma_flip is not None:
            self.gamma_flip = round(self.gamma_flip, 2)
        self.call_wall = round(self.call_wall, 2)
        self.put_wall = round(self.put_wall, 2)
        self.call_wall_oi = round(self.call_wall_oi, 2)
        self.put_wall_oi = round(self.put_wall_oi, 2)
        if self.z_score is not None:
            self.z_score = round(self.z_score, 2)
        return self


# ====================================================================== #
#  Summarize-блок: ключевые уровни, визуализация, Telegram HTML
# ====================================================================== #
class ResistanceLevelOut(_Base):
    """Уровни сопротивления."""

    primary_call_wall: float
    secondary_call_walls: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _round(self) -> "ResistanceLevelOut":
        self.primary_call_wall = round(self.primary_call_wall, 2)
        self.secondary_call_walls = [round(w, 2) for w in self.secondary_call_walls]
        return self


class SupportLevelOut(_Base):
    """Уровни поддержки."""

    primary_put_wall: float
    secondary_put_walls: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _round(self) -> "SupportLevelOut":
        self.primary_put_wall = round(self.primary_put_wall, 2)
        self.secondary_put_walls = [round(w, 2) for w in self.secondary_put_walls]
        return self


class GEXKeyLevelsOut(_Base):
    """Ключевые уровни GEX."""

    resistance: ResistanceLevelOut
    support: SupportLevelOut
    gamma_flip: Optional[float] = Field(
        None, description="Gamma Flip — уровень смены GEX-режима (None, если перехода нет)"
    )

    @model_validator(mode="after")
    def _round(self) -> "GEXKeyLevelsOut":
        if self.gamma_flip is not None:
            self.gamma_flip = round(self.gamma_flip, 2)
        return self


class GEXSummaryOut(_Base):
    """Сводный блок с визуализацией GEX-уровней."""

    symbol: str
    spot: float
    regime: str = Field(
        ...,
        description="Положительная гамма / Отрицательная гамма",
    )
    key_levels: GEXKeyLevelsOut
    text_visualization: str = Field(
        ...,
        description="ASCII-визуализация ключевых уровней",
    )
    telegram_html_message: str = Field(
        ...,
        description="HTML-сообщение для Telegram",
    )

    @model_validator(mode="after")
    def _round(self) -> "GEXSummaryOut":
        self.spot = round(self.spot, 2)
        return self


# ====================================================================== #
#  Главный ответ: полный анализ
# ====================================================================== #
class GEXAnalysisOut(_Base):
    """Итоговый JSON-ответ GET /gex/{ticker}.

    Содержит всё, что нужно для принятия решения:
    - направление (BULLISH/BEARISH/NEUTRAL) с % уверенности;
    - зоны поддержки и сопротивления;
    - вероятности движения;
    - GEX-профиль по страйкам;
    - summarize-блок с визуализацией.
    """

    symbol: str
    spot: float
    days: float = Field(..., description="Горизонт анализа, дней")

    # --- Направление ---
    direction: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(
        ..., description="Направление на основе GEX-режима и позиции спота"
    )
    confidence: float = Field(
        ..., ge=0.0, le=100.0, description="Уверенность в направлении, %"
    )

    # --- Вероятности ---
    p_up: float = Field(
        ..., ge=0.0, le=1.0, description="Вероятность роста за горизонт"
    )
    p_down: float = Field(
        ..., ge=0.0, le=1.0, description="Вероятность снижения за горизонт"
    )

    # --- Ключевые уровни ---
    support: float = Field(
        ..., description="Уровень поддержки (ближайший страйк ниже spot)"
    )
    resistance: float = Field(
        ..., description="Уровень сопротивления (ближайший страйк выше spot)"
    )
    gamma_flip: Optional[float] = Field(
        None, description="Gamma Flip — уровень смены GEX-режима"
    )

    # --- GEX-режим ---
    regime: Literal["POSITIVE", "NEGATIVE"] = Field(
        ..., description="POSITIVE = дилеры гасят волатильность, NEGATIVE = усиливают"
    )

    # --- GEX-профиль ---
    profile: GEXProfileOut

    # --- Summarize-блок ---
    summarize: GEXSummaryOut

    # --- AUTO-mode coverage metadata (MOEX path; design §5) ---
    auto: Optional[AutoCoverageOut] = Field(
        None,
        description="Метаданные AUTO-режима (охват, разреженность); null для mode=manual",
    )

    @model_validator(mode="after")
    def _round(self) -> "GEXAnalysisOut":
        self.spot = round(self.spot, 2)
        self.days = round(self.days, 1)
        self.confidence = round(self.confidence, 1)
        self.p_up = round(self.p_up, 2)
        self.p_down = round(self.p_down, 2)
        self.support = round(self.support, 2)
        self.resistance = round(self.resistance, 2)
        if self.gamma_flip is not None:
            self.gamma_flip = round(self.gamma_flip, 2)
        return self


# ====================================================================== #
#  Мапперы: dataclass → Pydantic
# ====================================================================== #
def profile_to_schema(p) -> GEXProfileOut:
    """GEXProfile (dataclass) → GEXProfileOut."""
    return GEXProfileOut(
        net_gex=p.net_gex,
        gamma_flip=p.gamma_flip,
        call_wall=p.call_wall,
        put_wall=p.put_wall,
        call_wall_oi=p.call_wall_oi,
        put_wall_oi=p.put_wall_oi,
        regime=p.regime,
        z_score=p.z_score,
        per_strike=[StrikeProfileOut(**r) for r in p.per_strike.to_dict(orient="records")],
    )

