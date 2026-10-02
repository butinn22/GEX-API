"""Pydantic-схемы для расширенного GEX-анализа (ручка ``/ext/gex/{ticker}``).

Схемы соответствуют dataclass-ам из :mod:`gex.extended` и ТЗ-файлу.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from gex.application.extended import (
    ExtendedGEXReport,
    ExtendedStrike,
    PowerZone,
    KeyLevel,
    HedgeScenario,
)
from gex.application.narrative import (
    GEXNarrative,
    LevelProbability,
    ExpiryBand,
)
from gex.domain.price_band import PriceBand
from gex.schemas.auto_coverage import AutoCoverageOut, auto_coverage_to_schema


# ====================================================================== #
#  Базовый класс с округлением
# ====================================================================== #
class _ExtBase(BaseModel):
    """Базовый класс с конфигом для сериализации."""

    model_config = {"from_attributes": True}


# ====================================================================== #
#  Схема страйка
# ====================================================================== #
class ExtendedStrikeOut(_ExtBase):
    """Aggregated GEX на одном страйке + производные метрики."""

    strike: float
    gex_call: float = Field(..., description="Σ Call GEX (взвеш. по экспирациям)")
    gex_put: float = Field(..., description="Σ Put GEX (всегда ≤ 0)")
    gex_net: float = Field(..., description="Call GEX + Put GEX")
    ag: float = Field(..., description="Aggregate Gamma = |Call GEX| + |Put GEX|")
    gamma_call: float = Field(..., description="Σ BSM-гамма по коллам")
    gamma_put: float = Field(..., description="Σ BSM-гамма по путам")
    oi_call: float = Field(..., description="Σ открытый интерес по коллам")
    oi_put: float = Field(..., description="Σ открытый интерес по путам")
    gamma_dollar: float = Field(
        ..., description="Net GEX × strike × 0.01 — влияние на хедж на страйке"
    )
    delta_hedge_ratio: Optional[float] = Field(
        None, description="Call GEX / |Put GEX|; None, если Put GEX = 0"
    )
    ag_normalized: float = Field(
        ..., ge=0.0, le=1.0, description="AG / max(AG) ∈ [0, 1] для area chart"
    )
    weight: float = Field(
        ..., description="Суммарный временной вес экспираций на страйке"
    )

    @model_validator(mode="after")
    def _round(self) -> "ExtendedStrikeOut":
        self.strike = round(self.strike, 4)
        self.gex_call = round(self.gex_call)
        self.gex_put = round(self.gex_put)
        self.gex_net = round(self.gex_net)
        self.ag = round(self.ag)
        self.gamma_call = round(self.gamma_call, 4)
        self.gamma_put = round(self.gamma_put, 4)
        self.oi_call = round(self.oi_call)
        self.oi_put = round(self.oi_put)
        self.gamma_dollar = round(self.gamma_dollar)
        if self.delta_hedge_ratio is not None:
            self.delta_hedge_ratio = round(self.delta_hedge_ratio, 3)
        self.ag_normalized = round(self.ag_normalized, 4)
        self.weight = round(self.weight, 4)
        return self


# ====================================================================== #
#  Power Zone
# ====================================================================== #
class ExtendedPowerZoneOut(_ExtBase):
    """Зона концентрации гаммы (топ-10% по AG, соседние страйки)."""

    center: float = Field(..., description="AG-взвешенный центр зоны")
    width: float = Field(..., description="max_strike − min_strike")
    total_ag: float = Field(..., description="Σ AG страйков зоны")
    dominant_type: Literal["CALL", "PUT"] = Field(
        ..., description="Знак Σ Net GEX зоны"
    )
    n_strikes: int = Field(..., ge=1)
    min_strike: float
    max_strike: float

    @model_validator(mode="after")
    def _round(self) -> "ExtendedPowerZoneOut":
        self.center = round(self.center, 4)
        self.width = round(self.width, 4)
        self.total_ag = round(self.total_ag)
        self.min_strike = round(self.min_strike, 4)
        self.max_strike = round(self.max_strike, 4)
        return self


# ====================================================================== #
#  Уровень поддержки/сопротивления
# ====================================================================== #
class ExtendedLevelOut(_ExtBase):
    """Уровень S/R из топ-|Net GEX|."""

    strike: float
    type: Literal["RESISTANCE", "SUPPORT"]
    strength: float = Field(..., description="|Net GEX|")
    distance_pct: float = Field(..., description="(strike − spot) / spot × 100")

    @model_validator(mode="after")
    def _round(self) -> "ExtendedLevelOut":
        self.strike = round(self.strike, 4)
        self.strength = round(self.strength)
        self.distance_pct = round(self.distance_pct, 2)
        return self


# ====================================================================== #
#  Сценарий хеджирования
# ====================================================================== #
class ExtendedHedgeOut(_ExtBase):
    """Hedge Requirement для одного сценария движения цены."""

    scenario_pct: float = Field(
        ..., description="Сценарий движения цены, % (напр. +1.0, −1.0)"
    )
    shares: float = Field(
        ...,
        description="Сколько акций/монет дилерам купить(+)/продать(−)",
    )
    dollar_value: float = Field(
        ..., description="Долларовый эквивалент хеджа (shares × spot)"
    )

    @model_validator(mode="after")
    def _round(self) -> "ExtendedHedgeOut":
        self.scenario_pct = round(self.scenario_pct, 2)
        self.shares = round(self.shares)
        self.dollar_value = round(self.dollar_value)
        return self


# ====================================================================== #
#  Narrative: аналитическая сводка GEX-профиля
# ====================================================================== #
class LevelProbabilityOut(_ExtBase):
    """Вероятность разворота на ключевом уровне."""

    strike: float
    label: str
    distance_pct: float
    reversal_pct: float = Field(..., ge=0.0, le=100.0)
    reasoning: str

    @model_validator(mode="after")
    def _round(self) -> "LevelProbabilityOut":
        self.strike = round(self.strike, 4)
        self.distance_pct = round(self.distance_pct, 2)
        self.reversal_pct = round(self.reversal_pct, 1)
        return self


class ExpiryBandOut(_ExtBase):
    """Доверительный диапазон для заданного срока экспирации."""

    days: int = Field(..., ge=1)
    support: float
    resistance: float
    band_width_pct: float
    n_strikes: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _round(self) -> "ExpiryBandOut":
        self.support = round(self.support, 4)
        self.resistance = round(self.resistance, 4)
        self.band_width_pct = round(self.band_width_pct, 2)
        return self


class GEXNarrativeOut(_ExtBase):
    """Аналитическая сводка GEX-профиля: структура, уровни, вероятности, диапазоны."""

    symbol: str
    spot: float
    net_gex: float
    regime: str
    summary: str = Field(..., description="1-2 предложения — главный вывод")

    # Рыночная структура
    direction: str
    structure: str = Field(..., description="HH=.. HL=.. LH=.. LL=.. + направление")
    ema_100: Optional[float] = None
    ema_100_distance_pct: Optional[float] = None
    global_regime: str

    # Ключевые уровни
    call_wall: Optional[float] = None
    call_wall_dist_pct: Optional[float] = None
    put_wall: Optional[float] = None
    put_wall_dist_pct: Optional[float] = None
    gamma_flip: Optional[float] = None
    gamma_flip_dist_pct: Optional[float] = None
    max_pain: Optional[float] = None
    max_pain_dist_pct: Optional[float] = None

    # Вероятность разворота
    level_probabilities: list[LevelProbabilityOut] = Field(default_factory=list)

    # Диапазоны по срокам
    expiry_bands: list[ExpiryBandOut] = Field(default_factory=list)

    # Итоговый доверительный интервал
    confidence_low: Optional[float] = None
    confidence_high: Optional[float] = None
    confidence_center: Optional[float] = None

    # EMA
    ema_cluster: bool = False
    ema_cluster_note: str = ""

    # Сырая структура
    fractal_structure: str = ""


class PriceBandOut(_ExtBase):
    """Доверительный интервал цены по ключевым GEX-уровням."""

    low: float = Field(..., description="Нижняя граница (ключевая поддержка)")
    high: float = Field(..., description="Верхняя граница (ключевое сопротивление)")
    center: float = Field(..., description="Центр диапазона")
    width_pct: float = Field(..., description="Ширина в % от spot")
    low_strength: float = Field(..., description="|Net GEX| нижней границы")
    high_strength: float = Field(..., description="|Net GEX| верхней границы")
    low_label: str = Field(..., description="Тип нижней границы")
    high_label: str = Field(..., description="Тип верхней границы")
    window_pct: float = Field(..., description="Окно поиска ±% от spot")
    n_strikes_in_window: int = Field(..., ge=0)
    key_levels: list[dict] = Field(default_factory=list)
    methodology: str = ""

    @model_validator(mode="after")
    def _round(self) -> "PriceBandOut":
        self.low = round(self.low, 4)
        self.high = round(self.high, 4)
        self.center = round(self.center, 4)
        self.width_pct = round(self.width_pct, 2)
        self.low_strength = round(self.low_strength)
        self.high_strength = round(self.high_strength)
        self.window_pct = round(self.window_pct, 1)
        return self


# ====================================================================== #
#  Volume Profile (donut chart)
# ====================================================================== #
class VolumeZoneOut(_ExtBase):
    """Одна зона денежности в volume profile."""
    zone: str = Field(..., description="ask_or_above | bid_or_below | between")
    label: str = Field(..., description="Ask or above | Bid or below | Between market")
    call_oi: float = Field(..., description="OI коллов в зоне")
    put_oi: float = Field(..., description="OI путов в зоне")
    call_pct: float = Field(..., ge=0.0, le=100.0)
    put_pct: float = Field(..., ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _round(self) -> "VolumeZoneOut":
        self.call_oi = round(self.call_oi)
        self.put_oi = round(self.put_oi)
        self.call_pct = round(self.call_pct, 2)
        self.put_pct = round(self.put_pct, 2)
        return self


class VolumeProfileOut(_ExtBase):
    """Профиль распределения OI по зонам денежности."""
    spot: float
    atm_pct: float = Field(..., description="Ширина ATM зоны в % от spot")
    total_call_oi: float
    total_put_oi: float
    total_oi: float
    call_pct: float
    put_pct: float
    zones: list[VolumeZoneOut] = Field(default_factory=list)

    @model_validator(mode="after")
    def _round(self) -> "VolumeProfileOut":
        self.spot = round(self.spot, 4)
        self.atm_pct = round(self.atm_pct, 4)
        self.total_call_oi = round(self.total_call_oi)
        self.total_put_oi = round(self.total_put_oi)
        self.total_oi = round(self.total_oi)
        self.call_pct = round(self.call_pct, 2)
        self.put_pct = round(self.put_pct, 2)
        return self


# ====================================================================== #
#  Главный ответ
# ====================================================================== #
class ExtendedGEXAnalysisOut(_ExtBase):
    """Итоговый JSON-ответ GET /ext/gex/{ticker}.

    Содержит все метрики расширенного GEX-анализа (крипта через Bybit,
    акции США через yfinance).
    """

    # --- Идентификация ---
    symbol: str
    source: Literal["crypto", "stock", "futures", "webull", "yfinance"] = Field(
        ..., description="Источник: crypto (Bybit), stock/futures/yfinance (yfinance), webull (primary)"
    )
    spot: float
    per_contract: int = Field(
        ..., description="Множитель контракта: 1 (крипта) или 100 (акции)"
    )
    days: float = Field(..., description="Горизонт анализа, дней")

    # --- Сводка по рынку ---
    net_gex: float = Field(..., description="Total Net GEX")
    total_call_gex: float = Field(..., description="Σ Call GEX (> 0)")
    total_put_gex: float = Field(..., description="Σ Put GEX (< 0)")
    total_ag: float = Field(..., description="Σ Aggregate Gamma по всем страйкам")
    regime: Literal["POSITIVE", "NEGATIVE"] = Field(
        ..., description="POSITIVE = дилеры гасят волатильность, NEGATIVE = усиливают"
    )
    directional_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"]

    # --- Ключевые уровни ---
    zero_gamma: Optional[float] = Field(
        None, description="Уровень нулевой гаммы (линейная интерполяция смены знака)"
    )
    call_wall: float = Field(..., description="Страйк с макс положительным Net GEX")
    call_wall_strength: float = Field(
        ..., ge=0.0, le=1.0, description="|Net GEX стены| / Σ|Net GEX|"
    )
    put_wall: float = Field(..., description="Страйк с макс отрицательным Net GEX")
    put_wall_strength: float = Field(
        ..., ge=0.0, le=1.0, description="|Net GEX стены| / Σ|Net GEX|"
    )
    power_zones: list[ExtendedPowerZoneOut] = Field(default_factory=list)
    key_levels: list[ExtendedLevelOut] = Field(
        default_factory=list, description="Топ-5 страйков по |Net GEX|"
    )

    # --- Метрики влияния ---
    gamma_dollar_total: float = Field(
        ..., description="Σ |Net GEX × strike × 0.01| — масштаб хеджирования"
    )
    put_call_ratio: float = Field(
        ..., ge=0.0, description="|Total Put GEX| / Total Call GEX"
    )
    gamma_exposure_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="|Net GEX| / Total AG × 100 — интегральная сила GEX"
    )
    max_pain: Optional[float] = Field(
        None, description="Страйк с макс потерями держателей опционов"
    )
    hedge_scenarios: list[ExtendedHedgeOut] = Field(
        default_factory=list,
        description="Hedge Requirement по сценариям движения цены",
    )

    # --- Профиль по страйкам ---
    per_strike: list[ExtendedStrikeOut] = Field(default_factory=list)

    # --- Динамическое описание GEX-профиля (narrative) ---
    narrative: Optional[GEXNarrativeOut] = Field(
        None,
        description="Динамическое описание: зоны, режим, HH/HL/LH/LL-сценарии, EMA-фильтры",
    )
    # --- Доверительный интервал цены (price band) ---
    price_band: Optional[PriceBandOut] = Field(
        None,
        description="Диапазон цены по ключевым GEX-уровням в окне ±10-20% от spot",
    )
    # --- Volume Profile (donut chart) ---
    volume_profile: Optional["VolumeProfileOut"] = Field(
        None,
        description="Распределение OI по зонам денежности (donut chart)",
    )
    # --- AUTO-mode coverage metadata (design §5) ---
    # Присутствует ТОЛЬКО для запросов mode=auto; для ручных — null (обратная совместимость).
    auto: Optional[AutoCoverageOut] = Field(
        None,
        description="Метаданные AUTO-режима (охват/эскалация/разреженность); null для mode=manual",
    )

    @model_validator(mode="after")
    def _round(self) -> "ExtendedGEXAnalysisOut":
        self.spot = round(self.spot, 4)
        self.days = round(self.days, 1)
        self.net_gex = round(self.net_gex)
        self.total_call_gex = round(self.total_call_gex)
        self.total_put_gex = round(self.total_put_gex)
        self.total_ag = round(self.total_ag)
        if self.zero_gamma is not None:
            self.zero_gamma = round(self.zero_gamma, 4)
        self.call_wall = round(self.call_wall, 4)
        self.put_wall = round(self.put_wall, 4)
        self.call_wall_strength = round(self.call_wall_strength, 4)
        self.put_wall_strength = round(self.put_wall_strength, 4)
        self.gamma_dollar_total = round(self.gamma_dollar_total)
        self.put_call_ratio = round(self.put_call_ratio, 3)
        self.gamma_exposure_score = round(self.gamma_exposure_score, 2)
        if self.max_pain is not None:
            self.max_pain = round(self.max_pain, 4)
        return self


# ====================================================================== #
#  Маппер: dataclass → Pydantic
# ====================================================================== #
def extended_report_to_schema(
    report: ExtendedGEXReport,
    days: float = 30.0,
) -> ExtendedGEXAnalysisOut:
    """ExtendedGEXReport (dataclass) → ExtendedGEXAnalysisOut."""
    return ExtendedGEXAnalysisOut(
        symbol=report.symbol,
        source=report.source,  # type: ignore[arg-type]
        spot=report.spot,
        per_contract=report.per_contract,
        days=days,
        net_gex=report.net_gex,
        total_call_gex=report.total_call_gex,
        total_put_gex=report.total_put_gex,
        total_ag=report.total_ag,
        regime=report.regime,  # type: ignore[arg-type]
        directional_bias=report.directional_bias,  # type: ignore[arg-type]
        zero_gamma=report.zero_gamma,
        call_wall=report.call_wall,
        call_wall_strength=report.call_wall_strength,
        put_wall=report.put_wall,
        put_wall_strength=report.put_wall_strength,
        power_zones=[_power_zone_to_schema(z) for z in report.power_zones],
        key_levels=[_key_level_to_schema(l) for l in report.key_levels],
        gamma_dollar_total=report.gamma_dollar_total,
        put_call_ratio=report.put_call_ratio,
        gamma_exposure_score=report.gamma_exposure_score,
        max_pain=report.max_pain,
        hedge_scenarios=[_hedge_to_schema(h) for h in report.hedge_scenarios],
        per_strike=[_strike_to_schema(s) for s in report.per_strike],
        narrative=_narrative_to_schema(report.narrative)
            if getattr(report, "narrative", None) is not None else None,
        price_band=_price_band_to_schema(report.price_band)
            if getattr(report, "price_band", None) is not None else None,
        volume_profile=_volume_profile_to_schema(report.volume_profile)
            if getattr(report, "volume_profile", None) is not None else None,
        auto=auto_coverage_to_schema(report.coverage)
            if getattr(report, "coverage", None) is not None else None,
    )


def _strike_to_schema(s: ExtendedStrike) -> ExtendedStrikeOut:
    return ExtendedStrikeOut(
        strike=s.strike,
        gex_call=s.gex_call,
        gex_put=s.gex_put,
        gex_net=s.gex_net,
        ag=s.ag,
        gamma_call=s.gamma_call,
        gamma_put=s.gamma_put,
        oi_call=s.oi_call,
        oi_put=s.oi_put,
        gamma_dollar=s.gamma_dollar,
        delta_hedge_ratio=s.delta_hedge_ratio,
        ag_normalized=s.ag_normalized,
        weight=s.weight,
    )


def _power_zone_to_schema(z: PowerZone) -> ExtendedPowerZoneOut:
    return ExtendedPowerZoneOut(
        center=z.center,
        width=z.width,
        total_ag=z.total_ag,
        dominant_type=z.dominant_type,  # type: ignore[arg-type]
        n_strikes=z.n_strikes,
        min_strike=z.min_strike,
        max_strike=z.max_strike,
    )


def _key_level_to_schema(l: KeyLevel) -> ExtendedLevelOut:
    return ExtendedLevelOut(
        strike=l.strike,
        type=l.type,  # type: ignore[arg-type]
        strength=l.strength,
        distance_pct=l.distance_pct,
    )


def _hedge_to_schema(h: HedgeScenario) -> ExtendedHedgeOut:
    return ExtendedHedgeOut(
        scenario_pct=h.scenario_pct,
        shares=h.shares,
        dollar_value=h.dollar_value,
    )


# --- Narrative мапперы (dataclass → Pydantic) ---
def _narrative_to_schema(n: GEXNarrative) -> GEXNarrativeOut:
    return GEXNarrativeOut(
        symbol=n.symbol,
        spot=n.spot,
        net_gex=n.net_gex,
        regime=n.regime,
        summary=n.summary,
        direction=n.direction,
        structure=n.structure,
        ema_100=n.ema_100,
        ema_100_distance_pct=n.ema_100_distance_pct,
        global_regime=n.global_regime,
        call_wall=n.call_wall,
        call_wall_dist_pct=n.call_wall_dist_pct,
        put_wall=n.put_wall,
        put_wall_dist_pct=n.put_wall_dist_pct,
        gamma_flip=n.gamma_flip,
        gamma_flip_dist_pct=n.gamma_flip_dist_pct,
        max_pain=n.max_pain,
        max_pain_dist_pct=n.max_pain_dist_pct,
        level_probabilities=[_lp_to_schema(lp) for lp in n.level_probabilities],
        expiry_bands=[_eb_to_schema(eb) for eb in n.expiry_bands],
        confidence_low=n.confidence_low,
        confidence_high=n.confidence_high,
        confidence_center=n.confidence_center,
        ema_cluster=n.ema_cluster,
        ema_cluster_note=n.ema_cluster_note,
        fractal_structure=n.fractal_structure,
    )


def _lp_to_schema(lp: LevelProbability) -> LevelProbabilityOut:
    return LevelProbabilityOut(
        strike=lp.strike, label=lp.label, distance_pct=lp.distance_pct,
        reversal_pct=lp.reversal_pct, reasoning=lp.reasoning,
    )


def _eb_to_schema(eb: ExpiryBand) -> ExpiryBandOut:
    return ExpiryBandOut(
        days=eb.days, support=eb.support, resistance=eb.resistance,
        band_width_pct=eb.band_width_pct, n_strikes=eb.n_strikes,
    )


def _price_band_to_schema(pb: PriceBand) -> PriceBandOut:
    return PriceBandOut(
        low=pb.low, high=pb.high, center=pb.center, width_pct=pb.width_pct,
        low_strength=pb.low_strength, high_strength=pb.high_strength,
        low_label=pb.low_label, high_label=pb.high_label,
        window_pct=pb.window_pct, n_strikes_in_window=pb.n_strikes_in_window,
        key_levels=list(pb.key_levels), methodology=pb.methodology,
    )


def _volume_profile_to_schema(vp) -> Optional["VolumeProfileOut"]:
    if vp is None:
        return None
    from gex.application.extended import VolumeProfile, VolumeZone
    return VolumeProfileOut(
        spot=vp.spot,
        atm_pct=vp.atm_pct,
        total_call_oi=vp.total_call_oi,
        total_put_oi=vp.total_put_oi,
        total_oi=vp.total_oi,
        call_pct=vp.call_pct,
        put_pct=vp.put_pct,
        zones=[VolumeZoneOut(
            zone=z.zone, label=z.label,
            call_oi=z.call_oi, put_oi=z.put_oi,
            call_pct=z.call_pct, put_pct=z.put_pct,
        ) for z in vp.zones],
    )
