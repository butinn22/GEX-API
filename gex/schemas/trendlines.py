"""Pydantic v2 — схемы сериализации для API."""

from datetime import date, datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator

from ._base import _Base


# ====================================================================== #
class TrendlineOut(_Base):
    """Одна построенная трендовая линия (поддержка/сопротивление).

    Координаты в системе (bar_index, price): ``x1 < x2`` (``x1`` — старший
    экстремум, ``x2`` — ближний). Линия экстраполируется вправо до текущего
    бара (``current_price``).
    """

    kind: Literal["support", "resistance"] = Field(
        ..., description="Тип линии: поддержка или сопротивление"
    )
    x1: int = Field(..., description="Бар-индекс старшего экстремума")
    x2: int = Field(..., description="Бар-индекс ближнего экстремума")
    price1: float = Field(..., description="Цена в точке x1")
    price2: float = Field(..., description="Цена в точке x2")
    angle_deg: float = Field(
        ..., description="Угол наклона линии, градусы (atan(Δprice/Δbars))"
    )
    slope: float = Field(..., description="Наклон: Δprice/Δbar (price per bar)")
    current_price: float = Field(
        ..., description="Экстраполированное значение линии на последний бар"
    )

    @model_validator(mode="after")
    def _round(self) -> "TrendlineOut":
        self.price1 = round(self.price1, 4)
        self.price2 = round(self.price2, 4)
        self.angle_deg = round(self.angle_deg, 4)
        self.slope = round(self.slope, 6)
        self.current_price = round(self.current_price, 4)
        return self


class FractalStructureOut(_Base):
    """Структура свингов HH/HL/LH/LL по фракталам."""

    n_swing_highs: int = Field(..., ge=0, description="Всего свинг-хаев")
    n_swing_lows: int = Field(..., ge=0, description="Всего свинг-лоев")
    higher_highs: int = Field(..., ge=0, description="Число Higher Highs")
    higher_lows: int = Field(..., ge=0, description="Число Higher Lows")
    lower_highs: int = Field(..., ge=0, description="Число Lower Highs")
    lower_lows: int = Field(..., ge=0, description="Число Lower Lows")
    last_high_price: Optional[float] = Field(
        None, description="Цена последнего свинг-хая"
    )
    last_low_price: Optional[float] = Field(
        None, description="Цена последнего свинг-лоя"
    )

    @model_validator(mode="after")
    def _round(self) -> "FractalStructureOut":
        if self.last_high_price is not None:
            self.last_high_price = round(self.last_high_price, 4)
        if self.last_low_price is not None:
            self.last_low_price = round(self.last_low_price, 4)
        return self


class TrendlineTimeframeOut(_Base):
    """Результат анализа трендовых линий по одному таймфрейму."""

    timeframe: str = Field(..., description="Таймфрейм (1h, 2h, 4h, 1d)")
    last_close: float = Field(..., description="Цена закрытия последнего бара")
    n_bars: int = Field(..., ge=0, description="Число баров в анализе")
    atr: float = Field(..., description="ATR(14) — для нормировки углов")

    support_lines: list[TrendlineOut] = Field(
        default_factory=list, description="Линии поддержки (зелёные)"
    )
    resistance_lines: list[TrendlineOut] = Field(
        default_factory=list, description="Линии сопротивления (розовые)"
    )
    strongest_support: Optional[TrendlineOut] = Field(
        None, description="Самая близкая к цене линия поддержки"
    )
    strongest_resistance: Optional[TrendlineOut] = Field(
        None, description="Самая близкая к цене линия сопротивления"
    )

    # --- Тренд по углу трендовых линий ---
    trend_direction: Literal["BULLISH", "BEARISH", "RANGE"] = Field(
        ..., description="Тренд по углу трендовых линий"
    )
    trend_strength: float = Field(
        ..., ge=0.0, le=100.0, description="Сила тренда по линиям, 0..100"
    )
    line_angle_deg: float = Field(
        ..., description="Усреднённый угол основных линий, градусы"
    )

    # --- Тренд по фракталам HH/HL/LH/LL ---
    fractals: FractalStructureOut = Field(
        ..., description="Структура свингов HH/HL/LH/LL"
    )
    fractal_trend: Literal["BULLISH", "BEARISH", "RANGE"] = Field(
        ..., description="Тренд по фрактальной структуре"
    )
    fractal_strength: float = Field(
        ..., ge=0.0, le=100.0, description="Сила фрактального тренда, 0..100"
    )

    # --- Объединённый вердикт ---
    combined_trend: Literal["BULLISH", "BEARISH", "RANGE"] = Field(
        ..., description="Объединённый тренд (согласие линий и фракталов)"
    )
    combined_strength: float = Field(
        ..., ge=0.0, le=100.0, description="Сила объединённого тренда, 0..100"
    )

    @model_validator(mode="after")
    def _round(self) -> "TrendlineTimeframeOut":
        self.last_close = round(self.last_close, 4)
        self.atr = round(self.atr, 4)
        self.trend_strength = round(self.trend_strength, 1)
        self.line_angle_deg = round(self.line_angle_deg, 2)
        self.fractal_strength = round(self.fractal_strength, 1)
        self.combined_strength = round(self.combined_strength, 1)
        return self


class TrendlineSummaryOut(_Base):
    """Объёмный summarize-блок: покадровый + консенсус + вердикт (HTML)."""

    consensus_trend: Literal["BULLISH", "BEARISH", "RANGE"] = Field(
        ..., description="Сводный тренд по всем таймфреймам"
    )
    consensus_strength: float = Field(
        ..., ge=0.0, le=100.0, description="Сила сводного тренда"
    )
    by_timeframe: dict[str, dict] = Field(
        default_factory=dict,
        description="Краткая сводка по каждому TF: направления + сила + углы",
    )
    nearest_support: Optional[float] = Field(
        None, description="Ближайшая поддержка (по 1d, если есть)"
    )
    nearest_resistance: Optional[float] = Field(
        None, description="Ближайшее сопротивление (по 1d, если есть)"
    )
    fractal_summary: dict = Field(
        default_factory=dict,
        description="Сводка фрактальной структуры по старшему таймфрейму",
    )
    verdict: str = Field(..., description="Итоговый текстовый вердикт")
    telegram_html_message: str = Field(
        ..., description="HTML-сообщение для Telegram (parse_mode=HTML)"
    )


class TrendlineAnalysisOut(_Base):
    """Итоговый JSON-ответ GET /trendlines/{ticker}.

    Объединяет анализ по всем доступным таймфреймам (1h, 2h, 4h, 1d): линии
    поддержки/сопротивления, тренд по углу линий и тренд по фракталам HH/HL,
    плюс сводный консенсус-вердикт и блок для Telegram.
    """

    symbol: str = Field(..., description="Тикер")
    asset_type: Literal["stock", "crypto", "moex", "commodity", "fx"] = Field(..., description="Тип актива")
    spot: float = Field(..., description="Текущая цена базиса")
    generated_at: datetime = Field(..., description="Метка времени анализа (UTC)")
    timeframes: list[TrendlineTimeframeOut] = Field(
        ..., min_length=1, description="Анализ по таймфреймам, от младшего к старшему"
    )
    consensus_trend: Literal["BULLISH", "BEARISH", "RANGE"] = Field(
        ..., description="Сводный тренд (взвешенный по таймфреймам)"
    )
    weights: dict[str, float] = Field(
        default_factory=dict, description="Веса таймфреймов в консенсусе"
    )
    summarize: Optional[TrendlineSummaryOut] = Field(
        None, description="Объёмный сводный отчёт + HTML для Telegram"
    )

    @model_validator(mode="after")
    def _round(self) -> "TrendlineAnalysisOut":
        self.spot = round(self.spot, 2)
        return self


def trendline_to_schema(
    analysis, *, symbol: str, asset_type: str, spot: float, generated_at: datetime,
    consensus_trend: str, weights: dict, summarize: Optional[dict],
) -> TrendlineAnalysisOut:
    """Маппер :class:`gex.trendlines.TrendlineAnalysis`-список → Pydantic-схема.

    Принимает уже готовый список dataclass-анализов (по таймфреймам) и
    дополнительные поля консенсуса/summarize, собирает канонический ответ.
    """
    tf_outs: list[TrendlineTimeframeOut] = []
    for a in analysis:
        fract = FractalStructureOut(
            n_swing_highs=len(a.fractals.swing_highs),
            n_swing_lows=len(a.fractals.swing_lows),
            higher_highs=a.fractals.higher_highs,
            higher_lows=a.fractals.higher_lows,
            lower_highs=a.fractals.lower_highs,
            lower_lows=a.fractals.lower_lows,
            last_high_price=(a.fractals.last_high.price if a.fractals.last_high else None),
            last_low_price=(a.fractals.last_low.price if a.fractals.last_low else None),
        )
        tf_outs.append(
            TrendlineTimeframeOut(
                timeframe=a.timeframe,
                last_close=a.last_close,
                n_bars=a.n_bars,
                atr=a.atr,
                support_lines=[TrendlineOut(**tl.as_dict()) for tl in a.support_lines],
                resistance_lines=[TrendlineOut(**tl.as_dict()) for tl in a.resistance_lines],
                strongest_support=(
                    TrendlineOut(**a.strongest_support.as_dict())
                    if a.strongest_support else None
                ),
                strongest_resistance=(
                    TrendlineOut(**a.strongest_resistance.as_dict())
                    if a.strongest_resistance else None
                ),
                trend_direction=a.trend_direction,
                trend_strength=a.trend_strength,
                line_angle_deg=a.line_angle_deg,
                fractals=fract,
                fractal_trend=a.fractal_trend,
                fractal_strength=a.fractal_strength,
                combined_trend=a.combined_trend,
                combined_strength=a.combined_strength,
            )
        )

    summary_obj = None
    if summarize is not None:
        summary_obj = TrendlineSummaryOut(**summarize)

    return TrendlineAnalysisOut(
        symbol=symbol,
        asset_type=asset_type,
        spot=spot,
        generated_at=generated_at,
        timeframes=tf_outs,
        consensus_trend=consensus_trend,
        weights=weights,
        summarize=summary_obj,
    )


# ====================================================================== #
#  Тренд по MACD (линии MACD/Signal + вероятность + теория игр)