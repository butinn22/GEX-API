"""Pydantic v2 — схемы сериализации для API."""

from datetime import date, datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator

from ._base import _Base


# ====================================================================== #
class MacdTrendTimeframeOut(_Base):
    """MACD-тренд по одному таймфрейму (последний бар).

    Поля — прямое отражение :class:`gex.macd_trend.BarResult` (значения на
    последнем баре стримингового/пакетного анализа).
    """

    timeframe: str = Field(..., description="Таймфрейм (1h, 2h, 4h, 1d)")
    n_bars: int = Field(..., ge=0, description="Число баров в анализе")

    avg_value: Optional[float] = Field(
        None, description="(macd_line + signal_line) / 2"
    )
    position: Optional[Literal["above", "below"]] = Field(
        None, description="Положение AVG-линии относительно ноля"
    )
    zero_cross: Optional[Literal["bull_cross", "bear_cross"]] = Field(
        None, description="Пересечение ноля AVG-линией за N баров"
    )
    norm_slope: Optional[float] = Field(
        None, description="Нормированный наклон AVG-линии"
    )
    angle_degrees: Optional[float] = Field(
        None, description="Угол наклона, градусы ∈ (-90, 90)"
    )
    quadrant: Optional[Literal[
        "BULLISH_STRENGTHENING",
        "BULLISH_WEAKENING",
        "BEARISH_STRENGTHENING",
        "BEARISH_WEAKENING",
        "FLAT",
    ]] = Field(None, description="Квадрант тренда (5 состояний)")
    strength_instant: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Мгновенная сила min(|angle|/90, 1)"
    )
    strength_percentile: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Перцентиль |angle| vs истории (H баров)"
    )
    markov_next_state_probs: dict[str, float] = Field(
        default_factory=dict,
        description="P(state[t+1] | state[t]) по 5 квадрантам",
    )
    composite_score: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="MWU-взвешенный голос 3 экспертов ∈ [-1, 1]"
    )
    conviction_multiplier: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Убеждённость по спреду MACD-Signal ∈ [0, 1]"
    )
    final_trend_score: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Итоговый score ∈ [-1, 1]"
    )
    kelly_fraction: Optional[float] = Field(
        None, description="Kelly fraction (только при kelly_enabled, исследовательский)"
    )
    trend: Literal["BULLISH", "BEARISH", "RANGE"] = Field(
        ..., description="Направление по знаку final_trend_score"
    )

    @model_validator(mode="after")
    def _round(self) -> "MacdTrendTimeframeOut":
        for attr in ("avg_value", "norm_slope"):
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, round(v, 6))
        for attr in ("angle_degrees", "strength_percentile", "kelly_fraction"):
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, round(v, 4))
        for attr in ("strength_instant", "conviction_multiplier", "final_trend_score", "composite_score"):
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, round(v, 4))
        return self


class MacdTrendSummaryOut(_Base):
    """Сводный summarize-блок MACD-тренда: консенсус + покадрово + HTML."""

    consensus_trend: Literal["BULLISH", "BEARISH", "RANGE"] = Field(
        ..., description="Сводный тренд по всем таймфреймам"
    )
    consensus_strength: float = Field(
        ..., ge=0.0, le=100.0, description="Сила сводного тренда"
    )
    by_timeframe: dict[str, dict] = Field(
        default_factory=dict,
        description="Краткая сводка по каждому TF (квадрант, угол, сила, score)",
    )
    verdict: str = Field(..., description="Итоговый текстовый вердикт")
    telegram_html_message: str = Field(
        ..., description="HTML-сообщение для Telegram (parse_mode=HTML)"
    )


class MacdTrendAnalysisOut(_Base):
    """Итоговый JSON-ответ GET /macd/trend/{ticker}.

    Объединяет MACD-анализ по всем доступным таймфреймам (1h, 2h, 4h, 1d):
    направление/силу тренда, квадрант, перцентиль силы, Markov-прогноз,
    MWU composite score, conviction и финальный trend score, плюс сводный
    консенсус и блок для Telegram.
    """

    symbol: str = Field(..., description="Тикер")
    asset_type: Literal["stock", "crypto", "moex", "commodity", "fx"] = Field(..., description="Тип актива")
    spot: float = Field(..., description="Текущая цена базиса")
    generated_at: datetime = Field(..., description="Метка времени анализа (UTC)")
    config: dict = Field(
        default_factory=dict,
        description="Использованная конфигурация AnalyzerConfig",
    )
    timeframes: list[MacdTrendTimeframeOut] = Field(
        ..., min_length=1, description="Анализ по таймфреймам, от младшего к старшему"
    )
    consensus_trend: Literal["BULLISH", "BEARISH", "RANGE"] = Field(
        ..., description="Сводный тренд (взвешенный по таймфреймам)"
    )
    weights: dict[str, float] = Field(
        default_factory=dict, description="Веса таймфреймов в консенсусе"
    )
    summarize: Optional[MacdTrendSummaryOut] = Field(
        None, description="Сводный отчёт + HTML для Telegram"
    )

    @model_validator(mode="after")
    def _round(self) -> "MacdTrendAnalysisOut":
        self.spot = round(self.spot, 2)
        return self


def macd_trend_to_schema(
    analyses: list,
    *,
    symbol: str,
    asset_type: str,
    spot: float,
    generated_at: datetime,
    consensus_trend: str,
    weights: dict,
    summarize: Optional[dict],
    config: Optional[object] = None,
) -> MacdTrendAnalysisOut:
    """Маппер списка ``(tf, df, BarResult)`` → Pydantic-схема.

    ``analyses`` — список кортежей ``(timeframe, dataframe, BarResult)`` из
    :class:`gex.macd_trend_service.MacdTrendService`.
    """
    import dataclasses

    tf_outs: list[MacdTrendTimeframeOut] = []
    for tf, df, bar in analyses:
        final = bar.final_trend_score
        trend = "BULLISH" if (final is not None and final > 1e-9) else (
            "BEARISH" if (final is not None and final < -1e-9) else "RANGE"
        )
        quad = bar.quadrant.value if bar.quadrant is not None else None
        tf_outs.append(
            MacdTrendTimeframeOut(
                timeframe=tf,
                n_bars=int(len(df)),
                avg_value=bar.avg_value,
                position=bar.position,
                zero_cross=bar.zero_cross,
                norm_slope=bar.norm_slope,
                angle_degrees=bar.angle_degrees,
                quadrant=quad,
                strength_instant=bar.strength_instant,
                strength_percentile=bar.strength_percentile,
                markov_next_state_probs=dict(bar.markov_next_state_probs),
                composite_score=bar.composite_score,
                conviction_multiplier=bar.conviction_multiplier,
                final_trend_score=bar.final_trend_score,
                kelly_fraction=bar.kelly_fraction,
                trend=trend,
            )
        )

    summary_obj = None
    if summarize is not None:
        summary_obj = MacdTrendSummaryOut(**summarize)

    config_dict: dict = {}
    if config is not None:
        config_dict = (
            dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)
        )

    return MacdTrendAnalysisOut(
        symbol=symbol,
        asset_type=asset_type,
        spot=spot,
        generated_at=generated_at,
        config=config_dict,
        timeframes=tf_outs,
        consensus_trend=consensus_trend,
        weights=weights,
        summarize=summary_obj,
    )


# ====================================================================== #