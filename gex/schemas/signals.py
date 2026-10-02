"""Pydantic v2 — схемы сериализации для API."""

from datetime import date, datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator

from ._base import _Base


# ====================================================================== #
#  Торговые сигналы: EMA Multi-Filter стратегия + GEX + verification
# ====================================================================== #
class GEXContextOut(_Base):
    """Снимок GEX-профиля, используемый при формировании сигнала."""

    regime: Literal["POSITIVE", "NEGATIVE"] = Field(
        ..., description="GEX-режим: POSITIVE (подавление волатильности) / NEGATIVE (усиление трендов)"
    )
    gamma_flip: Optional[float] = Field(None, description="Gamma Flip — уровень смены режима")
    net_gex: float = Field(..., description="Суммарный GEX, $/%spot")
    z_score: Optional[float] = Field(None, description="Z-score отклонения спота от Gamma Flip")
    call_wall: Optional[float] = Field(None, description="Call Wall (сопротивление)")
    put_wall: Optional[float] = Field(None, description="Put Wall (поддержка)")
    direction: Optional[Literal["BULLISH", "BEARISH", "NEUTRAL"]] = Field(
        None, description="GEX-направление"
    )
    confidence: Optional[float] = Field(None, ge=0.0, le=100.0, description="GEX-уверенность, %")


class SignalRecordOut(_Base):
    """Один исторический сигнал (entry-событие из backtest по истории)."""

    timestamp: datetime = Field(..., description="Время бара входа")
    action: Literal["buy", "sell", "hold"] = Field(..., description="Действие")
    reason: str = Field(..., description="Причина/тип сигнала (long_entry, short_entry, ...)")
    order_type: str = Field(..., description="entry_long / entry_short / ...")
    price: float = Field(..., description="Цена входа (close бара)")
    entry_score: float = Field(..., ge=0.0, le=1.0, description="Сила сетапа, 0..1")
    gex_multiplier: float = Field(..., description="GEX-множитель (0.7..1.30)")
    gex_reason: str = Field(..., description="Объяснение влияния GEX")
    verification_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Score verification-движка (None = не запускался)"
    )
    confidence_class: Literal["high", "medium", "low", "none"] = Field(
        ..., description="Класс уверенности сигнала"
    )
    atr: Optional[float] = Field(None, description="ATR на момент входа")
    tp_price: Optional[float] = Field(None, description="Цена тейк-профита (ATR-адаптивная)")
    sl_price: Optional[float] = Field(None, description="Цена стоп-лосса (trailing)")


class CurrentSignalOut(_Base):
    """Текущий (на последнем баре) сигнал с полным анализом."""

    action: Literal["buy", "sell", "hold"] = Field(..., description="Действие")
    reason: str = Field(..., description="Причина/тип сигнала")
    order_type: str = Field(..., description="entry_long / entry_short / hold / ...")
    price: float = Field(..., description="Текущая цена (close последнего бара)")
    entry_score: float = Field(..., ge=0.0, le=1.0, description="Сила сетапа, 0..1")
    gex_multiplier: float = Field(..., description="GEX-множитель (0.7..1.30)")
    gex_reason: str = Field(..., description="Объяснение влияния GEX")
    verification_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Score verification-движка"
    )
    verification_regime: Optional[str] = Field(
        None, description="Режим рынка по verification (strong_trend/weak_trend/...)"
    )
    confidence_class: Literal["high", "medium", "low", "none"] = Field(
        ..., description="Итоговый класс уверенности"
    )
    atr: Optional[float] = Field(None, description="Текущий ATR")
    tp_price: Optional[float] = Field(None, description="Тейк-профит")
    sl_price: Optional[float] = Field(None, description="Стоп-лосс (trailing)")
    trend_coefficient: Optional[float] = Field(None, description="Коэффициент тренда стратегии")

    # --- TA-контекст для подробного описания (на последнем баре) ---
    rsi: Optional[float] = Field(None, description="RSI(14) по close")
    ema_alignment: Optional[Literal["bullish", "bearish", "mixed"]] = Field(
        None, description="Выстраивание EMA fast/mid/slow"
    )
    vwap_position: Optional[Literal["above", "below", "at"]] = Field(
        None, description="Позиция цены относительно адаптивного VWAP"
    )
    trend_direction: Optional[float] = Field(
        None, description="Направление тренда стратегии (1.0 / -1.0 / 0.0)"
    )
    is_flat: Optional[bool] = Field(None, description="Флэт-зона (боковик)")
    spot_vs_gamma_flip: Optional[Literal["above", "below", "at"]] = Field(
        None, description="Позиция спота относительно Gamma Flip"
    )
    rr_ratio: Optional[float] = Field(
        None, description="Risk/reward соотношение TP/SL (None если нет входа)"
    )


class PositionOut(_Base):
    """Текущая позиция по машине состояний стратегии (сканеры сигналов).

    Позиция ведётся по сигнальным колонкам (зеркало ``evaluate()``):
    вход обязателен для любого выхода/добавления; без позиции — только входы.
    """

    side: Literal["flat", "long", "short"] = Field(
        "flat", description="Сторона текущей позиции (flat = позиции нет)"
    )
    avg_price: Optional[float] = Field(
        None, description="Средняя цена позиции (вход 1.0 + добавления по 0.1)"
    )
    since: Optional[datetime] = Field(
        None, description="Время открытия позиции (бар входа)"
    )


class SignalAnalysisOut(_Base):
    """Корневой ответ GET /signals/{ticker}."""
    symbol: str = Field(..., description="Тикер")
    asset_type: Literal["stock", "crypto", "moex", "commodity", "fx"] = Field(..., description="Тип актива")
    timeframe: str = Field(..., description="Таймфрейм анализа (1d, 4h, ...)")
    generated_at: datetime = Field(..., description="Время генерации (UTC)")
    spot: float = Field(..., description="Текущая цена")
    bars_analyzed: int = Field(..., ge=0, description="Число баров в истории")

    gex_context: Optional[GEXContextOut] = Field(
        None, description="GEX-контекст (None если GEX недоступен)"
    )
    current_signal: CurrentSignalOut = Field(..., description="Текущий сигнал")
    recent_signals: list[SignalRecordOut] = Field(
        default_factory=list, description="Последние N сигналов из истории"
    )
    position: PositionOut = Field(
        default_factory=PositionOut,
        description="Текущая позиция по машине состояний (вход/выход с очерёдностью)",
    )
    snapshot: Optional[Any] = Field(
        default=None,
        exclude=True,
        description="Служебный снапшот истории (переигровка машины позиции "
                    "в сканере); в ответы API не сериализуется",
    )
    regime: Optional["TrendRegimeOut"] = Field(
        None,
        description=("Режим рынка на окне 200 баров (тренд/флэт). "
                     "None = истории не хватило — верифицировать нечем"),
    )


# ====================================================================== #
#  Режим рынка: тренд / флэт на 200 барах (ATR + BBW + z-цена)
# ====================================================================== #
class TrendRegimeOut(_Base):
    """Вердикт детектора тренда/флэта (:mod:`gex.trend_regime`).

    ``metrics`` — слайдер-НЕЗАВИСИМЫЕ сырые числа (z, atr_change_n,
    bbw_change_n, ранги и т.д.); остальное — вердикт при текущем слайдере.
    """

    state: Literal["UP", "DOWN", "FLAT"] = Field(..., description="Состояние рынка")
    direction: int = Field(..., description="+1 вверх / -1 вниз / 0 флэт")
    trend_strength: float = Field(..., ge=0.0, le=100.0, description="Сила тренда, 0..100")
    flat_score: float = Field(..., ge=0.0, le=100.0, description="Оценка боковика, 0..100")
    is_flat: bool = Field(..., description="Флэт подтверждён (входы отсекаются)")
    is_chop: bool = Field(..., description="Высокая волатильность без направления")
    allowed_flat_pct_n: Optional[float] = Field(
        None, description="Допустимое изменение цены за N баров внутри флэта, %"
    )
    sliders: dict[str, float] = Field(default_factory=dict, description="Применённые слайдеры")
    metrics: dict[str, Optional[float]] = Field(
        default_factory=dict, description="Сырые метрики (не зависят от слайдера)"
    )
    flat_components: dict[str, float] = Field(
        default_factory=dict, description="Вклад ATR / BBW / цены в flat_score"
    )
    weights: dict[str, float] = Field(default_factory=dict, description="Веса компонент")
    thresholds: dict[str, float] = Field(default_factory=dict, description="Пороги после слайдера")
    multipliers: dict[str, float] = Field(
        default_factory=dict, description="Мультипликаторы порогов (m_atr, m_bbw, m_pct)"
    )


SignalAnalysisOut.model_rebuild()


# ====================================================================== #
#  Трендовые линии (порт Pinescript v5 «Trend lines Andreu»)
#  + анализ тренда по углу линий и фракталам HH/HL/LH/LL