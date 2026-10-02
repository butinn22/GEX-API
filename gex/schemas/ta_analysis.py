"""Pydantic v2 — схемы сериализации для API."""

from datetime import date, datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator

from ._base import _Base


# ====================================================================== #
class TAIndicatorsOut(_Base):
    """Технические индикаторы на последнем баре таймфрейма."""

    ema20: float = Field(..., description="EMA-20")
    ema50: float = Field(..., description="EMA-50")
    ema200: float = Field(..., description="EMA-200")
    rsi: float = Field(..., ge=0.0, le=100.0, description="RSI(14) Wilder, 0..100")
    macd: float = Field(..., description="Линия MACD(12,26)")
    macd_signal: float = Field(..., description="Сигнальная линия MACD(9)")
    macd_hist: float = Field(..., description="Гистограмма MACD")
    macd_bull_cross: bool = Field(False, description="Бычье пересечение MACD на последнем баре")
    macd_bear_cross: bool = Field(False, description="Медвежье пересечение MACD на последнем баре")
    ema_bull_stack: bool = Field(False, description="Идеальный бычий стек EMA (20>50>200)")
    ema_bear_stack: bool = Field(False, description="Идеальный медвежий стек EMA (20<50<200)")

    @model_validator(mode="after")
    def _round(self) -> "TAIndicatorsOut":
        self.ema20 = round(self.ema20, 2)
        self.ema50 = round(self.ema50, 2)
        self.ema200 = round(self.ema200, 2)
        self.rsi = round(self.rsi, 1)
        self.macd = round(self.macd, 4)
        self.macd_signal = round(self.macd_signal, 4)
        self.macd_hist = round(self.macd_hist, 4)
        return self


class MomentumOut(_Base):
    """Сила тренда по форме и объёму последних 20 свечей."""

    side: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(
        ..., description="Сторона импульса"
    )
    strength: float = Field(..., ge=0.0, le=100.0, description="Сила импульса, %")
    close_to_close_pct: float = Field(
        ..., description="Среднее %-изменение Close бар-к-бару по окну"
    )
    range_pct: float = Field(
        ..., description="Среднее %-расширение от прошлого Low до текущего High"
    )
    avg_body_pct: float = Field(..., description="Средний размер тела свечи, %")
    volume_trend: float = Field(
        ...,
        description="Коэффициент подтверждения объёмом: >1 = объём на стороне тренда",
    )
    net_move_pct: float = Field(..., description="Полное %-движение за окно")
    n_bars: int = Field(..., ge=0, description="Число баров в окне")

    @model_validator(mode="after")
    def _round(self) -> "MomentumOut":
        self.strength = round(self.strength, 1)
        self.close_to_close_pct = round(self.close_to_close_pct, 4)
        self.range_pct = round(self.range_pct, 4)
        self.avg_body_pct = round(self.avg_body_pct, 4)
        self.volume_trend = round(self.volume_trend, 3)
        self.net_move_pct = round(self.net_move_pct, 3)
        return self


class TrendOut(_Base):
    """Оценка рыночного тренда: свинги HH/LL + candle-momentum + стек EMA."""

    direction: Literal["BULLISH", "BEARISH", "RANGE"] = Field(
        ..., description="Направление тренда"
    )
    strength: float = Field(..., ge=0.0, le=100.0, description="Сила тренда, %")
    recent_high: float = Field(..., description="Недавний максимум")
    recent_low: float = Field(..., description="Недавний минимум")
    swing_highs: list[float] = Field(
        default_factory=list, description="Подтверждённые свинг-хай пивоты"
    )
    swing_lows: list[float] = Field(
        default_factory=list, description="Подтверждённые свинг-лоу пивоты"
    )
    higher_highs: int = Field(0, ge=0, description="Число Higher High")
    higher_lows: int = Field(0, ge=0, description="Число Higher Low")
    lower_highs: int = Field(0, ge=0, description="Число Lower High")
    lower_lows: int = Field(0, ge=0, description="Число Lower Low")
    momentum: Optional[MomentumOut] = Field(
        None, description="Candle-momentum (последние 20 свечей)"
    )

    @model_validator(mode="after")
    def _round(self) -> "TrendOut":
        self.strength = round(self.strength, 1)
        self.recent_high = round(self.recent_high, 2)
        self.recent_low = round(self.recent_low, 2)
        self.swing_highs = [round(s, 2) for s in self.swing_highs]
        self.swing_lows = [round(s, 2) for s in self.swing_lows]
        return self


class ReversalProbOut(_Base):
    """Стохастическая вероятность смены тренда (Markov + Monte-Carlo)."""

    p_reversal: float = Field(
        ..., ge=0.0, le=1.0, description="Итоговая вероятность разворота тренда"
    )
    p_markov: float = Field(
        ..., ge=0.0, le=1.0, description="Компонента: эмпирическая цепь Маркова"
    )
    p_mc: float = Field(
        ..., ge=0.0, le=1.0, description="Компонента: Монте-Карло GBM-путей"
    )
    method: str = Field(..., description="Метод оценки ('Markov+MonteCarlo')")
    n_paths: int = Field(..., ge=0, description="Число смоделированных путей")
    horizon_bars: int = Field(..., ge=1, description="Горизонт прогноза, баров")
    weights: dict[str, float] = Field(
        default_factory=dict, description="Веса компонентов Markov/MC"
    )

    @model_validator(mode="after")
    def _round(self) -> "ReversalProbOut":
        self.p_reversal = round(self.p_reversal, 3)
        self.p_markov = round(self.p_markov, 3)
        self.p_mc = round(self.p_mc, 3)
        return self


class MultiTFConfirmationOut(_Base):
    """Подтверждение тренда старшего таймфрейма младшими."""

    base_timeframe: str = Field(..., description="Базовый (старший) таймфрейм")
    base_direction: str = Field(..., description="Направление базового TF")
    agreeing: list[str] = Field(
        default_factory=list, description="Младшие TF, согласные с базовым"
    )
    disagreeing: list[str] = Field(
        default_factory=list, description="Младшие TF, противоречащие базовому"
    )
    neutral: list[str] = Field(default_factory=list, description="Младшие TF в RANGE")
    confirmation_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Взвешенная доля согласных младших TF"
    )
    adjustment: float = Field(
        ..., ge=-50.0, le=50.0, description="Поправка к силе тренда (−50..+50)"
    )

    @model_validator(mode="after")
    def _round(self) -> "MultiTFConfirmationOut":
        self.confirmation_ratio = round(self.confirmation_ratio, 3)
        self.adjustment = round(self.adjustment, 1)
        return self


class DivergenceItemOut(_Base):
    """Одна дивергенция осциллятора с ценой."""

    oscillator: str = Field(..., description="Осциллятор ('RSI' или 'MACD')")
    type: Literal["BULLISH", "BEARISH"] = Field(..., description="Тип дивергенции")
    type_label: str = Field(..., description="Человекочитаемый тип (рус.)")
    strength: float = Field(..., ge=0.0, le=100.0, description="Сила дивергенции, %")
    bars_ago: int = Field(..., ge=0, description="Сколько баров назад сформировалась")

    @model_validator(mode="after")
    def _round(self) -> "DivergenceItemOut":
        self.strength = round(self.strength, 1)
        return self


class DivergenceSummaryOut(_Base):
    """Сводка дивергенций осцилляторов на таймфрейме."""

    has_divergence: bool = Field(False, description="Есть ли дивергенции")
    items: list[DivergenceItemOut] = Field(
        default_factory=list, description="Список дивергенций"
    )


class TimeframeOut(_Base):
    """Полный анализ одного таймфрейма: индикаторы + тренд + вероятность."""

    timeframe: str = Field(..., description="Таймфрейм ('1h','2h','4h','1d')")
    last_close: float = Field(..., description="Цена закрытия последнего бара")
    n_bars: int = Field(..., ge=0, description="Число баров в истории")
    indicators: TAIndicatorsOut
    trend: TrendOut
    reversal: ReversalProbOut
    confirmation: Optional[MultiTFConfirmationOut] = Field(
        None, description="Подтверждение младшими TF (только в сводном анализе)"
    )
    divergence: Optional[DivergenceSummaryOut] = Field(
        None, description="Дивергенции осцилляторов (RSI/MACD)"
    )

    @model_validator(mode="after")
    def _round(self) -> "TimeframeOut":
        self.last_close = round(self.last_close, 2)
        return self


# ====================================================================== #
#  TA Summarize-блок (логичный объёмный сводный отчёт)
# ====================================================================== #
class TASRLevelOut(_Base):
    """Уровни поддержки/сопротивления."""

    support: float
    resistance: float

    @model_validator(mode="after")
    def _round(self) -> "TASRLevelOut":
        self.support = round(self.support, 2)
        self.resistance = round(self.resistance, 2)
        return self


class TAReversalSummaryOut(_Base):
    """Сводка потенциала разворота."""

    consensus_p_reversal: float = Field(..., ge=0.0, le=1.0)
    assessment: str = Field(..., description="Словесная оценка вероятности")
    by_timeframe: dict[str, dict] = Field(
        default_factory=dict, description="Вероятности разворота по TF"
    )

    @model_validator(mode="after")
    def _round(self) -> "TAReversalSummaryOut":
        self.consensus_p_reversal = round(self.consensus_p_reversal, 3)
        return self


class TASummaryOut(_Base):
    """«Объёмный» summarize-блок тех. анализа.

    Человекочитаемый сводный отчёт: тренд и сила, поддержка/сопротивление,
    потенциал разворота, состояние осцилляторов, дивергенции, multi-TF
    подтверждение и итоговый вердикт. Все строки — на русском.
    """

    headline: str = Field(..., description="Однострочный вердикт (эмодзи + тренд)")
    trend: dict = Field(..., description="Сводка по тренду (направление, сила, по TF)")
    support_resistance: dict = Field(
        ..., description="Поддержка/сопротивление (общая + по TF)"
    )
    reversal: dict = Field(..., description="Потенциал разворота (общий + по TF)")
    oscillators: dict = Field(..., description="RSI/MACD по TF")
    divergences: dict = Field(..., description="Дивергенции по TF + чистый сигнал")
    multi_tf: dict = Field(..., description="Multi-TF подтверждение (по старшему TF)")
    verdict: str = Field(..., description="Итоговый текстовый вердикт с рекомендацией")
    telegram_html_message: str = Field(
        ..., description="HTML-сообщение для Telegram (parse_mode=HTML)"
    )


class TAAnalysisOut(_Base):
    """Итоговый JSON-ответ GET /ta/{ticker} — тех. анализ по 4 таймфреймам.

    Содержит:
    - покадровый анализ (1h, 2h, 4h, 1d): индикаторы, тренд, вероятность разворота;
    - сводный консенсус-тренд и взвешенную вероятность разворота
      (старшие таймфреймы тяжелее);
    - ``summarize`` — объёмный сводный отчёт + HTML для Telegram.
    """

    symbol: str
    spot: float = Field(..., description="Текущая цена базиса")
    generated_at: datetime = Field(..., description="Метка времени анализа (UTC)")
    timeframes: list[TimeframeOut] = Field(
        ..., min_length=1, description="Анализ по таймфреймам, от младшего к старшему"
    )
    consensus_trend: Literal["BULLISH", "BEARISH", "RANGE"] = Field(
        ..., description="Сводный тренд (взвешенный по таймфреймам)"
    )
    consensus_p_reversal: float = Field(
        ..., ge=0.0, le=1.0, description="Взвешенная вероятность разворота"
    )
    weights: dict[str, float] = Field(
        default_factory=dict, description="Веса таймфреймов в консенсусе"
    )
    summarize: Optional[TASummaryOut] = Field(
        None, description="Объёмный сводный отчёт (тренд, уровни, разворот, дивергенции)"
    )

    @model_validator(mode="after")
    def _round(self) -> "TAAnalysisOut":
        self.spot = round(self.spot, 2)
        self.consensus_p_reversal = round(self.consensus_p_reversal, 3)
        return self


# ====================================================================== #
#  Маппер TA: dataclass → Pydantic
#  (импорты ta вынесены внутрь, чтобы избежать циклической зависимости
#   schemas → ta → (нет) — та не импортирует schemas, но ta_fetcher/ta
#   импортируются сервисом; держим импорт локальным для чистоты графа.)
# ====================================================================== #
def _momentum_to_schema(m) -> Optional[MomentumOut]:
    """MomentumStrength (dataclass) → MomentumOut. None → None."""
    if m is None:
        return None
    return MomentumOut(
        side=m.side,
        strength=m.strength,
        close_to_close_pct=m.close_to_close_pct,
        range_pct=m.range_pct,
        avg_body_pct=m.avg_body_pct,
        volume_trend=m.volume_trend,
        net_move_pct=m.net_move_pct,
        n_bars=m.n_bars,
    )


def _confirmation_to_schema(c) -> Optional[MultiTFConfirmationOut]:
    """TimeframeConfirmation (dataclass) → MultiTFConfirmationOut. None → None."""
    if c is None:
        return None
    return MultiTFConfirmationOut(
        base_timeframe=c.base_timeframe,
        base_direction=c.base_direction,
        agreeing=list(c.agreeing),
        disagreeing=list(c.disagreeing),
        neutral=list(c.neutral),
        confirmation_ratio=c.confirmation_ratio,
        adjustment=c.adjustment,
    )


def timeframe_to_schema(tf) -> TimeframeOut:
    """TimeframeAnalysis (dataclass из gex.ta) → TimeframeOut.

    Parameters
    ----------
    tf : TimeframeAnalysis
        Результат :func:`gex.ta.analyze_timeframe`.
    """
    return TimeframeOut(
        timeframe=tf.timeframe,
        last_close=tf.last_close,
        n_bars=tf.n_bars,
        indicators=TAIndicatorsOut(
            ema20=tf.indicators.ema20,
            ema50=tf.indicators.ema50,
            ema200=tf.indicators.ema200,
            rsi=tf.indicators.rsi,
            macd=tf.indicators.macd,
            macd_signal=tf.indicators.macd_signal,
            macd_hist=tf.indicators.macd_hist,
            macd_bull_cross=tf.indicators.macd_bull_cross,
            macd_bear_cross=tf.indicators.macd_bear_cross,
            ema_bull_stack=tf.indicators.ema_bull_stack,
            ema_bear_stack=tf.indicators.ema_bear_stack,
        ),
        trend=TrendOut(
            direction=tf.trend.direction,
            strength=tf.trend.strength,
            recent_high=tf.trend.recent_high,
            recent_low=tf.trend.recent_low,
            swing_highs=tf.trend.swing_highs[-8:],   # последние для краткости ответа
            swing_lows=tf.trend.swing_lows[-8:],
            higher_highs=tf.trend.higher_highs,
            higher_lows=tf.trend.higher_lows,
            lower_highs=tf.trend.lower_highs,
            lower_lows=tf.trend.lower_lows,
            momentum=_momentum_to_schema(tf.trend.momentum),
        ),
        reversal=ReversalProbOut(
            p_reversal=tf.reversal.p_reversal,
            p_markov=tf.reversal.p_markov,
            p_mc=tf.reversal.p_mc,
            method=tf.reversal.method,
            n_paths=tf.reversal.n_paths,
            horizon_bars=tf.reversal.horizon_bars,
            weights=tf.reversal.weights,
        ),
        confirmation=_confirmation_to_schema(getattr(tf, "confirmation", None)),
        divergence=_divergence_to_schema(getattr(tf, "divergence", None)),
    )


def _divergence_to_schema(d) -> Optional[DivergenceSummaryOut]:
    """DivergenceInfo (dataclass) → DivergenceSummaryOut. None → None."""
    if d is None:
        return None
    return DivergenceSummaryOut(
        has_divergence=d.has_bullish or d.has_bearish,
        items=[
            DivergenceItemOut(
                oscillator=div.oscillator,
                type=div.type,
                type_label=_DIVERGENCE_LABEL_RU.get(div.type, div.type),
                strength=div.strength,
                bars_ago=div.bars_ago,
            )
            for div in d.divergences
        ],
    )


# Подписи типов дивергенций (рус.) для маппера.
_DIVERGENCE_LABEL_RU: dict[str, str] = {
    "BULLISH": "Бычья 🔼",
    "BEARISH": "Медвежья 🔻",
}


# ====================================================================== #
#  Сканер: схемы результатов автосканирования TA + GEX
# ====================================================================== #