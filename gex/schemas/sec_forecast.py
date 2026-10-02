"""Pydantic v2 — схемы прогноза фундаментальных метрик и калькулятора оценки."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from ._base import _Base


# ══════════════════════════════════════════════════════════════════════ #
#  Прогноз метрик
# ══════════════════════════════════════════════════════════════════════ #
class ForecastPoint(_Base):
    """Фактическое значение метрики за период."""

    end: date = Field(..., description="Конец периода (ISO)")
    fy: int | None = Field(None, description="Фискальный год")
    value: float = Field(..., description="Значение метрики")


class GrowthInfo(_Base):
    """Темпы роста."""

    yoy: float | None = Field(None, description="Рост год к году (десятичная доля)")
    cagr_3y: float | None = Field(None, description="CAGR за 3 года")
    cagr_5y: float | None = Field(None, description="CAGR за 5 лет")
    quarterly: float | None = Field(None, description="Квартальный рост (только revenue)")


class ModelHorizons(_Base):
    """Прогноз по горизонтам 1/2/3 года."""

    h1: float | None = Field(None, description="Прогноз на 1 год")
    h2: float | None = Field(None, description="Прогноз на 2 года")
    h3: float | None = Field(None, description="Прогноз на 3 года")


class ForecastError(_Base):
    """Ошибка one-step-ahead прогноза на исторических данных."""

    rmse: float = Field(..., description="Корень из средней квадратичной ошибки")
    mae: float = Field(..., description="Средняя абсолютная ошибка")
    std: float = Field(..., description="Стандартное отклонение остатков (основа интервалов)")


class IntervalBand(_Base):
    """Доверительный интервал точки прогноза (80%)."""

    lower: float = Field(...)
    upper: float = Field(...)


class ScenarioBand(_Base):
    """Вероятностные сценарии будущего значения (±1σ·√h)."""

    pessimistic: float = Field(...)
    base: float = Field(...)
    optimistic: float = Field(...)


class ForecastModels(_Base):
    """Прогнозы трёх моделей + ошибка, интервалы и сценарии."""

    wma: ModelHorizons = Field(..., description="WMA-прогноз")
    linreg: ModelHorizons = Field(..., description="Линейная регрессия")
    combined: ModelHorizons = Field(..., description="Комбинированный α·WMA + (1−α)·LinReg")
    alpha: float = Field(..., description="Вес WMA в комбинированном прогнозе")
    error: ForecastError | None = Field(None, description="Ошибка прогноза на истории")
    interval: dict[str, IntervalBand] | None = Field(
        None, description="Доверительные интервалы по горизонтам (h1/h2/h3)"
    )
    scenarios: dict[str, ScenarioBand] | None = Field(
        None, description="Сценарии pessimistic/base/optimistic по горизонтам"
    )


class TrendRegime(_Base):
    """Характер движения метрики (рост/падение/ускорение/...)."""

    regime: str = Field(
        ...,
        description="growth | decline | acceleration | deceleration | reversal | stabilization | anomaly | unknown",
    )
    label: str = Field(..., description="Метка (RU)")
    direction: str = Field(..., description="up | down | flat")
    delta: float | None = Field(None, description="Последний темп роста")


class AnomalyInfo(_Base):
    """Индикатор аномального отклонения."""

    detected: bool = Field(..., description="Аномалия обнаружена")
    period: date | None = Field(None, description="Период аномального значения")
    change: float | None = Field(None, description="Изменение за период (десятичная доля)")
    note: str | None = Field(None, description="Пояснение")


class Scenarios(_Base):
    """Два сценария учёта аномалии (комбинированный прогноз)."""

    smooth: ModelHorizons = Field(..., description="Прогноз при сглаживании аномалии")
    keep: ModelHorizons = Field(..., description="Прогноз при полном учёте аномалии")


class QuarterlyHorizons(_Base):
    """Квартальные горизонты прогноза: 1/2/3 квартала."""

    q1: float | None = Field(None, description="Прогноз на 1 квартал")
    q2: float | None = Field(None, description="Прогноз на 2 квартала")
    q3: float | None = Field(None, description="Прогноз на 3 квартала")


class QuarterlyForecastModels(_Base):
    """Квартальные прогнозы моделей + интервалы."""

    wma: QuarterlyHorizons = Field(..., description="WMA-прогноз (кварталы)")
    linreg: QuarterlyHorizons = Field(..., description="Линейная регрессия")
    combined: QuarterlyHorizons = Field(..., description="Комбинированный прогноз")
    interval: dict[str, IntervalBand] | None = Field(None, description="Доверительные интервалы (q1/q2/q3)")
    error: ForecastError | None = Field(None, description="Ошибка one-step-ahead на квартальном ряду")
    alpha: float = Field(..., description="Вес WMA")


class QuarterlyForecast(_Base):
    """Квартальный ряд метрики (3M) с прогнозом — для переключателя «Год / Квартал»."""

    history: list[ForecastPoint] = Field(default_factory=list, description="Факт по кварталам")
    forecast: QuarterlyForecastModels | None = Field(None, description="Прогноз q4/q12/q20")


class MetricForecast(_Base):
    """Прогноз одной метрики."""

    history: list[ForecastPoint] = Field(default_factory=list, description="Факт по годам")
    growth: GrowthInfo | None = Field(None, description="Темпы роста")
    regime: TrendRegime = Field(..., description="Характер движения")
    forecast: ForecastModels | None = Field(None, description="Прогнозы моделей")
    quarterly: QuarterlyForecast | None = Field(None, description="Квартальный ряд (переключатель Год/Квартал)")
    anomaly: AnomalyInfo | None = Field(None, description="Аномалия")
    scenarios: Scenarios | None = Field(None, description="Сценарии учёта аномалии")


class PegInfo(_Base):
    """PEG: P/E ÷ темп роста прибыли."""

    growth: float | None = Field(None, description="Темп роста EPS (CAGR 3y/5y, иначе YoY)")
    pe: float | None = Field(None, description="Текущий P/E")
    peg: float | None = Field(None, description="Текущий PEG")
    warning: str | None = Field(None, description="Предупреждение (рост ≤ 0 и т.п.)")


class ValuationBase(_Base):
    """Базовые значения для калькулятора (текущий сценарий тикера)."""

    price: float | None = Field(None, description="Цена акции, USD")
    pe: float | None = Field(None, description="P/E")
    eps: float | None = Field(None, description="EPS, USD")
    earnings: float | None = Field(None, description="Чистая прибыль, USD")
    shares: float | None = Field(None, description="Акции в обращении, шт")
    market_cap: float | None = Field(None, description="Капитализация, USD")


class ForecastParameters(_Base):
    """Параметры прогнозной модели."""

    horizon_years: int = Field(..., ge=1, le=10)
    window: int = Field(..., ge=2, le=10, description="Окно WMA")
    alpha: float = Field(..., ge=0.0, le=1.0, description="Вес WMA в комбинации")
    anomaly_threshold: float = Field(..., ge=0.2, description="Порог аномалии (десятичная доля)")
    anomaly_mode: str = Field(..., description="both | smooth | keep")
    last_report_weight: float = Field(
        ..., ge=0.6, le=1.0, description="Влияние последнего отчёта на изменение прогноза (0.6 = 60% базово, 1.0 = 100%)"
    )


class PriceCrossValidation(_Base):
    """Кросс-валидация квартального прогноза цены с годовым."""

    adjusted: bool = Field(False, description="Квартальный прогноз скорректирован по годовой траектории")
    codes: list[str] = Field(default_factory=list, description="Коды правок: direction | pace")
    notes: list[str] = Field(default_factory=list, description="Пояснения правок")
    annual_implied: QuarterlyHorizons | None = Field(
        None, description="Годовая траектория, пересчитанная на кварталы (price·(1+g₁г)^(h/4))"
    )


class QuarterlyPriceForecast(_Base):
    """Прогноз цены на 1/2/3 квартала (TTM EPS × P/E × quality, согласован с годовым)."""

    q1: float | None = Field(None, description="Прогноз цены на 1 квартал, USD")
    q2: float | None = Field(None, description="Прогноз цены на 2 квартала, USD")
    q3: float | None = Field(None, description="Прогноз цены на 3 квартала, USD")
    pe_base: float | None = Field(None, description="P/E по TTM EPS")
    eps_ttm: float | None = Field(None, description="EPS за последние 4 квартала (факт)")
    quality: dict[str, float] | None = Field(None, description="Мультипликатор качества по кварталам")
    price_growth: float | None = Field(None, description="CAGR цены за 5 лет (десятичная доля)")
    cross_validation: PriceCrossValidation | None = Field(None, description="Согласование с годовым прогнозом")
    method: str | None = Field(None, description="Описание метода")


class PriceForecastModel(_Base):
    """Прогноз цены акции: EPS-прогноз × P/E × quality + поправка на 5-летний рост цены."""

    h1: float | None = Field(None, description="Прогноз цены на 1 год, USD")
    h2: float | None = Field(None, description="Прогноз цены на 2 года, USD")
    h3: float | None = Field(None, description="Прогноз цены на 3 года, USD")
    pe_base: float | None = Field(None, description="Текущий P/E (база)")
    quality: dict[str, float] | None = Field(
        None, description="Мультипликатор качества по горизонтам (revenue/EPS/FCF)"
    )
    price_growth: float | None = Field(
        None, description="Среднегодовой рост цены за 5 лет (CAGR, десятичная доля)"
    )
    method: str | None = Field(None, description="Описание метода")
    quarterly: QuarterlyPriceForecast | None = Field(
        None, description="Квартальный прогноз цены (переключатель Год/Квартал)"
    )


class SharesReconciliation(_Base):
    """Сверка количества акций: SEC EDGAR vs Finnhub (company profile2)."""

    ticker: str = Field(..., description="Тикер")
    source: str | None = Field(None, description="Источник: sec | finnhub | null")
    shares: float | None = Field(None, description="Количество акций (итоговое), шт")
    match: bool | None = Field(None, description="SEC и Finnhub согласованы (в пределах tolerance)")
    finnhub_raw: float | None = Field(None, description="Сырое значение Finnhub (масштаб неизвестен)")
    finnhub_shares: float | None = Field(None, description="Finnhub в подобранном масштабе, шт")
    finnhub_scale: float | None = Field(None, description="Подобранный масштаб (1/1e3/1e6/1e9)")
    diff_pct: float | None = Field(None, description="Относительная разница SEC vs Finnhub")
    warning: str | None = Field(None, description="Предупреждение (расхождение/фолбэк)")


class ForecastResponse(_Base):
    """Ответ GET /companies/{ticker}/forecast."""

    ticker: str = Field(...)
    cik: str | None = Field(None)
    price: dict | None = Field(None, description="Текущая цена (yfinance)")
    parameters: ForecastParameters = Field(...)
    metrics: dict[str, MetricForecast] = Field(..., description="Прогнозы по метрикам")
    quarterly: dict[str, QuarterlyForecast] = Field(
        default_factory=dict, description="Квартальные ряды (revenue/eps/fcf/operating_margin)"
    )
    peg: PegInfo | None = Field(None, description="PEG")
    valuation: ValuationBase | None = Field(None, description="Базовый сценарий для калькулятора")
    price_forecast: PriceForecastModel | None = Field(None, description="Прогноз цены акции (WMA+LinReg EPS × P/E × quality + поправка на 5-летний рост цены)")


# ══════════════════════════════════════════════════════════════════════ #
#  Калькулятор оценки
# ══════════════════════════════════════════════════════════════════════ #
class ValuationRequest(_Base):
    """Вход калькулятора: треугольник сценариев Цена ↔ P/E ↔ Прибыль.

    Основная связь ``P = (Прибыль / S) × P/E``: три переменные связаны одной
    формулой — выбирается режим, какие две задаём и что находим:

    * ``price_pe_to_earnings`` — цена + P/E → требуемая прибыль;
    * ``earnings_pe_to_price`` — прибыль + P/E → справедливая цена;
    * ``price_earnings_to_pe`` — цена + прибыль → подразумеваемый P/E;
      (в явных режимах ``shares`` обязателен);
    * ``auto`` — любые два из price/pe/eps (обратная совместимость).

    ``growth`` — десятичная доля (0.15 = 15%), нужен только для PEG.
    """

    mode: Literal["auto", "price_pe_to_earnings", "earnings_pe_to_price", "price_earnings_to_pe"] = Field(
        "auto", description="Режим калькулятора"
    )
    price: float | None = Field(None, gt=0, description="Цена акции, USD")
    pe: float | None = Field(None, gt=0, description="P/E")
    eps: float | None = Field(None, description="EPS, USD (может быть ≤ 0)")
    earnings: float | None = Field(None, gt=0, description="Чистая прибыль, USD")
    shares: float | None = Field(None, gt=0, description="Акции в обращении, шт")
    growth: float | None = Field(None, description="Темп роста прибыли (0.15 = 15%)")

    @model_validator(mode="after")
    def _check_mode_inputs(self) -> ValuationRequest:
        if self.mode == "price_pe_to_earnings":
            if self.price is None or self.pe is None:
                raise ValueError("Режим 'цена + P/E → прибыль': укажите price и pe")
            if self.shares is None:
                raise ValueError("Режим 'цена + P/E → прибыль': укажите shares (количество акций)")
        elif self.mode == "earnings_pe_to_price":
            if self.earnings is None or self.pe is None:
                raise ValueError("Режим 'прибыль + P/E → цена': укажите earnings и pe")
            if self.shares is None:
                raise ValueError("Режим 'прибыль + P/E → цена': укажите shares (количество акций)")
        elif self.mode == "price_earnings_to_pe":
            if self.price is None or self.earnings is None:
                raise ValueError("Режим 'цена + прибыль → P/E': укажите price и earnings")
            if self.shares is None:
                raise ValueError("Режим 'цена + прибыль → P/E': укажите shares (количество акций)")
        else:  # auto
            filled = sum(1 for v in (self.price, self.pe, self.eps) if v is not None)
            if filled < 2:
                raise ValueError(
                    "Укажите минимум два из трёх: price, P/E, EPS (или eps через earnings/shares)"
                )
        return self


class ValuationResponse(_Base):
    """Ответ калькулятора: все показатели достроены + режим и целевая величина."""

    mode: str = Field(..., description="Режим калькулятора")
    solved: str | None = Field(None, description="Что найдено: earnings | price | pe")
    price: float | None = Field(None)
    pe: float | None = Field(None)
    eps: float | None = Field(None)
    earnings: float | None = Field(None)
    shares: float | None = Field(None)
    market_cap: float | None = Field(None)
    growth: float | None = Field(None)
    peg: float | None = Field(None)
    peg_warning: str | None = Field(None)
