"""Pydantic v2 — схемы SEC fundamentals API (XBRL-показатели SEC EDGAR)."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field

from ._base import _Base


# ══════════════════════════════════════════════════════════════════════ #
#  Revenue (совместимая ручка)
# ══════════════════════════════════════════════════════════════════════ #
class RevenueRow(_Base):
    """Одна строка выручки за период (end → start)."""

    end: date = Field(..., description="Дата окончания периода (ISO)")
    start: date | None = Field(None, description="Дата начала периода (10-Q: 3M vs 9M накопит.)")
    val: float = Field(..., description="Выручка, USD")
    fy: int | None = Field(None, description="Фискальный год")
    fp: str | None = Field(None, description="Период внутри года: FY / Q1..Q4")
    form: str | None = Field(None, description="Форма отчётности: 10-K / 10-Q")
    frame: str | None = Field(None, description="XBRL frame (для кварталов)")
    filed: date | None = Field(None, description="Дата подачи отчётности (для дедупа)")
    tag: str = Field(..., description="XBRL-тег us-gaap")


class RevenueResponse(_Base):
    """Ответ GET /companies/{ticker}/revenue."""

    ticker: str = Field(..., description="Тикер (uppercase)")
    cik: str | None = Field(None, description="CIK компании (10 цифр)")
    period: Literal["FY", "Q", "ALL"] = Field(..., description="Запрошенный период")
    count: int = Field(..., ge=0, description="Число строк")
    source_tag: str | None = Field(None, description="Тег последней строки (актуальный)")
    rows: list[RevenueRow] = Field(default_factory=list, description="Строки выручки, по end↑")


# ══════════════════════════════════════════════════════════════════════ #
#  Fundamentals (полный набор показателей)
# ══════════════════════════════════════════════════════════════════════ #
class AnnualRow(_Base):
    """Один фискальный год: потоковые показатели + баланс на конец года."""

    end: date = Field(..., description="Конец фискального года (10-K)")
    fy: int | None = Field(None, description="Фискальный год")
    revenue: float | None = Field(None, description="Выручка, USD")
    net_income: float | None = Field(None, description="Чистая прибыль (NetIncomeLoss)")
    operating_income: float | None = Field(None, description="Операционная прибыль (EBIT)")
    eps_basic: float | None = Field(None, description="EPS basic, USD")
    eps_diluted: float | None = Field(None, description="EPS diluted, USD")
    cfo: float | None = Field(None, description="Операционный денежный поток, USD")
    capex: float | None = Field(None, description="CapEx, USD (знак как в XBRL)")
    free_cash_flow: float | None = Field(None, description="FCF = CFO − |CapEx|, USD")
    ebitda: float | None = Field(None, description="EBITDA = EBIT + D&A, USD")
    operating_margin: float | None = Field(None, description="Operating Margin = EBIT / Revenue")
    total_debt: float | None = Field(None, description="Total Debt на конец года, USD")
    net_debt: float | None = Field(None, description="Net Debt на конец года, USD")
    cash: float | None = Field(None, description="Cash + Cash Equivalents, USD")
    equity: float | None = Field(None, description="Stockholders Equity, USD")
    shares_outstanding: float | None = Field(None, description="Акции в обращении, шт")
    roe: float | None = Field(None, description="ROE = Net Income / Equity")


class BalanceSnapshot(_Base):
    """Последний доступный баланс (по всем формам, включая квартальные)."""

    end: date | None = Field(None, description="Дата баланса")
    cash: float | None = Field(None, description="Cash + Cash Equivalents, USD")
    st_investments: float | None = Field(None, description="Краткосрочные инвестиции, USD")
    total_debt: float | None = Field(None, description="Total Debt, USD")
    net_debt: float | None = Field(None, description="Net Debt = Debt − (Cash + ST inv.), USD")
    equity: float | None = Field(None, description="Stockholders Equity, USD")
    shares_outstanding: float | None = Field(None, description="Акции в обращении, шт")


class FundamentalsRatios(_Base):
    """Рыночные мультипликаторы (на основе текущей цены)."""

    pe: float | None = Field(None, description="P/E = Price / EPS последнего FY")
    ps: float | None = Field(None, description="P/S = Market Cap / Revenue последнего FY")
    net_debt_to_ebitda: float | None = Field(None, description="Net Debt / EBITDA (последний FY)")


class PriceInfo(_Base):
    """Текущая цена базиса."""

    value: float = Field(..., description="Цена, USD")
    source: str = Field("yfinance", description="Источник цены")
    currency: str = Field("USD", description="Валюта")


class FundamentalsResponse(_Base):
    """Ответ GET /companies/{ticker}/fundamentals."""

    ticker: str = Field(..., description="Тикер (uppercase)")
    cik: str | None = Field(None, description="CIK компании (10 цифр)")
    price: PriceInfo | None = Field(None, description="Текущая цена (yfinance)")
    market_cap: float | None = Field(None, description="Market Cap = Price × Shares, USD")
    balance: BalanceSnapshot = Field(..., description="Последний баланс")
    annual: list[AnnualRow] = Field(default_factory=list, description="Годовые ряды, по end↑")
    ratios: FundamentalsRatios = Field(..., description="Рыночные мультипликаторы")
