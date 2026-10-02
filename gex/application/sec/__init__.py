"""Слой приложения для SEC: чистый разбор XBRL и метрики отчётности (ring: application)."""

from gex.application.sec.ratios import (
    cagr,
    debt_to_equity,
    free_cash_flow,
    growth,
    margin,
    pe_ratio,
    safe_div,
)
from gex.application.sec.xbrl import AnnualFact, extract_first, extract_series, latest_value

__all__ = [
    "AnnualFact",
    "cagr",
    "debt_to_equity",
    "extract_first",
    "extract_series",
    "free_cash_flow",
    "growth",
    "latest_value",
    "margin",
    "pe_ratio",
    "safe_div",
]
