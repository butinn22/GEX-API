"""SEC EDGAR fundamentals routes: revenue + полный набор показателей.

Backend модуля фундаментального анализа:

* ``GET /companies/{ticker}/revenue`` — выручка компании из XBRL-фактов
  SEC EDGAR (годовые 10-K, квартальные 10-Q, или всё);
* ``GET /companies/{ticker}/fundamentals`` — полный набор: revenue,
  net income, operating income, EPS, FCF, EBITDA, Net Debt/EBITDA,
  Operating Margin, ROE, P/E, P/S (+ текущая цена из yfinance).

Данные проходят конвейер: Ticker → CIK resolver → EDGAR API → XBRL tag
normalizer → deduplicator (latest filed) → PostgreSQL (company_metrics) →
Redis SWR-кэш core-данных (TTL = SEC_FACTS_TTL_HOURS, max_age = 2×TTL).
Цена (yfinance) подмешивается на каждый запрос — её кэширует сам фетчер
(Redis, 300с), чтобы P/E/P/S не замораживались на TTL фундаментала.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from gex.auth.config import settings
from gex.auth.dependencies import require_subscription
from gex.adapters.cache.result_cache import cache_key, result_cache
from gex.schemas.sec_fundamentals import FundamentalsResponse, RevenueResponse
from gex.application.sec.sec_fundamentals import SecFundamentalsService

from ._helpers import handle

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_subscription("BASIC"))], tags=["sec-fundamentals"])

#: TTL SWR-кэша core-данных = свежести данных в PG (12ч), max_age = 24ч
_SEC_TTL = settings.SEC_FACTS_TTL_HOURS * 3600


def _make_service() -> SecFundamentalsService:
    from gex.deps import get_redis_client

    redis_client = get_redis_client()
    if redis_client and not redis_client.connected:
        redis_client = None
    return SecFundamentalsService(redis_client=redis_client)


@router.get("/companies/{ticker}/revenue", response_model=RevenueResponse)
def get_company_revenue(
    ticker: str,
    period: str = Query(
        "FY",
        pattern="^(FY|Q|ALL)$",
        description="Период: FY — годовые (10-K), Q — квартальные (10-Q), ALL — всё",
    ),
):
    """Выручка компании из XBRL-фактов SEC EDGAR.

    us-gaap теги revenue в порядке приоритета (фолбэк на финансовые теги
    для банков), дедуп по (end, start) → последняя подача (filed).
    """
    def _compute() -> dict:
        return _make_service().get_revenue(ticker, period)

    key = cache_key("res", "sec-revenue", ticker.strip().upper(), period)
    return result_cache.get(
        key, _SEC_TTL, lambda: handle(_compute, error_src="SEC EDGAR"), max_age=_SEC_TTL * 2
    )


@router.get("/companies/{ticker}/fundamentals", response_model=FundamentalsResponse)
def get_company_fundamentals(ticker: str):
    """Полный набор фундаментальных показателей компании из SEC EDGAR.

    Годовые ряды (revenue, net income, EBIT, EPS, FCF, EBITDA, margin,
    net debt, ROE) + последний баланс + рыночные мультипликаторы
    (P/E, P/S, Net Debt/EBITDA) на основе текущей цены из yfinance.
    """
    def _compute_core() -> dict:
        return _make_service().get_fundamentals_core(ticker)

    key = cache_key("res", "sec-fundamentals", ticker.strip().upper())
    core = result_cache.get(
        key, _SEC_TTL, lambda: handle(_compute_core, error_src="SEC EDGAR"), max_age=_SEC_TTL * 2
    )

    svc = _make_service()
    price = svc.fetch_price(ticker)
    return svc.with_market(core, price)
