"""SEC Forecast routes: прогноз метрик + сценарный калькулятор оценки.

Backend модуля «Прогноз фундаментальных метрик и сценарная оценка акций»:

* ``GET /companies/{ticker}/forecast`` — прогноз ключевых метрик (revenue,
  FCF, Net Debt/EBITDA, EPS, Operating Margin, ROE, PEG) моделями
  WMA + линейная регрессия (комбинированный α·WMA + (1−α)·LinReg),
  детект аномалий (порог настраиваемый) и два сценария учёта: сглаживать
  / учитывать полностью;
* ``POST /valuation/calculator`` — интерактивный калькулятор
  Цена ↔ P/E ↔ Прибыль/EPS (+ PEG): треугольник сценариев.

Оба эндпоинта ПУБЛИЧНЫЕ (без подписки): страница калькулятора открыта
для всех пользователей (product-решение). Core-прогноз кэшируется SWR
(ключ включает параметры модели); цена (yfinance, Redis 300с)
подмешивается на каждый запрос, чтобы PEG и valuation не замораживались.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from gex.auth.config import settings
from gex.adapters.ratelimit.rate_limiter import IpRateLimiter
from gex.adapters.cache.result_cache import cache_key, result_cache
from gex.schemas.sec_forecast import (
    ForecastResponse,
    SharesReconciliation,
    ValuationRequest,
    ValuationResponse,
)
from gex.application.sec.sec_forecast import SecForecastService, calculate_valuation

from ._helpers import handle

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sec-forecast"])

#: TTL SWR-кэша core-прогноза = свежести данных в PG (12ч), max_age = 24ч
_SEC_TTL = settings.SEC_FACTS_TTL_HOURS * 3600

# Per-IP лимит на публичные тяжёлые ручки (SEC EDGAR fetch до 7 МБ на промахе +
# CPU-normalization). ~1 запрос/10с в среднем на IP, короткие всплески до 6.
# См. security-аудит 2026-09-04: без лимита аноним перебирал тикеры/параметры
# и заставлял сервер качать EDGAR + yfinance бесконечно.
_SEC_IP_LIMITER = IpRateLimiter(rate=0.1, burst=6, scope="sec_forecast")


def _sec_rate_limit(request: Request) -> None:
    """429 при превышении числа публичных тяжёлых запросов с одного IP."""
    if settings.TESTING:
        return
    ip = request.client.host if request.client else "unknown"
    if ip == "testclient":
        return
    if not _SEC_IP_LIMITER.allow(ip):
        raise HTTPException(
            status_code=429,
            detail="Слишком много запросов к прогнозу. Подождите и повторите.",
        )


def _make_service() -> SecForecastService:
    from gex.deps import get_redis_client

    redis_client = get_redis_client()
    if redis_client and not redis_client.connected:
        redis_client = None
    return SecForecastService(redis_client=redis_client)


@router.get("/companies/{ticker}/forecast", response_model=ForecastResponse)
def get_company_forecast(
    ticker: str,
    _: None = Depends(_sec_rate_limit),
    horizon_years: int = Query(3, ge=1, le=10, description="Горизонт прогноза, лет (1/2/3)"),
    window: int = Query(5, ge=2, le=10, description="Окно WMA"),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Вес WMA в комбинированном прогнозе"),
    anomaly_threshold: float = Query(
        0.6, ge=0.2, le=10.0, description="Порог аномалии (0.2 = 20% и выше)"
    ),
    anomaly_mode: str = Query(
        "both", pattern="^(both|smooth|keep)$", description="both | smooth | keep"
    ),
    last_report_weight: float = Query(
        0.6, ge=0.6, le=1.0,
        description="Влияние последнего отчёта на изменение прогноза (0.6 = 60% базово, до 1.0 = 100%)",
    ),
):
    """Прогноз фундаментальных метрик компании из SEC EDGAR + yfinance.

    По каждой метрике: факт по годам, темпы роста (YoY/CAGR/квартальный),
    прогноз WMA / линейной регрессии / комбинированный (1/2/3 года) с
    учётом влияния последнего отчёта (``last_report_weight``), индикатор
    аномалии и два сценария учёта аномального скачка.
    """
    def _compute_core() -> dict:
        return _make_service().get_forecast_core(
            ticker,
            horizon_years=horizon_years,
            window=window,
            alpha=alpha,
            anomaly_threshold=anomaly_threshold,
            anomaly_mode=anomaly_mode,
            last_report_weight=last_report_weight,
        )

    key = cache_key(
        "res", "sec-forecast",
        ticker.strip().upper(), str(window), str(alpha),
        str(anomaly_threshold), anomaly_mode, str(horizon_years),
        str(last_report_weight),
    )
    core = result_cache.get(
        key, _SEC_TTL, lambda: handle(_compute_core, error_src="SEC EDGAR"), max_age=_SEC_TTL * 2
    )

    svc = _make_service()
    price = svc.fetch_price(ticker)
    return svc.with_price(core, price)


@router.get("/companies/{ticker}/shares", response_model=SharesReconciliation)
def get_company_shares(ticker: str, _: None = Depends(_sec_rate_limit)):
    """Сверка количества акций: SEC EDGAR vs Finnhub (company profile2).

    Если SEC не предоставил акции — берём Finnhub; если предоставил —
    проверяем, что Finnhub сходится по размеру (масштабы ×1/×10³/×10⁶/×10⁹
    подбираются автоматически), и предупреждаем о расхождении.
    """
    def _compute() -> dict:
        return _make_service().get_shares_reconciliation(ticker)

    key = cache_key("res", "sec-shares", ticker.strip().upper())
    return result_cache.get(
        key, _SEC_TTL, lambda: handle(_compute, error_src="SEC EDGAR / Finnhub"), max_age=_SEC_TTL * 2
    )


@router.post("/valuation/calculator", response_model=ValuationResponse)
def valuation_calculator(body: ValuationRequest):
    """Калькулятор оценки: Цена ↔ P/E ↔ Прибыль/EPS.

    Достаточно двух из трёх (price, P/E, EPS); earnings/shares/growth
    достраиваются. ``growth`` — десятичная доля (0.15 = 15%) для PEG.
    """
    return handle(
        lambda: calculate_valuation(body.model_dump()),
        error_src="калькулятора оценки",
    )
