"""Структура рынка HH/HL/LH/LL: GET /ta-structure/{ticker}."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from gex.application.hybrid_trend import HybridTrendParams, HybridTrendService
from gex.schemas.hybrid_trend import HybridStructureResponse

logger = logging.getLogger(__name__)

from fastapi import Depends
from gex.auth.dependencies import require_subscription
router = APIRouter(dependencies=[Depends(require_subscription("BASIC"))], tags=["technical-analysis", "chart"])


@router.get("/ta-structure/{ticker}", response_model=HybridStructureResponse)
def get_ta_structure(
    ticker: str,
    timeframe: str = Query("1d", description="Таймфрейм: 1h, 2h, 4h, 1d"),
    limit: int = Query(300, ge=50, le=1000, description="Число баров"),
    fractal_source: str = Query("median_candle", description="standard | candle | median | median_candle"),
    fractal_left: int = Query(2, ge=0, le=10, description="Левое окно фрактала"),
    fractal_right: int = Query(2, ge=0, le=10, description="Правое окно фрактала (лаг подтверждения)"),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Вес median в композите median_candle"),
    strict_fractals: bool = Query(True, description="Строгое сравнение фракталов"),
    atr_period: int = Query(14, ge=1, le=200, description="Период ATR"),
    atr_mult: float = Query(0.25, ge=0.0, le=5.0, description="Множитель ATR для порога HH/HL/LH/LL"),
    min_change: float = Query(0.0, ge=0.0, description="Минимальный абсолютный порог изменения"),
    max_event_age: int = Query(120, ge=1, le=2000, description="Макс. возраст события в барах"),
    use_novel_filter: bool = Query(True, description="Подтверждать тренд импульсом novelsrc"),
    novel_ema: int = Query(20, ge=2, le=200, description="Период EMA novelsrc"),
    min_fractal_distance: int = Query(0, ge=0, le=100, description="Мин. расстояние между фракталами в барах (0 = выкл)"),
    no_trendlines: bool = Query(False, description="Пропустить расчёт трендовых линий"),
    max_trendlines: int = Query(5, ge=1, le=20, description="Макс. число линий каждого типа"),
    pivot_left: int = Query(5, ge=1, le=30, description="Левое окно пивотов трендовых линий"),
    pivot_right: int = Query(5, ge=1, le=30, description="Правое окно пивотов трендовых линий"),
):
    """Структура рынка для графика на странице /ta.

    Отдаёт стандартные и Novel-бары, подтверждённые фракталы, события
    HH/HL/LH/LL, зигзаг, трендовые линии, строгий/финальный тренд и стадию
    рынка (UPTREND/DOWNTREND/REVERSAL/RANGE). Поддерживает US stocks,
    крипто (Bybit+yfinance), MOEX, commodities.
    """
    from gex.deps import get_redis_client

    redis_client = get_redis_client()
    if redis_client and not redis_client.connected:
        redis_client = None

    svc = HybridTrendService(redis_client=redis_client)

    cache_parts = [
        "structure", ticker, timeframe, str(limit), fractal_source,
        str(fractal_left), str(fractal_right), str(alpha), str(strict_fractals),
        str(atr_period), str(atr_mult), str(min_change), str(max_event_age),
        str(use_novel_filter), str(novel_ema), str(min_fractal_distance),
        str(no_trendlines), str(max_trendlines), str(pivot_left), str(pivot_right),
    ]

    # ── Redis cache (TTL 300 c) ──
    if redis_client and redis_client.connected:
        from gex.adapters.cache.redis_client import cache_key, deserialize_value
        ck = cache_key(*cache_parts)
        cached_data = redis_client.get(ck)
        if cached_data is not None:
            try:
                result = deserialize_value(cached_data)
                if isinstance(result, dict):
                    logger.info("TaStructure CACHE HIT for %s [%s]", ticker, timeframe)
                    return HybridStructureResponse(**result)
            except Exception:
                logger.debug("TaStructure cache deserialize error, refetching")

    from ._helpers import handle

    result = handle(
        lambda: svc.fetch_and_analyze(
            ticker,
            timeframe=timeframe,
            limit=limit,
            # Параметры строим здесь: ValueError от валидации → 404 через handle
            params=HybridTrendParams(
                fractal_left=fractal_left,
                fractal_right=fractal_right,
                fractal_source=fractal_source,
                alpha=alpha,
                strict_fractals=strict_fractals,
                atr_period=atr_period,
                atr_mult=atr_mult,
                min_change=min_change,
                max_event_age=max_event_age,
                use_novel_filter=use_novel_filter,
                novel_ema=novel_ema,
                min_fractal_distance=min_fractal_distance,
            ),
            with_trendlines=not no_trendlines,
            max_trendlines=max_trendlines,
            pivot_left=pivot_left,
            pivot_right=pivot_right,
        ),
        error_src="yfinance/Bybit/MOEX",
    )

    if redis_client and redis_client.connected:
        try:
            from gex.adapters.cache.redis_client import cache_key as ck_fn
            redis_client.set(ck_fn(*cache_parts), result, ex=300)
        except Exception:
            logger.debug("TaStructure cache write failed")

    return HybridStructureResponse(**result)
