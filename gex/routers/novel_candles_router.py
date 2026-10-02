"""Novel Candles routes: /novel-candles/{ticker}."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from gex.schemas.novel_candles import NovelCandlesResponse
from gex.application.novel_candles import NovelCandlesService, DEFAULT_EMA_PERIODS

logger = logging.getLogger(__name__)

from fastapi import Depends
from gex.auth.dependencies import require_subscription
router = APIRouter(dependencies=[Depends(require_subscription("BASIC"))], tags=["novel-candles"])


def _parse_ema_periods(raw: str | None) -> tuple[int, ...]:
    """Разобрать строку '10,20,50,100,200' в кортеж int."""
    if not raw:
        return DEFAULT_EMA_PERIODS
    try:
        periods = tuple(int(p.strip()) for p in raw.split(",") if p.strip())
        if not periods:
            return DEFAULT_EMA_PERIODS
        return periods
    except ValueError:
        return DEFAULT_EMA_PERIODS


@router.get("/novel-candles/{ticker}", response_model=NovelCandlesResponse)
def get_novel_candles(
    ticker: str,
    timeframe: str = Query("1d", description="Таймфрейм: 1h, 2h, 4h, 1d"),
    limit: int = Query(500, ge=10, le=1000, description="Максимальное число баров"),
    ema_periods: str | None = Query(None, description="Периоды EMA через запятую: 10,20,50,100,200"),
    trendline_resolution: int = Query(6, ge=2, le=50, description="Окно поиска экстремумов для трендовых линий"),
    max_trendlines: int = Query(5, ge=1, le=20, description="Макс. число линий каждого типа"),
    pivot_left: int = Query(5, ge=1, le=30, description="Левое окно фракталов"),
    pivot_right: int = Query(5, ge=1, le=30, description="Правое окно фракталов"),
    no_trendlines: bool = Query(False, description="Пропустить расчёт трендовых линий"),
    use_linreg: bool = Query(False, description="Использовать LinReg-сглаживание свечей"),
    linreg_length: int = Query(11, ge=1, le=200, description="Длина линейной регрессии"),
    use_linreg_for_ema: bool = Query(False, description="Использовать hybridsrc для EMA"),
    two_pole: bool = Query(False, description="Включить Two-Pole Filter"),
    tp_length: int = Query(20, ge=2, le=200, description="Длина Two-Pole Filter"),
    tp_damping: float = Query(0.9, ge=0.1, le=1.0, description="Damping Two-Pole Filter"),
    tp_bands: float = Query(1.0, ge=0.5, le=5.0, description="ATR bands для сигналов"),
    tp_ris_fal: int = Query(5, ge=1, le=50, description="Порог rising/falling для сигналов"),
    tp_signals: bool = Query(False, description="Показать сигналы Two-Pole Filter"),
):
    """Получить Novel Candles (гибрид std+Heikin-Ashi), EMA и трендовые линии.

    Поддерживает US stocks, крипто (Bybit+yfinance), MOEX, commodities.
    Режимы: LinReg-свечи, LinReg-EMA, Two-Pole Filter.
    """
    from gex.deps import get_redis_client

    redis_client = get_redis_client()
    if redis_client and not redis_client.connected:
        redis_client = None

    svc = NovelCandlesService(redis_client=redis_client)
    ema_tuple = _parse_ema_periods(ema_periods)

    # ── Redis cache: ключ включает все значимые параметры ──
    if redis_client and redis_client.connected:
        from gex.adapters.cache.redis_client import cache_key, deserialize_value
        ck_parts = [
            "novel", ticker, timeframe, str(limit),
            ",".join(map(str, ema_tuple)),
            str(no_trendlines),
            str(use_linreg), str(linreg_length), str(use_linreg_for_ema),
            str(two_pole),
        ]
        ck = cache_key(*ck_parts)
        cached_data = redis_client.get(ck)
        if cached_data is not None:
            try:
                result = deserialize_value(cached_data)
                if isinstance(result, dict):
                    logger.info("NovelCandles CACHE HIT for %s [%s]", ticker, timeframe)
                    return NovelCandlesResponse(**result)
            except Exception:
                logger.debug("NovelCandles cache deserialize error, refetching")

    # ── Fetch + compute ──
    # handle() отдаёт чистые 404/502 (KeyError/ValueError/RuntimeError),
    # как rsi-novel/ta/trendlines/macd — неизвестный тикер не должен давать 500.
    from ._helpers import handle

    result = handle(
        lambda: svc.fetch_and_analyze(
            ticker,
            timeframe=timeframe,
            limit=limit,
            ema_periods=ema_tuple,
            trendline_resolution=trendline_resolution,
            max_trendlines=0 if no_trendlines else max_trendlines,
            pivot_left=pivot_left,
            pivot_right=pivot_right,
            use_linreg=use_linreg,
            linreg_length=linreg_length,
            use_linreg_for_ema=use_linreg_for_ema,
            two_pole=two_pole,
            tp_length=tp_length,
            tp_damping=tp_damping,
            tp_bands=tp_bands,
            tp_ris_fal=tp_ris_fal,
            tp_signals=tp_signals,
        ),
        error_src="yfinance/Bybit/MOEX",
    )

    # ── Сохранить в Redis ──
    if redis_client and redis_client.connected:
        try:
            from gex.adapters.cache.redis_client import cache_key as ck_fn, serialize_value
            ck_parts = [
                "novel", ticker, timeframe, str(limit),
                ",".join(map(str, ema_tuple)),
                str(no_trendlines),
                str(use_linreg), str(linreg_length), str(use_linreg_for_ema),
                str(two_pole),
            ]
            ck = ck_fn(*ck_parts)
            redis_client.set(ck, result, ex=300)
        except Exception:
            logger.debug("NovelCandles cache write failed")

    return NovelCandlesResponse(**result)
