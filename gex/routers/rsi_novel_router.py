"""RSI Novel Candles routes: /rsi-novel/{ticker}.

Порт PineScript v5 «RSI CANDLES NOVEL BY ME»: RSI-свечи на базе Novel Candles
(гибрид стандартных свечей и Heikin-Ashi). Поддерживает US stocks, крипто
(Bybit + yfinance), MOEX и commodities — фетч OHLCV переиспользует
:class:`~gex.novel_candles.NovelCandlesService`.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from gex.application.rsi_novel import (
    RsiNovelService,
    DEFAULT_LENN,
    DEFAULT_LENG,
    DEFAULT_PERIOD100,
    DEFAULT_OB_LEVEL,
    DEFAULT_OS_LEVEL,
    DEFAULT_OM_LEVEL,
)
from gex.schemas.rsi_novel import RsiNovelResponse
from gex.adapters.cache.result_cache import cache_key, result_cache

from ._helpers import handle

logger = logging.getLogger(__name__)

from fastapi import Depends
from gex.auth.dependencies import require_subscription
router = APIRouter(dependencies=[Depends(require_subscription("BASIC"))], tags=["rsi-novel"])

#: TTL кэша финального результата (OHLCV уже кэшируется фетчером отдельно).
_RSI_NOVEL_TTL = 300


@router.get("/rsi-novel/{ticker}", response_model=RsiNovelResponse)
def get_rsi_novel(
    ticker: str,
    timeframe: str = Query("1d", description="Таймфрейм: 1h, 2h, 4h, 1d"),
    limit: int = Query(500, ge=10, le=1000, description="Максимальное число баров"),
    lenn: int = Query(DEFAULT_LENN, ge=1, le=100, description="Длина RSI (Length)"),
    wicks: bool = Query(True, description="Фитили на основе отдельного RSI каждой цены"),
    leng: int = Query(DEFAULT_LENG, ge=1, le=300, description="Длина ta.linreg"),
    period100: int = Query(DEFAULT_PERIOD100, ge=2, le=300, description="Период кастомной регрессии с MAD"),
    ob_level: float = Query(DEFAULT_OB_LEVEL, ge=1, le=99, description="RSI Overbought level"),
    os_level: float = Query(DEFAULT_OS_LEVEL, ge=1, le=99, description="RSI Oversold level"),
    om_level: float = Query(DEFAULT_OM_LEVEL, ge=1, le=99, description="RSI Middle level"),
    length12: int = Query(DEFAULT_LENN, ge=1, le=100, description="Длина RSI для динамических уровней"),
):
    """Рассчитать RSI Novel Candles для тикера.

    Свечи RSI (0..100) на базе novel-трансформации OHLCV + линии:
    EMA10/20/50, Trend MA (EMA33), кривая линейной регрессии,
    динамические resistance/support/mid.
    """
    def _compute() -> dict:
        from gex.deps import get_redis_client

        redis_client = get_redis_client()
        if redis_client and not redis_client.connected:
            redis_client = None

        svc = RsiNovelService(redis_client=redis_client)
        return svc.fetch_and_analyze(
            ticker,
            timeframe=timeframe,
            limit=limit,
            lenn=lenn,
            wicks=wicks,
            leng=leng,
            period100=period100,
            ob_level=ob_level,
            os_level=os_level,
            om_level=om_level,
            length12=length12,
        )

    _key = cache_key(
        "res", "rsi-novel",
        ticker.strip().upper(), timeframe, str(limit),
        str(lenn), str(wicks), str(leng), str(period100),
        str(ob_level), str(os_level), str(om_level), str(length12),
    )
    return result_cache.get(_key, _RSI_NOVEL_TTL, lambda: handle(_compute, error_src="yfinance/Bybit/MOEX"))
