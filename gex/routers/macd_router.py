"""MACD trend: /macd/trend/{ticker}."""
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from gex.deps import provide_macd_trend_service
from gex.domain.macd_trend import AnalyzerConfig
from gex.adapters.cache.result_cache import cache_key, result_cache
from gex.schemas import MacdTrendAnalysisOut
from gex.adapters.notifications.telegram_sender import notify_macd_trend_analysis

from ._helpers import handle

from fastapi import Depends
from gex.auth.dependencies import require_subscription
router = APIRouter(dependencies=[Depends(require_subscription("BASIC"))], tags=["macd-trend"])

@router.get("/macd/trend/{ticker}", response_model=MacdTrendAnalysisOut)
def get_macd_trend(ticker: str, timeframe: str = Query("all"),
    N: int = Query(5, ge=2, le=100), M: int = Query(20, ge=2, le=500),
    H: int = Query(200, ge=10, le=2000), flat_threshold_deg: float = Query(3.0, ge=0.0, le=45.0),
    eta: float = Query(0.1, ge=0.0, le=5.0), markov_min_obs: int = Query(30, ge=1, le=500),
    normalization_method: str = Query("rolling_std"), line_method: str = Query("two_point"),
    kelly_enabled: bool = Query(False), kelly_b: float = Query(2.0, ge=0.1, le=100.0),
    notify: bool = Query(False), background_tasks: BackgroundTasks = None,
    svc=Depends(provide_macd_trend_service)) -> MacdTrendAnalysisOut:
    if normalization_method not in ("rolling_std", "atr"):
        raise HTTPException(422, "normalization_method: rolling_std or atr")
    if line_method not in ("two_point", "linear_regression"):
        raise HTTPException(422, "line_method: two_point or linear_regression")
    config = AnalyzerConfig(N=N, M=M, H=H, flat_threshold_deg=flat_threshold_deg,
        eta=eta, markov_min_obs=markov_min_obs, normalization_method=normalization_method,
        line_method=line_method, kelly_enabled=kelly_enabled, kelly_b=kelly_b)
    def _do():
        if timeframe.strip().lower() == "all": return svc.analyze(ticker, config=config)
        return svc.analyze_timeframe(ticker, timeframe, config=config)
    _key = cache_key("res", "macd", ticker, timeframe, N, M, H, flat_threshold_deg, eta,
                     markov_min_obs, normalization_method, line_method, kelly_enabled, kelly_b)
    if notify:
        return handle(_do, notify_fn=notify_macd_trend_analysis, notify=True,
            background_tasks=background_tasks, error_src="yfinance/Bybit")
    return result_cache.get(_key, 600, lambda: handle(
        _do, notify_fn=notify_macd_trend_analysis, notify=False,
        background_tasks=background_tasks, error_src="yfinance/Bybit"))
