"""Live GEX: /live/gex/{ticker} (polling yfinance)."""
from fastapi import APIRouter, BackgroundTasks, Depends, Query

from gex.deps import provide_gex_service
from gex.adapters.cache.result_cache import cache_key, result_cache
from gex.schemas import GEXAnalysisOut, GEXProfileOut
from gex.adapters.notifications.telegram_sender import notify_gex_analysis, notify_gex_profile

from ._helpers import handle

from fastapi import Depends
from gex.auth.dependencies import require_subscription
router = APIRouter(dependencies=[Depends(require_subscription("EXTENDED"))], prefix="/live", tags=["live"])


@router.get("/gex/{ticker}", response_model=GEXAnalysisOut)
def get_live_gex(
    ticker: str,
    days: float = Query(30, ge=1, le=90),
    expiries: int = Query(5, ge=1, le=20),
    notify: bool = Query(False),
    background_tasks: BackgroundTasks = None,
    svc=Depends(provide_gex_service),
) -> GEXAnalysisOut:
    _key = cache_key("res", "livegex", ticker, days, expiries)
    if notify:
        return handle(
            lambda: svc.analyze_live(ticker, days=days, max_expiries=expiries),
            notify_fn=notify_gex_analysis, notify=True,
            background_tasks=background_tasks, error_src="yfinance")
    return result_cache.get(_key, 600, lambda: handle(
        lambda: svc.analyze_live(ticker, days=days, max_expiries=expiries),
        notify_fn=notify_gex_analysis, notify=False,
        background_tasks=background_tasks, error_src="yfinance"))


@router.get("/gex/{ticker}/profile", response_model=GEXProfileOut)
def get_live_gex_profile(
    ticker: str,
    days: float = Query(30, ge=1, le=90),
    expiries: int = Query(5, ge=1, le=20),
    notify: bool = Query(False),
    background_tasks: BackgroundTasks = None,
    svc=Depends(provide_gex_service),
) -> GEXProfileOut:
    def _do():
        return svc.analyze_profile_live(ticker, days=days, max_expiries=expiries)
    _key = cache_key("res", "livegexp", ticker, days, expiries)
    if notify:
        return handle(
            _do,
            notify_fn=lambda r: notify_gex_profile(r, ticker.upper()), notify=True,
            background_tasks=background_tasks, error_src="yfinance")
    return result_cache.get(_key, 600, lambda: handle(
        _do,
        notify_fn=lambda r: notify_gex_profile(r, ticker.upper()), notify=False,
        background_tasks=background_tasks, error_src="yfinance"))
