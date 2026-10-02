"""Vol-index GEX: /vix/gex/{index}."""
from fastapi import APIRouter, BackgroundTasks, Depends, Query
from gex.deps import provide_gex_service
from gex.schemas import GEXAnalysisOut, GEXProfileOut
from gex.adapters.notifications.telegram_sender import notify_gex_analysis, notify_gex_profile
from ._helpers import handle

from fastapi import Depends
from gex.auth.dependencies import require_subscription
router = APIRouter(dependencies=[Depends(require_subscription("EXTENDED"))], prefix="/vix", tags=["vol-index"])

@router.get("/gex/{index}", response_model=GEXAnalysisOut)
def get_vol_index_gex(index: str, days: float = Query(30, ge=1, le=90),
    expiries: int = Query(5, ge=1, le=20), notify: bool = Query(False),
    background_tasks: BackgroundTasks = None, svc=Depends(provide_gex_service)) -> GEXAnalysisOut:
    return handle(lambda: svc.analyze_vol_index(index, days=days, max_expiries=expiries),
        notify_fn=notify_gex_analysis, notify=notify, background_tasks=background_tasks, error_src="yfinance")

@router.get("/gex/{index}/profile", response_model=GEXProfileOut)
def get_vol_index_gex_profile(index: str, days: float = Query(30, ge=1, le=90),
    expiries: int = Query(5, ge=1, le=20), notify: bool = Query(False),
    background_tasks: BackgroundTasks = None, svc=Depends(provide_gex_service)) -> GEXProfileOut:
    return handle(lambda: svc.analyze_vol_index_profile(index, days=days, max_expiries=expiries),
        notify_fn=lambda r: notify_gex_profile(r, index.upper()),
        notify=notify, background_tasks=background_tasks, error_src="yfinance")
