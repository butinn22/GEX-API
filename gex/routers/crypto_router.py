"""Crypto GEX: /crypto/gex/{coin}."""
from fastapi import APIRouter, BackgroundTasks, Depends, Query
from gex.deps import provide_gex_service
from gex.schemas import GEXAnalysisOut, GEXProfileOut
from gex.adapters.notifications.telegram_sender import notify_gex_analysis, notify_gex_profile
from ._helpers import handle

from fastapi import Depends
from gex.auth.dependencies import require_subscription
router = APIRouter(dependencies=[Depends(require_subscription("EXTENDED"))], prefix="/crypto", tags=["crypto"])

@router.get("/gex/{coin}", response_model=GEXAnalysisOut)
def get_crypto_gex(coin: str, days: float = Query(30, ge=1, le=90),
    expiries: int = Query(5, ge=1, le=20), notify: bool = Query(False),
    background_tasks: BackgroundTasks = None, svc=Depends(provide_gex_service)) -> GEXAnalysisOut:
    return handle(lambda: svc.analyze_crypto(coin, days=days, max_expiries=expiries),
        notify_fn=notify_gex_analysis, notify=notify, background_tasks=background_tasks, error_src="Bybit")

@router.get("/gex/{coin}/profile", response_model=GEXProfileOut)
def get_crypto_gex_profile(coin: str, days: float = Query(30, ge=1, le=90),
    expiries: int = Query(5, ge=1, le=20), notify: bool = Query(False),
    background_tasks: BackgroundTasks = None, svc=Depends(provide_gex_service)) -> GEXProfileOut:
    return handle(lambda: svc.analyze_crypto_profile(coin, days=days, max_expiries=expiries),
        notify_fn=lambda r: notify_gex_profile(r, coin.upper()),
        notify=notify, background_tasks=background_tasks, error_src="Bybit")
