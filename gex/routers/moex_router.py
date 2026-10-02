"""MOEX GEX: /moex/gex/{asset}."""
from fastapi import APIRouter, BackgroundTasks, Depends, Query
from gex.application.auto_scope import AUTO_MAX_DAYS
from gex.deps import provide_gex_service
from gex.schemas import GEXAnalysisOut, GEXProfileOut
from gex.adapters.notifications.telegram_sender import notify_gex_analysis, notify_gex_profile
from ._helpers import handle

from fastapi import Depends
from gex.auth.dependencies import require_subscription
router = APIRouter(dependencies=[Depends(require_subscription("EXTENDED"))], prefix="/moex", tags=["moex"])


@router.get("/gex/{asset}", response_model=GEXAnalysisOut)
def get_moex_gex(
    asset: str, days: float = Query(30, ge=1, le=90),
    expiries: int = Query(0, ge=0, le=20),
    notify: bool = Query(False), background_tasks: BackgroundTasks = None,
    mode: str = Query("manual", pattern="^(manual|auto)$",
                      description="manual|auto — auto resolves to days=90, expiries=0 (ALL)"),
    svc=Depends(provide_gex_service),
) -> GEXAnalysisOut:
    if mode == "auto":
        # AUTO = max scope only for MOEX (ISS is the single provider, no fallback).
        # expiries=0 means ALL expirations to the MOEX fetcher (design decision Q2).
        return handle(
            lambda: svc.analyze_moex(asset, days=AUTO_MAX_DAYS, max_expiries=0, auto=True),
            notify_fn=notify_gex_analysis, notify=notify,
            background_tasks=background_tasks, error_src="MOEX ISS",
        )
    return handle(
        lambda: svc.analyze_moex(asset, days=days, max_expiries=expiries),
        notify_fn=notify_gex_analysis, notify=notify,
        background_tasks=background_tasks, error_src="MOEX ISS",
    )


@router.get("/gex/{asset}/profile", response_model=GEXProfileOut)
def get_moex_gex_profile(
    asset: str, days: float = Query(30, ge=1, le=90),
    expiries: int = Query(0, ge=0, le=20),
    notify: bool = Query(False), background_tasks: BackgroundTasks = None,
    svc=Depends(provide_gex_service),
) -> GEXProfileOut:
    return handle(
        lambda: svc.analyze_moex_profile(asset, days=days, max_expiries=expiries),
        notify_fn=lambda r: notify_gex_profile(r, asset.upper()),
        notify=notify, background_tasks=background_tasks, error_src="MOEX ISS",
    )
