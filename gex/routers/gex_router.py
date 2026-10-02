"""Static GEX analysis: /gex/{ticker}, /chains/{ticker}."""
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status

from gex.deps import provide_gex_service
from gex.domain.data_loader import GEXDataLoader, OptionSnapshot
from gex.schemas import ChainIn, GEXAnalysisOut, GEXProfileOut
from gex.adapters.notifications.telegram_sender import notify_gex_analysis, notify_gex_profile
from ._helpers import handle

from fastapi import Depends
from gex.auth.dependencies import require_subscription, require_master_admin
from gex.auth.models import User
router = APIRouter(dependencies=[Depends(require_subscription("EXTENDED"))], tags=["analysis", "data"])


def _ingest_chain_in(ticker: str, body: ChainIn) -> OptionSnapshot:
    loader = GEXDataLoader(spot=body.spot, symbol=ticker, contract_multiplier=body.per_contract)
    rows = []
    for r in body.chain:
        T = r.T
        if T is None and r.expiry is not None:
            days = (r.expiry - datetime.now(timezone.utc)).total_seconds() / 86400.0
            T = max(days / 365.0, 1e-6)
        rows.append({"strike": r.strike, "type": r.type, "oi": r.oi, "iv": r.iv, "T": T})
    import pandas as pd
    return loader.load_dataframe(pd.DataFrame(rows), as_of=body.as_of)


@router.get("/gex/{ticker}", response_model=GEXAnalysisOut)
def get_gex_analysis(
    ticker: str,
    days: float = Query(30, ge=1, le=90),
    notify: bool = Query(False),
    background_tasks: BackgroundTasks = None,
    svc=Depends(provide_gex_service),
) -> GEXAnalysisOut:
    return handle(
        lambda: svc.analyze(ticker, days=days),
        notify_fn=notify_gex_analysis,
        notify=notify,
        background_tasks=background_tasks,
    )


@router.get("/gex/{ticker}/profile", response_model=GEXProfileOut)
def get_gex_profile(
    ticker: str,
    days: float = Query(30, ge=1, le=90),
    notify: bool = Query(False),
    background_tasks: BackgroundTasks = None,
    svc=Depends(provide_gex_service),
) -> GEXProfileOut:
    def _do():
        return svc.analyze_profile(ticker, days=days)
    return handle(
        _do,
        notify_fn=lambda r: notify_gex_profile(r, ticker.upper()),
        notify=notify,
        background_tasks=background_tasks,
    )


@router.post("/chains/{ticker}", response_model=GEXAnalysisOut, status_code=status.HTTP_201_CREATED)
def post_chain(
    ticker: str,
    body: ChainIn,
    notify: bool = Query(False),
    background_tasks: BackgroundTasks = None,
    svc=Depends(provide_gex_service),
    user: User = Depends(require_master_admin),
) -> GEXAnalysisOut:
    snapshot = _ingest_chain_in(ticker, body)
    svc.ingest_chain(ticker, snapshot)
    return handle(
        lambda: svc.analyze(ticker, days=30),
        notify_fn=notify_gex_analysis,
        notify=notify,
        background_tasks=background_tasks,
    )


@router.delete("/chains/{ticker}")
def delete_chain(ticker: str, svc=Depends(provide_gex_service), user: User = Depends(require_master_admin)) -> dict:
    deleted = svc.repo.delete(ticker)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Тикер '{ticker}' не найден.")
    return {"deleted": ticker}
