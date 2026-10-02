"""Commodity routes: /commodity/*."""
from fastapi import APIRouter, Depends, Query

from gex.commodity_assets import COMMODITY_ASSETS
from gex.adapters.fetchers.commodity_fetcher import CommodityFetcher
from gex.deps import get_redis_client, provide_commodity_dynamics_service, provide_gex_service
from gex.adapters.cache.result_cache import cache_key, result_cache
from gex.schemas import GEXAnalysisOut, GEXProfileOut

from ._helpers import handle

from fastapi import Depends
from gex.auth.dependencies import require_subscription
router = APIRouter(dependencies=[Depends(require_subscription("BASIC"))], prefix="/commodity", tags=["commodity"])

@router.get("/analysis/{asset}", response_model=GEXAnalysisOut)
def get_commodity_analysis(asset: str, days: float = Query(30, ge=1, le=90),
    expiries: int = Query(5, ge=1, le=20), svc=Depends(provide_gex_service)) -> GEXAnalysisOut:
    return result_cache.get(cache_key("res", "comman", asset, days, expiries), 600,
        lambda: handle(lambda: svc.analyze_commodity(asset, days=days, max_expiries=expiries)))

@router.get("/analysis/{asset}/profile", response_model=GEXProfileOut)
def get_commodity_profile(asset: str, days: float = Query(30, ge=1, le=90),
    expiries: int = Query(5, ge=1, le=20), svc=Depends(provide_gex_service)) -> GEXProfileOut:
    return handle(lambda: svc.analyze_commodity_profile(asset, days=days, max_expiries=expiries))

@router.get("/ohlcv/{asset}")
def get_commodity_ohlcv(asset: str, timeframe: str = Query("1d"),
    limit: int = Query(200, ge=1, le=1000), svc=Depends(provide_gex_service)):
    return handle(lambda: svc.get_commodity_ohlcv(asset, timeframe=timeframe, limit=limit))

@router.get("/spot/{asset}")
def get_commodity_spot(asset: str):
    return handle(lambda: CommodityFetcher(redis_client=get_redis_client()).fetch_spot(asset))

@router.get("/tickers")
def list_commodity_tickers():
    return {asset: {"yf_symbol": c["yf_symbol"], "label": c["label"], "unit": c["unit"], "category": c["category"]}
            for asset, c in COMMODITY_ASSETS.items()}

@router.get("/dynamics")
def get_commodity_dynamics(bars: int = Query(90, ge=10, le=500),
    svc=Depends(provide_commodity_dynamics_service)):
    return result_cache.get(cache_key("res", "commdyn", bars), 600,
        lambda: handle(lambda: svc.compute(bars=bars)))
