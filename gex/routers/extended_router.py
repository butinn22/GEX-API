"""Extended GEX: /ext/gex/{ticker}."""

from fastapi import APIRouter, BackgroundTasks, Depends, Query

from gex.application.auto_scope import AUTO_CACHE_TTL, AUTO_MAX_DAYS, AUTO_MAX_EXPIRIES
from gex.deps import provide_extended_gex_service
from gex.schemas.extended_schemas import ExtendedGEXAnalysisOut, extended_report_to_schema
from gex.adapters.cache.result_cache import cache_key, result_cache
from gex.adapters.notifications.telegram_sender import notify_extended_gex

from ._helpers import handle

from fastapi import Depends
from gex.auth.dependencies import require_subscription
router = APIRouter(dependencies=[Depends(require_subscription("EXTENDED"))], prefix="/ext", tags=["extended-gex"])

@router.get("/gex/{ticker}", response_model=ExtendedGEXAnalysisOut)
def get_extended_gex(ticker: str, days: float = Query(30, ge=1, le=90),
    expiries: int = Query(5, ge=1, le=20), hedge_pct: list[float] | None = Query(None),
    notify: bool = Query(False), background_tasks: BackgroundTasks = None,
    source: str = Query("auto", description="auto|webull|yfinance"),
    mode: str = Query("manual", pattern="^(manual|auto)$",
                      description="manual|auto — auto forces days=90, expiries=20"),
    svc=Depends(provide_extended_gex_service)) -> ExtendedGEXAnalysisOut:
    pcts = hedge_pct if hedge_pct else [-1.0, 1.0]
    pcts_tag = "-".join(str(p) for p in pcts)

    if mode == "auto":
        # AUTO ignores any client days/expiries and resolves the API maxima.
        # The cache key uses a disjoint segment (``extgexA``) and a 10-minute TTL,
        # so a manual 30/5 entry can never poison an AUTO 90/20 entry (design §5.3).
        _key = cache_key(
            "res", "extgexA", ticker, int(AUTO_MAX_DAYS), AUTO_MAX_EXPIRIES,
            source, pcts_tag,
        )
        def _compute():
            return handle(
                lambda: extended_report_to_schema(
                    svc.analyze_auto(ticker, source=source, hedge_scenarios_pct=pcts),
                    days=AUTO_MAX_DAYS),
                notify_fn=notify_extended_gex, notify=False,
                background_tasks=background_tasks,
                error_src="Bybit/yfinance/Webull")
        if notify:
            return handle(
                lambda: extended_report_to_schema(
                    svc.analyze_auto(ticker, source=source, hedge_scenarios_pct=pcts),
                    days=AUTO_MAX_DAYS),
                notify_fn=notify_extended_gex, notify=True,
                background_tasks=background_tasks,
                error_src="Bybit/yfinance/Webull")
        return result_cache.get(_key, AUTO_CACHE_TTL, _compute)

    _key = cache_key("res", "extgex", ticker, days, expiries, source, pcts_tag)
    if notify:
        return handle(lambda: extended_report_to_schema(
            svc.analyze(ticker, days=days, max_expiries=expiries, hedge_scenarios_pct=pcts, source=source), days=days),
            notify_fn=notify_extended_gex, notify=True, background_tasks=background_tasks,
            error_src="Bybit/yfinance/Webull")
    return result_cache.get(_key, 600, lambda: handle(
        lambda: extended_report_to_schema(
            svc.analyze(ticker, days=days, max_expiries=expiries, hedge_scenarios_pct=pcts, source=source), days=days),
        notify_fn=notify_extended_gex, notify=False, background_tasks=background_tasks,
        error_src="Bybit/yfinance/Webull"))
