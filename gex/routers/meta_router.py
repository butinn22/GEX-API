"""Meta routes: /health, /tickers, /visit."""
from fastapi import APIRouter, Body, Depends, Request
from pydantic import BaseModel, Field

from gex.deps import provide_gex_service, provide_scan_service
from gex.deps import get_redis_client

router = APIRouter(tags=["meta"])


class VisitIn(BaseModel):
    """Путь SPA-страницы (хеш-роут) для аналитики посещений."""
    path: str = Field(..., max_length=200)


@router.get("/health")
def health(
    svc=Depends(provide_gex_service),
    scan=Depends(provide_scan_service),
) -> dict:
    redis_client = get_redis_client()
    return {
        "status": "ok",
        "tickers_loaded": len(svc.list_tickers()),
        "scanner_running": scan.is_running,
        "scanned_tickers": len(scan.list_records()),
        "redis": "connected" if (redis_client and redis_client.ping()) else "unavailable",
    }


@router.get("/tickers")
def tickers(svc=Depends(provide_gex_service)) -> dict:
    return {"tickers": svc.list_tickers()}


@router.post("/visit")
def record_visit(
    payload: VisitIn,
    request: Request,
) -> dict:
    """Записать визит SPA-страницы (хеш-роут). Публичный, без авторизации."""
    from gex.adapters.middleware.visits import _extract_user_id, get_visit_tracker

    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")
    user_id = _extract_user_id(dict(request.headers))
    get_visit_tracker().record(ip, payload.path, ua, user_id)
    return {"ok": True}


@router.get("/geoip")
def get_geoip(
    request: Request,
    lang: str | None = None,
) -> dict:
    """Определить страну и язык по IP клиента (для мультиязычного сайта).

    Публичный. ``?lang=ru|en`` — принудительно вернуть указанный язык
    (используется для тестов и ручного переопределения).
    """
    from gex.adapters.middleware.geoip import geoip_payload

    payload = geoip_payload(request)
    if lang in ("ru", "en"):
        payload["language"] = lang
        payload["forced"] = True
    return payload
