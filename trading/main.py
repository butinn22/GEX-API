"""GEX Trading API — FastAPI application entrypoint.

Run:  ``uvicorn trading.main:app --reload``
Then open http://127.0.0.1:8000/ for the API-keys frontend,
and http://127.0.0.1:8000/docs for the Swagger UI.
"""
from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from trading.adapters.fetchers import aclose_loop_registry
from trading.adapters.persistence.database import dispose, run_migrations
from trading.api import auth
from trading.api.deps import require_auth
from trading.api.local_client_ws import (
    router as local_client_router,
    start_dispatcher,
    stop_dispatcher,
)
from trading.api.middleware import RateLimitMiddleware
from trading.api.routers import (
    backtest,
    baskets,
    dashboard,
    data,
    export,
    keys,
    portfolio,
    presets,
    signal_keys,
    signals,
    strategies,
)
from trading.api.websockets import router as ws_router
from trading.config import settings
from trading.domain import RateLimitExceededError, TradingError
from trading.logging_config import CorrelationIdMiddleware, configure_logging
from trading.observability import metrics_response
from trading.application.signal_engine import signal_engine

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    await run_migrations()  # alembic upgrade head (idempotent)
    await start_dispatcher()  # fan signal_hub out to subscribed local clients
    # A previous process may have left open positions behind; adopting them is
    # safe (the engine only appends), so the ledger is never purged implicitly.
    yield
    if signal_engine.running:
        await signal_engine.stop()
    await stop_dispatcher()
    # Release the fetchers' HTTP connection pools; without this the loop-scoped
    # registry keeps live sockets open until the process exits.
    await aclose_loop_registry()
    await dispose()


_DOCS_ON = settings.enable_docs or not settings.is_production

app = FastAPI(
    title="GEX Trading API",
    version="0.1.0",
    description="Auto-trading platform: backtesting, BingX/TBANK brokers, API-key management.",
    lifespan=lifespan,
    # Swagger/ReDoc/OpenAPI expose the full route surface — disable in production
    # unless explicitly enabled (TRADING_ENABLE_DOCS=1). Guarding them with a JWT
    # would break the UI's own fetches, so they are simply not mounted.
    docs_url="/docs" if _DOCS_ON else None,
    redoc_url="/redoc" if _DOCS_ON else None,
    openapi_url="/openapi.json" if _DOCS_ON else None,
)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RateLimitMiddleware, requests=settings.rate_limit_requests, per_seconds=60.0)


@app.exception_handler(TradingError)
async def _trading_error_handler(request: Request, exc: TradingError) -> JSONResponse:
    status_code = 429 if isinstance(exc, RateLimitExceededError) else 502
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})

_API = "/api/v1"
#: Every business router is guarded **here, at the mount** — one file is the
#: single source of truth, so a future router cannot be forgotten. ``auth``
#: (login) and ``backtest.public_router`` (sendBeacon capability cancel) are
#: intentionally included without the guard; ``dashboard`` (/API_KEY/*) and the
#: WS routers authenticate themselves.
_GUARD = [Depends(require_auth)]
app.include_router(auth.router, prefix=_API)  # public (login)
app.include_router(keys.router, prefix=_API, dependencies=_GUARD)
app.include_router(backtest.router, prefix=_API, dependencies=_GUARD)
app.include_router(backtest.public_router, prefix=_API)  # POST /backtest/cancel/{token}
app.include_router(strategies.router, prefix=_API, dependencies=_GUARD)
app.include_router(portfolio.router, prefix=_API, dependencies=_GUARD)
app.include_router(data.router, prefix=_API, dependencies=_GUARD)
app.include_router(export.router, prefix=_API, dependencies=_GUARD)
app.include_router(presets.router, prefix=_API, dependencies=_GUARD)
app.include_router(signal_keys.router, prefix=_API, dependencies=_GUARD)
app.include_router(baskets.router, prefix=_API, dependencies=_GUARD)
app.include_router(signals.router, prefix=_API, dependencies=_GUARD)
app.include_router(ws_router)  # /ws/* — JWT validated before accept
app.include_router(local_client_router)  # /ws/client — handshake-frame token
app.include_router(dashboard.router)  # /API_KEY/{key} — key is the credential


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics(request: Request):
    """Prometheus scrape endpoint.

    Gated by ``TRADING_METRICS_TOKEN`` (Bearer header or ``?token=``). In
    production an unset token means "not exposed" (401) rather than open; in
    dev with no token configured it stays open for convenience.
    """
    token = settings.metrics_token
    if token:
        header = request.headers.get("authorization", "")
        bearer = header[7:].strip() if header.lower().startswith("bearer ") else ""
        query = request.query_params.get("token", "")
        if not (hmac.compare_digest(bearer, token) or hmac.compare_digest(query, token)):
            raise HTTPException(401, "metrics token required")
    elif settings.is_production:
        raise HTTPException(401, "metrics endpoint requires TRADING_METRICS_TOKEN")
    return metrics_response()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    # Browsers request this at the site root, not under /static — without an
    # explicit route every page load logs a noisy 404.
    return FileResponse(STATIC_DIR / "favicon.ico", media_type="image/x-icon")
