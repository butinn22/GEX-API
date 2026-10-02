"""GEX Trading API — FastAPI application entrypoint.

Run:  ``uvicorn trading.main:app --reload``
Then open http://127.0.0.1:8000/ for the API-keys frontend,
and http://127.0.0.1:8000/docs for the Swagger UI.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from trading.adapters.fetchers import aclose_loop_registry
from trading.adapters.persistence.database import dispose, init_db
from trading.api import auth
from trading.api.middleware import RateLimitMiddleware
from trading.api.routers import backtest, data, keys, portfolio, strategies
from trading.api.websockets import router as ws_router
from trading.config import settings
from trading.domain import RateLimitExceededError, TradingError
from trading.logging_config import CorrelationIdMiddleware, configure_logging
from trading.observability import metrics_response

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    await init_db()
    yield
    # Release the fetchers' HTTP connection pools; without this the loop-scoped
    # registry keeps live sockets open until the process exits.
    await aclose_loop_registry()
    await dispose()


app = FastAPI(
    title="GEX Trading API",
    version="0.1.0",
    description="Auto-trading platform: backtesting, BingX/TBANK brokers, API-key management.",
    lifespan=lifespan,
)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RateLimitMiddleware, requests=settings.rate_limit_requests, per_seconds=60.0)


@app.exception_handler(TradingError)
async def _trading_error_handler(request: Request, exc: TradingError) -> JSONResponse:
    status_code = 429 if isinstance(exc, RateLimitExceededError) else 502
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})

_API = "/api/v1"
app.include_router(auth.router, prefix=_API)
app.include_router(keys.router, prefix=_API)
app.include_router(backtest.router, prefix=_API)
app.include_router(strategies.router, prefix=_API)
app.include_router(portfolio.router, prefix=_API)
app.include_router(data.router, prefix=_API)
app.include_router(ws_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
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
