"""Market-data endpoint — real fetchers with fallback, plus a synthetic option."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from trading.adapters.fetchers import SyntheticFetcher, loop_registry
from trading.application.data_sources import detect_data_source
from trading.application.instruments import CATEGORIES, load_instruments, select_universe
from trading.domain import DataFetchError, Exchange

router = APIRouter(prefix="/data", tags=["data"])

_SOURCES = {
    "moex": [Exchange.MOEX],
    "yfinance": [Exchange.YFINANCE],
    "bybit": [Exchange.BYBIT],
    "webull": [Exchange.WEBULL],
    "auto": [Exchange.MOEX, Exchange.YFINANCE, Exchange.BYBIT],
}


@router.get("/instruments")
def instruments() -> list[dict]:
    """Ticker universe used by the auto_scanner (us/crypto/fx/ru/sectors)."""
    return load_instruments()


@router.get("/ohlcv/{symbol}")
async def ohlcv(
    symbol: str,
    timeframe: str = Query(default="1d"),
    limit: int = Query(default=200, ge=1, le=1000),
    source: str = Query(default="auto"),
) -> list[dict]:
    if source == "synthetic":
        bars = await SyntheticFetcher(seed=0).get_ohlcv(symbol, timeframe, limit=limit)
    else:
        exchanges = _SOURCES.get(source)
        if exchanges is None:
            raise HTTPException(400, f"unknown source '{source}'")
        try:
            # The loop-scoped registry reuses one HTTP client pool per exchange
            # and is closed at app shutdown; a per-request ``default_registry()``
            # built (and leaked) a fresh set of TLS connections every call.
            bars = await loop_registry().get_ohlcv(exchanges, symbol, timeframe, limit=limit)
        except DataFetchError as exc:
            raise HTTPException(502, str(exc)) from exc
    return [
        {
            "timestamp": b.timestamp.isoformat(),
            "open": b.open, "high": b.high, "low": b.low, "close": b.close,
            "volume": b.volume,
        }
        for b in bars
    ]


@router.get("/sources")
def sources() -> list[str]:
    return [*list(_SOURCES), "synthetic"]


@router.get("/detect/{symbol}")
def detect(symbol: str) -> dict:
    """Auto-detect the data source for a ticker (venue + fetch symbol)."""
    return detect_data_source(symbol)


@router.get("/categories")
def categories() -> list[str]:
    """Selectable universes for the multi-ticker portfolio backtest."""
    return list(CATEGORIES)


@router.get("/universe")
def universe(
    category: str = Query(default="all"),
    n: int = Query(default=10, ge=1, le=50),
) -> dict:
    """Preview the tickers ``n_tickers`` + ``category`` would auto-select.

    This is the "choose how many tickers" surface: ask for N names from a named
    universe and get a deterministic selection to feed ``/backtest/portfolio``.
    """
    if category.lower() not in CATEGORIES:
        raise HTTPException(400, f"unknown category '{category}' (known: {', '.join(CATEGORIES)})")
    picked = select_universe(category, n)
    return {"category": category, "n": n, "available": len(picked), "tickers": picked}
