"""Instrument universe (auto_scanner tickers) + symbol → data-source resolution.

The ticker CSVs ship with ``gex``. Each symbol maps to a data source:
  * ``ru`` → MOEX ISS (ticker as-is, e.g. SBER)
  * ``crypto`` → Bybit spot (ticker → ``<TICKER>USDT``, e.g. BTC → BTCUSDT)
  * ``fx`` → yFinance (display symbol → yFinance symbol, e.g. DXY → DX-Y.NYB, GOLD → GC=F)
  * ``us`` / ``sectors`` → yFinance (ticker as-is, e.g. NVDA, XLB)
"""
from __future__ import annotations

import csv
from pathlib import Path

__all__ = ["CATEGORIES", "load_instruments", "resolve_symbol", "select_universe"]

#: Selectable universes for auto-picking N tickers (``all`` = every category).
CATEGORIES: tuple[str, ...] = ("us", "crypto", "fx", "ru", "sectors", "all")

_GEX_DIR = Path(__file__).resolve().parents[2] / "gex"

_CATEGORY = {
    "auto_scanner_tickers.csv": "us",
    "auto_scanner_tickers_crypto.csv": "crypto",
    "auto_scanner_tickers_fx.csv": "fx",
    "auto_scanner_tickers_ru.csv": "ru",
    "auto_scanner_tickers_sectors.csv": "sectors",
}

#: Stablecoin quote suffixes that mark an explicit pair (``ETHUSDT``) as crypto
#: even when the ticker is not in the CSV universe. Restricted to stablecoins so
#: a fiat-looking symbol (``EURUSD``) is never misrouted to Bybit.
_CRYPTO_QUOTES = ("USDT", "USDC", "BUSD")

#: FX / commodity display symbols → yFinance symbols.
_FX_YF = {
    "DXY": "DX-Y.NYB",
    "EUR/USD": "EURUSD=X",
    "USD/CNY": "CNY=X",
    "USD/JPY": "JPY=X",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
}


def load_instruments() -> list[dict]:
    out: list[dict] = []
    for path in sorted(_GEX_DIR.glob("auto_scanner_tickers*.csv")):
        category = _CATEGORY.get(path.name, "other")
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        header = [h.strip() for h in rows[0]]
        has_name = "name" in header
        ticker_idx = header.index("ticker") if "ticker" in header else 0
        name_idx = header.index("name") if has_name else None
        for row in rows[1:]:
            if not row or not row[ticker_idx].strip():
                continue
            symbol = row[ticker_idx].strip()
            name = row[name_idx].strip() if (name_idx is not None and name_idx < len(row)) else symbol
            out.append({"symbol": symbol, "name": name, "category": category})
    return out


def select_universe(category: str = "all", n: int = 10) -> list[dict]:
    """Pick up to ``n`` instruments from ``category``.

    This is the backend for "choose the amount of tickers for the API": the
    client asks for N tickers from a named universe and gets a deterministic,
    reproducible selection to feed the portfolio backtest.

    A single category is walked in declared order. ``"all"`` is instead
    *round-robined* across categories, so asking for 50 names yields a spread
    over us/crypto/fx/ru/sectors rather than 50 US equities — a basket is far
    more useful when it is diversified.
    """
    if n < 1:
        return []
    cat = (category or "all").lower()
    if cat not in CATEGORIES:
        raise ValueError(f"unknown category '{category}' (known: {', '.join(CATEGORIES)})")
    instruments = load_instruments()
    if cat != "all":
        instruments = [i for i in instruments if i["category"] == cat]
        ordered = instruments
    else:
        ordered = _round_robin(instruments)
    # De-duplicate by symbol, preserving order.
    seen: set[str] = set()
    picked: list[dict] = []
    for inst in ordered:
        key = inst["symbol"].upper()
        if key in seen:
            continue
        seen.add(key)
        picked.append(inst)
        if len(picked) >= n:
            break
    return picked


def _round_robin(instruments: list[dict]) -> list[dict]:
    """Interleave instruments across categories, preserving per-category order."""
    buckets: dict[str, list[dict]] = {}
    for inst in instruments:
        buckets.setdefault(inst["category"], []).append(inst)
    # Keep the declared category order, then any extras.
    order = [c for c in CATEGORIES if c != "all" and c in buckets]
    order += [c for c in buckets if c not in order]
    out: list[dict] = []
    i = 0
    while any(len(buckets[c]) > i for c in order):
        for c in order:
            if len(buckets[c]) > i:
                out.append(buckets[c][i])
        i += 1
    return out


def resolve_symbol(symbol: str) -> dict:
    """Return ``{exchange, fetch_symbol, category}`` for a ticker."""
    up = symbol.upper()
    instruments = load_instruments()
    category = next((i["category"] for i in instruments if i["symbol"].upper() == up), None)

    if category == "ru":
        return {"exchange": "moex", "fetch_symbol": symbol, "category": category}
    if category == "crypto":
        fs = symbol if "USDT" in up else f"{symbol}USDT"
        return {"exchange": "bybit", "fetch_symbol": fs, "category": category}
    if category == "fx":
        return {"exchange": "yfinance", "fetch_symbol": _FX_YF.get(symbol, symbol), "category": category}
    if category is None:
        pair = _crypto_pair(symbol)
        if pair is not None:
            return {"exchange": "bybit", "fetch_symbol": pair, "category": "crypto"}
    # us / sectors / unknown → yFinance as-is
    return {"exchange": "yfinance", "fetch_symbol": symbol, "category": category or "us"}


def _crypto_pair(symbol: str) -> str | None:
    """``ETHUSDT`` / ``BTC-USDT`` → ``ETHUSDT`` (Bybit), otherwise ``None``."""
    up = symbol.upper().replace("-", "").replace("/", "")
    for quote in _CRYPTO_QUOTES:
        if up.endswith(quote) and len(up) > len(quote):
            return up
    return None
