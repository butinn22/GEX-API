"""Data source management: per-ticker auto-detection + synthesize toggle.

Auto-detection resolves a ticker to the exchange that actually lists it
(MOEX for RU names, Bybit for crypto pairs, yFinance for US/FX/unknown) via
the instrument universe, so callers never ask the wrong venue for a symbol.
The *synthesize* toggle overrides any detected source with the deterministic
generative feed (seeded by symbol, so results are reproducible).
"""
from __future__ import annotations

import zlib
from typing import Any

from trading.application.instruments import resolve_symbol

__all__ = ["detect_data_source", "synthetic_seed"]


def synthetic_seed(symbol: str) -> int:
    """Deterministic seed per symbol (stable across runs and processes)."""
    return 0 if symbol.upper() == "SYNTH" else zlib.crc32(symbol.upper().encode())


def detect_data_source(symbol: str) -> dict[str, Any]:
    """Resolve a ticker to its data source.

    Returns the detected ``source`` (exchange), the venue-specific
    ``fetch_symbol``, the instrument ``category``, and whether the
    generative feed can replace the real source (always — it needs no
    listing).
    """
    info = resolve_symbol(symbol)
    if symbol.upper() == "SYNTH":  # the built-in generative instrument
        info = {"exchange": "synthetic", "fetch_symbol": "SYNTH", "category": "synthetic"}
    return {
        "symbol": symbol.upper(),
        "source": info["exchange"],
        "fetch_symbol": info["fetch_symbol"],
        "category": info["category"],
        "synthetic_available": True,
    }
