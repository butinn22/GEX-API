"""Data validation: reject obviously-bad OHLCV before it reaches consumers."""
from __future__ import annotations

from trading.domain import Bar, DataFetchError

__all__ = ["validate_bars"]


def validate_bars(bars: list[Bar]) -> list[Bar]:
    """Return a validated, ascending, de-duplicated bar list; raise on gross errors."""
    if not bars:
        raise DataFetchError("no bars to validate")
    bars = sorted(bars, key=lambda b: b.timestamp)
    # de-duplicate by timestamp, keep the last
    dedup: dict = {}
    for b in bars:
        if b.close < 0 or b.high < b.low or b.volume < 0:
            raise DataFetchError(f"invalid bar values at {b.timestamp.isoformat()}")
        dedup[b.timestamp] = b
    out = list(dedup.values())
    out.sort(key=lambda b: b.timestamp)
    return out
