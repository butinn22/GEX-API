"""Tests for the GEX EMF+ADL strategy adapter."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from trading.application.strategies.gex_emf import GexEMFStrategy
from trading.domain import Bar, Side


def _bars(n: int = 150) -> list[Bar]:
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    bars = []
    for i in range(n):
        ts = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i)
        c = float(close[i])
        o = float(close[i - 1]) if i else c
        hi = max(o, c) * 1.01
        lo = min(o, c) * 0.99
        bars.append(Bar(ts, o, hi, lo, c, 1e6))
    return bars


async def test_gex_emf_generates_signals():
    bars = _bars(150)
    signals = await GexEMFStrategy("X").generate_signals(bars)
    assert len(signals) >= 1
    assert signals[0].side in (Side.BUY, Side.SELL)


async def test_gex_emf_on_bar_matches_batch():
    bars = _bars(120)
    batch = await GexEMFStrategy("X").generate_signals(bars)
    inc = GexEMFStrategy("X")
    inc_signals = []
    for b in bars:
        inc_signals.extend(await inc.on_bar(b))
    assert [s.side for s in batch] == [s.side for s in inc_signals]
    assert [s.reason for s in batch] == [s.reason for s in inc_signals]


async def test_gex_emf_insufficient_bars():
    signals = await GexEMFStrategy("X").generate_signals(_bars(30))
    assert signals == []  # below MIN_BARS


async def test_gex_emf_custom_settings():
    signals = await GexEMFStrategy("X", settings={"atr_length": 20, "rsi_length": 7}).generate_signals(_bars(150))
    assert len(signals) >= 1
