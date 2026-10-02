"""Tests for the signal pipeline."""
from __future__ import annotations

from trading.application.signal_pipeline import MinStrengthFilter, SignalPipeline, dedupe_by_key
from trading.domain import Side, Signal


def test_pipeline_filters_then_dedupes():
    sigs = [
        Signal("X", Side.BUY, "s", "r1", strength=0.5),
        Signal("X", Side.BUY, "s", "r2", strength=0.9),
        Signal("Y", Side.SELL, "s", "r3", strength=0.2),
    ]
    out = SignalPipeline([MinStrengthFilter(0.3)]).run(sigs)
    # the 0.2-strength signal is filtered; the other two survive
    assert {s.reason for s in out} == {"r1", "r2"}


def test_dedupe_keeps_latest_per_key():
    sigs = [
        Signal("X", Side.BUY, "s", "same", strength=0.5),
        Signal("X", Side.BUY, "s", "same", strength=0.8),
    ]
    out = dedupe_by_key(sigs)
    assert len(out) == 1 and out[0].strength == 0.8


def test_filter_rejects_bad_threshold():
    import pytest
    with pytest.raises(ValueError):
        MinStrengthFilter(1.5)
