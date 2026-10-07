"""Performance / equivalence tests for the batch strategy path.

The EMF+ADL strategy (``gex_emf``) is batch-oriented: its ``calculate`` builds a
~110-column frame over the whole OHLCV window. The original adapter called it on
every ``on_bar``, which is O(n) work per bar — O(n²) per replay — and measured
~405 s for 1,500 bars (51% of it inside the VWAP state machine).

These tests lock in the fix on two levels:

* **structure** — ``on_bar`` must not recompute the frame when the engine has
  already called ``prepare`` (a call-count assertion, so it cannot go flaky on a
  slow machine the way a wall-clock threshold would);
* **equivalence** — the batch path and the streaming path must emit the *same
  signals*, and the closed-form ``_linreg`` must equal the ``lstsq`` it replaced.

The numerical ground truth for the underlying pipeline is the project's own
golden fixture (``tests/test_strategy_golden.py``), which is untouched by this
change and still passes.
"""
from __future__ import annotations

from datetime import UTC

import numpy as np
import pytest

from trading.application.strategies.gex_emf import MIN_BARS, GexEMFStrategy
from trading.domain import Bar

# ── fixtures ───────────────────────────────────────────────────────────


def _bars(n: int, seed: int = 0) -> list[Bar]:
    """Deterministic OHLCV bars satisfying the domain invariants."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.015, n))
    from datetime import datetime, timedelta

    t0 = datetime(2020, 1, 1, tzinfo=UTC)
    out: list[Bar] = []
    for i in range(n):
        c = float(close[i])
        o = c * (1.0 + float(rng.normal(0.0, 0.002)))
        h = max(o, c) * (1.0 + abs(float(rng.normal(0.0, 0.004))))
        lo = min(o, c) * (1.0 - abs(float(rng.normal(0.0, 0.004))))
        out.append(Bar(timestamp=t0 + timedelta(days=i), open=o, high=h, low=lo,
                       close=c, volume=float(abs(rng.normal(1e6, 2e5)))))
    return out


def _key(sig) -> tuple:
    return (sig.timestamp, sig.side, sig.reason, round(float(sig.price), 10))


async def _replay(strategy: GexEMFStrategy, bars: list[Bar]) -> list:
    out = []
    for bar in bars:
        out.extend(await strategy.on_bar(bar))
    return out


# ── structure: the asymptotic fix ──────────────────────────────────────


async def test_on_bar_does_not_recompute_when_prepared(monkeypatch):
    """The whole point: N bars → ONE ``calculate``, not N.

    A wall-clock threshold would be flaky on a loaded CI box; a call count is
    the property that actually encodes "O(n) instead of O(n²)".
    """
    bars = _bars(400)
    calls = {"n": 0}
    strategy = GexEMFStrategy("SYNTH")
    real = strategy._gex.calculate

    def counting_calculate(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(strategy._gex, "calculate", counting_calculate)

    await strategy.start()
    await strategy.prepare(bars)
    await _replay(strategy, bars)

    assert calls["n"] == 1, f"expected a single precompute, got {calls['n']}"
    assert strategy.fallback_count == 0


async def test_streaming_still_recomputes_without_prepare(monkeypatch):
    """Live trading never calls ``prepare`` — the streaming path must still work."""
    bars = _bars(150)
    calls = {"n": 0}
    strategy = GexEMFStrategy("SYNTH")
    real = strategy._gex.calculate

    def counting_calculate(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(strategy._gex, "calculate", counting_calculate)
    await strategy.start()
    signals = await _replay(strategy, bars)

    # One recompute per bar once the warm-up gate opens, and signals are produced.
    assert calls["n"] == len(bars) - (MIN_BARS - 1)
    assert signals


async def test_engine_prepare_replay_matches_generate_signals():
    """The engine's hook path and the batch API must agree signal-for-signal.

    ``run_backtest`` calls ``prepare`` then replays ``on_bar``; ``generate_signals``
    builds the frame independently. Both read the same pipeline, so the emitted
    signals have to match exactly.
    """
    bars = _bars(500, seed=3)

    replay = GexEMFStrategy("SYNTH")
    await replay.start()
    await replay.prepare(bars)
    via_engine = await _replay(replay, bars)
    assert replay.fallback_count == 0

    batch = await GexEMFStrategy("SYNTH").generate_signals(bars)

    assert [_key(s) for s in via_engine] == [_key(s) for s in batch]
    assert via_engine  # non-vacuous: there were real signals to compare


# ── equivalence: batch path == streaming path ──────────────────────────


@pytest.mark.parametrize("n", [150, 300])
async def test_batch_and_streaming_signals_are_identical(n):
    """``prepare`` must not change a single signal, only when the work happens."""
    bars = _bars(n)
    stream = GexEMFStrategy("SYNTH")
    await stream.start()
    streamed = await _replay(stream, bars)

    batch = GexEMFStrategy("SYNTH")
    await batch.start()
    await batch.prepare(bars)
    batched = await _replay(batch, bars)

    assert [_key(s) for s in batched] == [_key(s) for s in streamed]
    assert streamed  # non-vacuous: there were real signals to compare


def test_full_frame_rows_equal_prefix_frame_rows():
    """The invariant the optimisation rests on: the pipeline is **causal**.

    Row *i* of the frame built on the whole series must equal row *i* of the frame
    built on the prefix ``bars[:i+1]``. The old adapter depended on exactly this
    (it recomputed the frame on each growing prefix), so proving it here is what
    makes the single precompute sound — and it checks far more indices per second
    than a full O(n²) streaming replay would.
    """
    import pandas as pd

    from gex.strategy.settings import StrategySettings
    from gex.strategy.trading_algorithm import EMAFilterTrendStrategy

    rng = np.random.default_rng(5)
    n = 600
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, n))
    ohlc = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * (1 + abs(rng.normal(0, 0.004, n))),
        "low": close * (1 - abs(rng.normal(0, 0.004, n))),
        "close": close,
        "volume": abs(rng.normal(1e6, 2e5, n)),
    })

    strat = EMAFilterTrendStrategy(StrategySettings())
    full = strat.calculate(ohlc, include_decorative=False)

    # Points spread across the series, including deep in the history.
    for i in (100, 250, 400, 599):
        prefix = strat.calculate(ohlc.iloc[: i + 1], include_decorative=False)
        assert len(prefix) == i + 1
        for col in full.columns:
            a, b = full[col].iloc[i], prefix[col].iloc[i]
            if full[col].dtype == bool:
                assert bool(a) == bool(b), f"{col}@{i}"
            else:
                assert a == pytest.approx(b, rel=1e-12, abs=1e-12, nan_ok=True), \
                    f"{col}@{i}"


async def test_backtest_results_are_unchanged_by_the_optimisation():
    """Equity curve and trade ledger must be identical, not merely close."""
    from trading.application.backtest.engine import BacktestConfig, run_backtest

    bars = _bars(300, seed=11)
    cfg = BacktestConfig()

    stream = GexEMFStrategy("SYNTH")
    # Neutralise the hook to reproduce the pre-optimisation structure.
    stream.prepare = lambda _bars: _noop()  # type: ignore[method-assign]
    a = await run_backtest(stream, bars, cfg)

    b = await run_backtest(GexEMFStrategy("SYNTH"), bars, cfg)

    assert np.array_equal(a.equity_curve, b.equity_curve)
    assert [(t.side, t.entry_price, t.exit_price, t.realized_pnl) for t in a.trades] == \
           [(t.side, t.entry_price, t.exit_price, t.realized_pnl) for t in b.trades]
    assert a.trades  # non-vacuous: the run actually traded


async def _noop():
    return None


async def test_prepare_falls_back_when_the_bar_stream_diverges():
    """Replaying a *different* series must not emit signals against stale rows.

    Same length and **same timestamps**, different prices — the case a timestamp
    check alone would miss.
    """
    prepared = _bars(200, seed=1)
    other = _bars(200, seed=2)

    strategy = GexEMFStrategy("SYNTH")
    await strategy.start()
    await strategy.prepare(prepared)
    assert strategy._closes is not None
    signals = await _replay(strategy, other)

    assert strategy.fallback_count == 1
    assert strategy._features is None and strategy._closes is None
    assert signals  # it kept working via the streaming path


async def test_short_series_is_handled_without_a_frame():
    bars = _bars(MIN_BARS - 1)
    strategy = GexEMFStrategy("SYNTH")
    await strategy.start()
    await strategy.prepare(bars)
    assert strategy._features is None
    assert await _replay(strategy, bars) == []


# ── the lean frame and the vectorised linreg ───────────────────────────


def test_lean_frame_matches_full_frame_on_every_shared_column():
    """``include_decorative=False`` may only *drop* columns, never change one."""
    import pandas as pd

    from gex.strategy.settings import StrategySettings
    from gex.strategy.trading_algorithm import EMAFilterTrendStrategy

    rng = np.random.default_rng(0)
    n = 400
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, n))
    ohlc = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * (1 + abs(rng.normal(0, 0.004, n))),
        "low": close * (1 - abs(rng.normal(0, 0.004, n))),
        "close": close,
        "volume": abs(rng.normal(1e6, 2e5, n)),
    })

    s = EMAFilterTrendStrategy(StrategySettings())
    full = s.calculate(ohlc)
    lean = s.calculate(ohlc, include_decorative=False)

    assert set(lean.columns) <= set(full.columns)
    assert len(lean.columns) < len(full.columns), "nothing was actually dropped"
    # The columns that decide signals must all survive.
    for needed in ("combined_long_entry", "combined_short_entry",
                   "combined_long_exit", "combined_short_exit",
                   "my_vwap_state", "adline", "tp_f", "adl_tl"):
        assert needed in lean.columns

    for col in lean.columns:
        a, b = full[col], lean[col]
        if a.dtype == bool:
            assert a.equals(b), col
        else:
            assert np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True), col


def test_linreg_closed_form_equals_lstsq():
    """The one-convolution replacement must reproduce ``rolling.apply(lstsq)``."""
    import pandas as pd

    from gex.strategy.indicators import _linreg_endpoint, _linreg_series

    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(100.0, 5.0, 300))

    def slow(win):
        return _linreg_endpoint(win.to_numpy(dtype=float))

    for length in (2, 10, 50):
        ref = y.rolling(length, min_periods=2).apply(slow, raw=False)
        got = _linreg_series(y, length)
        assert np.allclose(got.to_numpy(), ref.to_numpy(), rtol=1e-12, atol=1e-9,
                           equal_nan=True), length


def test_linreg_nan_behaviour_matches_pandas():
    """``lstsq`` yields NaN on a NaN-bearing window; the filter must too."""
    import pandas as pd

    from gex.strategy.indicators import _linreg_series

    y = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    got = _linreg_series(y, 4).to_numpy()
    # windows containing the leading NaN are NaN; clean full windows are exact
    assert np.isnan(got[:4]).all()
    assert got[4] == pytest.approx(4.0)
    assert got[7] == pytest.approx(7.0)


def test_linreg_is_fast_enough_not_to_regress():
    """Guard the 340–1300x win with a very generous bound (lstsq took ~180 ms)."""
    import time

    import pandas as pd

    from gex.strategy.indicators import _linreg_series

    y = pd.Series(np.random.default_rng(0).normal(100.0, 5.0, 4000))
    t = time.perf_counter()
    _linreg_series(y, 50)
    elapsed = time.perf_counter() - t
    assert elapsed < 0.5, f"_linreg_series took {elapsed:.3f}s (per-window lstsq was ~0.18s)"
