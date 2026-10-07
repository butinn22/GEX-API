"""The take-profit / trailing-stop overlay on the EMF+ADL strategy.

Why this exists
---------------
``gex_emf``'s ``combined_*_exit`` columns are pure indicator crossovers. The
TP/trailing rules live in ``gex.strategy.risk._RiskMixin`` and are only reached
by ``EMAFilterTrendStrategy.evaluate`` — which the column-driven backtest never
calls. So the ``tp_percent`` / ``trailing_percent`` / ``atr_tp_mult`` knobs the
UI offers were **silently inert**: the same request with different values returned
byte-identical results.

The overlay makes them live. Contract locked in here:

* **default OFF** — ``use_risk_exits`` defaults to False, so every previously
  recorded result is unchanged;
* **ON** — TP and trailing exits fire, and each knob changes the outcome;
* **no intrabar lookahead** — the trailing level tested against a bar's close is
  built only from *earlier* bars' extremes.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from trading.application.backtest.engine import BacktestConfig, run_backtest
from trading.application.strategies.gex_emf import GexEMFStrategy
from trading.domain import Bar

# ── fixtures ───────────────────────────────────────────────────────────


def _bars(n: int, seed: int = 0) -> list[Bar]:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.015, n))
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


def _strategy(use_risk_exits: bool, **settings):
    return GexEMFStrategy("SYNTH", settings=settings or None, use_risk_exits=use_risk_exits)


async def _signals(strategy: GexEMFStrategy, bars: list[Bar]) -> list:
    await strategy.start()
    await strategy.prepare(bars)
    out = []
    for bar in bars:
        out.extend(await strategy.on_bar(bar))
    return out


def _reasons(signals: list) -> list[str]:
    return [s.reason for s in signals]


# ── the flag itself ────────────────────────────────────────────────────


def test_use_risk_exits_defaults_to_off_and_is_plumbed_from_the_factory():
    from trading.application.strategy_factory import build_strategy

    assert build_strategy("gex_emf", "E", {})._use_risk_exits is False
    assert build_strategy("gex_emf", "E", {"settings": {}})._use_risk_exits is False
    # accepted from inside ``settings`` (how the UI sends it) …
    assert build_strategy("gex_emf", "E", {"settings": {"use_risk_exits": True}})._use_risk_exits
    # … and at the top level
    assert build_strategy("gex_emf", "E", {"use_risk_exits": True})._use_risk_exits
    # the flag is not leaked into StrategySettings (it is not a settings field)
    s = build_strategy("gex_emf", "E", {"settings": {"use_risk_exits": True, "tp_percent": 3.0}})
    assert s._settings.tp_percent == 3.0


async def test_disabled_by_default_emits_only_indicator_exits():
    bars = _bars(400, seed=4)
    signals = await _signals(_strategy(False, use_atr_stops=False), bars)
    reasons = set(_reasons(signals))
    assert "take_profit" not in reasons and "trailing_stop" not in reasons
    assert reasons <= {"long_entry", "short_entry", "long_exit", "short_exit"}


async def test_enabled_emits_risk_exits():
    bars = _bars(400, seed=4)
    signals = await _signals(
        _strategy(True, use_atr_stops=False, tp_percent=2.0, trailing_percent=1.0), bars
    )
    reasons = set(_reasons(signals))
    assert "take_profit" in reasons or "trailing_stop" in reasons


async def test_default_run_is_byte_identical_to_a_run_without_the_feature():
    """Guard the previously recorded results: OFF must reproduce them exactly."""
    bars = _bars(500, seed=9)
    cfg = BacktestConfig()

    a = await run_backtest(_strategy(False), bars, cfg)
    b = await run_backtest(GexEMFStrategy("SYNTH"), bars, cfg)  # flag never mentioned

    assert np.array_equal(a.equity_curve, b.equity_curve)
    assert a.trades == b.trades


# ── the knobs now bite ─────────────────────────────────────────────────


async def test_tighter_take_profit_exits_sooner():
    """A tighter target must not produce fewer exits than a wider one."""
    bars = _bars(600, seed=2)

    async def exits(tp: float) -> int:
        s = _strategy(True, use_atr_stops=False, tp_percent=tp, use_trailing=False)
        return _reasons(await _signals(s, bars)).count("take_profit")

    tight, wide = await exits(1.0), await exits(8.0)
    assert tight > wide > 0


async def test_tighter_trailing_stop_exits_sooner():
    bars = _bars(600, seed=2)

    async def exits(tr: float) -> int:
        s = _strategy(True, use_atr_stops=False, trailing_percent=tr, use_take_profit=False)
        return _reasons(await _signals(s, bars)).count("trailing_stop")

    tight, wide = await exits(0.3), await exits(10.0)
    assert tight > wide > 0


async def test_each_rule_can_be_switched_off_independently():
    bars = _bars(600, seed=2)

    no_tp = _reasons(await _signals(
        _strategy(True, use_atr_stops=False, use_take_profit=False), bars))
    assert "take_profit" not in no_tp and "trailing_stop" in no_tp

    no_trail = _reasons(await _signals(
        _strategy(True, use_atr_stops=False, use_trailing=False), bars))
    assert "trailing_stop" not in no_trail and "take_profit" in no_trail


async def test_atr_mode_uses_the_multipliers_instead_of_the_percentages():
    bars = _bars(600, seed=2)

    async def exits(mult: float) -> int:
        s = _strategy(True, use_atr_stops=True, atr_tp_mult=mult, use_trailing=False)
        return _reasons(await _signals(s, bars)).count("take_profit")

    assert await exits(1.0) > await exits(6.0) > 0

    # in ATR mode the percentage is irrelevant ...
    same_a = await _signals(
        _strategy(True, use_atr_stops=True, tp_percent=1.0, use_trailing=False), bars)
    same_b = await _signals(
        _strategy(True, use_atr_stops=True, tp_percent=9.0, use_trailing=False), bars)
    assert _reasons(same_a) == _reasons(same_b)


async def test_settings_change_the_backtest_outcome_through_the_engine():
    """End to end: the UI's knobs must move the equity curve."""
    bars = _bars(500, seed=6)
    cfg = BacktestConfig()

    off = await run_backtest(_strategy(False), bars, cfg)
    on = await run_backtest(
        _strategy(True, use_atr_stops=False, tp_percent=1.0, trailing_percent=0.5), bars, cfg)

    assert not np.array_equal(off.equity_curve, on.equity_curve)
    assert len(on.trades) != len(off.trades)


# ── no intrabar lookahead ──────────────────────────────────────────────


async def test_trailing_level_ignores_the_current_bars_high(monkeypatch):
    """The stop tested against bar *i*'s close must come from bars *< i*.

    Bar 1 is the trap: a huge upper wick (high 200) closing at 100. Two orderings
    disagree about that bar —

    * extremes updated **before** the check → trail jumps to 200·(1−1%) = 198,
      which is above the close of 100, so the bar fires;
    * extremes updated **after** the check → the trail is still the entry-based
      100·(1−1%) = 99, so nothing fires.

    Bar 2 then fires legitimately, because bar 1's wick *did* raise the trail by
    the time bar 2 is judged (198 vs close 100).
    """
    s = _strategy(True, use_atr_stops=False, tp_percent=2.0, trailing_percent=1.0)

    frame = pd.DataFrame({
        "open": [100.0, 100.0, 100.0],
        "high": [100.0, 200.0, 101.0],   # the wick on bar 1 is the trap
        "low": [100.0, 99.0, 100.0],
        "close": [100.0, 100.0, 100.0],
        "atr": [np.nan] * 3,
        "long_entry_signal": [True, False, False],
        "short_entry_signal": [False] * 3,
        "long_exit_signal": [False] * 3,
        "short_exit_signal": [False] * 3,
    })
    # And a control run where bar 1 has no wick: neither bar fires, proving the
    # bar-2 exit above is caused by the extreme, not by something else.
    calm = frame.copy()
    calm.loc[1, "high"] = 100.5

    t0 = datetime(2021, 1, 1, tzinfo=UTC)
    stamps = [t0 + timedelta(days=i) for i in range(3)]

    assert _reasons(s._signal_at(frame, 0, stamps[0])) == ["long_entry"]
    assert s._signal_at(frame, 1, stamps[1]) == []      # ← the ordering assertion
    assert _reasons(s._signal_at(frame, 2, stamps[2])) == ["trailing_stop"]

    control = _strategy(True, use_atr_stops=False, tp_percent=2.0, trailing_percent=1.0)
    assert _reasons(control._signal_at(calm, 0, stamps[0])) == ["long_entry"]
    assert control._signal_at(calm, 1, stamps[1]) == []
    assert control._signal_at(calm, 2, stamps[2]) == []


async def test_no_exit_is_emitted_on_the_entry_bar():
    """A position cannot be opened and risk-exited on the same bar."""
    bars = _bars(500, seed=8)
    signals = await _signals(
        _strategy(True, use_atr_stops=False, tp_percent=0.5, trailing_percent=0.2), bars)
    prev_entry = None
    for sig in signals:
        if sig.reason.endswith("_entry"):
            prev_entry = sig.timestamp
        else:
            assert sig.timestamp != prev_entry, "risk exit on the entry bar"
            prev_entry = None
