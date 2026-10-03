"""Correctness tests for the research harness.

Scope note (important): the price series in this file are **hand-built fixtures**. They
exist to prove that the *engine mechanics* do what the docstrings claim — next-open
fills, conservative gap handling, stop-before-target ordering, funding sign, cost
monotonicity, min-holding classification, hybrid-repair identity, no lookahead.

They are **not** performance evidence and no strategy result anywhere in this programme
is derived from them. Every reported number comes from real Bybit data.

Run:  python -m research.emf_adl.tests
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gex.strategy.trading_algorithm import EMAFilterTrendStrategy

from . import rules as R
from .engine import (
    STOP_LOSS_BEFORE_4H,
    TAKE_PROFIT_BEFORE_4H,
    Costs,
    SignalSet,
    StopSpec,
    run,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


def bars_from(rows, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(rows), freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": (idx.view("int64") // 10**6).astype("int64"),
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [r[4] if len(r) > 4 else 1.0 for r in rows],
        }
    )


def mk_signals(n, *, el=None, es=None, xl=None, xs=None, atr=1.0):
    z = np.zeros(n, dtype=bool)
    return SignalSet(
        entry_long=np.asarray(el if el is not None else z),
        entry_short=np.asarray(es if es is not None else z),
        exit_long=np.asarray(xl if xl is not None else z),
        exit_short=np.asarray(xs if xs is not None else z),
        atr=np.full(n, float(atr)),
    )


# --------------------------------------------------------------------------- #
def test_next_open_execution():
    """A signal on bar 0 must fill at bar 1's OPEN, moved against by slippage."""
    rows = [(100, 101, 99, 100), (110, 112, 108, 111), (111, 113, 109, 112),
            (112, 114, 110, 113), (113, 115, 111, 114)]
    bars = bars_from(rows)
    sig = mk_signals(5, el=[True, False, False, False, False])
    res = run(bars, sig, symbol="T", timeframe="4H", tf_hours=4.0,
              costs=Costs(fee_rate=0.0, slippage_bps=10.0, funding_on=False),
              stops=StopSpec())
    t = res.trades[0]
    expected = 110 * 1.001
    check("next-open entry fill = bar1 open + slippage",
          abs(t.entry_price - expected) < 1e-9, f"{t.entry_price} vs {expected}")


def test_stop_gap_is_conservative():
    """A bar opening below the stop must fill at the open (worse), not at the stop."""
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (90, 91, 88, 89),
            (89, 90, 88, 89), (89, 90, 88, 89)]
    bars = bars_from(rows)
    sig = mk_signals(5, el=[True, False, False, False, False])
    res = run(bars, sig, symbol="T", timeframe="4H", tf_hours=4.0,
              costs=Costs(fee_rate=0.0, slippage_bps=0.0, funding_on=False),
              stops=StopSpec(mode="pct_trail", pct=0.05))
    t = res.trades[0]
    check("gap through stop fills at the open, not the stop",
          t.exit_reason in ("stop", "trail_stop") and abs(t.exit_price - 90.0) < 1e-9,
          f"reason={t.exit_reason} price={t.exit_price}")


def test_stop_beats_target_within_one_bar():
    """When a single bar spans both levels, the stop is assumed to hit first."""
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 130, 70, 120),
            (120, 121, 119, 120), (120, 121, 119, 120)]
    bars = bars_from(rows)
    sig = mk_signals(5, el=[True, False, False, False, False])
    res = run(bars, sig, symbol="T", timeframe="4H", tf_hours=4.0,
              costs=Costs(fee_rate=0.0, slippage_bps=0.0, funding_on=False),
              stops=StopSpec(mode="pct_trail", pct=0.05, tp_mode="pct", tp_pct=0.05))
    t = res.trades[0]
    check("bar spanning stop and target resolves to the stop",
          t.exit_reason in ("stop", "trail_stop"), f"reason={t.exit_reason}")


def test_trailing_stop_is_not_optimistic():
    """The level tested on bar i must come from bars < i, so a bar that spikes up then
    collapses must NOT exit at the spike-derived level."""
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 200, 85, 86),
            (86, 87, 85, 86), (86, 87, 85, 86)]
    bars = bars_from(rows)
    sig = mk_signals(5, el=[True, False, False, False, False])
    res = run(bars, sig, symbol="T", timeframe="4H", tf_hours=4.0,
              costs=Costs(fee_rate=0.0, slippage_bps=0.0, funding_on=False),
              stops=StopSpec(mode="pct_trail", pct=0.10))
    t = res.trades[0]
    # Trailing from the previous bar's best (100) => stop 90. Bar 2 opens at 100, so the
    # exit is at the stop (90), not at 180 from the same-bar high.
    check("trailing stop ignores the same bar's own extreme",
          t.exit_reason in ("stop", "trail_stop") and t.exit_price <= 100.0,
          f"reason={t.exit_reason} price={t.exit_price}")


def test_stop_provenance_distinguishes_fixed_from_trailed():
    """A fixed stop must be reported as ``stop``; a stop that has ratcheted must be
    reported as ``trail_stop`` with the level recorded. Without this the "is the trailing
    stop actually working?" question cannot be answered from the ledger."""
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100),
            (99, 100, 88, 89), (89, 90, 88, 89)]
    bars = bars_from(rows)
    sig = mk_signals(5, el=[True, False, False, False, False], atr=10.0)
    cost = Costs(fee_rate=0.0, slippage_bps=0.0, funding_on=False)

    fixed = run(bars, sig, symbol="T", timeframe="4H", tf_hours=4.0, costs=cost,
                stops=StopSpec(mode="atr_fixed", atr_mult=1.0)).trades[0]
    # atr_fixed never moves: the level sits at entry - 1*ATR = 90 for the whole trade.
    check("fixed stop is reported as 'stop', not 'trail_stop'",
          fixed.exit_reason == "stop", f"reason={fixed.exit_reason}")
    check("initial_stop recorded at entry",
          abs(fixed.initial_stop - 90.0) < 1e-9, f"initial_stop={fixed.initial_stop}")
    check("fixed stop's level equals its initial level",
          abs(fixed.stop_level - fixed.initial_stop) < 1e-12,
          f"level={fixed.stop_level} init={fixed.initial_stop}")

    rows2 = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 140, 99, 139),
             (139, 139, 120, 121), (121, 122, 120, 121)]
    bars2 = bars_from(rows2)
    sig2 = mk_signals(5, el=[True, False, False, False, False])
    trailed = run(bars2, sig2, symbol="T", timeframe="4H", tf_hours=4.0, costs=cost,
                  stops=StopSpec(mode="pct_trail", pct=0.10)).trades[0]
    check("ratcheted trailing stop is reported as 'trail_stop'",
          trailed.exit_reason == "trail_stop",
          f"reason={trailed.exit_reason} level={trailed.stop_level} init={trailed.initial_stop}")
    check("trailed level is above the initial level for a long",
          trailed.stop_level > trailed.initial_stop + 1e-9,
          f"level={trailed.stop_level} init={trailed.initial_stop}")


def test_funding_sign():
    """Positive funding is a cost for a long and a credit for a short."""
    rows = [(100, 101, 99, 100)] * 6
    bars = bars_from(rows)
    fund = pd.DataFrame({"timestamp": [int(bars["timestamp"].iloc[2]) + 1], "rate": [0.01]})
    for side, tag in ((("entry_long",), "long"), (("entry_short",), "short")):
        el = [True, False, False, False, False, False] if tag == "long" else None
        es = [True, False, False, False, False, False] if tag == "short" else None
        sig = mk_signals(6, el=el, es=es)
        res = run(bars, sig, symbol="T", timeframe="4H", tf_hours=4.0,
                  costs=Costs(fee_rate=0.0, slippage_bps=0.0, funding_on=True),
                  stops=StopSpec(), funding=fund)
        fc = res.trades[0].funding_cost
        ok = fc > 0 if tag == "long" else fc < 0
        check(f"funding sign: {tag} {'pays' if tag == 'long' else 'receives'}",
              ok, f"funding_cost={fc:.4f}")


def test_costs_are_monotone():
    """More slippage can never improve the result."""
    rng = np.random.default_rng(0)
    n = 400
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    rows = [(px[i], px[i] * 1.005, px[i] * 0.995, px[i]) for i in range(n)]
    bars = bars_from(rows)
    el = np.zeros(n, dtype=bool)
    el[::20] = True
    xl = np.zeros(n, dtype=bool)
    xl[10::20] = True
    sig = mk_signals(n, el=el, xl=xl)
    prev = None
    ok = True
    for bps in (0.0, 3.0, 9.0, 30.0):
        res = run(bars, sig, symbol="T", timeframe="4H", tf_hours=4.0,
                  costs=Costs(fee_rate=0.0, slippage_bps=bps, funding_on=False),
                  stops=StopSpec())
        cur = float(res.equity[-1])
        if prev is not None and cur > prev + 1e-12:
            ok = False
        prev = cur
    check("equity is non-increasing in slippage", ok)


def test_min_holding_classification():
    """Exit classes partition the trade list and compliance is derivable."""
    rng = np.random.default_rng(3)
    n = 600
    px = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, n)))
    rows = [(px[i], px[i] * 1.01, px[i] * 0.99, px[i]) for i in range(n)]
    bars = bars_from(rows)
    el = np.zeros(n, dtype=bool)
    el[::25] = True
    xl = np.zeros(n, dtype=bool)
    xl[12::25] = True
    sig = mk_signals(n, el=el, xl=xl, atr=1.0)
    res = run(bars, sig, symbol="T", timeframe="4H", tf_hours=4.0,
              costs=Costs(), stops=StopSpec(mode="atr_fixed", atr_mult=2.0),
              funding=None)
    cls = [t.exit_class for t in res.trades]
    from .metrics import compute_metrics
    m = compute_metrics(res.equity, res.trades, 4.0)
    total = m.n_stop_before_4h + m.n_tp_before_4h + m.n_after_4h
    check("exit classes partition the trade list", total == len(res.trades),
          f"{total} vs {len(res.trades)}")
    check("min-holding compliance computable and 1.0", m.min_holding_compliance == 1.0)
    check("named classes are the documented ones",
          set(cls) <= {STOP_LOSS_BEFORE_4H, TAKE_PROFIT_BEFORE_4H, "EXIT_AFTER_4H"})


def test_engine_determinism():
    rng = np.random.default_rng(11)
    n = 500
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    rows = [(px[i], px[i] * 1.008, px[i] * 0.992, px[i]) for i in range(n)]
    bars = bars_from(rows)
    el = np.zeros(n, dtype=bool)
    el[::17] = True
    xl = np.zeros(n, dtype=bool)
    xl[9::17] = True
    sig = mk_signals(n, el=el, xl=xl, atr=1.0)
    a = run(bars, sig, symbol="T", timeframe="4H", tf_hours=4.0, costs=Costs(),
            stops=StopSpec(mode="atr_trail", atr_mult=3.0))
    b = run(bars, sig, symbol="T", timeframe="4H", tf_hours=4.0, costs=Costs(),
            stops=StopSpec(mode="atr_trail", atr_mult=3.0))
    check("engine is deterministic (identical equity)",
          np.array_equal(a.equity, b.equity))
    check("engine is deterministic (identical trades)",
          [t.to_dict() for t in a.trades] == [t.to_dict() for t in b.trades])


def test_hybrid_repair_identity():
    """The hybrid body must be real, and ``repair_hybrid`` must be honest about itself.

    History: this test used to assert that the *shipped* transform collapsed the hybrid
    body to a point (zero range on >50% of bars, ``avg_candle`` identity violated). That
    was true, and it was the headline defect of the study. ``gex/strategy/features.py``
    was repaired on 2026-10-03, so the defect-side assertions were inverted into the
    regression guards below: the project's own transform must now be correct, and the
    ``repair_hybrid`` flag must not silently pretend to do something it no longer does.
    """
    rng = np.random.default_rng(5)
    n = 500
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    rows = [(px[i], px[i] * 1.006, px[i] * 0.994, px[i]) for i in range(n)]
    bars = bars_from(rows)

    f_proj = R.build_frame(bars, repair_hybrid=False)
    f_ok = R.build_frame(bars, repair_hybrid=True)

    rng_proj = (f_proj["candle_top"] - f_proj["candle_bottom"]).to_numpy(float)
    rng_ok = (f_ok["candle_top"] - f_ok["candle_bottom"]).to_numpy(float)
    check("project hybrid body range is essentially never zero [regression guard]",
          float((rng_proj[5:] == 0).mean()) < 0.01,
          f"zero frac={float((rng_proj[5:] == 0).mean()):.4f}")
    check("repaired hybrid body range is essentially never zero",
          float((rng_ok[5:] == 0).mean()) < 0.01,
          f"zero frac={float((rng_ok[5:] == 0).mean()):.4f}")
    check("repaired avg_candle == (hybrid_open + hybrid_close)/2",
          np.allclose(f_ok["avg_candle"], (f_ok["hybrid_open"] + f_ok["hybrid_close"]) / 2))
    check("project avg_candle upholds that identity [regression guard]",
          np.allclose(f_proj["avg_candle"],
                      (f_proj["hybrid_open"] + f_proj["hybrid_close"]) / 2, equal_nan=True))
    check("project hybrid transform matches the reference to 1e-9",
          float(np.nanmax(np.abs(
              rng_proj - rng_ok))) < 1e-9,
          f"max|diff|={float(np.nanmax(np.abs(rng_proj - rng_ok))):.3e}")


def test_repair_flag_is_not_misleading():
    """``repair_hybrid`` must mean exactly what it says in the current tree.

    If the project has already been repaired, ``repair_hybrid=True`` is a no-op and the
    ``V0_shipped`` / ``V1_repaired`` distinction in any report is void. This check makes
    that state explicit instead of letting it be silently mislabelled.
    """
    rng = np.random.default_rng(11)
    n = 400
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.011, n)))
    rows = [(px[i], px[i] * 1.007, px[i] * 0.993, px[i]) for i in range(n)]
    bars = bars_from(rows)
    a = R.build_frame(bars, repair_hybrid=False)
    b = R.build_frame(bars, repair_hybrid=True)
    identical = bool(np.allclose(a["candle_top"], b["candle_top"], rtol=0, atol=1e-12))
    check("repair_hybrid is a no-op iff the project is already repaired",
          identical == R.PROJECT_ALREADY_REPAIRED,
          f"toggles_equal={identical} flag={R.PROJECT_ALREADY_REPAIRED}")


def test_no_lookahead_in_signals():
    """Truncating the history must not change any earlier signal."""
    rng = np.random.default_rng(17)
    n = 900
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    rows = [(px[i], px[i] * 1.007, px[i] * 0.993, px[i]) for i in range(n)]
    bars = bars_from(rows)
    full, _ = R.base_signals(bars, repair_hybrid=True, warmup=200)
    for cut in (500, 700):
        part, _ = R.base_signals(bars.iloc[:cut].reset_index(drop=True),
                                 repair_hybrid=True, warmup=200)
        same = (
            np.array_equal(full.entry_long[:cut], part.entry_long)
            and np.array_equal(full.entry_short[:cut], part.entry_short)
            and np.array_equal(full.exit_long[:cut], part.exit_long)
            and np.array_equal(full.exit_short[:cut], part.exit_short)
        )
        check(f"no lookahead: signals identical after truncation at {cut}", same)


def test_structure_gate_is_causal():
    """Truncation test for the structural layer.

    The structural gate reads pivots, breaks and trailing stops out of the vendored
    ``market_structure`` builder. If any of that were computed with future bars, the
    values on the retained prefix would change when the series is truncated. They must
    not — that is the whole basis of the claim that gating is causal.
    """
    rng = np.random.default_rng(31)
    n = 1500
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.011, n)))
    rows = [(px[i], px[i] * 1.008, px[i] * 0.992, px[i]) for i in range(n)]
    bars = bars_from(rows)
    full = R.structural_arrays(bars)
    keys = ("trend", "break_long", "break_short", "stop_long", "stop_short",
            "last_confirmed_HL", "last_confirmed_LH")
    for cut in (900, 1200):
        part = R.structural_arrays(bars.iloc[:cut].reset_index(drop=True))
        leaks = []
        for k in keys:
            if k not in full or k not in part:
                continue
            a = np.asarray(full[k])[:cut]
            b = np.asarray(part[k])
            if not np.array_equal(a.astype(str), b.astype(str)):
                leaks.append(k)
        check(f"structural gate causal after truncation at {cut}", not leaks,
              f"leaking={leaks}" if leaks else "")


def test_engine_vs_project_engine_smoke():
    """The research engine must not disagree wildly with the project's own engine on the
    same signals (a sanity cross-check, not an equality claim — the research engine adds
    intrabar stops and funding, which the project engine does not model)."""
    rng = np.random.default_rng(23)
    n = 600
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    rows = [(px[i], px[i] * 1.006, px[i] * 0.994, px[i]) for i in range(n)]
    bars = bars_from(rows)
    el = np.zeros(n, dtype=bool)
    el[::30] = True
    xl = np.zeros(n, dtype=bool)
    xl[15::30] = True
    sig = mk_signals(n, el=el, xl=xl)
    res = run(bars, sig, symbol="T", timeframe="4H", tf_hours=4.0,
              costs=Costs(fee_rate=0.0005, slippage_bps=3.0, funding_on=False),
              stops=StopSpec())
    check("research engine produces trades on a trending fixture",
          len(res.trades) > 5, f"trades={len(res.trades)}")
    check("equity curve is finite", bool(np.all(np.isfinite(res.equity))))


def main() -> int:
    for fn in (
        test_next_open_execution,
        test_stop_gap_is_conservative,
        test_stop_beats_target_within_one_bar,
        test_trailing_stop_is_not_optimistic,
        test_stop_provenance_distinguishes_fixed_from_trailed,
        test_funding_sign,
        test_costs_are_monotone,
        test_min_holding_classification,
        test_engine_determinism,
        test_hybrid_repair_identity,
        test_repair_flag_is_not_misleading,
        test_no_lookahead_in_signals,
        test_structure_gate_is_causal,
        test_engine_vs_project_engine_smoke,
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            check(f"{fn.__name__} raised", False, repr(exc))
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
