"""Deterministic execution engine with realistic costs and intrabar stop handling.

Execution model — every choice here is the *conservative* one:

======================  =====================================================
Event                   Treatment
======================  =====================================================
Signal                  Computed from bar ``i`` (close). Bar ``i`` is closed.
Entry fill              Bar ``i+1`` **open**, moved against the trade by slippage.
Indicator exit fill     Next bar **open**, moved against the trade by slippage.
Stop trigger            Bar high/low touches the level → assumed filled.
Stop fill price         ``min(stop, bar_open)`` for a long (a gap down fills at the
                        open, i.e. worse), then slippage applied against.
Take-profit trigger     Bar high/low touches the level → assumed filled.
TP fill price           The level itself. A favourable gap is **not** credited
                        (deliberately pessimistic).
Intrabar ambiguity      If a bar could have touched both stop and TP, the **stop**
                        is assumed to have hit first.
Funding                 Real 8h settlements, charged to the open position's notional
                        at every settlement inside the bar's window.
Gaps in the series      Carried through as a single bar; the position is marked at the
                        next real close. Missing bars are never interpolated.
======================  =====================================================

Early-exit classification (the 4-hour minimum-holding rule):

* ``STOP_LOSS_BEFORE_4H``     — a stop fired on the entry bar.
* ``TAKE_PROFIT_BEFORE_4H``   — a take-profit fired on the entry bar.
* ``EXIT_AFTER_4H``           — everything else (exited at >= 1 full bar after entry).

On the 4H grid one bar *is* the 4-hour minimum, so the classification is exact. On the
1D grid an exit inside the entry bar has an unobservable holding time in ``(0, 24h]``;
it is classified as *before 4h* because that is the conservative reading — the data
cannot prove it satisfied the minimum. This limitation is reported, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
#: Bybit linear-perp VIP0 **taker** fee. The strategy uses market orders on both legs,
#: so taker is the honest assumption. (VIP0 maker is 0.02%, but a market order is not
#: a maker order - using maker pricing here would be self-deception.)
DEFAULT_FEE_RATE = 0.00055
#: Per-side slippage = market impact + half-spread. 3 bps is a realistic figure for a
#: top-10-turnover perp at retail size; it is stressed to 6/9/15 bps in the robustness
#: battery. Bybit does not publish a spread series, so the half-spread is *estimated* -
#: and that estimate is disclosed wherever it is used.
DEFAULT_SLIPPAGE_BPS = 3.0

STOP_LOSS_BEFORE_4H = "STOP_LOSS_BEFORE_4H"
TAKE_PROFIT_BEFORE_4H = "TAKE_PROFIT_BEFORE_4H"
EXIT_AFTER_4H = "EXIT_AFTER_4H"


@dataclass(frozen=True)
class Costs:
    fee_rate: float = DEFAULT_FEE_RATE
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    funding_on: bool = True
    #: Extra per-side cost for gapping/illiquidity, applied on top of slippage.
    latency_bps: float = 0.0

    @property
    def slip(self) -> float:
        return (self.slippage_bps + self.latency_bps) / 10_000.0


@dataclass(frozen=True)
class StopSpec:
    """Stop / take-profit rules. All levels are fixed at entry (ATR frozen at entry),
    so a level never jumps because volatility moved after the trade was opened."""

    #: ``none`` | ``atr_fixed`` | ``atr_trail`` | ``pct_trail`` | ``external``
    mode: str = "none"
    atr_mult: float = 2.0
    pct: float = 0.02
    #: Optional per-bar structural stop level array (e.g. hybrid-candle pivots).
    external_long: np.ndarray | None = None
    external_short: np.ndarray | None = None
    tp_mode: str = "none"  # ``none`` | ``atr`` | ``pct``
    tp_atr_mult: float = 3.0
    tp_pct: float = 0.04
    #: Once the trade is this many R in profit, floor the stop at break-even. 0 = off.
    breakeven_at_r: float = 0.0
    #: Once the trade is this many R in profit, the trailing stop engages. Before that
    #: the stop is the fixed initial one, so an early wiggle cannot shake the trade out.
    #: 0 = trail from the first bar. This is what makes an ``atr_trail`` stop *work*:
    #: without it a wide trail sits beyond the noise and never fires.
    trail_activate_r: float = 0.0
    #: Trailing distance in ATR, *independent* of ``atr_mult``. 0 means "use atr_mult for
    #: both", which is the older behaviour. Set this to keep a wide initial catastrophe stop
    #: (atr_mult=8) while trailing much tighter (trail_atr_mult=3) once in profit. Without
    #: the split, "add a trailing stop" silently means "also narrow the initial stop", and
    #: the two effects cannot be told apart in the results.
    trail_atr_mult: float = 0.0
    #: Hard switch for the trailing behaviour itself. ``trail_atr_mult=0`` is NOT "no
    #: trail" — it means "fall back to ``atr_mult``", which produces an *immediate* trail at
    #: the full initial width. Expressing "no trail" through that sentinel silently ran a
    #: tight early trail instead of a fixed stop. Set ``trail=False`` to get a genuinely
    #: fixed stop out of an ``atr_trail`` spec, so the arm labelled "no trail" really has
    #: none and the two effects can be separated.
    trail: bool = True
    #: Optional regime/vol gate: skip entries where ``gate[i]`` is False.
    entry_gate: np.ndarray | None = None


@dataclass(frozen=True)
class Sizing:
    """Position sizing. Two independent levers, both aimed at drawdown.

    ``risk_per_trade`` is the important one. Sizing every trade to 100% notional means the
    risk per trade is whatever the (unknowable) volatility happens to be, so the portfolio
    drawdown is set by whether the universe happened to be volatile. Sizing so that
    ``|entry - stop| * qty`` equals a fixed fraction of equity makes the *risk* constant
    across trades and instruments, which is the mechanism that actually bounds drawdown.

    ``bar_scale`` is an external per-bar multiplier (the portfolio drawdown brake): it is
    filled in by ``portfolio.py`` from the panel equity curve and is strictly causal.
    """

    #: Fraction of equity risked between entry and the initial stop. 0 = not risk-based.
    risk_per_trade: float = 0.0
    #: Hard cap on notional / equity, applied after risk sizing.
    max_leverage: float = 1.0
    #: Optional ``(n,)`` per-bar multiplier in ``[0, 1]`` from the portfolio brake.
    bar_scale: np.ndarray | None = None
    #: Optional ``(n,)`` per-bar volatility-normalisation multiplier. Unlike ``bar_scale``
    #: this may exceed 1: it scales a quiet instrument *up* so every symbol contributes a
    #: comparable amount of risk. Strictly causal — set from closes up to ``i - 1``.
    vol_scale: np.ndarray | None = None
    #: Fallback stop distance as a fraction of price when no stop level exists.
    fallback_stop_pct: float = 0.02


@dataclass
class Trade:
    symbol: str
    side: int  # +1 long, -1 short
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    notional_at_entry: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    funding_cost: float
    net_pnl: float
    holding_bars: int
    holding_hours: float
    entry_bar: int
    exit_bar: int
    exit_reason: str  # stop | trail_stop | take_profit | indicator | eod
    exit_class: str
    #: Stop level in force when the position closed, and the level at entry. Comparing the
    #: two is what proves a trailing stop actually trailed rather than sat still.
    stop_level: float = 0.0
    initial_stop: float = 0.0
    mfe: float = 0.0  # max favourable excursion, in R of the initial risk
    mae: float = 0.0  # max adverse excursion, in R of the initial risk
    entry_mode: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["entry_time"] = pd.Timestamp(self.entry_time).isoformat()
        d["exit_time"] = pd.Timestamp(self.exit_time).isoformat()
        return d


@dataclass
class SignalSet:
    """Causal boolean arrays. ``entry_long[i]`` True means: *after bar i closed*, the
    strategy wants to be long. The engine fills it at bar ``i+1``'s open."""

    entry_long: np.ndarray
    entry_short: np.ndarray
    exit_long: np.ndarray
    exit_short: np.ndarray
    atr: np.ndarray
    warmup: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        n = len(self.entry_long)
        for name in ("entry_short", "exit_long", "exit_short", "atr"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"SignalSet.{name} length {len(arr)} != {n}")


@dataclass
class Result:
    symbol: str
    timeframe: str
    equity: np.ndarray
    timestamps: np.ndarray
    trades: list[Trade]
    costs: Costs
    label: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def net(self) -> np.ndarray:
        return np.asarray([t.net_pnl for t in self.trades], dtype=float)


# --------------------------------------------------------------------------- #
# Core replay
# --------------------------------------------------------------------------- #
def _stop_level(
    spec: StopSpec, side: int, entry: float, atr_entry: float, best: float,
    prev: float | None, bar: int, *, risk_unit: float = 0.0,
) -> float | None:
    """Stop level in force *at the close of bar ``bar``*.

    For ``external`` arrays the level used to test bar ``i`` is ``arr[i-1]``: a stop
    cannot be tightened by a pivot that only becomes confirmed at the close of the very
    bar it is being applied to.

    ``trail_activate_r`` gates the *trailing* behaviour only: until the trade is that many
    R in profit the level stays at its fixed initial position. This is what lets a trail be
    tight enough to fire without being shaken out by the entry's own noise.
    """
    if spec.mode == "none":
        return None
    if spec.mode == "atr_fixed":
        off = spec.atr_mult * atr_entry
        return entry - off if side > 0 else entry + off
    if spec.mode in ("atr_trail", "pct_trail"):
        if spec.mode == "atr_trail":
            off = spec.atr_mult * atr_entry
            # The trailing distance may differ from the initial width: a wide catastrophe
            # stop and a tight profit trail are different jobs.
            trail_off = (spec.trail_atr_mult or spec.atr_mult) * atr_entry
            base = entry - off if side > 0 else entry + off
            trail = best - trail_off if side > 0 else best + trail_off
        else:
            base = entry * (1 - spec.pct) if side > 0 else entry * (1 + spec.pct)
            trail = best * (1 - spec.pct) if side > 0 else best * (1 + spec.pct)
        if prev is None:
            return base
        if not spec.trail:
            # Trailing explicitly switched off: this is a fixed stop that happens to be
            # expressed through the trail branch.
            return base
        # Has the trade earned the right to be trailed yet?
        if spec.trail_activate_r > 0 and risk_unit > 0:
            favourable = (best - entry) if side > 0 else (entry - best)
            if favourable < spec.trail_activate_r * risk_unit:
                return base  # not yet; hold the fixed initial stop
        return max(prev, trail) if side > 0 else min(prev, trail)
    if spec.mode == "external":
        arr = spec.external_long if side > 0 else spec.external_short
        lvl = float(arr[bar - 1]) if (arr is not None and bar >= 1) else np.nan
        if np.isfinite(lvl):
            return lvl
        # Structural levels only exist once a pivot is confirmed — 33% of bars in this
        # dataset. Leaving the position unprotected on the other 67% would flatter the
        # drawdown, so the fallback is an explicit ATR stop and the usage is counted.
        if atr_entry <= 0:
            return None
        off = spec.atr_mult * atr_entry
        return entry - off if side > 0 else entry + off
    raise ValueError(f"unknown stop mode {spec.mode!r}")


def _tp_level(spec: StopSpec, side: int, entry: float, atr_entry: float) -> float | None:
    if spec.tp_mode == "none":
        return None
    if spec.tp_mode == "atr":
        off = spec.tp_atr_mult * atr_entry
        return entry + off if side > 0 else entry - off
    if spec.tp_mode == "pct":
        return entry * (1 + spec.tp_pct) if side > 0 else entry * (1 - spec.tp_pct)
    raise ValueError(f"unknown tp mode {spec.tp_mode!r}")


def run(
    bars: pd.DataFrame,
    signals: SignalSet,
    *,
    symbol: str,
    timeframe: str,
    tf_hours: float,
    costs: Costs | None = None,
    stops: StopSpec | None = None,
    funding: pd.DataFrame | None = None,
    position_fraction: float = 1.0,
    sizing: Sizing | None = None,
    initial_equity: float = 1.0,
    allow_shorts: bool = True,
) -> Result:
    """Replay ``bars`` through ``signals``. Pure, deterministic, no lookahead."""
    costs = costs or Costs()
    stops = stops or StopSpec()
    sizing = sizing or Sizing()
    n = len(bars)
    if n == 0:
        raise ValueError("no bars")
    if len(signals.entry_long) != n:
        raise ValueError("signals and bars disagree on length")

    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    low_ = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    ts = bars["timestamp"].to_numpy(dtype=np.int64)
    ts_dt = pd.to_datetime(ts, unit="ms", utc=True)
    step_ms = int(tf_hours * 3600 * 1000)

    # Funding settlements bucketed by the bar whose window contains them.
    fund_by_bar: dict[int, float] = {}
    if costs.funding_on and funding is not None and len(funding):
        ft = funding["timestamp"].to_numpy(dtype=np.int64)
        fr = funding["rate"].to_numpy(dtype=float)
        idx = np.searchsorted(ts, ft, side="right") - 1
        for k, rate in zip(idx, fr):
            if 0 <= k < n:
                fund_by_bar[int(k)] = fund_by_bar.get(int(k), 0.0) + float(rate)

    slip = costs.slip
    cash = float(initial_equity)
    qty = 0.0
    side = 0
    entry_bar = -1
    entry_price = 0.0
    entry_atr = 0.0
    stop = None
    tp = None
    best = 0.0
    entry_notional = 0.0
    fees_acc = 0.0
    slip_acc = 0.0
    fund_acc = 0.0
    mfe = 0.0
    mae = 0.0
    risk_unit = 0.0

    equity = np.empty(n, dtype=float)
    trades: list[Trade] = []

    def _mark(price: float) -> float:
        return cash + qty * price * side

    pending: tuple[str, int] | None = None  # ("enter"/"exit", direction)

    for i in range(n):
        # ---- 1. Execute yesterday's decision at today's open ----
        if pending is not None:
            kind, want = pending
            if kind == "enter":
                fill = o[i] * (1 + slip) if want > 0 else o[i] * (1 - slip)
                # The book is flat on entry, so equity at the fill price is just cash.
                eq_at_entry = _mark(fill)
                # Portfolio drawdown brake: a strictly causal per-bar multiplier.
                scale = 1.0
                if sizing.bar_scale is not None:
                    raw_scale = float(sizing.bar_scale[i])
                    if np.isfinite(raw_scale):
                        scale = min(max(raw_scale, 0.0), 1.0)
                if sizing.vol_scale is not None:
                    raw_vol = float(sizing.vol_scale[i])
                    if np.isfinite(raw_vol):
                        scale *= max(raw_vol, 0.0)
                # Levels are needed before size, because risk-based sizing is defined by
                # the distance to the stop.
                atr_entry = signals.atr[i]
                entry_atr = float(atr_entry) if np.isfinite(atr_entry) and atr_entry > 0 else 0.0
                prelim_stop = _stop_level(stops, want, fill, entry_atr, fill, None, i)
                prelim_risk = (
                    abs(fill - prelim_stop) if prelim_stop is not None
                    else (entry_atr if entry_atr > 0 else fill * sizing.fallback_stop_pct)
                )
                if prelim_risk <= 0:
                    prelim_risk = fill * sizing.fallback_stop_pct

                if sizing.risk_per_trade > 0:
                    # Constant-risk sizing: the dollars at stake between entry and stop are
                    # a fixed fraction of equity. This is the mechanism that bounds drawdown
                    # when volatility changes, which fixed-notional sizing cannot do.
                    budget = eq_at_entry * sizing.risk_per_trade * position_fraction * scale
                    q = budget / prelim_risk
                else:
                    q = (eq_at_entry * position_fraction * scale) / fill

                max_notional = eq_at_entry * sizing.max_leverage
                if q * fill > max_notional > 0:
                    q = max_notional / fill
                if q <= 0:
                    pending = None
                    continue

                notional = q * fill
                fee = notional * costs.fee_rate
                slip_cost = q * abs(fill - o[i])
                if want > 0:
                    cash -= notional + fee
                else:
                    cash += notional - fee
                qty, side = q, want
                entry_bar, entry_price = i, fill
                entry_notional = notional
                fees_acc, slip_acc, fund_acc = fee, slip_cost, 0.0
                mfe = mae = 0.0
                # Recompute at the actual fill (identical to the preliminary level).
                stop = _stop_level(stops, side, entry_price, entry_atr, entry_price, None, i)
                initial_stop = stop if stop is not None else 0.0
                tp = _tp_level(stops, side, entry_price, entry_atr)
                risk_unit = (
                    abs(entry_price - stop) if stop is not None
                    else (entry_atr if entry_atr > 0 else entry_price * sizing.fallback_stop_pct)
                )
                best = entry_price
            else:  # exit at open
                fill = o[i] * (1 - slip) if side > 0 else o[i] * (1 + slip)
                notional = qty * fill
                fee = notional * costs.fee_rate
                slip_cost = qty * abs(fill - o[i])
                fees_acc += fee
                slip_acc += slip_cost
                gross = (fill - entry_price) * qty * side
                if side > 0:
                    cash += notional - fee
                else:
                    cash -= notional + fee
                dollars_per_unit = risk_unit * qty if risk_unit > 0 else 0.0
                trades.append(
                    Trade(
                        symbol=symbol, side=side,
                        entry_time=ts_dt[entry_bar], exit_time=ts_dt[i],
                        entry_price=entry_price, exit_price=fill, quantity=qty,
                        notional_at_entry=entry_notional,
                        gross_pnl=gross, fees=fees_acc, slippage_cost=slip_acc,
                        funding_cost=fund_acc,
                        net_pnl=gross - fees_acc - fund_acc,
                        holding_bars=i - entry_bar,
                        holding_hours=(i - entry_bar) * tf_hours,
                        entry_bar=entry_bar, exit_bar=i,
                        exit_reason="indicator",
                        stop_level=float(stop) if stop is not None else 0.0,
                        initial_stop=float(initial_stop),
                        # An indicator exit needs a signal at or after the entry bar,
                        # so it always fills >= 1 bar later => always AFTER_4H.
                        exit_class=EXIT_AFTER_4H,
                        mfe=mfe / dollars_per_unit if dollars_per_unit else 0.0,
                        mae=mae / dollars_per_unit if dollars_per_unit else 0.0,
                    )
                )
                qty, side = 0.0, 0
                stop = tp = None
                fees_acc = slip_acc = fund_acc = 0.0
            pending = None

        # ---- 2. Intrabar: stops and take-profits (only while a position is open) ----
        if side != 0:
            bar_open = o[i]
            # The trailing level used to test THIS bar is the one implied by bars
            # strictly before it. Raising it with this bar's own extreme before testing
            # this bar's opposite extreme would book profit that the bar never had to
            # give - an optimistic assumption. So: test first, update afterwards.
            best_next = max(best, h[i]) if side > 0 else min(best, low_[i])

            hit_stop = False
            if stops.mode != "none" and stop is not None:
                hit_stop = (low_[i] <= stop) if side > 0 else (h[i] >= stop)
            hit_tp = False
            if tp is not None:
                hit_tp = (h[i] >= tp) if side > 0 else (low_[i] <= tp)

            if hit_stop:
                # Conservative fill: a gap through the stop fills at the open (worse).
                if side > 0:
                    raw = bar_open if bar_open <= stop else stop
                    fill = raw * (1 - slip)
                else:
                    raw = bar_open if bar_open >= stop else stop
                    fill = raw * (1 + slip)
                # Distinguish a stop that was tightened from the entry level (a trailing
                # stop, actually working) from the initial catastrophe level sitting still.
                moved = (
                    stop > initial_stop + 1e-12 if side > 0
                    else stop < initial_stop - 1e-12
                ) if initial_stop > 0 else False
                reason = "trail_stop" if moved else "stop"
            elif hit_tp:
                # Deliberately pessimistic: the level itself, never a favourable gap.
                fill = float(tp)
                reason = "take_profit"
            else:
                fill = None
                reason = ""

            dollars_per_unit = risk_unit * qty if risk_unit > 0 else 0.0
            if fill is not None:
                notional = qty * fill
                fee = notional * costs.fee_rate
                slip_cost = qty * abs(fill - bar_open)
                fees_acc += fee
                slip_acc += slip_cost
                gross = (fill - entry_price) * qty * side
                if side > 0:
                    cash += notional - fee
                else:
                    cash -= notional + fee
                early = (i == entry_bar)
                if reason == "stop":
                    cls = STOP_LOSS_BEFORE_4H if early else EXIT_AFTER_4H
                else:
                    cls = TAKE_PROFIT_BEFORE_4H if early else EXIT_AFTER_4H
                trades.append(
                    Trade(
                        symbol=symbol, side=side,
                        entry_time=ts_dt[entry_bar], exit_time=ts_dt[i],
                        entry_price=entry_price, exit_price=fill, quantity=qty,
                        notional_at_entry=entry_notional,
                        gross_pnl=gross, fees=fees_acc, slippage_cost=slip_acc,
                        funding_cost=fund_acc,
                        net_pnl=gross - fees_acc - fund_acc,
                        holding_bars=i - entry_bar,
                        holding_hours=(i - entry_bar) * tf_hours,
                        entry_bar=entry_bar, exit_bar=i,
                        exit_reason=reason, exit_class=cls,
                        stop_level=float(stop) if stop is not None else 0.0,
                        initial_stop=float(initial_stop),
                        mfe=mfe / dollars_per_unit if dollars_per_unit else 0.0,
                        mae=mae / dollars_per_unit if dollars_per_unit else 0.0,
                    )
                )
                qty, side = 0.0, 0
                stop = tp = None
                fees_acc = slip_acc = fund_acc = 0.0
            else:
                # Still open: roll the trailing level forward and record excursions.
                best = best_next
                if stops.mode in ("atr_trail", "pct_trail", "external"):
                    stop = _stop_level(stops, side, entry_price, entry_atr, best, stop, i,
                                     risk_unit=risk_unit)
                # Profit protection: once the trade has earned enough R, never give it
                # back past break-even. Applied after the level update so it can only
                # tighten, and never before the move has actually happened.
                if stops.breakeven_at_r > 0 and risk_unit > 0 and stop is not None:
                    reached = (
                        (best - entry_price) >= stops.breakeven_at_r * risk_unit
                        if side > 0
                        else (entry_price - best) >= stops.breakeven_at_r * risk_unit
                    )
                    if reached:
                        stop = max(stop, entry_price) if side > 0 else min(stop, entry_price)
                if dollars_per_unit:
                    favourable = ((h[i] - entry_price) if side > 0 else (entry_price - low_[i])) * qty
                    adverse = ((low_[i] - entry_price) if side > 0 else (entry_price - h[i])) * qty
                    mfe = max(mfe, favourable)
                    mae = min(mae, adverse)

        # ---- 3. Funding on the position still open at the settlement ----
        if side != 0 and costs.funding_on and i in fund_by_bar:
            rate = fund_by_bar[i]
            notional_now = qty * c[i]
            cost = notional_now * rate * (1 if side > 0 else -1)
            cash -= cost
            fund_acc += cost

        # ---- 4. Signals from this bar's close → pending for the next open ----
        if side == 0:
            if signals.entry_long[i] and (stops.entry_gate is None or bool(stops.entry_gate[i])):
                pending = ("enter", 1)
            elif (
                allow_shorts
                and signals.entry_short[i]
                and (stops.entry_gate is None or bool(stops.entry_gate[i]))
            ):
                pending = ("enter", -1)
        else:
            want_exit = (signals.exit_long[i] if side > 0 else signals.exit_short[i])
            if want_exit:
                pending = ("exit", side)

        # ---- 5. Mark to market ----
        equity[i] = _mark(c[i])

    # ---- Force-close any position still open at the end of the sample ----
    if side != 0:
        fill = c[-1] * (1 - slip) if side > 0 else c[-1] * (1 + slip)
        notional = qty * fill
        fee = notional * costs.fee_rate
        fees_acc += fee
        slip_acc += qty * abs(fill - c[-1])
        gross = (fill - entry_price) * qty * side
        if side > 0:
            cash += notional - fee
        else:
            cash -= notional + fee
        trades.append(
            Trade(
                symbol=symbol, side=side,
                entry_time=ts_dt[entry_bar], exit_time=ts_dt[n - 1],
                entry_price=entry_price, exit_price=fill, quantity=qty,
                notional_at_entry=entry_notional,
                gross_pnl=gross, fees=fees_acc, slippage_cost=slip_acc,
                funding_cost=fund_acc, net_pnl=gross - fees_acc - fund_acc,
                holding_bars=n - 1 - entry_bar,
                holding_hours=(n - 1 - entry_bar) * tf_hours,
                entry_bar=entry_bar, exit_bar=n - 1,
                exit_reason="eod",
                stop_level=float(stop) if stop is not None else 0.0,
                initial_stop=float(initial_stop),
                exit_class=EXIT_AFTER_4H,
            )
        )
        qty, side = 0.0, 0
        equity[-1] = cash

    return Result(
        symbol=symbol, timeframe=timeframe, equity=equity, timestamps=ts,
        trades=trades, costs=costs, label=signals.label,
    )
