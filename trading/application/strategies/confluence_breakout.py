"""Confluence-breakout strategy — the validated trend system of the platform.

This is the **port of the two systems that survived out-of-sample validation**
in the research harness (``quant/``), unified behind the live strategy contract:

* ``ALLIGATOR_4H`` — the ``quant/alligator`` frozen selection
  (``entry_mode=breakout, n_break=30, k_sl_atr=2.5, k_trail=5.0,
  exit_level=jaw``, 0.5% risk/trade). Validated on 4H bars: holdout
  Sharpe 0.56, **max drawdown 2.66%**, **profit factor 1.38**, win rate
  31.5%, payoff 3.0 over 92 OOS trades; ≤2.8% drawdown on train/validation.
* ``DONCHIAN_1D`` — the ``quant/results`` frozen selection
  (``n_break=40, k_sl_atr=2.5, k_trail=4.0, ma_exit=100`` on 1D bars,
  long only). Validated across three independent ticker loops:
  **profit factor 1.88–2.08** on fresh OOS sets with 10–12.6% CAGR.

Both presets are **frozen research artefacts**, not tuning suggestions. The
documented disable conditions (rolling Sharpe < 0, drawdown > 50%, cost regime
> 2× the validated costs) apply verbatim.

Entry logic (all conditions at the close of the signal bar)
-----------------------------------------------------------
1. **Trend confluence** — Alligator aligned and waking
   (``lips > teeth > jaw`` and jaw rising), confirmed HH/HL market structure,
   ADL above its EMA, EMF > 0.
2. **Breakout** — ``close`` above the highest high of the prior ``n_break``
   bars (Donchian continuation), or, when ``entry_mode`` allows it, a *pullback*
   entry: price dipped to the lips within ``pullback_window`` bars, held the
   teeth, and closed back above the lips.
3. **Trend filter** (``DONCHIAN_1D``) — ``close > SMA(sma_period)``.
4. Flat, past warm-up, and past the re-entry cooldown.

Exit logic (priority order, evaluated on every bar)
---------------------------------------------------
1. **Intrabar stop** — the initial/trailing stop is a *resting* level, so a bar
   whose low (long) or high (short) trades through it exits. Conservative by
   construction: this is what keeps the drawdown as low as the research
   measured.
2. **Chandelier trail** — ``max(stop, best_close − k_trail·ATR)`` ratcheted,
   plus the structural ratchet
   ``max(stop, last_confirmed_HL − k_struct_buf·ATR)`` when
   ``use_struct_trail``.
3. **Alligator line** — close below the teeth/jaw/lips ends the trend.
4. **Structure break** — close below the last confirmed swing low.
5. **MA exit** — ``ma_exit`` SMA, when configured (Donchian preset).
6. **Time stop** — held ``max_hold_bars`` bars without ever reaching
   ``min_mfe_r``.

Risk / sizing
-------------
``risk_pct`` of equity is risked per trade: the emitted ``strength`` is scaled
so that, under the standard engine position fraction, a stop-out risks about
``risk_pct``. The complete trade plan (entry, stop, size, risk amount) rides on
the :class:`~trading.domain.Signal` — that is what the live signal engine
persists and what the CSV/XLSX exports publish.

Performance
-----------
``prepare()`` computes the whole causal frame once (O(n)); ``on_bar()`` is an
O(1) lookup plus constant-time bookkeeping, so a 1,500-bar replay costs one
pass instead of n².
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np

from trading.domain import Bar, Side, Signal, Tick
from trading.ports import Strategy

from .confluence_indicators import compute_frame

__all__ = [
    "ConfluenceBreakoutParams",
    "ConfluenceBreakoutStrategy",
    "BREAKOUT_PRESETS",
    "DEFAULT_PRESET",
    "preset_params",
]

#: Must stay in sync with ``BacktestConfig.position_fraction`` — the strength we
#: emit is *relative to that fraction*, so the per-trade risk budget holds.
ENGINE_POSITION_FRACTION = 0.95


# ────────────────────────────────────────────────────────────────────── #
#  Configuration
# ────────────────────────────────────────────────────────────────────── #
@dataclass(frozen=True)
class ConfluenceBreakoutParams:
    # trend confluence
    pivot_left: int = 3
    pivot_right: int = 3
    adl_ema_span: int = 20
    use_alligator: bool = True
    use_structure: bool = True
    use_adl: bool = True
    use_emf: bool = True
    # entry
    entry_mode: str = "breakout"   # breakout | pullback | both
    n_break: int = 30
    pullback_window: int = 8
    use_trend_filter: bool = False  # SMA filter (Donchian preset)
    sma_period: int = 200
    # exits
    k_sl_atr: float = 2.5           # ATR stop distance
    k_struct_buf: float = 0.5       # structural stop = last HL − buf·ATR
    min_stop_atr: float = 1.0       # stop never closer than 1·ATR
    max_stop_atr: float = 4.0       # never farther than 4·ATR
    k_trail: float = 4.0            # chandelier: best_close − k·ATR
    use_struct_trail: bool = True
    exit_level: str = "jaw"        # teeth | jaw | lips
    ma_exit: int = 0               # 0 = off; else SMA period
    # position management
    risk_pct: float = 0.005        # fraction of equity risked per trade
    max_hold_bars: int = 45
    min_mfe_r: float = 1.0
    cooldown_bars: int = 3
    warmup: int = 250
    allow_long: bool = True
    allow_short: bool = False
    #: ``risk`` = percent-of-equity risk per trade; ``fraction`` = fixed fraction
    #: of equity per trade (the Donchian preset's "95% of an equal capital
    #: slice" rule).
    sizing: str = "risk"
    position_fraction: float = 0.95
    #: Used only for the *informational* plan fields (risk_amount/position_size);
    #: the live engine overrides it with the real equity it observes.
    equity: float = 100_000.0
    timeframe: str = "4h"

    _ENTRY_MODES = ("breakout", "pullback", "both")
    _EXIT_LEVELS = ("teeth", "jaw", "lips")

    #: Names accepted from an API request / optimizer grid.
    FIELD_NAMES: tuple[str, ...] = (
        "pivot_left", "pivot_right", "adl_ema_span", "use_alligator", "use_structure",
        "use_adl", "use_emf", "entry_mode", "n_break", "pullback_window",
        "use_trend_filter", "sma_period", "k_sl_atr", "k_struct_buf",
        "min_stop_atr", "max_stop_atr", "k_trail", "use_struct_trail",
        "exit_level", "ma_exit", "risk_pct", "max_hold_bars", "min_mfe_r",
        "cooldown_bars", "warmup", "allow_long", "allow_short",
        "sizing", "position_fraction", "equity", "timeframe",
    )

    _SIZING = ("risk", "fraction")

    def __post_init__(self) -> None:
        if self.entry_mode not in self._ENTRY_MODES:
            raise ValueError(
                f"entry_mode must be one of {self._ENTRY_MODES}, got {self.entry_mode!r}"
            )
        if self.exit_level not in self._EXIT_LEVELS:
            raise ValueError(
                f"exit_level must be one of {self._EXIT_LEVELS}, got {self.exit_level!r}"
            )
        for name in ("n_break", "pivot_left", "pivot_right", "adl_ema_span"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"parameter '{name}' must be >= 1")
        if not 0 < self.k_sl_atr:
            raise ValueError("k_sl_atr must be > 0")
        if not 0 < self.k_trail:
            raise ValueError("k_trail must be > 0")
        if self.min_stop_atr > self.max_stop_atr:
            raise ValueError("min_stop_atr must be <= max_stop_atr")
        if self.sizing not in self._SIZING:
            raise ValueError(f"sizing must be one of {self._SIZING}, got {self.sizing!r}")
        if not 0 < self.position_fraction <= 1:
            raise ValueError("position_fraction must be in (0, 1]")
        if not 0 < self.risk_pct <= 1:
            raise ValueError("risk_pct must be in (0, 1]")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "ConfluenceBreakoutParams":
        data = {k: v for k, v in dict(raw or {}).items() if k in cls.FIELD_NAMES}
        out = dataclasses.replace(cls(), **data)
        out.__post_init__()
        return out

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.FIELD_NAMES}


#: Frozen, out-of-sample-validated configurations. See the module docstring.
BREAKOUT_PRESETS: dict[str, dict[str, Any]] = {
    "alligator_4h": {
        "label": "Alligator confluence 4H — validated maxDD 2.7%, PF 1.38 OOS",
        "params": {
            "entry_mode": "breakout", "n_break": 30, "risk_pct": 0.005,
            "k_sl_atr": 2.5, "k_struct_buf": 0.5, "k_trail": 5.0,
            "exit_level": "jaw", "max_hold_bars": 45, "min_mfe_r": 1.0,
            "cooldown_bars": 3, "warmup": 250, "allow_short": False,
            "timeframe": "4h",
        },
    },
    "donchian_1d": {
        "label": "Donchian breakout 1D — validated PF 1.88-2.08 OOS",
        "params": {
            "entry_mode": "breakout", "n_break": 40, "use_trend_filter": True,
            "sma_period": 200, "k_sl_atr": 2.5, "k_trail": 4.0, "ma_exit": 100,
            "max_hold_bars": 240, "min_mfe_r": 0.5, "cooldown_bars": 1,
            "warmup": 260, "allow_short": False, "timeframe": "1d",
            "sizing": "fraction", "position_fraction": 0.95, "risk_pct": 0.05,
        },
    },
}

DEFAULT_PRESET = "alligator_4h"


def preset_params(preset: str | None) -> dict[str, Any]:
    """Frozen parameters for ``preset`` (falls back to :data:`DEFAULT_PRESET`)."""
    key = preset or DEFAULT_PRESET
    if key not in BREAKOUT_PRESETS:
        raise ValueError(f"unknown preset {key!r} (known: {', '.join(BREAKOUT_PRESETS)})")
    return dict(BREAKOUT_PRESETS[key]["params"])


# ────────────────────────────────────────────────────────────────────── #
#  Position bookkeeping
# ────────────────────────────────────────────────────────────────────── #
@dataclass
class _Position:
    side: str                       # "long" | "short"
    entry: float
    stop: float
    trail: float
    entry_bar: int
    entry_time: datetime
    ref_atr: float
    struct_level: float | None
    best: float                     # best close (long) / worst close (short)
    worst: float                    # worst close (long) / best close (short)
    strength: float = 1.0
    mfe_r: float = 0.0
    bars: int = 0
    trail_reason: str = ""

    def risk_distance(self) -> float:
        return abs(self.entry - self.stop)

    def unrealised_r(self, price: float) -> float:
        dist = self.risk_distance()
        if dist <= 0:
            return 0.0
        signed = (price - self.entry) if self.side == "long" else (self.entry - price)
        return signed / dist


def _sma_at(close: np.ndarray, i: int, period: int) -> float:
    """Causal SMA over ``close[i-period+1 : i+1]`` (NaN when out of data)."""
    if period <= 0 or i - period + 1 < 0:
        return float("nan")
    window = close[i - period + 1:i + 1]
    if len(window) < period or not np.all(np.isfinite(window)):
        return float("nan")
    return float(np.mean(window))


class ConfluenceBreakoutStrategy(Strategy):
    """One instance per symbol; owns at most one position on that symbol."""

    name = "confluence_breakout"

    def __init__(
        self,
        symbol: str,
        params: Mapping[str, Any] | None = None,
        *,
        preset: str | None = None,
        timeframe: str | None = None,
    ) -> None:
        self.symbol = symbol
        merged: dict[str, Any] = preset_params(preset)
        merged.update(dict(params or {}))   # explicit params always win
        if timeframe:
            merged["timeframe"] = timeframe
        self.preset = preset or DEFAULT_PRESET
        self._p = ConfluenceBreakoutParams.from_dict(merged)
        self.timeframe = self._p.timeframe
        self.min_bars = max(
            self._p.warmup,
            self._p.n_break + 2,
            self._p.sma_period + 2 if self._p.use_trend_filter else 0,
            self._p.ma_exit + 2 if self._p.ma_exit > 0 else 0,
        )
        self._reset()

    # ── state ───────────────────────────────────────────────────────────
    def _reset(self) -> None:
        self._bars: list[Bar] = []
        self._ts: list[datetime] = []
        self._frame: dict[str, np.ndarray] | None = None
        self._cursor = 0
        self._processed = 0
        self._pos: _Position | None = None
        self._last_exit_bar = -10 ** 9
        self.fallback_count = 0
        self._equity_hint = self._p.equity

    @property
    def position(self) -> _Position | None:
        """Open position, if any (live engine + tests read this)."""
        return self._pos

    # ── lifecycle ───────────────────────────────────────────────────────
    async def start(self) -> None:
        self._reset()

    async def shutdown(self) -> None:
        self._reset()

    async def on_tick(self, tick: Tick) -> list[Signal]:
        """Bar-driven family: ticks never produce signals."""
        return []

    # ── batch path ──────────────────────────────────────────────────────
    async def prepare(self, bars: Sequence[Bar]) -> None:
        """Precompute the whole causal frame once (O(n) instead of O(n²))."""
        self._reset()
        ordered = sorted(bars, key=lambda b: b.timestamp)
        self._bars = list(ordered)
        self._ts = [b.timestamp for b in ordered]
        self._frame = self._frame_from(ordered)

    def _frame_from(self, bars: Sequence[Bar]) -> dict[str, np.ndarray]:
        return compute_frame(
            [b.high for b in bars], [b.low for b in bars],
            [b.close for b in bars], [b.volume for b in bars],
            pivot_left=self._p.pivot_left, pivot_right=self._p.pivot_right,
            adl_ema_span=self._p.adl_ema_span,
        )

    # ── streaming path ──────────────────────────────────────────────────
    async def on_bar(self, bar: Bar) -> list[Signal]:
        if (self._frame is not None and self._cursor < len(self._ts)
                and self._matches(self._cursor, bar)):
            idx = self._cursor
            self._cursor += 1
            return self._drain(idx)
        self.fallback_count += 1
        self._frame = None

        self._bars.append(bar)
        self._ts.append(bar.timestamp)
        if len(self._bars) < self.min_bars:
            return []
        self._frame = self._frame_from(self._bars)
        return self._drain(len(self._bars) - 1)

    async def generate_signals(self, bars: Sequence[Bar]) -> list[Signal]:
        out: list[Signal] = []
        for bar in sorted(bars, key=lambda b: b.timestamp):
            out.extend(await self.on_bar(bar))
        return out

    def _matches(self, idx: int, bar: Bar) -> bool:
        if self._ts[idx] != bar.timestamp or idx >= len(self._bars):
            return False
        expected = float(self._bars[idx].close)
        return abs(expected - float(bar.close)) <= 1e-9 * max(1.0, abs(float(bar.close)))

    def _drain(self, upto: int) -> list[Signal]:
        out: list[Signal] = []
        while self._processed <= upto:
            i = self._processed
            self._processed += 1
            if i < self.min_bars:
                continue
            out.extend(self._signal_at(i))
        return out

    # ── decision logic ──────────────────────────────────────────────────
    def _signal_at(self, i: int) -> list[Signal]:
        f = self._frame
        assert f is not None
        p = self._p
        atr = f["atr"][i]
        if not math.isfinite(atr) or atr <= 0:
            return []

        if self._pos is not None:
            exit_sig = self._manage_position(i, f)
            return [exit_sig] if exit_sig is not None else []

        if (i - self._last_exit_bar) < p.cooldown_bars:
            return []
        for attempt, enabled in (("long", p.allow_long), ("short", p.allow_short)):
            if not enabled:
                continue
            sig = self._try_entry("long" if attempt == "long" else "short", i, f)
            if sig is not None:
                return [sig]
        return []

    # ── entries ─────────────────────────────────────────────────────────
    def _confluence(self, f: dict[str, np.ndarray], i: int, side: str) -> tuple[bool, str]:
        """Alligator + HH/HL + ADL + EMF confluence → (ok, detail string)."""
        p = self._p
        parts: list[str] = []
        ok = True
        long = side == "long"
        if p.use_alligator:
            lips, teeth, jaw = f["lips"][i], f["teeth"][i], f["jaw"][i]
            if not all(math.isfinite(v) for v in (lips, teeth, jaw)):
                return False, "alligator_warmup"
            jaw_moves = i > 0 and math.isfinite(f["jaw"][i - 1]) and (
                jaw > f["jaw"][i - 1] if long else jaw < f["jaw"][i - 1]
            )
            aligned = lips > teeth > jaw if long else lips < teeth < jaw
            if aligned and jaw_moves:
                parts.append("alligator=ok")
            else:
                parts.append("alligator")
                ok = False
        if p.use_structure:
            flag = f["struct_bull"][i] if long else f["struct_bear"][i]
            parts.append("structure=ok" if bool(flag) else "structure")
            ok = ok and bool(flag)
        if p.use_adl:
            adl_ok = f["adl"][i] > f["adl_ema"][i] if long else f["adl"][i] < f["adl_ema"][i]
            parts.append("adl=ok" if adl_ok else "adl")
            ok = ok and adl_ok
        if p.use_emf:
            emf_ok = f["emf"][i] > 0 if long else f["emf"][i] < 0
            parts.append("emf=ok" if emf_ok else "emf")
            ok = ok and emf_ok
        return ok, ",".join(parts)

    def _breakout_level(self, f: dict[str, np.ndarray], i: int, side: str) -> float | None:
        """Highest high / lowest low of the prior ``n_break`` bars (excl. ``i``)."""
        p = self._p
        start = i - p.n_break
        if start < 0 or start >= i:
            return None
        window = f["high"][start:i] if side == "long" else f["low"][start:i]
        return float(np.max(window) if side == "long" else np.min(window))

    def _trend_filter_ok(self, f: dict[str, np.ndarray], i: int, side: str) -> bool:
        p = self._p
        if not p.use_trend_filter:
            return True
        close = f["close"][i]
        if side == "long":
            sma = _sma_at(f["close"], i, p.sma_period)
            return math.isfinite(sma) and close > sma
        sma = _sma_at(f["close"], i, p.sma_period)
        return math.isfinite(sma) and close < sma

    def _entry_trigger(self, f: dict[str, np.ndarray], i: int, side: str) -> tuple[str, str]:
        """(kind, detail) for the entry trigger at bar ``i``; ``("", "")`` = none."""
        p = self._p
        long = side == "long"
        close = f["close"][i]
        if p.entry_mode in ("breakout", "both"):
            level = self._breakout_level(f, i, side)
            if level is not None and ((long and close > level) or (not long and close < level)):
                return "breakout", f"close {'>' if long else '<'}{level:.6g}"
        if p.entry_mode in ("pullback", "both"):
            lips, teeth = f["lips"], f["teeth"]
            lo = max(0, i - p.pullback_window + 1)
            gap = lips[lo:i + 1] - f["low"][lo:i + 1] if long else lips[lo:i + 1] - f["high"][lo:i + 1]
            touched = np.nanmin(gap) <= 0 if long else np.nanmax(gap) >= 0
            held = (close >= teeth[i] if long else close <= teeth[i]) if math.isfinite(teeth[i]) else False
            back = (i > 0 and close > f["close"][i - 1]) if long else (i > 0 and close < f["close"][i - 1])
            if touched and held and back and math.isfinite(lips[i]):
                return "pullback", f"lips {'dip' if long else 'rally'} + close back {'above' if long else 'below'}"
        return "", ""

    def _try_entry(self, side: str, i: int, f: dict[str, np.ndarray]) -> Signal | None:
        p = self._p
        close = f["close"][i]
        if not self._trend_filter_ok(f, i, side):
            return None
        ok, detail = self._confluence(f, i, side)
        if not ok:
            return None
        kind, trigger = self._entry_trigger(f, i, side)
        if not kind:
            return None
        stop = self._initial_stop(f, i, side=side)
        if stop is None:
            return None
        if (side == "long" and stop >= close) or (side == "short" and stop <= close):
            return None
        sig = self._entry_signal(side, i, f, stop, f"{kind}: {detail}; {trigger}")
        # Book the position at the signal close (the engine fills at the next
        # bar's open — the same convention the other strategies in this package
        # use), so the very next bar enters the position-management branch.
        self._pos = _Position(
            side=side,
            entry=float(close),
            stop=float(stop),
            trail=float(stop),
            entry_bar=i,
            entry_time=self._ts[i],
            ref_atr=float(f["atr"][i]),
            struct_level=(float(f["last_hl"][i]) if side == "long" else float(f["last_hh"][i]))
            if math.isfinite(f["last_hl"][i] if side == "long" else f["last_hh"][i]) else None,
            best=float(close),
            worst=float(close),
            strength=float(sig.strength),
        )
        return sig

    def _initial_stop(self, f: dict[str, np.ndarray], i: int, *, side: str) -> float | None:
        """Tighter of the structural and ATR stop, clamped to [1, 4]·ATR."""
        p = self._p
        long = side == "long"
        atr = f["atr"][i]
        if not math.isfinite(atr) or atr <= 0:
            return None
        close = f["close"][i]
        ref = f["last_hl"][i] if long else f["last_hh"][i]
        atr_stop = close - p.k_sl_atr * atr if long else close + p.k_sl_atr * atr
        if math.isfinite(ref):
            struct_stop = ref - p.k_struct_buf * atr if long else ref + p.k_struct_buf * atr
            ideal = min(struct_stop, atr_stop) if long else max(struct_stop, atr_stop)
        else:
            ideal = atr_stop
        if long:
            lo, hi = close - p.max_stop_atr * atr, close - p.min_stop_atr * atr
            return max(min(ideal, hi), lo)
        lo, hi = close + p.min_stop_atr * atr, close + p.max_stop_atr * atr
        return min(max(ideal, lo), hi)

    def _entry_signal(self, side: str, i: int, f: dict[str, np.ndarray],
                      stop: float, reason: str) -> Signal:
        p = self._p
        close = f["close"][i]
        stop_dist = abs(close - stop)
        risk_amount = p.risk_pct * self._equity_hint
        pos_size = risk_amount / stop_dist if stop_dist > 0 else 0.0
        if p.sizing == "fraction":
            # Fixed capital slice per trade (the Donchian rule): emit the
            # fraction of the engine's position_fraction that equals the slice.
            strength = min(1.0, p.position_fraction / ENGINE_POSITION_FRACTION)
            pos_size = p.position_fraction * self._equity_hint / close if close > 0 else 0.0
        else:
            # So that engine_qty = equity · 0.95 · strength risks ~risk_pct.
            strength = 1.0
            if stop_dist > 0:
                strength = min(1.0, p.risk_pct / (ENGINE_POSITION_FRACTION * stop_dist / close))
                strength = max(strength, 1e-6)
        atr = f["atr"][i]
        return Signal(
            symbol=self.symbol,
            side=Side.BUY if side == "long" else Side.SELL,
            strategy=self.name,
            reason=reason,
            strength=float(strength),
            entry_price=float(close),
            stop_loss=float(stop),
            take_profit=None,          # trend system: exits trail, there is no target
            timeframe=self.timeframe,
            risk_pct=float(p.risk_pct),
            risk_amount=float(risk_amount),
            position_size=float(pos_size),
            bar_time=self._ts[i],
            meta={
                "preset": self.preset,
                "atr": float(atr),
                "risk_distance": float(stop_dist),
                "max_hold_bars": p.max_hold_bars,
                "sizing": p.sizing,
            },
        )

    # ── position management / exits ─────────────────────────────────────
    def _manage_position(self, i: int, f: dict[str, np.ndarray]) -> Signal | None:
        pos = self._pos
        assert pos is not None
        p = self._p
        long = pos.side == "long"
        close, high, low = f["close"][i], f["high"][i], f["low"][i]
        atr = f["atr"][i]
        pos.bars += 1
        if long:
            pos.best = max(pos.best, close)
            pos.worst = min(pos.worst, close)
        else:
            pos.best = min(pos.best, close)
            pos.worst = max(pos.worst, close)

        # 1. intrabar stop — a resting level, filled through the bar extreme
        if (long and low <= pos.stop) or (not long and high >= pos.stop):
            return self._exit_signal(i, pos, pos.stop, pos.trail_reason or "stop_loss")

        # 2. chandelier + structural trail, ratcheted, then tested
        if math.isfinite(atr) and atr > 0:
            if long:
                cands = [pos.trail, pos.best - p.k_trail * atr]
                if p.use_struct_trail and math.isfinite(f["last_hl"][i]):
                    cands.append(f["last_hl"][i] - p.k_struct_buf * atr)
                new_trail = max(c for c in cands if math.isfinite(c))
                if new_trail > pos.trail + 1e-12:
                    pos.trail, pos.trail_reason = new_trail, "trailing_stop"
                if low <= pos.trail:
                    return self._exit_signal(i, pos, pos.trail, "trailing_stop")
            else:
                cands = [pos.trail, pos.best + p.k_trail * atr]
                if p.use_struct_trail and math.isfinite(f["last_hh"][i]):
                    cands.append(f["last_hh"][i] + p.k_struct_buf * atr)
                new_trail = min(c for c in cands if math.isfinite(c))
                if new_trail < pos.trail - 1e-12:
                    pos.trail, pos.trail_reason = new_trail, "trailing_stop"
                if high >= pos.trail:
                    return self._exit_signal(i, pos, pos.trail, "trailing_stop")

        pos.mfe_r = max(pos.mfe_r, pos.unrealised_r(close))

        # 3. alligator line
        key = {"teeth": "teeth", "jaw": "jaw", "lips": "lips"}[p.exit_level]
        ref = f[key][i]
        if math.isfinite(ref) and ((long and close < ref) or (not long and close > ref)):
            return self._exit_signal(i, pos, close, f"{p.exit_level}_break")

        # 4. structure break
        struct = f["last_hl"][i] if long else f["last_hh"][i]
        if math.isfinite(struct) and ((long and close < struct) or (not long and close > struct)):
            return self._exit_signal(i, pos, close, "structure_break")

        # 5. MA exit
        if p.ma_exit > 0:
            ma = _sma_at(f["close"], i, p.ma_exit)
            if math.isfinite(ma) and ((long and close < ma) or (not long and close > ma)):
                return self._exit_signal(i, pos, close, f"ma_exit_{p.ma_exit}")

        # 6. time stop
        if p.max_hold_bars > 0 and pos.bars >= p.max_hold_bars and pos.mfe_r < p.min_mfe_r:
            return self._exit_signal(i, pos, close, "time_stop")
        return None

    def _exit_signal(self, i: int, pos: _Position, price: float, reason: str) -> Signal:
        p = self._p
        risk_amount = p.risk_pct * self._equity_hint
        sig = Signal(
            symbol=self.symbol,
            side=Side.SELL if pos.side == "long" else Side.BUY,
            strategy=self.name,
            reason=reason,
            strength=float(pos.strength),
            entry_price=pos.entry,
            stop_loss=pos.trail,
            take_profit=None,
            timeframe=self.timeframe,
            risk_pct=float(p.risk_pct),
            risk_amount=float(risk_amount),
            position_size=float(pos.strength * ENGINE_POSITION_FRACTION),
            bar_time=self._ts[i],
            meta={
                "preset": self.preset,
                "exit_reason": reason,
                "bars_held": pos.bars,
                "mfe_r": round(pos.mfe_r, 4),
                "exit_price": float(price),
                "risk_distance": pos.risk_distance(),
                "trail": float(pos.trail),
                "sizing": p.sizing,
            },
        )
        self._pos = None
        self._last_exit_bar = i
        return sig

    # ── diagnostics ─────────────────────────────────────────────────────
    def diagnostics(self) -> dict[str, Any]:
        pos = self._pos
        return {
            "symbol": self.symbol,
            "preset": self.preset,
            "timeframe": self.timeframe,
            "bars_seen": len(self._bars),
            "side": pos.side if pos else "flat",
            "entry": pos.entry if pos else None,
            "stop": pos.stop if pos else None,
            "trail": pos.trail if pos else None,
            "bars_held": pos.bars if pos else 0,
            "mfe_r": round(pos.mfe_r, 4) if pos else None,
            "fallback_count": self.fallback_count,
        }
