"""Editable-parameter schema for the strategy catalogue (UI + optimizer).

``trend_confluence_pine`` carries ~60 tunables across five families. The
backtest only needs names; the *console* needs types, ranges, groups and
defaults so it can render a form, build a sweep grid and explain what each
knob does. That metadata lives here — one place, JSON-serialisable, served by
``GET /strategies/{name}/schema``.

Key convention
--------------
The unified strategy keeps every EMF+ADL tunable in a nested ``emf`` block, so
those keys are dotted (``emf.damping``). Both the optimizer's ``_expand_params``
(dotted grid keys) and the UI's form builder understand dotted paths, so a
dotted key behaves exactly like a flat one end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ParamSpec", "PARAM_SCHEMA", "schema_for", "defaults_for", "grid_for"]

#: Parameter families shown as sections in the console.
GROUPS: tuple[str, ...] = ("TC", "EMF", "ADL", "MOM", "PINE")


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    kind: str  # "int" | "float" | "bool" | "enum"
    group: str  # one of GROUPS
    default: Any
    help: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: tuple[Any, ...] = ()
    sweep: tuple[Any, ...] = ()  # candidate values offered by the optimizer UI

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "label": self.label, "kind": self.kind,
            "group": self.group, "default": self.default, "help": self.help,
            "min": self.min, "max": self.max, "step": self.step,
            "options": list(self.options), "sweep": list(self.sweep),
        }


def _i(key: str, label: str, group: str, default: int, *, lo: int, hi: int,
       help_: str = "", sweep: tuple[Any, ...] = ()) -> ParamSpec:
    return ParamSpec(key, label, "int", group, default, help_, lo, hi, 1, (), sweep)


def _f(key: str, label: str, group: str, default: float, *, lo: float, hi: float,
       step: float = 0.1, help_: str = "", sweep: tuple[Any, ...] = ()) -> ParamSpec:
    return ParamSpec(key, label, "float", group, default, help_, lo, hi, step, (), sweep)


def _b(key: str, label: str, group: str, default: bool, help_: str = "",
       sweep: tuple[Any, ...] = ()) -> ParamSpec:
    return ParamSpec(key, label, "bool", group, default, help_, None, None, None, (), sweep)


def _e(key: str, label: str, group: str, default: Any, options: tuple[Any, ...],
       help_: str = "", sweep: tuple[Any, ...] = ()) -> ParamSpec:
    return ParamSpec(key, label, "enum", group, default, help_, None, None, None,
                     options, sweep)


#: The pine strategy's editable schema (TC × EMF × ADL × Momentum + Pine TP/Trail).
PINE_SCHEMA: tuple[ParamSpec, ...] = (
    # ── Trend-confluence core ──────────────────────────────────────────
    _i("ema_fast", "EMA fast", "TC", 20, lo=2, hi=100,
       help_="Fast EMA of the confluence level stack.", sweep=(10, 20, 33)),
    _i("ema_mid", "EMA mid", "TC", 50, lo=5, hi=200,
       help_="Mid EMA — regime anchor with the slow EMA."),
    _i("ema_slow", "EMA slow", "TC", 200, lo=20, hi=400,
       help_="Slow EMA — price above mid+slow = bullish regime."),
    _f("zone_atr", "Zone width (ATR)", "TC", 0.5, lo=0.1, hi=2.0, step=0.1,
       help_="Half-width of the confluence zone, in ATRs.", sweep=(0.3, 0.5, 0.8)),
    _i("min_confluence", "Min confluence", "TC", 2, lo=1, hi=5,
       help_="Independent levels that must sit in the zone.",
       sweep=(2, 3)),
    _i("pullback_lookback", "Pullback lookback", "TC", 10, lo=2, hi=40,
       help_="Bars that define 'pulled back off the high/low'."),
    _b("use_trendlines", "Use trendlines", "TC", True,
       help_="Feed the Pine trendline engine into the regime vote."),
    _i("trendline_refresh", "Trendline refresh", "TC", 5, lo=1, hi=40,
       help_="Re-run the trendline engine every N bars."),
    _f("atr_trail_mult", "ATR trail mult", "TC", 2.5, lo=1.0, hi=6.0, step=0.5,
       help_="TC's ATR trailing stop (off by default in this strategy).",
       sweep=(2.0, 3.0)),
    _f("stop_atr", "Hard stop (ATR)", "TC", 2.0, lo=0.0, hi=6.0, step=0.5,
       help_="Catastrophic stop distance in ATRs (0 = off).",
       sweep=(1.5, 2.5)),
    _f("risk_pct", "Risk per trade", "TC", 0.01, lo=0.0, hi=0.05, step=0.005,
       help_="Fraction of equity risked per trade (0 = full size)."),
    _i("reentry_cooldown", "Re-entry cooldown", "TC", 3, lo=0, hi=20,
       help_="Bars to sit out after any exit (breaks whipsaws)."),
    _b("need_rejection", "Confirmation candle", "TC", False,
       help_="Require the previous bar to close in trade direction.",
       sweep=(True, False)),

    # ── EMF: trend analysis + adaptive VWAP (Pine Strategy A) ──────────
    _i("emf.length_bars", "Trend bar count", "EMF", 15, lo=3, hi=60,
       help_="Bars scored by the weighted bar-trend analysis.",
       sweep=(10, 15, 25)),
    _f("emf.movement_threshold", "Movement threshold %", "EMF", 0.314, lo=0.05,
       hi=2.0, step=0.001, help_="A bar must move this % to count as trend."),
    _f("emf.bull_trend_threshold", "Bull threshold", "EMF", 0.3, lo=-1.0, hi=1.0,
       step=0.05, help_="Trend coefficient needed to allow longs."),
    _f("emf.bear_trend_threshold", "Bear threshold", "EMF", -0.3, lo=-1.0, hi=1.0,
       step=0.05, help_="Trend coefficient below which shorts are allowed."),
    _b("emf.flat_filter_enabled", "Flat filter", "EMF", True,
       help_="Tighten thresholds inside low-volatility flat zones."),
    _f("emf.flat_threshold_d", "Flat zone threshold", "EMF", 1.2, lo=0.5, hi=3.0,
       step=0.1, help_="% move / range that defines a flat zone."),
    _i("emf.smooth_len", "Trend smoothing", "EMF", 3, lo=1, hi=20,
       help_="Smoothing length of the trend coefficient."),
    _i("emf.swing_period", "Swing period", "EMF", 50, lo=2, hi=200,
       help_="Bars for the adaptive-VWAP pivot detection."),
    _f("emf.base_apt", "Adaptive price tracking", "EMF", 20.0, lo=1.0, hi=300.0,
       step=1.0, help_="VWAP half-life in bars (bigger = slower VWAP)."),
    _b("emf.use_adaptive_apt", "Adapt APT by ATR", "EMF", False,
       help_="Scale the VWAP half-life by the ATR ratio."),
    _f("emf.volatility_bias", "Volatility bias", "EMF", 10.0, lo=0.1, hi=50.0,
       step=0.1, help_="Exponent used when adapting APT."),
    _i("emf.atr_length", "ATR length (EMF)", "EMF", 50, lo=5, hi=200,
       help_="ATR length used by the EMF engine."),
    _i("emf.cooldown_bars", "Add cooldown (EMF)", "EMF", 10, lo=1, hi=60,
       help_="Bars between Pine add signals."),

    # ── ADL: accumulation/distribution + MACD (Pine Strategy B) ────────
    _i("emf.rsi_length", "RSI length", "ADL", 14, lo=2, hi=50,
       help_="RSI length behind the ADL upper/middle/lower bands.",
       sweep=(9, 14, 21)),
    _i("emf.ob_level", "RSI overbought", "ADL", 75, lo=50, hi=95,
       help_="Upper band level."),
    _i("emf.os_level", "RSI oversold", "ADL", 30, lo=5, hi=50,
       help_="Lower band level."),
    _i("emf.om_level", "RSI middle", "ADL", 50, lo=20, hi=80,
       help_="Middle band level."),
    _i("emf.length_adl", "Two-pole length", "ADL", 20, lo=2, hi=100,
       help_="Two-pole filter length applied to the AD line.",
       sweep=(10, 20, 30)),
    _f("emf.damping", "Two-pole damping", "ADL", 0.9, lo=0.1, hi=1.0, step=0.01,
       help_="Damping of the two-pole filter (lower = smoother).",
       sweep=(0.7, 0.9)),
    _i("emf.rising_falling", "Rising/falling bars", "ADL", 5, lo=1, hi=30,
       help_="Bars counted for the rising/falling ADL check."),
    _i("emf.length_bb", "BB length (ADL)", "ADL", 33, lo=5, hi=200,
       help_="Bollinger length on the AD line."),
    _f("emf.mult_bb", "BB multiplier", "ADL", 2.618, lo=0.1, hi=10.0, step=0.001,
       help_="Bollinger deviation multiplier."),
    _f("emf.percent_bb", "BB percent", "ADL", 61.8, lo=1.0, hi=100.0, step=0.1,
       help_="Fraction of the deviation used for the bands."),
    _i("emf.signal_length", "MACD signal length", "ADL", 9, lo=2, hi=30,
       help_="Signal SMA of the ADL MACD.", sweep=(5, 9, 14)),

    # ── Momentum overlay ───────────────────────────────────────────────
    _b("use_momentum", "Use momentum", "MOM", True,
       help_="Rate-of-change filter on entries."),
    _i("momentum_period", "Momentum period", "MOM", 10, lo=2, hi=60,
       help_="ROC lookback in bars.", sweep=(5, 10, 20)),
    _e("momentum_mode", "Momentum mode", "MOM", "bonus", ("gate", "bonus", "off"),
       help_="gate = veto entries, bonus = size up on agreement.",
       sweep=("gate", "bonus")),
    _f("momentum_bonus", "Momentum bonus", "MOM", 0.15, lo=0.0, hi=1.0, step=0.05,
       help_="Size bonus when ROC agrees (bonus mode)."),
    _b("use_emf", "Use EMF/ADL entries", "EMF", True,
       help_="Apply the EMF+ADL combined entry logic."),
    _e("emf_mode", "EMF mode", "EMF", "bonus", ("require", "bonus", "off"),
       help_="require = strict intersection with TC entries.",
       sweep=("require", "bonus")),
    _f("emf_bonus", "EMF bonus", "EMF", 0.25, lo=0.0, hi=1.0, step=0.05,
       help_="Size bonus when EMF+ADL agrees."),
    _b("use_emf_exits", "EMF exits", "EMF", True,
       help_="Use the EMF+ADL combined exit columns."),

    # ── Pine take-profit / trailing stop ───────────────────────────────
    _b("tp_enabled", "Take profit", "PINE", True,
       help_="Percent take profit on/off.", sweep=(True, False)),
    _f("tp_percent", "TP %", "PINE", 2.0, lo=0.1, hi=20.0, step=0.1,
       help_="Profit target measured from the entry price, in %.",
       sweep=(1.0, 2.0, 3.5)),
    _f("tp_close_pct", "TP close %", "PINE", 50.0, lo=5.0, hi=100.0, step=5.0,
       help_="Share of the position closed per TP hit.",
       sweep=(25.0, 50.0, 100.0)),
    _i("tp_cooldown_bars", "TP cooldown bars", "PINE", 3, lo=0, hi=20,
       help_="Bars between partial TP closes.", sweep=(0, 3, 6)),
    _b("trailing_enabled", "Trailing stop", "PINE", True,
       help_="Percent trailing stop on/off (independent of TP).",
       sweep=(True, False)),
    _f("trailing_percent", "Trailing %", "PINE", 1.0, lo=0.1, hi=10.0, step=0.1,
       help_="Trail distance, in % of the extreme price since entry.",
       sweep=(0.5, 1.0, 2.0)),
    _b("allow_adds", "Allow adds", "PINE", False,
       help_="Pyramiding: add only into an open position."),
    _i("add_cooldown_bars", "Add cooldown bars", "PINE", 10, lo=1, hi=60,
       help_="Bars between adds."),
    _f("add_size_mult", "Add size multiplier", "PINE", 0.25, lo=0.05, hi=1.0,
       step=0.05, help_="Add size relative to the entry strength."),
)

PARAM_SCHEMA: dict[str, tuple[ParamSpec, ...]] = {
    "trend_confluence_pine": PINE_SCHEMA,
}


def schema_for(name: str) -> list[dict[str, Any]]:
    """JSON-ready parameter schema for ``name`` (empty when unknown)."""
    return [p.as_dict() for p in PARAM_SCHEMA.get(name, ())]


def defaults_for(name: str) -> dict[str, Any]:
    """Defaults by key (dotted keys resolve into nested blocks)."""
    return {p.key: p.default for p in PARAM_SCHEMA.get(name, ())}


def grid_for(name: str) -> dict[str, list[Any]]:
    """Sweep candidates offered by the console for ``name``."""
    return {p.key: list(p.sweep) for p in PARAM_SCHEMA.get(name, ()) if p.sweep}
