"""The optimisation loop: hypotheses, walk-forward validation, robustness battery.

Discipline enforced by construction:

* **Ticker rotation.** Each loop draws a *new* group of liquid perps from a ranked pool
  and never reuses a group until the pool is exhausted.
* **Train / validate / holdout.** Variants are *ranked* on validation only; the holdout
  slice is touched once, for the final answer, and never used for selection.
* **Rotation, not cherry-picking.** Every variant is run on every ticker of the loop's
  group and both timeframes. A variant that only wins on one ticker is reported as such.
* **Nothing is silently dropped.** Excluded series, gaps and degenerate fits are logged.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from gex.strategy.settings import StrategySettings

from . import data as D
from . import metrics as M
from . import rules as R
from .engine import Costs, StopSpec, run
from .metrics import compute_metrics, periods_per_year

log = logging.getLogger(__name__)
OUT = Path(__file__).with_name("out")

#: Liquidity pool — real Bybit USDT perps, ranked live by 24h turnover (see
#: data.fetch_top_symbols). Groups are rotated so no loop repeats a ticker set.
POOL_SIZE = 24
GROUP_SIZE = 8


@dataclass
class Variant:
    """One strategy hypothesis. Kept small: gates and stops, nothing exotic."""

    name: str
    repair_hybrid: bool = False
    gate: str = "none"          # none | vol_band | trend | struct_break | struct_break_event
    gate2: str = "none"         # optional second gate, currently only "vol_band"
    vol_low: float = 0.5
    vol_high: float = 2.5
    vol_window: int = 100
    vol_quantile_window: int = 300
    trend_ema: int = 200
    stop_mode: str = "none"     # none | atr_fixed | atr_trail | pct_trail | external
    stop_atr_mult: float = 2.0
    stop_pct: float = 0.03
    #: Multiplier on the structural stop's buffer. 1.0 = the vendor's own buffered
    #: level; larger values push the stop further from price (wider, looser).
    ext_buffer_mult: float = 1.0
    #: Move the stop to break-even once the trade is this many R in profit (0 = off).
    breakeven_at_r: float = 0.0
    tp_mode: str = "none"       # none | atr | pct
    tp_atr_mult: float = 3.0
    tp_pct: float = 0.06
    long_only: bool = False
    short_only: bool = False
    settings_overrides: dict = field(default_factory=dict)
    hypothesis: str = ""

    def key(self) -> str:
        return self.name


def _settings(v: Variant) -> StrategySettings:
    if v.settings_overrides:
        return replace(StrategySettings(), **v.settings_overrides)
    return StrategySettings()


def _stops(v: Variant, bars: pd.DataFrame) -> StopSpec:
    ext_l = ext_s = None
    if v.stop_mode == "external":
        ext_l, ext_s = R.structural_stop_arrays(bars)
    return StopSpec(
        mode=v.stop_mode,
        atr_mult=v.stop_atr_mult,
        pct=v.stop_pct,
        external_long=ext_l,
        external_short=ext_s,
        tp_mode=v.tp_mode,
        tp_atr_mult=v.tp_atr_mult,
        tp_pct=v.tp_pct,
    )


def evaluate_variant(
    v: Variant,
    bars: pd.DataFrame,
    funding: pd.DataFrame | None,
    symbol: str,
    timeframe: str,
    costs: Costs,
    *,
    warmup: int = R.WARMUP,
) -> tuple[M.Metrics, object]:
    """Run one variant on one series. Returns metrics + the raw result."""
    tf_hours = D.TIMEFRAMES[timeframe][1]
    settings = _settings(v)
    ss, f = R.base_signals(
        bars, settings, repair_hybrid=v.repair_hybrid,
        warmup=warmup, long_only=v.long_only, short_only=v.short_only,
    )

    # Gates are built per side (the trend / structural gates are directional).
    spec = _stops(v, bars)
    if v.gate != "none":
        gs = R.GateSpec(
            kind=v.gate, vol_low=v.vol_low, vol_high=v.vol_high,
            allow_shorts=not v.long_only,
        )
        gl = R.build_gate(bars, f, gs, "long")
        gsh = R.build_gate(bars, f, gs, "short")
        if gl is None and gsh is None:
            gate = None
        else:
            n = len(bars)
            gate = np.zeros(n, dtype=bool)
            if gl is not None:
                gate |= gl
            if gsh is not None:
                gate |= gsh
            # Directional gates must not open the wrong side: mask the entries instead.
            if v.gate in ("trend", "struct_break", "struct_break_event"):
                if gl is not None:
                    ss.entry_long &= gl
                if gsh is not None:
                    ss.entry_short &= gsh
                gate = None
        spec = replace(spec, entry_gate=gate)
    else:
        spec = replace(spec, entry_gate=None)

    res = run(
        bars, ss, symbol=symbol, timeframe=timeframe, tf_hours=tf_hours,
        costs=costs, stops=spec, funding=funding, position_fraction=1.0,
    )
    met = compute_metrics(
        res.equity, res.trades, tf_hours,
        timestamps=bars["timestamp"].to_numpy(dtype=np.int64),
    )
    return met, res


# --------------------------------------------------------------------------- #
# Panel aggregation
# --------------------------------------------------------------------------- #
def panel_equity(results: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight panel equity built from the per-ticker curves, aligned on time.

    Each timestamp is averaged over the tickers that are *live* at that timestamp
    (``skipna``), never over the intersection of everyone's history. Requiring every
    ticker to exist at every bar would silently collapse a five-year window to the
    listing date of the newest coin in the panel — and a panel that quietly starts in
    2023 is a different test from the one being reported.
    """
    if not results:
        return np.zeros(0), np.zeros(0)
    per = {}
    for sym, r in results.items():
        r_ = M.equity_returns(r.equity)
        # equity_returns is one shorter than the bar series (it differences), so the
        # index must start at the second bar or the alignment is off by one.
        stamps = pd.to_datetime(r.timestamps[1: len(r_) + 1], unit="ms", utc=True)
        per[sym] = pd.Series(r_, index=stamps)
    df = pd.DataFrame(per)
    df = df.loc[df.notna().any(axis=1)]
    if df.empty:
        return np.zeros(0), np.zeros(0)
    port_ret = df.mean(axis=1, skipna=True)
    eq = (1.0 + port_ret).cumprod().to_numpy()
    eq = np.concatenate([[1.0], eq])
    idx = np.concatenate([[df.index[0] - pd.Timedelta(seconds=1)], df.index.values])
    return eq, np.asarray(idx, dtype="datetime64[ns]")


def pooled_trades(trades: list, tf_hours: float) -> dict:
    """Trade-level statistics pooled across every ticker of a panel.

    Per-ticker metrics cannot be averaged (a ticker with 3 trades would weigh as much as
    one with 300), and computing them on the panel curve with an empty trade list yields
    zeros that read like bad performance. Both are wrong, so the pooled set is used for
    everything trade-derived; the portfolio curve is used only for curve-derived metrics.
    """
    n = len(trades)
    if n == 0:
        return {"total_trades": 0}
    net = np.array([t.net_pnl for t in trades], dtype=float)
    gross = np.array([t.gross_pnl for t in trades], dtype=float)
    notionals = np.array([t.notional_at_entry for t in trades], dtype=float)
    hold = np.array([t.holding_hours for t in trades], dtype=float)
    wins, losses = net[net > 0], net[net <= 0]
    pos_gross = float(gross[gross > 0].sum())
    rel = np.divide(net, notionals, out=np.zeros_like(net), where=notionals > 0)
    # Concentration is measured on the *relative* P&L (net / notional), not on currency.
    # A currency sum weights by position size, so a compounding equity curve makes late
    # trades look dominant and the "best single trade" share means nothing.
    order_rel = np.sort(rel)[::-1]
    rel_pos = float(rel[rel > 0].sum())
    sides = np.array([t.side for t in trades])
    out = {
        "total_trades": n,
        "win_rate": float(len(wins) / n),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum()))
        if losses.sum() < 0 else float("inf"),
        "expectancy_ccy": float(net.mean()),
        "expectancy_pct": float(rel.mean()),
        "total_fees": float(sum(t.fees for t in trades)),
        "total_slippage": float(sum(t.slippage_cost for t in trades)),
        "total_funding": float(sum(t.funding_cost for t in trades)),
        "gross_profit": pos_gross,
        "cost_drag_ratio": (
            float(sum(t.fees + t.slippage_cost for t in trades)) / pos_gross
            if pos_gross > 0 else 0.0
        ),
        "net_over_gross": float(net.sum()) / pos_gross if pos_gross > 0 else 0.0,
        "avg_holding_hours": float(hold.mean()),
        "median_holding_hours": float(np.median(hold)),
        "n_long": int((sides > 0).sum()),
        "n_short": int((sides < 0).sum()),
        "long_win_rate": float((net[sides > 0] > 0).mean()) if (sides > 0).any() else 0.0,
        "short_win_rate": float((net[sides < 0] > 0).mean()) if (sides < 0).any() else 0.0,
        "best_trade_share": float(order_rel[0] / rel_pos) if rel_pos > 0 else 0.0,
        "top5_share": float(order_rel[:5].sum() / rel_pos) if rel_pos > 0 else 0.0,
        "n_stop_before_4h": sum(1 for t in trades if t.exit_class == M.STOP_LOSS_BEFORE_4H),
        "n_tp_before_4h": sum(1 for t in trades if t.exit_class == M.TAKE_PROFIT_BEFORE_4H),
        "n_after_4h": sum(1 for t in trades if t.exit_class == M.EXIT_AFTER_4H),
        "avg_mfe_r": float(np.mean([t.mfe for t in trades])),
        "avg_mae_r": float(np.mean([t.mae for t in trades])),
        "net_per_trade": float(net.sum()) / n,
    }
    allowed_early = out["n_stop_before_4h"] + out["n_tp_before_4h"]
    out["min_holding_compliance"] = 1.0 if allowed_early + out["n_after_4h"] == n else 0.0
    out["payoff"] = abs(out["avg_win"] / out["avg_loss"]) if out["avg_loss"] else 0.0
    return out


def panel_metrics(results: dict[str, object], timeframe: str) -> dict:
    """Portfolio-curve metrics + pooled trade statistics for one panel.

    ``sharpe``/``max_dd``/``cagr`` describe the equal-weight portfolio of every ticker in
    the panel. ``median_ticker_*`` describe the typical single ticker. A variant is only
    interesting when both agree: a good portfolio built from one good ticker is not a
    result, it is a ticker.
    """
    tf_hours = D.TIMEFRAMES[timeframe][1]
    eq, idx = panel_equity(results)
    if len(eq) < 3:
        return {"n_tickers": 0}
    per_ticker = {sym: compute_metrics(r.equity, r.trades, tf_hours)
                  for sym, r in results.items()}
    pooled = pooled_trades([t for r in results.values() for t in r.trades], tf_hours)

    d = compute_metrics(eq, [], tf_hours, timestamps=idx).as_dict()
    d.update(pooled)
    d["n_trades"] = pooled.get("total_trades", 0)
    d.update(
        {
            "n_tickers": len(results),
            "median_ticker_sharpe": float(np.median([m.sharpe for m in per_ticker.values()])),
            "median_ticker_calmar": float(np.median([m.calmar for m in per_ticker.values()])),
            "median_ticker_return": float(
                np.median([m.total_return for m in per_ticker.values()])
            ),
            "share_tickers_positive": float(
                np.mean([m.total_return > 0 for m in per_ticker.values()])
            ),
            # Mean per-ticker exposure: a fraction, unlike the pooled bar sum.
            "exposure": float(np.mean([m.exposure for m in per_ticker.values()])),
        }
    )
    return d


def per_ticker_table(results: dict[str, object], timeframe: str) -> list[dict]:
    tf_hours = D.TIMEFRAMES[timeframe][1]
    rows = []
    for sym, r in sorted(results.items()):
        m = compute_metrics(r.equity, r.trades, tf_hours)
        rows.append(
            {
                "symbol": sym,
                "n_trades": m.n_trades,
                "net_return": round(m.total_return, 4),
                "sharpe": round(m.sharpe, 3),
                "calmar": round(m.calmar, 3),
                "max_dd": round(m.max_dd, 4),
                "win_rate": round(m.win_rate, 3),
                "profit_factor": round(m.profit_factor, 3),
                "expectancy_pct": round(m.expectancy_pct, 5),
                "exposure": round(m.exposure, 3),
                "funding": round(m.total_funding, 4),
                "fees": round(m.total_fees, 4),
            }
        )
    return rows


def robust_score(pm: dict) -> float:
    """Ranking score that rewards *consistency*, not the best ticker.

    Median ticker Sharpe, shrunk by sample size and by how much of the panel failed to
    make money. Designed so a single lucky ticker cannot carry a variant.
    """
    if pm.get("n_tickers", 0) < 3:
        return -9.9
    sharpe = float(pm.get("sharpe", 0.0))
    med = float(pm.get("median_ticker_sharpe", 0.0))
    trades = float(pm.get("total_trades", 0))
    pos = float(pm.get("share_tickers_positive", 0.0))
    shrink = min(1.0, trades / (40.0 * max(1, pm.get("n_tickers", 1))))
    return (0.5 * sharpe + 0.5 * med) * pos * shrink


def concentration_flags(pm: dict) -> list[str]:
    flags = []
    if pm.get("total_trades", 0) < 30:
        flags.append(f"too few trades ({pm.get('total_trades')})")
    if pm.get("share_tickers_positive", 1.0) < 0.6:
        flags.append(f"only {pm.get('share_tickers_positive', 0):.0%} of tickers profitable")
    if pm.get("best_trade_share", 0.0) > 0.25:
        flags.append(f"best single trade = {pm.get('best_trade_share', 0):.0%} of net")
    if pm.get("max_dd", 0.0) < -0.60:
        flags.append(f"max DD {pm.get('max_dd', 0):.0%}")
    return flags
