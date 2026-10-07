"""Win/loss trade analysis + rule-based strategy adaptation.

Two pieces:

1. :func:`analyze_trades` — turns a backtest's trade ledger into a structured
   win/loss breakdown: rates, expectancy, payoff, streaks, per-side stats and a
   PnL histogram. This is the *diagnostic* half of the improvement loop.

2. :func:`recommend_adjustments` — maps diagnostics onto concrete parameter
   changes (``min_confluence``, ``atr_trail_mult``, side toggles, …) with a
   human-readable rationale for each. This is the *adaptive* half: every rule
   is a small, auditable heuristic rather than an opaque refit, so the UI can
   show *why* a change is suggested.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from trading.domain import PositionSide

from .engine import Trade

__all__ = ["SideStats", "TradeAnalysis", "analyze_trades", "recommend_adjustments"]


@dataclass
class SideStats:
    n: int = 0
    wins: int = 0
    pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0


@dataclass
class TradeAnalysis:
    n_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float | None = None
    avg_win: float = 0.0
    avg_loss: float = 0.0  # negative number (mean of losing trades)
    payoff_ratio: float | None = None  # avg_win / |avg_loss|
    expectancy: float = 0.0  # mean pnl per trade
    max_win_streak: int = 0
    max_loss_streak: int = 0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    long: SideStats = field(default_factory=SideStats)
    short: SideStats = field(default_factory=SideStats)
    histogram: dict[str, list] = field(default_factory=lambda: {"counts": [], "edges": []})

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "gross_profit": round(self.gross_profit, 2),
            "gross_loss": round(self.gross_loss, 2),
            "profit_factor": _r(self.profit_factor),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "payoff_ratio": _r(self.payoff_ratio),
            "expectancy": round(self.expectancy, 2),
            "max_win_streak": self.max_win_streak,
            "max_loss_streak": self.max_loss_streak,
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "long": {"n": self.long.n, "wins": self.long.wins,
                     "win_rate": round(self.long.win_rate, 4), "pnl": round(self.long.pnl, 2)},
            "short": {"n": self.short.n, "wins": self.short.wins,
                      "win_rate": round(self.short.win_rate, 4), "pnl": round(self.short.pnl, 2)},
            "histogram": self.histogram,
        }


def _r(x: float | None, nd: int = 3) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return round(x, nd)


def _streaks(pnls: Sequence[float]) -> tuple[int, int]:
    max_w = max_l = cur_w = cur_l = 0
    for p in pnls:
        if p > 0:
            cur_w, cur_l = cur_w + 1, 0
        elif p < 0:
            cur_l, cur_w = cur_l + 1, 0
        else:
            cur_w = cur_l = 0
        max_w, max_l = max(max_w, cur_w), max(max_l, cur_l)
    return max_w, max_l


def _histogram(pnls: Sequence[float], bins: int = 10) -> dict[str, list]:
    if not pnls:
        return {"counts": [], "edges": []}
    lo, hi = min(pnls), max(pnls)
    if math.isclose(lo, hi):
        return {"counts": [len(pnls)], "edges": [lo, hi]}
    width = (hi - lo) / bins
    counts = [0] * bins
    for p in pnls:
        idx = min(int((p - lo) / width), bins - 1)
        counts[idx] += 1
    edges = [round(lo + i * width, 2) for i in range(bins + 1)]
    return {"counts": counts, "edges": edges}


def analyze_trades(trades: Sequence[Trade]) -> TradeAnalysis:
    """Full win/loss breakdown of a backtest trade ledger."""
    out = TradeAnalysis()
    if not trades:
        return out
    pnls = [float(t.realized_pnl) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    out.n_trades = len(pnls)
    out.wins = len(wins)
    out.losses = len(losses)
    out.win_rate = len(wins) / len(pnls)
    out.gross_profit = sum(wins)
    out.gross_loss = sum(losses)
    out.profit_factor = (out.gross_profit / abs(out.gross_loss)) if out.gross_loss else None
    out.avg_win = sum(wins) / len(wins) if wins else 0.0
    out.avg_loss = sum(losses) / len(losses) if losses else 0.0
    out.payoff_ratio = (out.avg_win / abs(out.avg_loss)) if out.avg_loss else None
    out.expectancy = sum(pnls) / len(pnls)
    out.max_win_streak, out.max_loss_streak = _streaks(pnls)
    out.largest_win = max(pnls)
    out.largest_loss = min(pnls)
    out.histogram = _histogram(pnls)

    for t in trades:
        bucket = out.long if t.side is PositionSide.LONG else out.short
        bucket.n += 1
        bucket.wins += 1 if t.realized_pnl > 0 else 0
        bucket.pnl += float(t.realized_pnl)
    return out


# ────────────────────────────────────────────────────────────────────── #
#  Rule-based adaptation
# ────────────────────────────────────────────────────────────────────── #
def recommend_adjustments(
    analysis: TradeAnalysis,
    params: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Suggest parameter/logic changes from the win/loss diagnostics.

    Every suggestion is ``{parameter, current, suggested, rationale}`` so the
    caller (UI or optimizer) can apply or display it directly. Rules only fire
    with a minimal sample (``n_trades >= 5``) — below that the honest answer is
    "more data", which is itself the first recommendation.
    """
    p = dict(params or {})
    recs: list[dict[str, Any]] = []

    def add(parameter: str, suggested: Any, rationale: str) -> None:
        recs.append({
            "parameter": parameter,
            "current": p.get(parameter),
            "suggested": suggested,
            "rationale": rationale,
        })

    if analysis.n_trades < 5:
        add("min_confluence", 1,
            "Too few trades to judge the edge; loosening the confluence "
            "requirement produces a meaningful sample faster.")
        return recs

    # 1. Low hit rate but winners pay → the filter is fine, the exits are loose.
    if analysis.win_rate < 0.40 and (analysis.payoff_ratio or 0) >= 1.5:
        add("atr_trail_mult", round(max(1.5, float(p.get("atr_trail_mult", 3.0)) - 0.5), 2),
            f"Win rate {analysis.win_rate:.0%} with payoff {analysis.payoff_ratio:.1f}: "
            "entries select well but winners are given back; tighten the trailing stop.")

    # 2. Low hit rate AND poor payoff → entries are weak, raise the bar.
    if analysis.win_rate < 0.40 and (analysis.payoff_ratio or 0) < 1.5:
        add("min_confluence", int(p.get("min_confluence", 2)) + 1,
            f"Win rate {analysis.win_rate:.0%} with payoff "
            f"{_r(analysis.payoff_ratio) or '—'}: entries are not selective enough; "
            "require one more overlapping level.")
        add("zone_atr", round(max(0.2, float(p.get("zone_atr", 0.5)) - 0.1), 2),
            "Narrow the confluence zone so only precise touches qualify.")

    # 3. Losses dominate on one side → disable or de-emphasise that side.
    if analysis.short.n >= 4 and analysis.short.win_rate < 0.30 and analysis.short.pnl < 0 \
            and analysis.long.pnl > 0:
        add("allow_short", False,
            f"Shorts lose ({analysis.short.win_rate:.0%} win rate, "
            f"{analysis.short.pnl:.0f} PnL) while longs pay; trade the winning side only.")
    if analysis.long.n >= 4 and analysis.long.win_rate < 0.30 and analysis.long.pnl < 0 \
            and analysis.short.pnl > 0:
        add("allow_long", False,
            f"Longs lose ({analysis.long.win_rate:.0%} win rate, "
            f"{analysis.long.pnl:.0f} PnL) while shorts pay; trade the winning side only.")

    # 4. Deep loss streaks → cut size in unclear regimes.
    if analysis.max_loss_streak >= 4:
        add("allow_range", False,
            f"{analysis.max_loss_streak} consecutive losses — chop is toxic for this "
            "configuration; stay flat when the regime is unclear.")

    # 5. Consistently profitable → press the edge a little.
    if analysis.win_rate >= 0.55 and (analysis.profit_factor or 0) >= 1.5:
        add("position_fraction", "increase (via backtest config)",
            f"Win rate {analysis.win_rate:.0%}, profit factor {analysis.profit_factor:.2f}: "
            "the edge supports a larger allocation per signal.")
    return recs
