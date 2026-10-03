"""Extended metric suite for the loop reports (portfolio + per-ticker)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from .backtest import Trade

__all__ = ["portfolio_metrics", "trade_stats", "regime_split", "PPY"]


def _ppy(tf: str) -> int:
    return 365 * 6 if tf == "4h" else 365  # crypto trades 24/7


def _returns(equity: np.ndarray) -> np.ndarray:
    e = np.asarray(equity, float)
    return e[1:] / e[:-1] - 1.0


def max_dd_duration(equity: np.ndarray, index: pd.DatetimeIndex) -> tuple[float, str, str]:
    """Longest underwater stretch (days) + its start/end."""
    e = np.asarray(equity, float)
    peak = np.maximum.accumulate(e)
    under = e < peak * (1 - 1e-12)
    best, best_s, best_e, s = 0.0, None, None, None
    for i, u in enumerate(under):
        if u and s is None:
            s = i
        elif not u and s is not None:
            dur = (index[i - 1] - index[s]).total_seconds() / 86400
            if dur > best:
                best, best_s, best_e = dur, str(index[s].date()), str(index[i - 1].date())
            s = None
    if s is not None:  # still underwater at the end
        dur = (index[-1] - index[s]).total_seconds() / 86400
        if dur > best:
            best, best_s, best_e = dur, str(index[s].date()), str(index[-1].date())
    return best, best_s or "", best_e or ""


def max_drawdown(equity: np.ndarray) -> float:
    e = np.asarray(equity, float)
    peak = np.maximum.accumulate(e)
    return float(((peak - e) / peak).max())


def sharpe(r: np.ndarray, ppy: int) -> float:
    if r.size < 2 or r.std(ddof=1) < 1e-12:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ppy))


def sortino(r: np.ndarray, ppy: int) -> float:
    dn = r[r < 0]
    if dn.size == 0 or r.size == 0:
        return 0.0
    dd = float(np.sqrt(np.mean(dn ** 2)))
    return float(r.mean() / dd * np.sqrt(ppy)) if dd > 1e-12 else 0.0


@dataclass
class Metrics:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_dd: float
    dd_duration_days: float
    dd_window: str
    profit_factor: float
    expectancy: float          # net pnl per trade, in % of slice equity at entry
    win_rate: float
    avg_win: float
    avg_loss: float
    payoff: float              # avg_win / |avg_loss|
    n_trades: int
    exposure: float
    turnover_pa: float         # traded notional / initial equity / years
    avg_hold_h: float
    med_hold_h: float
    n_stop_before_4h: int
    n_tp_before_4h: int
    n_exit_after_4h: int
    min_hold_violations: int   # classified BEFORE_4H for non-stop reasons (must be 0)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def portfolio_metrics(
    equity: np.ndarray,
    index: pd.DatetimeIndex,
    trades: list[Trade],
    tf: str,
    initial: float,
    exposures: dict[str, np.ndarray],
) -> Metrics:
    ppy = _ppy(tf)
    e = np.asarray(equity, float)
    r = _returns(e)
    years = len(e) / ppy
    tr = e[-1] / e[0] - 1.0
    cagr = (1 + tr) ** (1 / years) - 1 if years > 0 and tr > -1 else -1.0
    mdd = max_drawdown(e)
    dddur, ddw, _ = max_dd_duration(e, index)
    pnls = [t.net_pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    notional = sum(t.entry_price * t.qty + t.exit_price * t.qty for t in trades)
    exp_frac = float(np.mean([ex.mean() for ex in exposures.values()]))
    cls = [t.classification for t in trades]
    notes = []
    if any(t.exit_reason == "END_OF_DATA" for t in trades):
        notes.append("final open position force-closed at last bar (END_OF_DATA)")
    return Metrics(
        total_return=tr, cagr=cagr, sharpe=sharpe(r, ppy), sortino=sortino(r, ppy),
        calmar=(cagr / mdd) if mdd > 1e-9 else 0.0, max_dd=mdd,
        dd_duration_days=dddur, dd_window=ddw,
        profit_factor=(gross_win / gross_loss) if gross_loss > 1e-9 else float("inf"),
        expectancy=float(np.mean(pnls)) if pnls else 0.0,
        win_rate=len(wins) / len(pnls) if pnls else 0.0,
        avg_win=avg_win, avg_loss=avg_loss,
        payoff=(avg_win / abs(avg_loss)) if avg_loss < 0 else float("inf"),
        n_trades=len(pnls), exposure=exp_frac,
        turnover_pa=notional / initial / years if years > 0 else 0.0,
        avg_hold_h=float(np.mean([t.holding_hours for t in trades])) if trades else 0.0,
        med_hold_h=float(np.median([t.holding_hours for t in trades])) if trades else 0.0,
        n_stop_before_4h=cls.count("STOP_LOSS_BEFORE_4H"),
        n_tp_before_4h=cls.count("TAKE_PROFIT_BEFORE_4H"),
        n_exit_after_4h=cls.count("EXIT_AFTER_4H"),
        min_hold_violations=int(any(
            t.classification == "TAKE_PROFIT_BEFORE_4H" for t in trades)),
        notes=notes,
    )


def trade_stats(trades: list[Trade]) -> dict:
    """Per-trade diagnostics for robustness checks (drop best trade etc.)."""
    if not trades:
        return {"n": 0}
    pnls = np.array([t.net_pnl for t in trades])
    return {
        "n": len(trades),
        "total_net": float(pnls.sum()),
        "best_trade": float(pnls.max()),
        "worst_trade": float(pnls.min()),
        "best_trade_share": float(pnls.max() / pnls.sum()) if pnls.sum() > 0 else float("inf"),
    }


def regime_split(equity: np.ndarray, index: pd.DatetimeIndex, ref_close: pd.Series) -> dict:
    """Split portfolio returns into bull/bear/sideways using the reference
    ticker's 90-bar return (regime known point-in-time at each bar close)."""
    e = np.asarray(equity, float)
    r = _returns(e)
    lookback_ret = ref_close.pct_change(90)
    regime = np.where(lookback_ret > 0.20, "bull",
                      np.where(lookback_ret < -0.20, "bear", "sideways"))
    out: dict[str, dict] = {}
    aligned = regime[1:len(r) + 1]  # regime at t-1 governs return t -> no lookahead
    for g in ("bull", "bear", "sideways"):
        mask = aligned == g
        sub = r[mask]
        out[g] = {
            "share_of_bars": float(mask.mean()),
            "ret_sum": float((np.prod(1 + sub) - 1)) if sub.size else 0.0,
        }
    return out
