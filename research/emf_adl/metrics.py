"""Metric computation for the research programme.

Every number reported in the final deliverable comes from here, so the definitions are
explicit and the annualisation is stated rather than implied.

Annualisation factor
--------------------
Crypto perpetuals trade 24/7/365, so ``periods_per_year = 365 * 24 / tf_hours``:
``4H -> 2190``, ``1D -> 365``. Using 252 (equity convention) here would inflate Sharpe
by ~20% and is therefore wrong for this asset class.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .engine import (
    EXIT_AFTER_4H,
    STOP_LOSS_BEFORE_4H,
    TAKE_PROFIT_BEFORE_4H,
    Trade,
)


def periods_per_year(tf_hours: float) -> float:
    return 365.0 * 24.0 / tf_hours


@dataclass
class Metrics:
    n_bars: int = 0
    n_trades: int = 0
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_dd: float = 0.0
    max_dd_bars: int = 0
    max_dd_days: float = 0.0
    calmar: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    expectancy_ccy: float = 0.0
    expectancy_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff: float = 0.0
    exposure: float = 0.0
    turnover_x: float = 0.0
    avg_holding_hours: float = 0.0
    median_holding_hours: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0
    cost_drag_ratio: float = 0.0  # costs / gross profit (gross = sum of positive gross)
    gross_profit: float = 0.0
    net_over_gross: float = 0.0
    n_stop_before_4h: int = 0
    n_tp_before_4h: int = 0
    n_after_4h: int = 0
    min_holding_compliance: float = 0.0
    avg_mfe_r: float = 0.0
    avg_mae_r: float = 0.0
    n_long: int = 0
    n_short: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    best_trade_share: float = 0.0   # share of net profit from the single best trade
    top5_share: float = 0.0
    profitable_months: float = 0.0
    n_months: int = 0
    final_equity: float = 1.0

    def as_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _safe(x: float, default: float = 0.0) -> float:
    return float(x) if np.isfinite(x) else default


def compute_metrics(
    equity: np.ndarray,
    trades: list[Trade],
    tf_hours: float,
    timestamps: np.ndarray | None = None,
) -> Metrics:
    m = Metrics()
    n = len(equity)
    m.n_bars = n
    m.n_trades = len(trades)
    if n < 2:
        return m

    eq = np.asarray(equity, dtype=float)
    m.final_equity = float(eq[-1])
    m.total_return = float(eq[-1] / eq[0] - 1.0) if eq[0] > 0 else float("nan")

    ppy = periods_per_year(tf_hours)
    ret = np.diff(eq) / np.where(eq[:-1] == 0, np.nan, eq[:-1])
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    sd = float(np.std(ret, ddof=1)) if len(ret) > 1 else 0.0
    m.sharpe = _safe(np.mean(ret) / sd * math.sqrt(ppy)) if sd > 0 else 0.0
    downside = ret[ret < 0]
    dsd = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    m.sortino = _safe(np.mean(ret) / dsd * math.sqrt(ppy)) if dsd > 0 else 0.0

    yrs = n / ppy
    if eq[0] > 0 and eq[-1] > 0 and yrs > 0:
        m.cagr = _safe((eq[-1] / eq[0]) ** (1.0 / yrs) - 1.0)
    else:
        m.cagr = -1.0

    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    m.max_dd = float(dd.min())
    # Longest underwater stretch.
    under = dd < -1e-12
    run = best = 0
    for u in under:
        run = run + 1 if u else 0
        best = max(best, run)
    m.max_dd_bars = int(best)
    m.max_dd_days = best * tf_hours / 24.0
    m.calmar = _safe(m.cagr / abs(m.max_dd)) if m.max_dd < 0 else 0.0

    if trades:
        net = np.array([t.net_pnl for t in trades], dtype=float)
        gross = np.array([t.gross_pnl for t in trades], dtype=float)
        wins = net[net > 0]
        losses = net[net <= 0]
        m.win_rate = float(len(wins) / len(net))
        m.avg_win = float(wins.mean()) if len(wins) else 0.0
        m.avg_loss = float(losses.mean()) if len(losses) else 0.0
        m.payoff = abs(m.avg_win / m.avg_loss) if m.avg_loss else 0.0
        m.profit_factor = (
            float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf")
        )
        m.expectancy_ccy = float(net.mean())
        notionals = np.array([t.notional_at_entry for t in trades], dtype=float)
        rel = np.divide(net, notionals, out=np.zeros_like(net), where=notionals > 0)
        m.expectancy_pct = float(rel.mean())
        m.total_fees = float(sum(t.fees for t in trades))
        m.total_slippage = float(sum(t.slippage_cost for t in trades))
        m.total_funding = float(sum(t.funding_cost for t in trades))
        positive_gross = float(gross[gross > 0].sum())
        m.gross_profit = positive_gross
        m.cost_drag_ratio = (
            (m.total_fees + m.total_slippage) / positive_gross if positive_gross > 0 else 0.0
        )
        m.net_over_gross = float(net.sum() / positive_gross) if positive_gross > 0 else 0.0

        hold = np.array([t.holding_hours for t in trades], dtype=float)
        m.avg_holding_hours = float(hold.mean())
        m.median_holding_hours = float(np.median(hold))
        m.exposure = float(np.sum([t.holding_bars for t in trades]) / n)
        total_notional = float(sum(t.notional_at_entry * 2 for t in trades))
        m.turnover_x = total_notional / float(np.mean(eq))

        m.n_long = int(sum(1 for t in trades if t.side > 0))
        m.n_short = int(sum(1 for t in trades if t.side < 0))
        for sgn, attr in ((1, "long_win_rate"), (-1, "short_win_rate")):
            sub = net[np.array([t.side for t in trades]) == sgn]
            setattr(m, attr, float((sub > 0).mean()) if len(sub) else 0.0)

        order = np.sort(net)[::-1]
        tot = net.sum()
        if tot > 0:
            m.best_trade_share = float(order[0] / tot)
            m.top5_share = float(order[:5].sum() / tot)

        m.n_stop_before_4h = sum(1 for t in trades if t.exit_class == STOP_LOSS_BEFORE_4H)
        m.n_tp_before_4h = sum(1 for t in trades if t.exit_class == TAKE_PROFIT_BEFORE_4H)
        m.n_after_4h = sum(1 for t in trades if t.exit_class == EXIT_AFTER_4H)
        # Compliance = every trade is either >= 4h or an allowed early exit.
        allowed_early = m.n_stop_before_4h + m.n_tp_before_4h
        m.min_holding_compliance = 1.0 if allowed_early + m.n_after_4h == len(trades) else 0.0
        m.avg_mfe_r = float(np.mean([t.mfe for t in trades]))
        m.avg_mae_r = float(np.mean([t.mae for t in trades]))

    if timestamps is not None and len(timestamps) == n:
        months = np.array(
            [int(pd_ts.year) * 12 + int(pd_ts.month)
             for pd_ts in np.asarray(timestamps, dtype="datetime64[ms]").astype(object)]
        )
        uniq, inv = np.unique(months, return_inverse=True)
        monthly = np.array([eq[inv == k][-1] / (eq[inv == k][0]) - 1.0 for k in range(len(uniq))])
        m.n_months = int(len(uniq))
        m.profitable_months = float((monthly > 0).mean()) if len(monthly) else 0.0

    return m


def equity_returns(equity: np.ndarray) -> np.ndarray:
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return np.zeros(0)
    r = np.diff(eq) / np.where(eq[:-1] == 0, np.nan, eq[:-1])
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def segment_by(df: dict, key: np.ndarray, trades: list[Trade], equity: np.ndarray,
               timestamps: np.ndarray) -> dict[str, dict]:
    """Equity-curve performance per segment label (regime, year, vol tercile)."""
    out: dict[str, dict] = {}
    labels = np.unique(key)
    for lab in labels:
        mask = key == lab
        seg = equity[mask]
        if len(seg) < 2:
            continue
        ret = np.diff(seg) / np.where(seg[:-1] == 0, np.nan, seg[:-1])
        ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        tsel = [t for t in trades if mask[t.exit_bar] if t.exit_bar < len(mask)]
        out[str(lab)] = {
            "n_bars": int(mask.sum()),
            "return": float(seg[-1] / seg[0] - 1.0) if seg[0] > 0 else float("nan"),
            "mean_bar_return": float(np.mean(ret)),
            "bar_sharpe_unannualised": _safe(
                np.mean(ret) / np.std(ret, ddof=1), 0.0
            ) if len(ret) > 1 and np.std(ret, ddof=1) > 0 else 0.0,
            "n_trades": len(tsel),
            "net": float(sum(t.net_pnl for t in tsel)),
            "win_rate": float(np.mean([t.net_pnl > 0 for t in tsel])) if tsel else 0.0,
        }
    return out
