"""Backtest reporters: JSON + self-contained HTML (with SVG charts) + PDF.

Three reporters share the same chart library (:mod:`trading.application.backtest.charts`):

* :class:`BacktestReporter`  — single-symbol backtest;
* :class:`PortfolioReporter` — multi-ticker portfolio (combined + per-ticker equity,
  drawdown, correlation heatmap);
* :class:`MonteCarloReporter`— percentile fan chart + final-return histogram.

The HTML output inlines the SVGs, so a single ``.html`` file is a complete,
shareable report with no external assets and no JavaScript.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence

from .charts import (
    correlation_heatmap,
    drawdown_chart,
    fan_chart,
    histogram_chart,
    line_chart,
)
from .engine import BacktestResult
from .metrics import BacktestMetrics
from .monte_carlo import MonteCarloResult
from .portfolio import PortfolioBacktestResult

__all__ = ["BacktestReporter", "PortfolioReporter", "MonteCarloReporter"]


def _f(x: float) -> float | None:
    return x if math.isfinite(x) else None


def _metric_rows(m: BacktestMetrics | dict) -> str:
    if isinstance(m, BacktestMetrics):
        items = {
            "total_return": m.total_return,
            "annualized_return": m.annualized_return,
            "sharpe": m.sharpe,
            "sortino": _f(m.sortino),
            "calmar": _f(m.calmar),
            "max_drawdown": m.max_drawdown,
            "var_95": m.var_95,
            "cvar_95": m.cvar_95,
            "win_rate": m.win_rate,
            "profit_factor": _f(m.profit_factor),
            "n_trades": m.n_trades,
            "n_periods": m.n_periods,
        }
    else:
        items = dict(m)
    return "".join(
        f"<tr><td>{k}</td><td>{v if v is not None else '—'}</td></tr>" for k, v in items.items()
    )


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>"
        "body{font:14px/1.5 -apple-system,'Segoe UI',Roboto,sans-serif;color:#3d4757;"
        "background:#fff;margin:0;padding:28px;max-width:960px;margin:0 auto}"
        "h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:26px 0 6px}"
        ".sub{color:#6b7280;margin-bottom:18px}"
        "table{border-collapse:collapse;width:100%;max-width:480px}"
        "td{padding:5px 8px;border-bottom:1px solid #eceef1;font-size:13px}"
        "td:first-child{color:#6b7280}"
        ".chart{margin:6px 0 4px}figure{margin:0 0 8px}"
        "</style></head><body>" + body + "</body></html>"
    )


def _labels(times: Sequence) -> list[str]:
    return [str(t)[:10] for t in times]


class BacktestReporter:
    """Single-symbol backtest report."""

    def __init__(self, result: BacktestResult, *, strategy: str = "", symbol: str = "",
                 times: Sequence | None = None) -> None:
        self.result = result
        self.strategy = strategy
        self.symbol = symbol
        self.times = list(times) if times is not None else []

    def _sanitized_metrics(self) -> dict:
        m = self.result.metrics
        return {
            "total_return": m.total_return,
            "annualized_return": m.annualized_return,
            "sharpe": m.sharpe,
            "sortino": _f(m.sortino),
            "calmar": _f(m.calmar),
            "max_drawdown": m.max_drawdown,
            "var_95": m.var_95,
            "cvar_95": m.cvar_95,
            "win_rate": m.win_rate,
            "profit_factor": _f(m.profit_factor),
            "n_trades": m.n_trades,
            "n_periods": m.n_periods,
        }

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "metrics": self._sanitized_metrics(),
            "n_trades": len(self.result.trades),
            "final_equity": float(self.result.equity_curve[-1]),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def equity_svg(self) -> str:
        labels = _labels(self.times) if self.times else [str(i) for i in range(len(self.result.equity_curve))]
        return line_chart(
            labels,
            [(self.symbol or self.strategy or "equity", list(map(float, self.result.equity_curve)))],
            title=f"{self.strategy or 'backtest'} {self.symbol} — equity",
        )

    def drawdown_svg(self) -> str:
        labels = _labels(self.times) if self.times else [str(i) for i in range(len(self.result.equity_curve))]
        return drawdown_chart(labels, list(map(float, self.result.equity_curve)))

    def to_html(self) -> str:
        header = f"<h1>{self.strategy or 'Backtest'} — {self.symbol or ''}</h1>"
        intro = f"<div class='sub'>strategy <b>{self.strategy or '—'}</b> · symbol <b>{self.symbol or '—'}</b></div>"
        body = (
            header + intro
            + "<h2>Metrics</h2><table>" + _metric_rows(self.result.metrics) + "</table>"
            + "<h2>Equity curve</h2><div class='chart'>" + self.equity_svg() + "</div>"
            + "<h2>Drawdown</h2><div class='chart'>" + self.drawdown_svg() + "</div>"
        )
        return _page(f"Backtest {self.symbol}", body)

    def to_pdf(self, path: str) -> str:
        """Write a simple PDF report; returns the path."""
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(path, pagesize=A4)
        width, height = A4
        y = height - 72
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, y, f"Backtest: {self.strategy or 'backtest'} {self.symbol}")
        y -= 32
        c.setFont("Helvetica", 10)
        for k, v in self._sanitized_metrics().items():
            c.drawString(72, y, f"{k}: {v if v is not None else 'n/a'}")
            y -= 16
        c.save()
        return path


class PortfolioReporter:
    """Multi-ticker portfolio report."""

    def __init__(self, result: PortfolioBacktestResult, *, title: str = "Portfolio backtest") -> None:
        self.result = result
        self.title = title

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "initial_cash": self.result.initial_cash,
            "final_equity": float(self.result.equity_curve[-1]),
            "n_tickers": self.result.n_tickers,
            "metrics": _series(self.result.metrics),
            "tickers": [
                {
                    "symbol": t.symbol, "strategy": t.strategy, "weight": t.weight,
                    "capital": t.capital, "n_trades": t.result.metrics.n_trades,
                    "total_return": t.total_return,
                }
                for t in self.result.tickers
            ],
            "errors": list(self.result.errors),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def equity_svg(self) -> str:
        labels = _labels(self.result.times)
        series = [("Portfolio", list(map(float, self.result.equity_curve)))]
        for k, t in enumerate(self.result.tickers):
            curve = (
                self.result.aligned_equity[k]
                if k < len(self.result.aligned_equity)
                else t.result.equity_curve
            )
            series.append((t.symbol, list(map(float, curve))))
        return line_chart(labels, series, title="Equity — portfolio vs per-ticker")

    def drawdown_svg(self) -> str:
        return drawdown_chart(_labels(self.result.times), list(map(float, self.result.equity_curve)))

    def correlation_svg(self) -> str:
        corr = self.result.correlation
        if corr is None:
            return ""
        return correlation_heatmap(list(corr.symbols), [list(r) for r in corr.matrix])

    def to_html(self) -> str:
        rows = "".join(
            f"<tr><td>{t.symbol}</td><td>{t.strategy}</td><td>{t.weight:.1%}</td>"
            f"<td>{t.result.metrics.n_trades}</td><td>{t.total_return:.2%}</td></tr>"
            for t in self.result.tickers
        )
        errors = "".join(
            f"<tr><td>{e['symbol']}</td><td style='color:#e5534b'>{e['error']}</td></tr>"
            for e in self.result.errors
        )
        body = (
            f"<h1>{self.title}</h1>"
            f"<div class='sub'>{self.result.n_tickers} tickers · capital "
            f"${self.result.initial_cash:,.0f} · {len(self.result.times)} bars</div>"
            + "<h2>Portfolio metrics</h2><table>" + _metric_rows(self.result.metrics) + "</table>"
            + "<h2>Equity</h2><div class='chart'>" + self.equity_svg() + "</div>"
            + "<h2>Drawdown</h2><div class='chart'>" + self.drawdown_svg() + "</div>"
            + ("<h2>Correlation</h2><div class='chart'>" + self.correlation_svg() + "</div>"
               if self.result.correlation else "")
            + "<h2>Per-ticker</h2><table style='max-width:none'><tr>"
              "<td><b>Symbol</b></td><td><b>Strategy</b></td><td><b>Weight</b></td>"
              "<td><b>Trades</b></td><td><b>Return</b></td></tr>" + rows + "</table>"
            + ("<h2>Errors</h2><table>" + errors + "</table>" if errors else "")
        )
        return _page(self.title, body)


class MonteCarloReporter:
    """Monte-Carlo distribution report."""

    def __init__(self, result: MonteCarloResult, *, label: str = "portfolio") -> None:
        self.result = result
        self.label = label

    def to_dict(self) -> dict:
        r = self.result
        return {
            "label": self.label, "method": r.method, "n_paths": r.n_paths, "n_steps": r.n_steps,
            "prob_profit": r.prob_profit, "var_95": r.var_95, "cvar_95": r.cvar_95,
            "final_percentiles": r.final_percentiles,
            "metrics_mean": r.metrics_mean, "metrics_ci": r.metrics_ci,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_html(self) -> str:
        r = self.result
        ci_rows = "".join(
            f"<tr><td>{k}</td><td>{r.metrics_mean[k]:.4f}</td>"
            f"<td>[{lo:.4f}, {hi:.4f}]</td></tr>"
            for k, (lo, hi) in r.metrics_ci.items()
        )
        body = (
            f"<h1>Monte-Carlo — {self.label}</h1>"
            f"<div class='sub'>{r.method} · {r.n_paths:,} paths · {r.n_steps} steps · "
            f"P(profit) {r.prob_profit:.1%} · VaR95 {r.var_95:.2%} · CVaR95 {r.cvar_95:.2%}</div>"
            + "<h2>Equity fan</h2><div class='chart'>"
            + fan_chart(list(r.steps), r.bands, title="Monte-Carlo equity paths") + "</div>"
            + "<h2>Final return distribution</h2><div class='chart'>"
            + histogram_chart(r.histogram.counts, r.histogram.centers, var_95=r.var_95) + "</div>"
            + "<h2>Metric confidence intervals (5–95%)</h2><table>" + ci_rows + "</table>"
        )
        return _page(f"Monte-Carlo {self.label}", body)


def _series(m: BacktestMetrics) -> dict:
    return {
        "total_return": m.total_return, "annualized_return": m.annualized_return,
        "sharpe": m.sharpe, "sortino": _f(m.sortino), "calmar": _f(m.calmar),
        "max_drawdown": m.max_drawdown, "var_95": m.var_95, "cvar_95": m.cvar_95,
        "win_rate": m.win_rate, "profit_factor": _f(m.profit_factor),
        "n_trades": m.n_trades, "n_periods": m.n_periods,
    }
