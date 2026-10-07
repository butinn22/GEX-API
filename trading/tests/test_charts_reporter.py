"""Tests for the SVG chart library + the three reporters."""
from __future__ import annotations

import asyncio
import json

import numpy as np

from trading.application.backtest.charts import (
    DARK,
    LIGHT,
    correlation_heatmap,
    drawdown_chart,
    fan_chart,
    histogram_chart,
    line_chart,
)
from trading.application.backtest.engine import BacktestConfig, run_backtest
from trading.application.backtest.monte_carlo import MonteCarloConfig, run_monte_carlo
from trading.application.backtest.portfolio import (
    PortfolioBacktestConfig,
    TickerSpec,
    run_portfolio_backtest,
)
from trading.application.backtest.reporter import (
    BacktestReporter,
    MonteCarloReporter,
    PortfolioReporter,
)
from trading.application.strategies.buy_and_hold import BuyAndHold

# ── helpers ────────────────────────────────────────────────────────────


def _equity(n=120, seed=0):
    r = np.random.default_rng(seed).normal(0.0008, 0.015, n)
    return 100_000.0 * np.cumprod(1.0 + np.concatenate([[0.0], r]))[: n + 1]


def _labels(n):
    return [f"2024-{(i % 12) + 1:02d}-01" for i in range(n)]


def _is_svg(s: str) -> bool:
    return s.startswith("<svg") and s.rstrip().endswith("</svg>")


def _well_formed(svg: str) -> bool:
    """Cheap well-formedness: every opened tag is closed, no stray brackets."""
    import xml.etree.ElementTree as ET

    try:
        ET.fromstring(svg)
        return True
    except ET.ParseError:
        return False


# ── line_chart ─────────────────────────────────────────────────────────


def test_line_chart_is_well_formed_svg():
    svg = line_chart(_labels(60), [("equity", list(_equity(60)))], title="EQ")
    assert _is_svg(svg)
    assert _well_formed(svg)
    assert "EQ" in svg and "equity" in svg


def test_line_chart_multi_series_uses_distinct_colors():
    svg = line_chart(
        _labels(50),
        [("a", list(_equity(50, 1))), ("b", list(_equity(50, 2))), ("c", list(_equity(50, 3)))],
    )
    for color in (LIGHT.series[0], LIGHT.series[1], LIGHT.series[2]):
        assert color in svg
    assert svg.count("<polyline") == 3  # one polyline per series
    assert len({LIGHT.series[0], LIGHT.series[1], LIGHT.series[2]}) == 3


def test_line_chart_draws_axes_and_gridlines():
    svg = line_chart(_labels(80), [("e", list(_equity(80)))])
    assert svg.count("<line") >= 4  # 5 gridlines
    assert svg.count("<text") >= 6  # y ticks + x labels


def test_line_chart_handles_empty_data():
    svg = line_chart([], [], title="nothing")
    assert _is_svg(svg) and _well_formed(svg)
    assert "not enough data" in svg


def test_line_chart_handles_single_point():
    svg = line_chart(["2024-01-01"], [("e", [100.0])])
    assert _is_svg(svg) and _well_formed(svg)


def test_line_chart_flat_series_does_not_divide_by_zero():
    svg = line_chart(_labels(20), [("flat", [100.0] * 20)])
    assert _is_svg(svg) and _well_formed(svg)


def test_line_chart_y_kind_pct_formats_labels():
    svg = line_chart(_labels(20), [("r", [0.01 * i for i in range(20)])], y_kind="pct")
    assert "%" in svg


def test_line_chart_dark_palette_changes_colors():
    light = line_chart(_labels(20), [("e", list(_equity(20)))], palette=LIGHT)
    dark = line_chart(_labels(20), [("e", list(_equity(20)))], palette=DARK)
    assert LIGHT.series[0] in light and DARK.series[0] in dark
    assert light != dark


def test_line_chart_escapes_title_html():
    svg = line_chart(_labels(10), [("e", list(_equity(10)))], title="<script>x</script>")
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


# ── drawdown_chart ─────────────────────────────────────────────────────


def test_drawdown_chart_is_well_formed_and_shows_worst():
    eq = [100.0, 120.0, 90.0, 110.0, 80.0]
    svg = drawdown_chart(_labels(5), eq)
    assert _is_svg(svg) and _well_formed(svg)
    assert "#e5534b" in svg  # drawdown fill
    assert "%" in svg


def test_drawdown_chart_monotonic_rise_has_no_deep_drawdown():
    svg = drawdown_chart(_labels(6), [100, 110, 120, 130, 140, 150])
    assert _is_svg(svg) and _well_formed(svg)


def test_drawdown_chart_empty():
    assert "not enough data" in drawdown_chart([], [])


# ── fan_chart ──────────────────────────────────────────────────────────


def _bands(n=100, seed=0):
    r = np.random.default_rng(seed).normal(0.0005, 0.02, (500, n))
    paths = 100_000.0 * np.cumprod(1.0 + r, axis=1)
    paths = np.hstack([np.full((500, 1), 100_000.0), paths])
    return {f"p{p}": tuple(float(x) for x in np.percentile(paths, p, axis=0))
            for p in (5, 25, 50, 75, 95)}


def test_fan_chart_is_well_formed_with_all_bands():
    b = _bands()
    svg = fan_chart(list(range(101)), b)
    assert _is_svg(svg) and _well_formed(svg)
    assert svg.count("<path") >= 2  # p5–p95 and p25–p75 bands
    assert svg.count("<polyline") >= 1  # median line
    assert "median" in svg


def test_fan_chart_missing_bands_degrades_gracefully():
    svg = fan_chart([0, 1, 2], {"p50": [1, 2, 3]})
    assert _is_svg(svg) and "not enough data" in svg


def test_fan_chart_empty_steps():
    assert "not enough data" in fan_chart([], {})


# ── histogram_chart ────────────────────────────────────────────────────


def test_histogram_is_well_formed_with_bars():
    counts, edges = np.histogram(np.random.default_rng(0).normal(0, 0.1, 1000), bins=25)
    centers = (edges[:-1] + edges[1:]) / 2
    svg = histogram_chart(counts.tolist(), centers.tolist(), var_95=-0.15)
    assert _is_svg(svg) and _well_formed(svg)
    assert svg.count('class="bar"') == 25
    assert svg.count('class="chart-bg"') == 1
    assert "VaR95" in svg


def test_histogram_without_var_marker():
    counts, edges = np.histogram(np.random.default_rng(1).normal(0, 0.1, 500), bins=10)
    centers = (edges[:-1] + edges[1:]) / 2
    svg = histogram_chart(counts.tolist(), centers.tolist())
    assert "VaR95" not in svg
    assert _well_formed(svg)


def test_histogram_var_outside_range_is_not_drawn():
    counts, edges = np.histogram([0.0, 0.1, 0.2], bins=3)
    centers = (edges[:-1] + edges[1:]) / 2
    svg = histogram_chart(counts.tolist(), centers.tolist(), var_95=-99.0)
    assert "VaR95" not in svg


def test_histogram_empty():
    assert "not enough data" in histogram_chart([], [])


# ── correlation_heatmap ────────────────────────────────────────────────


def test_heatmap_is_well_formed_and_sizes_to_matrix():
    syms = ["A", "B", "C"]
    mat = [[1.0, 0.5, -0.3], [0.5, 1.0, 0.1], [-0.3, 0.1, 1.0]]
    svg = correlation_heatmap(syms, mat)
    assert _is_svg(svg) and _well_formed(svg)
    assert svg.count('class="cell"') == 9  # 3x3 cells
    for s in syms:
        assert s in svg


def test_heatmap_color_scales_with_sign():
    """Positive correlation → blue end, negative → red end of the diverging scale."""
    import re as _re

    def cell_colors(svg: str) -> list[tuple[int, int, int]]:
        out = []
        for m in _re.finditer(r'class="cell"[^>]*fill="rgb\((\d+),(\d+),(\d+)\)"', svg):
            out.append(tuple(int(g) for g in m.groups()))
        return out

    pos = cell_colors(correlation_heatmap(["A", "B"], [[1.0, 0.9], [0.9, 1.0]]))
    neg = cell_colors(correlation_heatmap(["A", "B"], [[1.0, -0.9], [-0.9, 1.0]]))
    assert pos and neg
    # Strong positive → blue dominant; strong negative → red dominant.
    assert any(b > r for r, g, b in pos)
    assert any(r > b for r, g, b in neg)


def test_heatmap_empty_symbols():
    assert _is_svg(correlation_heatmap([], []))


# ── BacktestReporter ───────────────────────────────────────────────────


def _single_result():
    bars = []
    from datetime import datetime, timedelta

    import numpy as np

    rng = np.random.default_rng(0)
    price = 100.0
    t0 = datetime(2024, 1, 1)
    for i in range(150):
        price *= 1.0 + 0.001 + rng.normal(0, 0.01)
        from trading.domain import Bar

        bars.append(Bar(t0 + timedelta(days=i), price * 0.99, price * 1.01, price * 0.98, price, 1000))
    return bars


def test_backtest_reporter_json_keeps_legacy_shape():
    bars = _single_result()
    res = asyncio.run(run_backtest(BuyAndHold("X"), bars, BacktestConfig()))
    rep = BacktestReporter(res, strategy="buy_and_hold", symbol="X",
                           times=[b.timestamp for b in bars])
    data = json.loads(rep.to_json())
    assert data["strategy"] == "buy_and_hold"
    assert data["symbol"] == "X"
    assert set(data["metrics"]) >= {"sharpe", "calmar", "max_drawdown"}
    assert data["final_equity"] > 0


def test_backtest_reporter_html_contains_metrics_and_charts():
    bars = _single_result()
    res = asyncio.run(run_backtest(BuyAndHold("X"), bars, BacktestConfig()))
    rep = BacktestReporter(res, strategy="buy_and_hold", symbol="X",
                           times=[b.timestamp for b in bars])
    html = rep.to_html()
    assert "buy_and_hold" in html and "max_drawdown" in html
    assert html.count("<svg") == 2  # equity + drawdown
    assert "<!doctype html>" in html


def test_backtest_reporter_equity_svg_uses_provided_times():
    bars = _single_result()
    res = asyncio.run(run_backtest(BuyAndHold("X"), bars, BacktestConfig()))
    svg = BacktestReporter(res, strategy="s", symbol="X",
                           times=[b.timestamp for b in bars]).equity_svg()
    assert "2024-" in svg


def test_backtest_reporter_pdf():
    import os
    import tempfile

    bars = _single_result()
    res = asyncio.run(run_backtest(BuyAndHold("X"), bars, BacktestConfig()))
    rep = BacktestReporter(res, strategy="buy_and_hold", symbol="X")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "r.pdf")
        rep.to_pdf(path)
        with open(path, "rb") as f:
            assert f.read(5) == b"%PDF-"


# ── PortfolioReporter ──────────────────────────────────────────────────


def _portfolio_result():
    from datetime import datetime, timedelta

    from trading.domain import Bar
    bars = {}
    t0 = datetime(2024, 1, 1)
    for k, sym in enumerate(("A", "B", "C")):
        rng = np.random.default_rng(k)
        price = 100.0
        rows = []
        for i in range(180):
            price *= 1.0 + 0.0008 + rng.normal(0, 0.012)
            rows.append(Bar(t0 + timedelta(days=i), price * 0.99, price * 1.01, price * 0.98, price, 1000))
        bars[sym] = rows
    return asyncio.run(run_portfolio_backtest(
        [TickerSpec(s, strategy="sma_crossover", params={"fast": 5, "slow": 20}, source="synthetic")
         for s in ("A", "B", "C")],
        PortfolioBacktestConfig(initial_cash=120_000.0),
        bars_by_symbol=bars,
    ))


def test_portfolio_reporter_json():
    res = _portfolio_result()
    data = json.loads(PortfolioReporter(res).to_json())
    assert data["n_tickers"] == 3
    assert data["initial_cash"] == 120_000.0
    assert len(data["tickers"]) == 3
    assert "sharpe" in data["metrics"]
    assert data["errors"] == []


def test_portfolio_reporter_html_has_all_charts():
    html = PortfolioReporter(_portfolio_result()).to_html()
    assert html.count("<svg") == 3  # equity + drawdown + correlation
    assert "Correlation" in html
    assert "Per-ticker" in html
    for s in ("A", "B", "C"):
        assert f">{s}<" in html


def test_portfolio_reporter_omits_correlation_for_single_ticker():
    from datetime import datetime, timedelta

    from trading.domain import Bar
    t0 = datetime(2024, 1, 1)
    rng = np.random.default_rng(9)
    price = 100.0
    rows = []
    for i in range(150):
        price *= 1.0 + 0.001 + rng.normal(0, 0.01)
        rows.append(Bar(t0 + timedelta(days=i), price, price * 1.02, price * 0.98, price, 1000))
    res = asyncio.run(run_portfolio_backtest(
        [TickerSpec("SOLO", source="synthetic")], bars_by_symbol={"SOLO": rows}
    ))
    html = PortfolioReporter(res).to_html()
    assert "Correlation" not in html
    assert html.count("<svg") == 2


def test_portfolio_reporter_lists_errors():
    from datetime import datetime, timedelta

    from trading.domain import Bar
    t0 = datetime(2024, 1, 1)
    rows = [Bar(t0 + timedelta(days=i), 100 + i, 101 + i, 99 + i, 100 + i, 1000) for i in range(120)]
    res = asyncio.run(run_portfolio_backtest(
        [TickerSpec("OK", source="synthetic"), TickerSpec("BAD", source="nope")],
        bars_by_symbol={"OK": rows},
    ))
    html = PortfolioReporter(res).to_html()
    assert "Errors" in html and "BAD" in html


# ── MonteCarloReporter ─────────────────────────────────────────────────


def test_monte_carlo_reporter_json_and_html():
    mc = run_monte_carlo(
        np.random.default_rng(0).normal(0.0006, 0.02, 260),
        MonteCarloConfig(n_paths=1500, method="block_bootstrap", seed=4),
    )
    rep = MonteCarloReporter(mc, label="basket")
    data = json.loads(rep.to_json())
    assert data["label"] == "basket"
    assert data["n_paths"] == 1500
    assert "metrics_ci" in data and "final_percentiles" in data

    html = rep.to_html()
    assert html.count("<svg") == 2  # fan + histogram
    assert "P(profit)" in html
    assert "Metric confidence intervals" in html
    assert _well_formed(html.split("<svg")[1].join(["<svg", ""]).split("</svg>")[0] + "</svg>")


def test_monte_carlo_reporter_lists_every_ci_metric():
    mc = run_monte_carlo(np.random.default_rng(1).normal(0, 0.02, 200),
                         MonteCarloConfig(n_paths=500))
    html = MonteCarloReporter(mc).to_html()
    for key in mc.metrics_ci:
        assert key in html


def test_reports_are_self_contained_no_external_refs():
    html = PortfolioReporter(_portfolio_result()).to_html()
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in html
    assert "<script" not in html
