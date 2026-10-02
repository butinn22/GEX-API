"""Self-contained SVG chart builders for backtest reports.

Pure functions: they take plain numbers (no matplotlib, no network, no DOM) and
return an SVG string. That keeps the reporter dependency-light and lets the same
charts be embedded in an HTML report, an email, or a PDF (via a rasteriser).

Charts
------
* :func:`line_chart`         — multi-series equity curves (portfolio + per-ticker)
* :func:`drawdown_chart`     — underwater / drawdown area
* :func:`fan_chart`          — Monte-Carlo percentile bands (p5–p95, p25–p75, median)
* :func:`histogram_chart`    — distribution of Monte-Carlo final returns
* :func:`correlation_heatmap`— pairwise return correlation matrix
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "Palette",
    "LIGHT",
    "DARK",
    "line_chart",
    "drawdown_chart",
    "fan_chart",
    "histogram_chart",
    "correlation_heatmap",
]


@dataclass(frozen=True)
class Palette:
    bg: str
    panel: str
    grid: str
    axis: str
    text: str
    muted: str
    series: tuple[str, ...]
    band: str


LIGHT = Palette(
    bg="#ffffff", panel="#fbfcfd", grid="#e6e8eb", axis="#9aa4b2", text="#3d4757",
    muted="#6b7280",
    series=("#2f81f7", "#f0883e", "#3fb950", "#a371f7", "#e5534b", "#39c5cf", "#d29922"),
    band="#2f81f7",
)
DARK = Palette(
    bg="#0d1117", panel="#11161d", grid="#21262d", axis="#6e7681", text="#c9d1d9",
    muted="#8b949e",
    series=("#58a6ff", "#f0883e", "#3fb950", "#a371f7", "#f85149", "#39c5cf", "#d29922"),
    band="#58a6ff",
)


# ── helpers ────────────────────────────────────────────────────────────


def _esc(s: object) -> str:
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _money(v: float) -> str:
    a = abs(v)
    if a >= 1e9:
        return f"${v / 1e9:.1f}B"
    if a >= 1e6:
        return f"${v / 1e6:.2f}M"
    if a >= 1e3:
        return f"${v / 1e3:.1f}k"
    return f"${v:.0f}"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt(v: float, kind: str) -> str:
    if kind == "money":
        return _money(v)
    if kind == "pct":
        return _pct(v)
    return f"{v:.2f}"


def _nice_bounds(lo: float, hi: float) -> tuple[float, float]:
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.06
    return lo - pad, hi + pad


def _x_positions(n: int, x0: float, x1: float) -> list[float]:
    if n <= 1:
        return [x0]
    return [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]


def _label_indices(n: int, count: int = 6) -> list[int]:
    if n <= count:
        return list(range(n))
    return sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})


# ── charts ─────────────────────────────────────────────────────────────


def line_chart(
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float]]],
    *,
    title: str = "",
    width: int = 860,
    height: int = 340,
    y_kind: str = "money",
    palette: Palette = LIGHT,
) -> str:
    """Multi-series line chart with gridlines, x/y axes and a legend."""
    L, R, T, B = 76, 16, 34, 40
    pw, ph = width - L - R, height - T - B
    values = [v for _, vals in series for v in vals]
    if not values or not labels:
        return _empty_svg(width, height, title, palette)
    y0, y1 = _nice_bounds(min(values), max(values))
    n = len(labels)
    xs = _x_positions(n, L, width - R)

    def y_of(v: float) -> float:
        return T + (1 - (v - y0) / (y1 - y0)) * ph

    parts: list[str] = [
        f'<rect class="chart-bg" x="0" y="0" width="{width}" height="{height}" fill="{palette.bg}"/>'
    ]
    if title:
        parts.append(
            f'<text x="{L}" y="20" font-size="13" font-weight="600" '
            f'fill="{palette.text}">{_esc(title)}</text>'
        )
    for t in range(5):
        v = y0 + (y1 - y0) * t / 4
        y = y_of(v)
        parts.append(
            f'<line x1="{L}" y1="{y:.1f}" x2="{width - R}" y2="{y:.1f}" '
            f'stroke="{palette.grid}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{L - 8}" y="{y + 3:.1f}" font-size="10" fill="{palette.axis}" '
            f'text-anchor="end">{_fmt(v, y_kind)}</text>'
        )
    for i in _label_indices(n):
        parts.append(
            f'<text x="{xs[i]:.1f}" y="{height - 14}" font-size="10" fill="{palette.axis}" '
            f'text-anchor="middle">{_esc(labels[i])}</text>'
        )
    for k, (name, vals) in enumerate(series):
        color = palette.series[k % len(palette.series)]
        if len(vals) != n:
            vals = list(vals)[:n] + [vals[-1]] * max(n - len(vals), 0)
        pts = " ".join(f"{xs[i]:.1f},{y_of(v):.1f}" for i, v in enumerate(vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        lx = L + k * 150
        parts.append(f'<rect class="legend" x="{lx}" y="{height - 10}" width="9" height="9" fill="{color}" rx="2"/>')
        parts.append(
            f'<text x="{lx + 14}" y="{height - 2}" font-size="10" fill="{palette.muted}">{_esc(name)}</text>'
        )
    return _svg(width, height, parts)


def drawdown_chart(
    labels: Sequence[str],
    equity: Sequence[float],
    *,
    title: str = "Drawdown",
    width: int = 860,
    height: int = 220,
    palette: Palette = LIGHT,
) -> str:
    """Underwater chart: drawdown (negative %) over time."""
    L, R, T, B = 76, 16, 34, 30
    pw, ph = width - L - R, height - T - B
    eq = list(equity)
    if len(eq) < 2 or not labels:
        return _empty_svg(width, height, title, palette)
    peak = eq[0]
    dd: list[float] = []
    for v in eq:
        peak = max(peak, v)
        dd.append((v - peak) / peak if peak > 0 else 0.0)
    worst = min(dd) if min(dd) < 0 else -0.01
    y0, y1 = worst * 1.08, 0.0
    xs = _x_positions(len(dd), L, width - R)

    def y_of(v: float) -> float:
        return T + (1 - (v - y0) / (y1 - y0)) * ph

    base_y = y_of(0.0)
    parts: list[str] = [
        f'<rect class="chart-bg" x="0" y="0" width="{width}" height="{height}" fill="{palette.bg}"/>'
    ]
    parts.append(
        f'<text x="{L}" y="20" font-size="13" font-weight="600" fill="{palette.text}">{_esc(title)}</text>'
    )
    for t in range(3):
        v = y0 + (y1 - y0) * t / 2
        y = y_of(v)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{width - R}" y2="{y:.1f}" stroke="{palette.grid}"/>')
        parts.append(
            f'<text x="{L - 8}" y="{y + 3:.1f}" font-size="10" fill="{palette.axis}" '
            f'text-anchor="end">{_pct(v)}</text>'
        )
    area = f"M {xs[0]:.1f},{base_y:.1f} " + " ".join(
        f"L {xs[i]:.1f},{y_of(v):.1f}" for i, v in enumerate(dd)
    ) + f" L {xs[-1]:.1f},{base_y:.1f} Z"
    parts.append(f'<path d="{area}" fill="#e5534b" fill-opacity="0.28" stroke="#e5534b" stroke-width="1.4"/>')
    for i in _label_indices(len(labels)):
        parts.append(
            f'<text x="{xs[i]:.1f}" y="{height - 8}" font-size="10" fill="{palette.axis}" '
            f'text-anchor="middle">{_esc(labels[i])}</text>'
        )
    return _svg(width, height, parts)


def fan_chart(
    steps: Sequence[int],
    bands: dict[str, Sequence[float]],
    *,
    title: str = "Monte-Carlo equity paths",
    width: int = 860,
    height: int = 360,
    palette: Palette = LIGHT,
) -> str:
    """Fan chart from percentile bands (keys ``p5``/``p25``/``p50``/``p75``/``p95``)."""
    L, R, T, B = 76, 16, 34, 30
    pw, ph = width - L - R, height - T - B
    req = ("p5", "p25", "p50", "p75", "p95")
    if not steps or any(k not in bands or not bands[k] for k in req):
        return _empty_svg(width, height, title, palette)
    lo = min(min(bands[k]) for k in req)
    hi = max(max(bands[k]) for k in req)
    y0, y1 = _nice_bounds(lo, hi)
    n = len(steps)
    xs = _x_positions(n, L, width - R)

    def y_of(v: float) -> float:
        return T + (1 - (v - y0) / (y1 - y0)) * ph

    def band_path(low: Sequence[float], high: Sequence[float]) -> str:
        up = " ".join(f"L {xs[i]:.1f},{y_of(v):.1f}" for i, v in enumerate(high))
        down = " ".join(f"L {xs[i]:.1f},{y_of(v):.1f}" for i, v in reversed(list(enumerate(low))))
        return f"M {xs[0]:.1f},{y_of(high[0]):.1f} {up} {down} Z"

    parts: list[str] = [
        f'<rect class="chart-bg" x="0" y="0" width="{width}" height="{height}" fill="{palette.bg}"/>'
    ]
    parts.append(
        f'<text x="{L}" y="20" font-size="13" font-weight="600" fill="{palette.text}">{_esc(title)}</text>'
    )
    for t in range(5):
        v = y0 + (y1 - y0) * t / 4
        y = y_of(v)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{width - R}" y2="{y:.1f}" stroke="{palette.grid}"/>')
        parts.append(
            f'<text x="{L - 8}" y="{y + 3:.1f}" font-size="10" fill="{palette.axis}" '
            f'text-anchor="end">{_money(v)}</text>'
        )
    parts.append(f'<path d="{band_path(bands["p5"], bands["p95"])}" fill="{palette.band}" fill-opacity="0.16"/>')
    parts.append(f'<path d="{band_path(bands["p25"], bands["p75"])}" fill="{palette.band}" fill-opacity="0.24"/>')
    med = " ".join(f"{xs[i]:.1f},{y_of(v):.1f}" for i, v in enumerate(bands["p50"]))
    parts.append(f'<polyline points="{med}" fill="none" stroke="{palette.band}" stroke-width="2"/>')
    legend = [("p95–p5", "0.16"), ("p75–p25", "0.30"), ("median", "1")]
    for i, (lbl, op) in enumerate(legend):
        lx = L + i * 130
        parts.append(
            f'<rect class="legend" x="{lx}" y="{height - 10}" width="9" height="9" fill="{palette.band}" '
            f'fill-opacity="{op}" rx="2"/>'
        )
        parts.append(f'<text x="{lx + 14}" y="{height - 2}" font-size="10" fill="{palette.muted}">{lbl}</text>')
    for i in _label_indices(n):
        parts.append(
            f'<text x="{xs[i]:.1f}" y="{height - 16}" font-size="10" fill="{palette.axis}" '
            f'text-anchor="middle">{steps[i]}</text>'
        )
    return _svg(width, height, parts)


def histogram_chart(
    counts: Sequence[int],
    centers: Sequence[float],
    *,
    title: str = "Final return distribution",
    var_95: float | None = None,
    width: int = 860,
    height: int = 280,
    palette: Palette = LIGHT,
) -> str:
    """Bar histogram of final returns, with an optional VaR 95 % marker."""
    L, R, T, B = 60, 16, 34, 34
    pw, ph = width - L - R, height - T - B
    if not counts or not centers:
        return _empty_svg(width, height, title, palette)
    lo, hi = min(centers), max(centers)
    if hi <= lo:
        hi = lo + 1e-6
    cmax = max(counts) or 1
    bw = pw / len(counts)

    def x_of(v: float) -> float:
        return L + (v - lo) / (hi - lo) * pw

    parts: list[str] = [
        f'<rect class="chart-bg" x="0" y="0" width="{width}" height="{height}" fill="{palette.bg}"/>'
    ]
    parts.append(
        f'<text x="{L}" y="20" font-size="13" font-weight="600" fill="{palette.text}">{_esc(title)}</text>'
    )
    parts.append(f'<line x1="{L}" y1="{T + ph}" x2="{width - R}" y2="{T + ph}" stroke="{palette.axis}"/>')
    for i, c in enumerate(counts):
        h = (c / cmax) * ph
        x = L + i * bw + 0.5
        parts.append(
            f'<rect class="bar" x="{x:.1f}" y="{T + ph - h:.1f}" width="{max(bw - 1.4, 0.8):.1f}" height="{h:.1f}" '
            f'fill="{palette.series[0]}" fill-opacity="0.85" rx="1"/>'
        )
    for v in (lo, (lo + hi) / 2, hi):
        parts.append(
            f'<text x="{x_of(v):.1f}" y="{height - 12}" font-size="10" fill="{palette.axis}" '
            f'text-anchor="middle">{_pct(v)}</text>'
        )
    if var_95 is not None and lo <= var_95 <= hi:
        xv = x_of(var_95)
        parts.append(
            f'<line x1="{xv:.1f}" y1="{T}" x2="{xv:.1f}" y2="{T + ph}" stroke="#e5534b" '
            f'stroke-width="1.6" stroke-dasharray="4 3"/>'
        )
        parts.append(
            f'<text x="{xv:.1f}" y="{T - 4}" font-size="10" fill="#e5534b" text-anchor="middle">'
            f'VaR95 {_pct(var_95)}</text>'
        )
    return _svg(width, height, parts)


def correlation_heatmap(
    symbols: Sequence[str],
    matrix: Sequence[Sequence[float]],
    *,
    title: str = "Return correlation",
    cell: int = 54,
    palette: Palette = LIGHT,
) -> str:
    """Square correlation heatmap (blue = +1, red = −1)."""
    n = len(symbols)
    if n == 0 or len(matrix) != n:
        return _empty_svg(n * cell + 120, cell + 60, title, palette)
    left, top = 96, 44
    width = left + n * cell + 20
    height = top + n * cell + 40

    def color(v: float) -> str:
        v = max(-1.0, min(1.0, v))
        if v >= 0:
            mix = v
            r = int(255 + (47 - 255) * mix)
            g = int(255 + (129 - 255) * mix)
            b = int(255 + (247 - 255) * mix)
        else:
            mix = -v
            r = int(255 + (229 - 255) * mix)
            g = int(255 + (83 - 255) * mix)
            b = int(255 + (75 - 255) * mix)
        return f"rgb({r},{g},{b})"

    parts: list[str] = [
        f'<rect class="chart-bg" x="0" y="0" width="{width}" height="{height}" fill="{palette.bg}"/>'
    ]
    parts.append(
        f'<text x="16" y="22" font-size="13" font-weight="600" fill="{palette.text}">{_esc(title)}</text>'
    )
    for j, sym in enumerate(symbols):
        parts.append(
            f'<text x="{left + j * cell + cell / 2:.1f}" y="{top - 8}" font-size="10" '
            f'fill="{palette.muted}" text-anchor="middle">{_esc(sym)}</text>'
        )
    for i in range(n):
        parts.append(
            f'<text x="{left - 8}" y="{top + i * cell + cell / 2 + 3:.1f}" font-size="10" '
            f'fill="{palette.muted}" text-anchor="end">{_esc(symbols[i])}</text>'
        )
        for j in range(n):
            v = float(matrix[i][j])
            x = left + j * cell
            y = top + i * cell
            parts.append(
                f'<rect class="cell" data-r="{i}" data-c="{j}" x="{x}" y="{y}" '
                f'width="{cell - 2}" height="{cell - 2}" fill="{color(v)}" rx="3"/>'
            )
            parts.append(
                f'<text x="{x + cell / 2 - 1:.1f}" y="{y + cell / 2 + 3:.1f}" font-size="9" '
                f'fill="#11161d" text-anchor="middle">{v:.2f}</text>'
            )
    return _svg(width, height, parts)


# ── svg wrapper ────────────────────────────────────────────────────────


def _svg(width: int, height: int, parts: Sequence[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" style="max-width:{width}px" role="img">' + "".join(parts) + "</svg>"
    )


def _empty_svg(width: int, height: int, title: str, palette: Palette) -> str:
    return _svg(width, height, [
        f'<rect class="chart-bg" x="0" y="0" width="{width}" height="{height}" fill="{palette.bg}"/>',
        f'<text x="16" y="{height / 2:.0f}" font-size="12" fill="{palette.muted}">'
        f'{_esc(title or "no data")} — not enough data</text>',
    ])
