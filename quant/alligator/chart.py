"""Build the full-period equity curve SVG for the final frozen configuration.

Dark-theme chart (IDE): reads the cached trainval (risk 0.5%) and holdout
equity slices, concatenates them into the full 2021-07..2026-10 curve and
marks the TRAIN / VAL / HOLDOUT boundaries.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
VAL_END = pd.Timestamp("2025-07-01", tz="UTC")
TRAIN_END = pd.Timestamp("2024-07-01", tz="UTC")

W, H = 960, 420
PAD_L, PAD_R, PAD_T, PAD_B = 70, 20, 30, 46
BG, FG, GRID = "#16181d", "#d7dae0", "#2a2e37"
EQ_LINE, EQ_FILL = "#4da3ff", "rgba(77,163,255,0.12)"
MARK = "#8f96a3"
BH = None  # band color reused from JS-less svg


def _load_full_equity() -> pd.Series:
    tv = pd.read_csv(RESULTS / "equity_trainval_risk0.005.csv", index_col=0,
                     parse_dates=True).iloc[:, 0]
    ho = pd.read_csv(RESULTS / "equity_holdout_4h.csv", index_col=0,
                     parse_dates=True).iloc[:, 0]
    tv.index = pd.to_datetime(tv.index, utc=True)
    ho.index = pd.to_datetime(ho.index, utc=True)
    full = pd.concat([tv[tv.index < VAL_END], ho])
    return full[~full.index.duplicated(keep="first")]


def build_svg(out: Path | None = None, *, width: int = W,
              height: int = H) -> str:
    eq = _load_full_equity()
    # downsample to <= 1400 points for a compact polyline
    step = max(1, len(eq) // 1400)
    eqd = eq.iloc[::step]
    x = np.arange(len(eqd), dtype=float)
    yv = eqd.to_numpy(float)

    plot_w, plot_h = width - PAD_L - PAD_R, height - PAD_T - PAD_B
    xmin, xmax = 0.0, float(len(eqd) - 1)
    ymin, ymax = float(yv.min()), float(yv.max())
    yrange = max(ymax - ymin, 1e-9)
    ymin -= yrange * 0.05
    ymax += yrange * 0.08
    sx = lambda i: PAD_L + (i - xmin) / (xmax - xmin) * plot_w
    sy = lambda v: PAD_T + (ymax - v) / (ymax - ymin) * plot_h

    pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(yv))
    base_y = sy(ymin)

    def xpos(ts: pd.Timestamp) -> float:
        i = float(np.searchsorted(eqd.index.values, np.datetime64(ts)))
        return sx(min(max(i, 0.0), xmax))

    marks = [
        (xpos(TRAIN_END), "TRAIN end 24-07", "#7f8896"),
        (xpos(VAL_END), "HOLDOUT start 25-07", "#c9a86a"),
    ]

    # year ticks
    years = sorted({d.year for d in eqd.index})
    year_x = []
    for yr in years:
        ts = pd.Timestamp(f"{yr}-01-01", tz="UTC")
        if eqd.index[0] <= ts <= eqd.index[-1]:
            year_x.append((xpos(ts), str(yr)))

    peak = np.maximum.accumulate(yv)
    dd = (peak - yv) / peak
    max_dd = float(dd.max())

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="Consolas, monospace">',
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
    ]
    # horizontal grid + y labels
    for i in range(5):
        v = ymin + (ymax - ymin) * i / 4
        yy = sy(v)
        parts.append(f'<line x1="{PAD_L}" y1="{yy:.1f}" x2="{width-PAD_R}" '
                     f'y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{PAD_L-8}" y="{yy+4:.1f}" fill="{MARK}" '
                     f'font-size="11" text-anchor="end">{v:,.0f}</text>')
    # split markers
    for xx, label, color in marks:
        parts.append(f'<line x1="{xx:.1f}" y1="{PAD_T}" x2="{xx:.1f}" '
                     f'y2="{PAD_T+plot_h}" stroke="{color}" stroke-width="1.5" '
                     f'stroke-dasharray="6 4"/>')
        parts.append(f'<text x="{xx+5:.1f}" y="{PAD_T+13}" fill="{color}" '
                     f'font-size="11">{label}</text>')
    # equity area + line
    parts.append(f'<polygon points="{PAD_L},{base_y:.1f} {pts} '
                 f'{width-PAD_R},{base_y:.1f}" fill="{EQ_FILL}"/>')
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{EQ_LINE}" '
                 f'stroke-width="1.6"/>')
    # year ticks
    for xx, label in year_x:
        parts.append(f'<text x="{xx:.1f}" y="{height-PAD_B+18}" fill="{MARK}" '
                     f'font-size="11" text-anchor="middle">{label}</text>')
    parts.append(
        f'<text x="{PAD_L}" y="{PAD_T-10}" fill="{FG}" font-size="13">'
        f'Alligator Confluence 4H — portfolio equity (final params, '
        f'risk 0.5%/trade) — max DD {max_dd:.2%}</text>')
    parts.append("</svg>")
    svg = "\n".join(parts)
    if out is not None:
        out.write_text(svg)
    return svg


if __name__ == "__main__":
    p = RESULTS / "equity_curve.svg"
    build_svg(p)
    print("saved ->", p)
