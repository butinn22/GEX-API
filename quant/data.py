"""Real-data fetcher: Bybit v5 public spot klines, paginated, cached, validated.

Rules enforced:
- Only real exchange candles are stored. If the API returns nothing for a range,
  that range stays missing and is REPORTED, never interpolated.
- OHLC sanity checks: high >= max(open, close), low <= min(open, close), all > 0.
- Duplicate timestamps are dropped (keep first), bars are strictly increasing.
- Cache file + a sidecar .meta.json with sha256 for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

BASE = "https://api.bybit.com"
INTERVAL = {"4h": "240", "1d": "D"}
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

__all__ = ["fetch_klines", "load_or_fetch", "validate_ohlcv", "DataQualityReport"]


@dataclass
class DataQualityReport:
    symbol: str
    timeframe: str
    n_bars: int
    first: str
    last: str
    n_bad_ohlc: int          # bars failing high/low sanity
    expected_cadence_h: float
    n_gaps: int               # missing bars vs expected cadence
    max_gap_bars: int
    largest_gap: str
    sha256: str

    def summary(self) -> str:
        return (f"{self.symbol} {self.timeframe}: {self.n_bars} bars "
                f"[{self.first} .. {self.last}], bad_ohlc={self.n_bad_ohlc}, "
                f"gaps={self.n_gaps} (max {self.max_gap_bars} bars)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_klines(symbol: str, timeframe: str, *, sleep: float = 0.15) -> pd.DataFrame:
    """Fetch the ENTIRE available spot history for symbol/timeframe from Bybit.

    Paginates backwards with `end` = oldest_seen - 1 until the API returns an
    empty page. Returns a DataFrame indexed by UTC timestamp, columns
    open/high/low/close/volume.
    """
    interval = INTERVAL[timeframe]
    out: list[list] = []
    end_ms: int | None = None
    with httpx.Client(base_url=BASE, timeout=30) as c:
        while True:
            params = {"category": "spot", "symbol": symbol, "interval": interval,
                      "limit": 1000}
            if end_ms is not None:
                params["end"] = end_ms
            r = c.get("/v5/market/kline", params=params)
            r.raise_for_status()
            d = r.json()
            if d["retCode"] != 0:
                raise RuntimeError(f"bybit error {d['retCode']}: {d['retMsg']}")
            rows = d["result"]["list"]
            if not rows:
                break
            out.extend(rows)
            oldest = int(rows[-1][0])
            if end_ms is not None and oldest >= end_ms:
                break  # no progress (safety)
            end_ms = oldest - 1
            time.sleep(sleep)
    if not out:
        raise RuntimeError(f"no data for {symbol} {timeframe}")
    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col])
    df = (df.drop_duplicates(subset="ts", keep="first")
            .sort_values("ts")
            .set_index("ts"))[["open", "high", "low", "close", "volume"]]
    return df


def validate_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str,
                   cadence_h: float) -> DataQualityReport:
    """Sanity-check OHLC relationships and detect missing bars (gaps).

    Gaps are only reported — the caller decides how to treat them. No synthetic
    fills are ever created here.
    """
    bad = int(((df["high"] < df[["open", "close"]].max(axis=1) - 1e-9) |
               (df["low"] > df[["open", "close"]].min(axis=1) + 1e-9) |
               (df[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum())
    step = pd.Timedelta(hours=cadence_h)
    expected = df.index.to_series().diff()
    gaps = expected > step * 1.5
    n_gaps = int(gaps.sum())
    if n_gaps:
        gap_sizes = ((expected[gaps] / step).round().astype(int) - 1)
        max_gap = int(gap_sizes.max())
        worst_idx = gap_sizes.idxmax()
        largest = f"{worst_idx} (+{max_gap} bars)"
    else:
        max_gap, largest = 0, "none"
    return DataQualityReport(
        symbol=symbol, timeframe=timeframe, n_bars=len(df),
        first=str(df.index[0]), last=str(df.index[-1]),
        n_bad_ohlc=bad, expected_cadence_h=cadence_h,
        n_gaps=n_gaps, max_gap_bars=max_gap, largest_gap=largest,
        sha256="",
    )


def load_or_fetch(symbol: str, timeframe: str) -> tuple[pd.DataFrame, DataQualityReport]:
    """Cache-through loader. Returns (df, quality_report) — real data only."""
    cache = CACHE_DIR / f"{symbol}_{timeframe}.csv"
    meta_path = CACHE_DIR / f"{symbol}_{timeframe}.meta.json"
    if cache.exists() and meta_path.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        df = fetch_klines(symbol, timeframe)
        df.to_csv(cache)
    report = validate_ohlcv(df, symbol, timeframe, 4.0 if timeframe == "4h" else 24.0)
    report.sha256 = _sha256(cache)
    meta = asdict(report) | {"source": "bybit/v5/spot/kline",
                             "fetched_at": datetime.now(timezone.utc).isoformat()}
    meta_path.write_text(json.dumps(meta, indent=2))
    return df, report
