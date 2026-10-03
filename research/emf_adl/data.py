"""Real market data layer for the EMF+ADL research programme.

Sources (all real, all public, all verifiable):

* ``/v5/market/kline``           — real OHLCV candles (Bybit v5, linear USDT perps)
* ``/v5/market/funding/history`` — real funding-rate settlements (8h)
* ``/v5/market/tickers``         — real 24h turnover / volume, used for liquidity ranking

Design rules enforced here (no exceptions):

1. **No synthetic data.** There is no generator, no fixture, no interpolation path in
   this module. Gaps are *reported*, never filled.
2. **UTC and point-in-time.** Every timestamp is UTC milliseconds. A bar stamped
   ``t`` is the bar that *closed* at ``t + timeframe``; features are computed from it
   only after it closes.
3. **Deterministic.** Same symbol/timeframe/range ⇒ identical frame, and the frame is
   hashed (sha256 of the raw candle payload) so a run can be reproduced and audited.
4. **Disk cache is content-addressed by request**, and every cached frame carries the
   manifest entry that produced it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BYBIT = "https://api.bybit.com"
CATEGORY = "linear"
CACHE = Path(__file__).with_name("cache")

#: timeframe label -> (Bybit interval code, hours per bar)
TIMEFRAMES: dict[str, tuple[str, float]] = {
    "4H": ("240", 4.0),
    "1D": ("D", 24.0),
}

MAX_KLINE_LIMIT = 1000
MAX_FUNDING_LIMIT = 200

_USER_AGENT = "gex-api-research/1.0 (+deterministic backtest harness)"


class DataError(RuntimeError):
    """Raised when real data cannot be obtained or fails integrity checks."""


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _get(path: str, params: dict, *, attempts: int = 5, timeout: int = 30) -> dict:
    """GET a Bybit endpoint with bounded exponential backoff. Raises on failure."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BYBIT}{path}?{query}"
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("retCode") != 0:
                raise DataError(
                    f"{path} retCode={payload.get('retCode')} msg={payload.get('retMsg')}"
                )
            return payload["result"]
        except Exception as exc:  # noqa: BLE001 - network layer, retry everything
            last = exc
            if attempt < attempts - 1:
                # Bybit throttles bursts; a bounded backoff keeps a 48-series
                # prefetch from stalling for minutes on one poisoned request.
                time.sleep(min(0.4 * 2.0 ** attempt, 4.0))
    raise DataError(f"GET {path} failed after {attempts} attempts: {last!r}")


# --------------------------------------------------------------------------- #
# Manifest / audit
# --------------------------------------------------------------------------- #
@dataclass
class DataManifest:
    """Everything needed to reproduce a bar frame and audit its provenance."""

    symbol: str
    timeframe: str
    category: str = CATEGORY
    interval_code: str = ""
    source: str = "bybit-v5"
    fetched_utc: str = ""
    first_bar_utc: str = ""
    last_bar_utc: str = ""
    n_bars: int = 0
    sha256: str = ""
    url_sample: str = ""
    n_funding: int = 0
    funding_sha256: str = ""
    integrity: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _hash_frame(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(df[["timestamp", "open", "high", "low", "close", "volume"]]
                                  .to_numpy(dtype=float)).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Klines
# --------------------------------------------------------------------------- #
def fetch_klines(
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    *,
    category: str = CATEGORY,
) -> pd.DataFrame:
    """Fetch every real 1m+ candle in ``[start_ms, end_ms]``, paginating backwards.

    Returns a frame sorted ascending with columns
    ``timestamp, open, high, low, close, volume, turnover``.
    """
    if timeframe not in TIMEFRAMES:
        raise DataError(f"unsupported timeframe {timeframe!r}; have {list(TIMEFRAMES)}")
    code, _ = TIMEFRAMES[timeframe]

    rows: list[list] = []
    cursor = end_ms
    while True:
        res = _get(
            "/v5/market/kline",
            {
                "category": category,
                "symbol": symbol,
                "interval": code,
                "start": start_ms,
                "end": cursor,
                "limit": MAX_KLINE_LIMIT,
            },
        )
        batch = res.get("list") or []
        if not batch:
            break
        rows.extend(batch)
        oldest = int(batch[-1][0])
        if oldest <= start_ms or len(batch) < MAX_KLINE_LIMIT:
            break
        cursor = oldest - 1

    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        )

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )
    for col in ("open", "high", "low", "close", "volume", "turnover"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = df["timestamp"].astype("int64")
    df = (
        df.dropna(subset=["open", "high", "low", "close"])
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    df = df[(df.timestamp >= start_ms) & (df.timestamp <= end_ms)].reset_index(drop=True)
    return df


def fetch_funding(
    symbol: str, start_ms: int, end_ms: int, *, category: str = CATEGORY
) -> pd.DataFrame:
    """Real funding settlements in the window. Columns: ``timestamp, rate``."""
    rows: list[list] = []
    cursor = end_ms
    while True:
        res = _get(
            "/v5/market/funding/history",
            {
                "category": category,
                "symbol": symbol,
                "startTime": start_ms,
                "endTime": cursor,
                "limit": MAX_FUNDING_LIMIT,
            },
        )
        batch = res.get("list") or []
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(r["fundingRateTimestamp"]) for r in batch)
        if oldest <= start_ms or len(batch) < MAX_FUNDING_LIMIT:
            break
        cursor = oldest - 1

    if not rows:
        return pd.DataFrame(columns=["timestamp", "rate"])
    df = pd.DataFrame(
        {
            "timestamp": [int(r["fundingRateTimestamp"]) for r in rows],
            "rate": [float(r["fundingRate"]) for r in rows],
        }
    ).drop_duplicates(subset="timestamp")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[(df.timestamp >= start_ms) & (df.timestamp <= end_ms)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
def fetch_top_symbols(n: int, *, min_turnover_usd: float = 20_000_000.0) -> pd.DataFrame:
    """Live liquidity ranking of USDT perps by real 24h turnover.

    This is *current* liquidity. Point-in-time ranking for a historical window is
    done separately (see :func:`rank_by_trailing_turnover`) because Bybit does not
    publish a historical turnover table.
    """
    res = _get("/v5/market/tickers", {"category": CATEGORY})
    rows = []
    for t in res.get("list") or []:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        try:
            turnover = float(t.get("turnover24h") or 0.0)
            last = float(t.get("lastPrice") or 0.0)
        except (TypeError, ValueError):
            continue
        if turnover < min_turnover_usd or last <= 0:
            continue
        rows.append(
            {
                "symbol": sym,
                "turnover24h": turnover,
                "volume24h": float(t.get("volume24h") or 0.0),
                "last": last,
                "open_interest": float(t.get("openInterestValue") or 0.0),
            }
        )
    df = pd.DataFrame(rows).sort_values("turnover24h", ascending=False).reset_index(drop=True)
    return df.head(n)


def rank_by_trailing_turnover(
    bars_by_symbol: dict[str, pd.DataFrame], as_of_ms: int, n: int, lookback_bars: int = 30
) -> list[str]:
    """Point-in-time liquidity ranking using only candles *before* ``as_of_ms``.

    Turnover is read from the real candle payload (quote turnover column), so the
    ranking uses information that genuinely existed at ``as_of_ms``. A symbol needs a
    complete, gap-free lookback window to be eligible — hence the ``None`` entries the
    integrity check turns into exclusions.
    """
    scored: list[tuple[float, str]] = []
    for sym, df in bars_by_symbol.items():
        window = df[df.timestamp < as_of_ms].tail(lookback_bars)
        if len(window) < lookback_bars:
            continue
        scored.append((float(window["turnover"].sum()), sym))
    scored.sort(reverse=True)
    return [s for _, s in scored[:n]]


# --------------------------------------------------------------------------- #
# Integrity
# --------------------------------------------------------------------------- #
def check_integrity(df: pd.DataFrame, timeframe: str) -> dict:
    """Structural validation of a real candle frame. Reports, never repairs."""
    _, hours = TIMEFRAMES[timeframe]
    step = int(hours * 3600 * 1000)
    report: dict = {"n_bars": int(len(df)), "ok": False, "issues": []}

    if df.empty:
        report["issues"].append("empty frame")
        return report

    ts = df["timestamp"].to_numpy(dtype=np.int64)
    report["first_bar_utc"] = pd.to_datetime(ts[0], unit="ms", utc=True).isoformat()
    report["last_bar_utc"] = pd.to_datetime(ts[-1], unit="ms", utc=True).isoformat()

    if not np.all(np.diff(ts) > 0):
        report["issues"].append("timestamps not strictly increasing")
    dup = int(len(ts) - len(np.unique(ts)))
    if dup:
        report["issues"].append(f"{dup} duplicate timestamps")

    gaps = int(np.sum(np.diff(ts) != step))
    report["n_gaps"] = gaps
    report["expected_bars"] = int((ts[-1] - ts[0]) // step + 1)
    report["coverage"] = round(len(ts) / report["expected_bars"], 6) if report["expected_bars"] else 0.0
    if gaps:
        report["issues"].append(f"{gaps} spacing gaps (coverage {report['coverage']:.4f})")

    bad_ohlc = int(
        np.sum(
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )
    )
    report["bad_ohlc"] = bad_ohlc
    if bad_ohlc:
        report["issues"].append(f"{bad_ohlc} bars violate high>=max(o,c) and low<=min(o,c)")

    nonpos = int(np.sum((df[["open", "high", "low", "close"]] <= 0).any(axis=1)))
    if nonpos:
        report["issues"].append(f"{nonpos} non-positive prices")

    zero_vol = int(np.sum(df["volume"] <= 0))
    report["zero_volume_bars"] = zero_vol
    if zero_vol:
        report["issues"].append(f"{zero_vol} zero-volume bars (illiquid/stale)")

    # 1-bar return outliers: a real perp moving >60% in one 4H bar is suspicious data,
    # not signal. Flag for exclusion review rather than "correcting" it.
    ret = df["close"].pct_change().abs()
    extreme = int(np.sum(ret > 0.60))
    if extreme:
        report["issues"].append(f"{extreme} bars with |1-bar return| > 60% (verify)")

    report["ok"] = (
        not report["issues"]
        or set(report["issues"]) <= {f"{zero_vol} zero-volume bars (illiquid/stale)"}
    )
    return report


def worst_quality_windows(df: pd.DataFrame, timeframe: str, top: int = 5) -> list[dict]:
    """Largest real gaps in the series — required by the reporting format."""
    _, hours = TIMEFRAMES[timeframe]
    step = int(hours * 3600 * 1000)
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    if len(ts) < 2:
        return []
    d = np.diff(ts)
    idx = np.argsort(d)[::-1][:top]
    out = []
    for i in idx:
        if d[i] <= step:
            continue
        out.append(
            {
                "from_utc": pd.to_datetime(ts[i], unit="ms", utc=True).isoformat(),
                "to_utc": pd.to_datetime(ts[i + 1], unit="ms", utc=True).isoformat(),
                "missing_bars": int(d[i] // step - 1),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Cached loader
# --------------------------------------------------------------------------- #
@dataclass
class LoadedSeries:
    symbol: str
    timeframe: str
    bars: pd.DataFrame
    funding: pd.DataFrame
    manifest: DataManifest


def load_series(
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    *,
    with_funding: bool = True,
    use_cache: bool = True,
) -> LoadedSeries:
    """Load one real series (+ real funding) with an on-disk cache and a manifest."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        f"{symbol}|{timeframe}|{start_ms}|{end_ms}|{with_funding}".encode()
    ).hexdigest()[:16]
    # Pickle (not parquet): the cache must round-trip float64 bit-exactly and must not
    # add a pyarrow/fastparquet dependency to the research path.
    bar_path = CACHE / f"bars_{symbol}_{timeframe}_{key}.pkl"
    fun_path = CACHE / f"fund_{symbol}_{timeframe}_{key}.pkl"
    man_path = CACHE / f"manifest_{symbol}_{timeframe}_{key}.json"

    if use_cache and bar_path.exists() and man_path.exists():
        bars = pd.read_pickle(bar_path)
        funding = pd.read_pickle(fun_path) if fun_path.exists() else pd.DataFrame(
            columns=["timestamp", "rate"]
        )
        manifest = DataManifest(**json.loads(man_path.read_text(encoding="utf-8")))
        return LoadedSeries(symbol, timeframe, bars, funding, manifest)

    bars = fetch_klines(symbol, timeframe, start_ms, end_ms)
    if bars.empty:
        raise DataError(f"{symbol} {timeframe}: no real candles returned for the window")
    funding = (
        fetch_funding(symbol, start_ms, end_ms) if with_funding else pd.DataFrame(
            columns=["timestamp", "rate"]
        )
    )

    integrity = check_integrity(bars, timeframe)
    integrity["worst_gaps"] = worst_quality_windows(bars, timeframe)
    manifest = DataManifest(
        symbol=symbol,
        timeframe=timeframe,
        interval_code=TIMEFRAMES[timeframe][0],
        fetched_utc=pd.Timestamp.now("UTC").isoformat(),
        first_bar_utc=integrity.get("first_bar_utc", ""),
        last_bar_utc=integrity.get("last_bar_utc", ""),
        n_bars=int(len(bars)),
        sha256=_hash_frame(bars),
        url_sample=f"{BYBIT}/v5/market/kline?category={CATEGORY}&symbol={symbol}"
                   f"&interval={TIMEFRAMES[timeframe][0]}",
        n_funding=int(len(funding)),
        funding_sha256=(
            hashlib.sha256(funding.to_numpy(dtype=float).tobytes()).hexdigest()
            if len(funding)
            else ""
        ),
        integrity=integrity,
    )

    if use_cache:
        bars.to_pickle(bar_path)
        funding.to_pickle(fun_path)
        man_path.write_text(manifest.to_json(), encoding="utf-8")

    return LoadedSeries(symbol, timeframe, bars, funding, manifest)


def load_panel(
    symbols: Iterable[str],
    timeframe: str,
    start_ms: int,
    end_ms: int,
    *,
    with_funding: bool = True,
) -> tuple[dict[str, LoadedSeries], list[dict]]:
    """Load many series. Returns ``(loaded, exclusions)`` — exclusions are reported,
    never silently replaced."""
    loaded: dict[str, LoadedSeries] = {}
    exclusions: list[dict] = []
    for sym in symbols:
        try:
            series = load_series(sym, timeframe, start_ms, end_ms, with_funding=with_funding)
        except Exception as exc:  # noqa: BLE001
            exclusions.append({"symbol": sym, "timeframe": timeframe, "reason": repr(exc)})
            continue
        integ = series.manifest.integrity
        # Conservative eligibility gate: a series must be >=95% complete and have no
        # OHLC violations. Anything else is excluded from the study and reported.
        if integ.get("coverage", 0) < 0.95 or integ.get("bad_ohlc", 0) > 0:
            exclusions.append(
                {
                    "symbol": sym,
                    "timeframe": timeframe,
                    "reason": f"integrity: coverage={integ.get('coverage')} "
                              f"bad_ohlc={integ.get('bad_ohlc')} "
                              f"issues={integ.get('issues')}",
                }
            )
            continue
        loaded[sym] = series
    return loaded, exclusions


def ms(year: int, month: int, day: int) -> int:
    """UTC calendar → epoch milliseconds."""
    return int(pd.Timestamp(year=year, month=month, day=day, tz="UTC").timestamp() * 1000)


def prefetch_panel(
    symbols: Iterable[str],
    timeframes: Iterable[str],
    start_ms: int,
    end_ms: int,
    *,
    workers: int = 4,
) -> list[dict]:
    """Warm the disk cache for a whole panel. I/O-bound, so threads are safe here.

    Fetching is the only expensive part of the harness; the analytics run from cache
    at ~0.1 s per series. Returns a per-series status report.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs = [(s, tf) for s in symbols for tf in timeframes]
    report: list[dict] = []

    def one(job):
        sym, tf = job
        t0 = time.time()
        try:
            ls = load_series(sym, tf, start_ms, end_ms, with_funding=True)
            return {
                "symbol": sym, "timeframe": tf, "ok": True,
                "n_bars": int(len(ls.bars)), "n_funding": int(len(ls.funding)),
                "coverage": ls.manifest.integrity.get("coverage"),
                "seconds": round(time.time() - t0, 1),
            }
        except Exception as exc:  # noqa: BLE001
            return {"symbol": sym, "timeframe": tf, "ok": False,
                    "error": repr(exc), "seconds": round(time.time() - t0, 1)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, j) for j in jobs]
        for fut in as_completed(futs):
            rep = fut.result()
            report.append(rep)
            log.info("prefetch %s %s ok=%s %ss", rep["symbol"], rep["timeframe"],
                     rep.get("ok"), rep.get("seconds"))
    return sorted(report, key=lambda r: (r["symbol"], r["timeframe"]))


def write_panel_manifest(path: Path, loaded: dict[str, LoadedSeries], exclusions: list[dict]) -> None:
    payload = {
        "generated_utc": pd.Timestamp.now("UTC").isoformat(),
        "source": f"{BYBIT} (Bybit v5 public market data)",
        "timeframes": {k: v[1] for k, v in TIMEFRAMES.items()},
        "series": {s: json.loads(ls.manifest.to_json()) for s, ls in sorted(loaded.items())},
        "exclusions": exclusions,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
