#!/usr/bin/env python
"""CLI: fetch bulk OHLCV history (real sources with fallback, or synthetic).

Usage:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/fetch_data.py SBER --source=moex --limit=200
    PYTHONPATH=. .venv/Scripts/python.exe scripts/fetch_data.py SYNTH --source=synthetic --limit=100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def _main(args: argparse.Namespace) -> None:
    from trading.adapters.fetchers import SyntheticFetcher, default_registry
    from trading.domain import DataFetchError, Exchange

    sources = {
        "moex": [Exchange.MOEX],
        "yfinance": [Exchange.YFINANCE],
        "bybit": [Exchange.BYBIT],
        "auto": [Exchange.MOEX, Exchange.YFINANCE, Exchange.BYBIT],
    }
    try:
        if args.source == "synthetic":
            bars = await SyntheticFetcher(seed=args.seed).get_ohlcv(
                args.symbol, args.timeframe, limit=args.limit
            )
        else:
            bars = await default_registry().get_ohlcv(
                sources[args.source], args.symbol, args.timeframe, limit=args.limit
            )
    except DataFetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    for b in bars:
        print(json.dumps({
            "t": b.timestamp.isoformat(), "o": b.open, "h": b.high,
            "l": b.low, "c": b.close, "v": b.volume,
        }))
    print(f"# {len(bars)} bars", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch OHLCV history")
    p.add_argument("symbol")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--source", default="synthetic",
                   choices=["synthetic", "moex", "yfinance", "bybit", "auto"])
    p.add_argument("--seed", type=int, default=0)
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
