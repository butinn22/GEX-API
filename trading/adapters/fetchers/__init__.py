"""Fetcher adapters implementing ``BaseFetcher``.

Real public sources: MOEX ISS, Bybit, yFinance, Webull (best-effort), plus a
deterministic synthetic source for dev/testing.
"""
from .bybit import BybitFetcher
from .moex import MoexIssFetcher
from .registry import (
    FetcherRegistry,
    aclose_loop_registry,
    default_registry,
    loop_registry,
)
from .synthetic import SyntheticFetcher
from .validate import validate_bars
from .webull import WebullFetcher
from .yfinance import YFinanceFetcher

__all__ = [
    "BybitFetcher",
    "MoexIssFetcher",
    "WebullFetcher",
    "YFinanceFetcher",
    "SyntheticFetcher",
    "FetcherRegistry",
    "default_registry",
    "loop_registry",
    "aclose_loop_registry",
    "validate_bars",
]
