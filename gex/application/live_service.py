"""Live GEX analysis: polling via Yahoo Finance (US stocks/ETFs).

Standalone service (was LiveServiceMixin). Created by GEXService via composition.
"""
from __future__ import annotations

from typing import Optional

from gex.assets_config import DEFAULT_ASSETS
from gex.domain.data_loader import OptionSnapshot
from gex.domain.pipeline import GEXPipeline
from gex.application.pipeline_runner import GEXPipelineRunner
from gex.adapters.cache.redis_client import RedisClient
from gex.adapters.persistence.repository import ChainRepository
from gex.schemas import GEXAnalysisOut, GEXProfileOut
from gex.adapters.fetchers.yf_fetcher import YFOptionsFetcher


class LiveGEXService:
    """Live GEX analysis via Yahoo Finance (US stocks/ETFs).

    Parameters
    ----------
    repo : ChainRepository
        Snapshot storage (shared with other services).
    runner : GEXPipelineRunner
        Shared pipeline orchestrator.
    redis_client : RedisClient or None
        Redis cache client.
    """

    def __init__(
        self,
        repo: ChainRepository,
        runner: GEXPipelineRunner,
        redis_client: Optional[RedisClient] = None,
    ):
        self._repo = repo
        self._runner = runner
        self._redis = redis_client

    def analyze(
        self, ticker: str, days: float = 30.0, max_expiries: int = 5,
    ) -> GEXAnalysisOut:
        ticker = ticker.upper()
        fetcher = YFOptionsFetcher(max_expiries=max_expiries, redis_client=self._redis)
        snapshot = fetcher.fetch(ticker)

        DEFAULT_ASSETS.setdefault(ticker, {"spot": snapshot.spot, "q": 0.0})
        DEFAULT_ASSETS[ticker]["spot"] = snapshot.spot

        self._repo.put(ticker, snapshot)

        pipeline = GEXPipeline(
            spot=snapshot.spot, symbol=ticker,
            r=DEFAULT_ASSETS.get(ticker, {}).get("r", 0.045),
            q=DEFAULT_ASSETS.get(ticker, {}).get("q", 0.0),
        )
        return self._runner.run_analysis(ticker, snapshot, days, pipeline)

    def analyze_profile(
        self, ticker: str, days: float = 30.0, max_expiries: int = 5,
    ) -> GEXProfileOut:
        ticker = ticker.upper()
        fetcher = YFOptionsFetcher(max_expiries=max_expiries, redis_client=self._redis)
        snapshot = fetcher.fetch(ticker)

        DEFAULT_ASSETS.setdefault(ticker, {"spot": snapshot.spot, "q": 0.0})
        DEFAULT_ASSETS[ticker]["spot"] = snapshot.spot

        self._repo.put(ticker, snapshot)

        pipeline = GEXPipeline(
            spot=snapshot.spot, symbol=ticker,
            r=DEFAULT_ASSETS.get(ticker, {}).get("r", 0.045),
            q=DEFAULT_ASSETS.get(ticker, {}).get("q", 0.0),
        )
        return self._runner.run_profile(ticker, snapshot, days, pipeline)
