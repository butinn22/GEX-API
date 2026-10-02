"""Commodity GEX analysis: ETF-proxy options + OHLCV.

Standalone service (was CommodityServiceMixin). Created by GEXService via composition.
"""
from __future__ import annotations

import logging
from typing import Optional

from gex.commodity_assets import COMMODITY_ASSETS, resolve_commodity
from gex.adapters.fetchers.commodity_fetcher import CommodityFetcher
from gex.domain.pipeline import GEXPipeline
from gex.application.pipeline_runner import GEXPipelineRunner
from gex.adapters.cache.redis_client import RedisClient
from gex.adapters.persistence.repository import ChainRepository
from gex.schemas import GEXAnalysisOut, GEXProfileOut

logger = logging.getLogger(__name__)


class CommodityGEXService:
    """Commodity market analysis via ETF proxy options.

    Parameters
    ----------
    repo : ChainRepository
    runner : GEXPipelineRunner
    redis_client : RedisClient or None
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
        self, asset: str, days: float = 30.0, max_expiries: int = 5,
    ) -> GEXAnalysisOut:
        snapshot, _cfg = self._require_option_chain(asset, max_expiries)
        pipeline = GEXPipeline(
            spot=snapshot.spot, symbol=asset.upper(),
            r=0.045, q=0.0, per_contract=100,
            call_sign=+1.0, put_sign=-1.0,
        )
        self._repo.put(asset.upper(), snapshot)
        return self._runner.run_analysis(asset.upper(), snapshot, days, pipeline)

    def analyze_profile(
        self, asset: str, days: float = 30.0, max_expiries: int = 5,
    ) -> GEXProfileOut:
        snapshot, _cfg = self._require_option_chain(asset, max_expiries)
        pipeline = GEXPipeline(
            spot=snapshot.spot, symbol=asset.upper(),
            r=0.045, q=0.0, per_contract=100,
            call_sign=+1.0, put_sign=-1.0,
        )
        self._repo.put(asset.upper(), snapshot)
        return self._runner.run_profile(asset.upper(), snapshot, days, pipeline)

    def get_ohlcv(
        self, asset: str, timeframe: str = "1d", limit: int = 200,
    ) -> dict:
        asset = resolve_commodity(asset)
        fetcher = CommodityFetcher(redis_client=self._redis)
        return fetcher.fetch_ohlcv(asset.strip().upper(), timeframe, limit)

    def _require_option_chain(self, asset: str, max_expiries: int = 5):
        asset = resolve_commodity(asset)
        cfg = COMMODITY_ASSETS.get(asset.upper())
        if cfg is None:
            raise ValueError(
                f"Unsupported commodity '{asset}'. "
                f"Available: {list(COMMODITY_ASSETS)}."
            )
        if not cfg.get("has_options"):
            raise ValueError(
                f"Data Not Provided: опционы на {cfg['label']} ({asset}) "
                f"недоступны через yfinance."
            )
        fetcher = CommodityFetcher(redis_client=self._redis)
        snapshot = fetcher.fetch_option_chain(asset, max_expiries=max_expiries)
        if snapshot is None:
            raise RuntimeError(
                f"Data Not Provided: не удалось загрузить опционную цепочку "
                f"ETF-прокси {cfg['etf_proxy']} для {cfg['label']} ({asset})."
            )
        return snapshot, cfg
