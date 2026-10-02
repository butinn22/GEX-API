"""Vol-Index GEX analysis: polling via yfinance (VIX/VVIX).

Standalone service (was VolIndexServiceMixin). Created by GEXService via composition.
"""
from __future__ import annotations

from typing import Optional

from gex.assets_config import DEFAULT_ASSETS, VOL_INDEX_ASSETS
from gex.domain.data_loader import OptionSnapshot
from gex.domain.pipeline import GEXPipeline
from gex.application.pipeline_runner import GEXPipelineRunner
from gex.adapters.cache.redis_client import RedisClient
from gex.adapters.persistence.repository import ChainRepository
from gex.schemas import GEXAnalysisOut, GEXProfileOut
from gex.adapters.fetchers.yf_fetcher import YFOptionsFetcher


class VolIndexGEXService:
    """GEX analysis for volatility indices (VIX/VVIX).

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
        self, index: str, days: float = 30.0, max_expiries: int = 5,
    ) -> GEXAnalysisOut:
        snapshot, cfg = self._fetch(index, max_expiries)
        pipeline = GEXPipeline(
            spot=snapshot.spot, symbol=index.upper(),
            r=cfg["r"], q=cfg["q"], per_contract=cfg["per_contract"],
            call_sign=cfg["call_sign"], put_sign=cfg["put_sign"],
        )
        self._repo.put(index.upper(), snapshot)
        return self._runner.run_analysis(index.upper(), snapshot, days, pipeline)

    def analyze_profile(
        self, index: str, days: float = 30.0, max_expiries: int = 5,
    ) -> GEXProfileOut:
        snapshot, cfg = self._fetch(index, max_expiries)
        pipeline = GEXPipeline(
            spot=snapshot.spot, symbol=index.upper(),
            r=cfg["r"], q=cfg["q"], per_contract=cfg["per_contract"],
            call_sign=cfg["call_sign"], put_sign=cfg["put_sign"],
        )
        self._repo.put(index.upper(), snapshot)
        return self._runner.run_profile(index.upper(), snapshot, days, pipeline)

    def _fetch(self, index: str, max_expiries: int):
        idx = index.strip().upper()
        cfg = VOL_INDEX_ASSETS.get(idx)
        if cfg is None:
            raise ValueError(
                f"Неподдерживаемый индекс волатильности '{idx}'. "
                f"Доступны: {list(VOL_INDEX_ASSETS)}."
            )
        if not cfg["has_chain"]:
            raise ValueError(
                f"Цепочка опционов для '{idx}' недоступна через yfinance "
                f"(только spot). {cfg.get('note', '')}."
            )
        fetcher = YFOptionsFetcher(max_expiries=max_expiries, redis_client=self._redis)
        snapshot = fetcher.fetch(cfg["yf_ticker"])
        DEFAULT_ASSETS[idx] = {"spot": snapshot.spot, "q": cfg["q"]}
        return snapshot, cfg
