"""Crypto GEX analysis: polling via Bybit V5 (BTC/ETH/SOL/XRP/DOGE).

Standalone service (was CryptoServiceMixin). Created by GEXService via composition.
"""
from __future__ import annotations

from typing import Optional

from gex.assets_config import DEFAULT_ASSETS
from gex.adapters.fetchers.bybit_fetcher import BybitOptionsFetcher, _CRYPTO_ASSETS
from gex.domain.data_loader import OptionSnapshot
from gex.domain.pipeline import GEXPipeline
from gex.application.pipeline_runner import GEXPipelineRunner
from gex.adapters.cache.redis_client import RedisClient
from gex.adapters.persistence.repository import ChainRepository
from gex.schemas import GEXAnalysisOut, GEXProfileOut


class CryptoGEXService:
    """GEX analysis for cryptocurrencies via Bybit V5.

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
        self, coin: str, days: float = 30.0, max_expiries: int = 5,
    ) -> GEXAnalysisOut:
        snapshot = self._fetch(coin, max_expiries)
        cfg = _CRYPTO_ASSETS[snapshot.symbol]
        pipeline = GEXPipeline(
            spot=snapshot.spot, symbol=snapshot.symbol,
            r=cfg["r"], q=cfg["q"], per_contract=cfg["per_contract"],
            call_sign=cfg["call_sign"], put_sign=cfg["put_sign"],
        )
        self._repo.put(snapshot.symbol, snapshot)
        return self._runner.run_analysis(snapshot.symbol, snapshot, days, pipeline)

    def analyze_profile(
        self, coin: str, days: float = 30.0, max_expiries: int = 5,
    ) -> GEXProfileOut:
        snapshot = self._fetch(coin, max_expiries)
        cfg = _CRYPTO_ASSETS[snapshot.symbol]
        pipeline = GEXPipeline(
            spot=snapshot.spot, symbol=snapshot.symbol,
            r=cfg["r"], q=cfg["q"], per_contract=cfg["per_contract"],
            call_sign=cfg["call_sign"], put_sign=cfg["put_sign"],
        )
        self._repo.put(snapshot.symbol, snapshot)
        return self._runner.run_profile(snapshot.symbol, snapshot, days, pipeline)

    def _fetch(self, coin: str, max_expiries: int) -> OptionSnapshot:
        idx = coin.strip().upper()
        cfg = _CRYPTO_ASSETS.get(idx)
        if cfg is None:
            raise ValueError(
                f"Неподдерживаемая криптовалюта '{idx}'. "
                f"Доступны: {list(_CRYPTO_ASSETS)}."
            )
        fetcher = BybitOptionsFetcher(max_expiries=max_expiries, redis_client=self._redis)
        snapshot = fetcher.fetch(idx)
        DEFAULT_ASSETS[idx] = {"spot": snapshot.spot, "q": cfg["q"]}
        return snapshot
