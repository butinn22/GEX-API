"""MOEX GEX analysis: polling via MOEX ISS (RTS/MIX/CNY/Si).

Standalone service (was MOEXServiceMixin). Created by GEXService via composition.
"""
from __future__ import annotations

import time
from typing import Optional

from gex.assets_config import DEFAULT_ASSETS, MOEX_ASSETS
from gex.domain.data_loader import OptionSnapshot
from gex.adapters.fetchers.moex_fetcher import MOEXOptionsFetcher
from gex.domain.pipeline import GEXPipeline
from gex.application.pipeline_runner import GEXPipelineRunner
from gex.adapters.cache.redis_client import RedisClient
from gex.adapters.persistence.repository import ChainRepository
from gex.schemas import GEXAnalysisOut, GEXProfileOut
from gex.schemas.auto_coverage import AutoCoverageOut
from gex.application.auto_scope import (
    StrikeLite,
    count_expiries,
    detect_sparse,
)

#: Provider marker used in MOEX coverage metadata (no second provider → no fallback).
MOEX_PRIMARY_SOURCE = "moex_iss"


class MOEXGEXService:
    """GEX analysis for MOEX futures options (RTS/MIX/CNY/Si).

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
        self, asset: str, days: float = 30.0, max_expiries: int = 0,
        auto: bool = False,
    ) -> GEXAnalysisOut:
        t0 = time.perf_counter()
        snapshot, cfg = self._fetch(asset, max_expiries)
        self._repo.put(asset, snapshot)
        pipeline = GEXPipeline(
            spot=snapshot.spot, symbol=asset, r=cfg["r"], q=cfg["r"],
            per_contract=cfg["per_contract"],
        )
        out = self._runner.run_analysis(asset, snapshot, days, pipeline)
        # Аудит 2026-09-17: охват считается и в ручном режиме (``auto=False``),
        # иначе качество выборки по российским тикерам видно только при AUTO.
        # Сам ISS-фетчер не меняется — меняется только измерение результата.
        out.auto = self._auto_coverage(
            snapshot, out, days=days, max_expiries=max_expiries,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            auto=auto,
        )
        return out

    def _auto_coverage(
        self,
        snapshot: OptionSnapshot,
        out: GEXAnalysisOut,
        days: float,
        max_expiries: int,
        elapsed_ms: int,
        auto: bool = True,
    ) -> AutoCoverageOut:
        """Измерить охват MOEX (максимальный охват, без эскалации).

        ISS — единственный источник, поэтому ``escalated``/``fallback_used`` всегда
        ``False``; метаданные описывают фактически собранный профиль (страйки, OI,
        число экспираций, разреженность). ``resolved_expiries=0`` означает «все
        экспирации» на стороне MOEX API (design decision Q2).

        ``auto=False`` → ``mode="manual"``, ``elapsed_ms`` не имеет смысла как
        «время AUTO» и кладётся как есть (совместимо: поле и раньше было
        необязательным для чтения).
        """
        profile = getattr(out, "profile", None)
        per_strike = getattr(profile, "per_strike", None) or []
        strikes = [
            StrikeLite(
                strike=float(getattr(s, "strike", 0.0)),
                oi_call=float(getattr(s, "oi_call", 0.0) or 0.0),
                oi_put=float(getattr(s, "oi_put", 0.0) or 0.0),
                gex_net=float(getattr(s, "gex_net", 0.0) or 0.0),
            )
            for s in per_strike
        ]
        call_wall = getattr(profile, "call_wall", None)
        put_wall = getattr(profile, "put_wall", None)
        reasons = detect_sparse(strikes, float(getattr(out, "spot", 0.0) or 0.0),
                                call_wall, put_wall,
                                expiries=count_expiries(getattr(snapshot, "chain", None)))
        total_oi = float(sum(s.oi_call + s.oi_put for s in strikes))
        # Additive Phase-4 поля: диапазон страйков и ближайшая экспирация.
        chain = getattr(snapshot, "chain", None)
        if chain is not None and len(chain) > 0:
            strike_min = float(chain["strike"].min())
            strike_max = float(chain["strike"].max())
            tdays = chain["T"].astype(float) * 365.0
            nearest_expiry_days = float(tdays.min())
            furthest_expiry_days = float(tdays.max())
        else:
            strike_min = strike_max = nearest_expiry_days = furthest_expiry_days = None
        return AutoCoverageOut(
            mode="auto" if auto else "manual",
            resolved_days=float(days),
            resolved_expiries=int(max_expiries),
            sources_used=[MOEX_PRIMARY_SOURCE],
            primary_source=MOEX_PRIMARY_SOURCE,
            fallback_used=False,
            escalated=False,
            partial=False,
            expirations_merged=count_expiries(chain),
            strike_count=len(strikes),
            total_oi=total_oi,
            sparse=bool(reasons),
            sparse_reasons=list(reasons),
            elapsed_ms=int(elapsed_ms),
            strike_min=strike_min,
            strike_max=strike_max,
            nearest_expiry_days=nearest_expiry_days,
            furthest_expiry_days=furthest_expiry_days,
        )

    def analyze_profile(
        self, asset: str, days: float = 30.0, max_expiries: int = 0,
    ) -> GEXProfileOut:
        snapshot, cfg = self._fetch(asset, max_expiries)
        self._repo.put(asset, snapshot)
        pipeline = GEXPipeline(
            spot=snapshot.spot, symbol=asset, r=cfg["r"], q=cfg["r"],
            per_contract=cfg["per_contract"],
        )
        return self._runner.run_profile(asset, snapshot, days, pipeline)

    def _fetch(self, asset: str, max_expiries: int):
        asset = asset.upper()
        cfg = MOEX_ASSETS.get(asset)
        if cfg is None:
            raise ValueError(
                f"Неподдерживаемый MOEX-актив '{asset}'. "
                f"Доступны: {list(MOEX_ASSETS)}."
            )
        fetcher = MOEXOptionsFetcher(max_expiries=max_expiries, redis_client=self._redis)
        snapshot = fetcher.fetch(asset)
        DEFAULT_ASSETS[asset] = {"spot": snapshot.spot, "q": cfg["r"]}
        return snapshot, cfg
