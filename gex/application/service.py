"""GEX Service layer — фасад над GEX-пайплайном.

**Architecture (Clean Architecture / Composition over Inheritance):**

- ``GEXPipelineRunner`` — Application layer, единая точка построения GEXAnalysisOut
- ``LiveGEXService``, ``MOEXGEXService``, ``CryptoGEXService``,
  ``VolIndexGEXService``, ``CommodityGEXService`` — инфраструктурные адаптеры
  под источники данных (yfinance, Bybit, MOEX ISS, CBOE)
- ``GEXService`` — фасад: владеет репозиторием + runner'ом, делегирует субсервисам

Было (mixin-наследование, 5 родителей):
    class GEXService(LiveServiceMixin, MOEXServiceMixin, CryptoServiceMixin, ...)

Стало (композиция):
    class GEXService:
        self.live = LiveGEXService(repo, runner, redis)
        self.moex = MOEXGEXService(repo, runner, redis)
        ...

**Dependency rule:** Service → PipelineRunner → Pipeline (domain).
Fetchers — инфраструктура, внедряются в субсервисы.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from gex.assets_config import DEFAULT_ASSETS
from gex.application.commodity_service import CommodityGEXService
from gex.application.crypto_service import CryptoGEXService
from gex.application.gex_engine import GexEngineResult
from gex.domain.data_loader import GEXDataLoader, OptionSnapshot
from gex.application.live_service import LiveGEXService
from gex.domain.metrics import GEXProfile
from gex.application.moex_service import MOEXGEXService
from gex.domain.pipeline import GEXPipeline
from gex.application.pipeline_runner import GEXPipelineRunner, SupportsDirection, SupportsSummary
from gex.adapters.cache.redis_client import RedisClient, get_redis
from gex.adapters.persistence.repository import ChainRepository, InMemoryRepository
from gex.schemas import (
    GEXAnalysisOut,
    GEXKeyLevelsOut,
    GEXProfileOut,
    GEXSummaryOut,
    ResistanceLevelOut,
    SupportLevelOut,
    profile_to_schema,
)
from gex.domain.visualization import (
    levels_from_profile,
    regime_label,
    render_telegram_html,
    render_text_visualization,
)
from gex.application.vol_index_service import VolIndexGEXService


# ═══════════════════════════════════════════════════════════════════════
# Direction provider (implements SupportsDirection)
# ═══════════════════════════════════════════════════════════════════════

class _DirectionProvider:
    """Adapter: вызывает compute_direction из ``gex.direction``."""

    def compute(self, spot, profile, horizon_years, atm_vol, ticker=None):
        from gex.domain.direction import compute_direction

        result = compute_direction(
            spot=spot, profile=profile,
            horizon_years=horizon_years, atm_vol=atm_vol,
            ticker=ticker,
        )
        return result.p_up, result.p_down, result.direction, result.confidence


# ═══════════════════════════════════════════════════════════════════════
# Summary provider (implements SupportsSummary)
# ═══════════════════════════════════════════════════════════════════════

class _SummaryProvider:
    """Adapter: строит GEXSummaryOut из профиля."""

    def build(self, symbol: str, spot: float, profile: "GEXProfile") -> GEXSummaryOut:
        call_walls, put_walls, gamma_flip = levels_from_profile(spot, profile)
        if not call_walls:
            call_walls = [spot * 1.05]
        if not put_walls:
            put_walls = [spot * 0.95]

        key_levels = GEXKeyLevelsOut(
            resistance=ResistanceLevelOut(
                primary_call_wall=call_walls[0],
                secondary_call_walls=call_walls[1:],
            ),
            support=SupportLevelOut(
                primary_put_wall=put_walls[0],
                secondary_put_walls=put_walls[1:],
            ),
            gamma_flip=gamma_flip,
        )
        return GEXSummaryOut(
            symbol=symbol,
            spot=spot,
            regime=regime_label(profile.regime),
            key_levels=key_levels,
            text_visualization=render_text_visualization(
                spot, gamma_flip, call_walls, put_walls,
            ),
            telegram_html_message=render_telegram_html(
                symbol, spot, profile.regime, gamma_flip, call_walls, put_walls,
            ),
        )


# ═══════════════════════════════════════════════════════════════════════
# GEXService
# ═══════════════════════════════════════════════════════════════════════

class GEXService:
    """Фасад над GEX-пайплайном для HTTP/API-слоя.

    **Архитектура:** композиция субсервисов (не наследование mixin'ов).

    Public API (сохранён для обратной совместимости):
    - ``analyze(ticker, days)`` / ``analyze_profile(ticker, days)`` — статические цепочки
    - ``analyze_live(ticker, ...)`` → self.live.analyze()
    - ``analyze_crypto(coin, ...)`` → self.crypto.analyze()
    - ``analyze_moex(asset, ...)`` → self.moex.analyze()
    - ``analyze_vol_index(index, ...)`` → self.vol_index.analyze()
    - ``analyze_commodity(asset, ...)`` → self.commodity.analyze()
    - ``get_commodity_ohlcv(asset, ...)`` → self.commodity.get_ohlcv()
    - ``ingest_chain(ticker, snapshot)`` / ``list_tickers()`` / ``seed_defaults()``
    """

    def __init__(
        self,
        repo: Optional[ChainRepository] = None,
        r: float = 0.045,
        seed_on_init: bool = True,
        redis_client: Optional[RedisClient] = None,
    ):
        self.repo: ChainRepository = repo if repo is not None else InMemoryRepository()
        self.r = float(r)
        self._redis = redis_client or get_redis()

        # Core orchestrator — единая точка для построения GEXAnalysisOut
        self._runner = GEXPipelineRunner(
            direction_provider=_DirectionProvider(),
            summary_provider=_SummaryProvider(),
        )

        # ── Composition: sub-services (was mixin inheritance) ────────
        self.live = LiveGEXService(
            repo=self.repo, runner=self._runner, redis_client=self._redis,
        )
        self.moex = MOEXGEXService(
            repo=self.repo, runner=self._runner, redis_client=self._redis,
        )
        self.crypto = CryptoGEXService(
            repo=self.repo, runner=self._runner, redis_client=self._redis,
        )
        self.vol_index = VolIndexGEXService(
            repo=self.repo, runner=self._runner, redis_client=self._redis,
        )
        self.commodity = CommodityGEXService(
            repo=self.repo, runner=self._runner, redis_client=self._redis,
        )

        if seed_on_init:
            self.seed_defaults()

    # ── Инициализация демо-данными ────────────────────────────────────

    def seed_defaults(self) -> None:
        """Заполнить репозиторий синтетическими цепочками для тикеров по умолчанию."""
        rng_seed = 0
        for ticker, params in DEFAULT_ASSETS.items():
            if self.repo.get(ticker) is not None:
                continue
            loader = GEXDataLoader(spot=params["spot"], symbol=ticker)
            spot = params["spot"]
            strikes = np.arange(spot * 0.96, spot * 1.041, spot * 0.005)
            base_iv = 0.18 if ticker != "IWM" else 0.25

            all_frames = []
            last_as_of = None
            for days in (1, 7, 14, 21, 30):
                snap = loader.synthetic_chain(
                    strikes=strikes, expiry_years=days / 365.0,
                    atm_iv=base_iv + 0.01 * np.log1p(days),
                    skew=1.1, oi_seed=rng_seed,
                )
                all_frames.append(snap.chain)
                last_as_of = snap.as_of
                rng_seed += 1

            merged = pd.concat(all_frames, ignore_index=True)
            self.repo.put(ticker, OptionSnapshot(
                symbol=ticker, spot=spot, as_of=last_as_of, chain=merged,
            ))

    # ── CRUD ──────────────────────────────────────────────────────────

    def ingest_chain(self, ticker: str, snapshot: OptionSnapshot) -> None:
        self.repo.put(ticker, snapshot)

    def list_tickers(self) -> list[str]:
        return self.repo.list_tickers()

    # ── Core Use Cases (статическая цепочка из репозитория) ──────────

    def analyze(self, ticker: str, days: float = 30.0) -> GEXAnalysisOut:
        ticker = ticker.upper()
        snapshot = self.repo.get(ticker)
        if snapshot is None:
            raise KeyError(f"Тикер '{ticker}' не найден в репозитории.")
        pipeline = GEXPipeline(
            spot=snapshot.spot, symbol=ticker, r=self.r,
            q=DEFAULT_ASSETS.get(ticker, {}).get("q", 0.0),
        )
        return self._runner.run_analysis(ticker, snapshot, days, pipeline)

    def analyze_profile(self, ticker: str, days: float = 30.0) -> GEXProfileOut:
        ticker = ticker.upper()
        snapshot = self.repo.get(ticker)
        if snapshot is None:
            raise KeyError(f"Тикер '{ticker}' не найден в репозитории.")
        pipeline = GEXPipeline(
            spot=snapshot.spot, symbol=ticker, r=self.r,
            q=DEFAULT_ASSETS.get(ticker, {}).get("q", 0.0),
        )
        return self._runner.run_profile(ticker, snapshot, days, pipeline)

    # ── Delegation to sub-services (composition, was mixin) ───────────

    def analyze_live(self, ticker: str, **kwargs) -> GEXAnalysisOut:
        return self.live.analyze(ticker, **kwargs)

    def analyze_profile_live(self, ticker: str, **kwargs) -> GEXProfileOut:
        return self.live.analyze_profile(ticker, **kwargs)

    # ── Канонический движок для GEX-конуса ────────────────────────────
    # Конус обязан считать GEX-метрики (стены, Gamma Flip, режим, Net GEX,
    # AG) тем же движком и по той же цепочке, что и главная страница GEX —
    # поэтому здесь нет собственного вызова GEXMetrics, только параметры
    # источника (build_engine_params) и единый прогон (run_gex_profile_domain).

    def analyze_cone_engine(
        self,
        symbol: str,
        snapshot: OptionSnapshot,
        source_name: str,
        days: float,
    ) -> GexEngineResult:
        """Прогнать канонический GEX-движок для конуса.

        ``snapshot`` — свежая цепочка (уже выбранная фетчером конуса);
        ``days`` — горизонт (дней), как параметр ``days`` главной страницы.
        Возвращает доменный профиль + отфильтрованную по days цепочку и
        адаптированный per-strike кадр для уровней/агрегатов конуса.
        """
        from gex.application.gex_engine import (
            build_engine_params,
            stk_all_from_profile,
        )

        params = build_engine_params(symbol, snapshot.spot, source_name)
        engine = self._runner.run_gex_profile_domain(snapshot, days, params.pipeline)
        return GexEngineResult(
            profile=engine.profile,
            snapshot=engine.snapshot,
            r=params.r,
            q=params.q,
            horizon_years=engine.horizon_years,
            atm_vol=engine.atm_vol,
            stk_all=stk_all_from_profile(engine.profile),
        )

    def analyze_crypto(self, coin: str, **kwargs) -> GEXAnalysisOut:
        return self.crypto.analyze(coin, **kwargs)

    def analyze_crypto_profile(self, coin: str, **kwargs) -> GEXProfileOut:
        return self.crypto.analyze_profile(coin, **kwargs)

    def analyze_moex(self, asset: str, **kwargs) -> GEXAnalysisOut:
        return self.moex.analyze(asset, **kwargs)

    def analyze_moex_profile(self, asset: str, **kwargs) -> GEXProfileOut:
        return self.moex.analyze_profile(asset, **kwargs)

    def analyze_vol_index(self, index: str, **kwargs) -> GEXAnalysisOut:
        return self.vol_index.analyze(index, **kwargs)

    def analyze_vol_index_profile(self, index: str, **kwargs) -> GEXProfileOut:
        return self.vol_index.analyze_profile(index, **kwargs)

    def analyze_commodity(self, asset: str, **kwargs) -> GEXAnalysisOut:
        return self.commodity.analyze(asset, **kwargs)

    def analyze_commodity_profile(self, asset: str, **kwargs) -> GEXProfileOut:
        return self.commodity.analyze_profile(asset, **kwargs)

    def get_commodity_ohlcv(self, asset: str, **kwargs) -> dict:
        return self.commodity.get_ohlcv(asset, **kwargs)
