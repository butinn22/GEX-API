"""GEX Pipeline Orchestrator — единая точка входа для всех GEX-анализов.

Устраняет дублирование логики в 4 mixin'ах (live, crypto, moex, vol_index).
Все они повторяли один и тот же паттерн:

    fetch → filter → auto_params → pipeline.run → direction → GEXAnalysisOut

Теперь это делает один класс :class:`GEXPipelineRunner`.

Архитектурный принцип: **Single Source of Truth для GEX-пайплайна.**
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from gex.domain.data_loader import OptionSnapshot
from gex.domain.pipeline import GEXPipeline, GEXReport
from gex.schemas import GEXAnalysisOut, GEXProfileOut, GEXSummaryOut, profile_to_schema


@dataclass
class GexProfileDomainResult:
    """Результат канонического прогона движка: доменный профиль + контекст.

    ``snapshot`` — цепочка ПОСЛЕ фильтра по days (та же, что у главной
    страницы GEX); ``horizon_years``/``atm_vol`` — авто-параметры прогона.
    """

    profile: object  # gex.domain.metrics.GEXProfile
    snapshot: OptionSnapshot
    horizon_years: float
    atm_vol: float


# ═══════════════════════════════════════════════════════════════════════
# Protocol interfaces — заменяют неявный контракт mixin'ов
# ═══════════════════════════════════════════════════════════════════════

class SupportsDirection(Protocol):
    """Контракт: объект умеет определять направление рынка."""

    def compute(
        self,
        spot: float,
        profile: object,
        horizon_years: float,
        atm_vol: float,
        ticker: Optional[str] = None,
    ) -> tuple[float, float, str, float]:
        """Вернуть (p_up, p_down, direction, confidence)."""
        ...


class SupportsSummary(Protocol):
    """Контракт: объект умеет строить summarize-блок."""

    def build(
        self,
        symbol: str,
        spot: float,
        profile: object,
    ) -> GEXSummaryOut:
        """Вернуть GEXSummaryOut."""
        ...


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════

class GEXPipelineRunner:
    """Оркестратор GEX-пайплайна: filter → params → run → direction → output.

    Используется всеми сервисными mixin'ами вместо дублирования ~30 строк
    идентичной логики построения ``GEXAnalysisOut``.

    Parameters
    ----------
    direction_provider : SupportsDirection
        Объект с методом ``compute(...) → (p_up, p_down, direction, confidence)``.
    summary_provider : SupportsSummary
        Объект с методом ``build(symbol, spot, profile) → GEXSummaryOut``.
    """

    def __init__(
        self,
        direction_provider: SupportsDirection,
        summary_provider: SupportsSummary,
    ):
        self._direction = direction_provider
        self._summary = summary_provider

    # ── Полный анализ ─────────────────────────────────────────────────

    def run_gex_profile_domain(
        self,
        snapshot: OptionSnapshot,
        days: float,
        pipeline: GEXPipeline,
    ) -> "GexProfileDomainResult":
        """Канонический прогон GEX-пайплайна: filter → params → pipeline.run.

        Возвращает доменный ``GEXProfile`` (с per_strike DataFrame), а не
        Pydantic-схему — это внутренняя точка, которой пользуются и
        ``run_analysis``/``run_profile``, и GEX-конус (конус обязан считать
        GEX-метрики ровно тем же движком, что и главная страница GEX).

        Returns
        -------
        GexProfileDomainResult
            ``profile`` (доменный), ``snapshot`` (после фильтра по days),
            ``horizon_years``, ``atm_vol``.
        """
        filtered = _filter_by_days(snapshot, days)
        params = _auto_params(filtered, days)

        report: GEXReport = pipeline.run(
            snapshot=filtered,
            sigma=params.atm_vol,
            T=params.horizon_years,
            mu=0.0,
            run_put_wall_setup=False,
            verbose=False,
        )
        return GexProfileDomainResult(
            profile=report.profile,
            snapshot=filtered,
            horizon_years=params.horizon_years,
            atm_vol=params.atm_vol,
        )

    def run_analysis(
        self,
        ticker: str,
        snapshot: OptionSnapshot,
        days: float,
        pipeline: GEXPipeline,
    ) -> GEXAnalysisOut:
        """Полный цикл: filter → params → pipeline.run → direction → output.

        Это ЕДИНСТВЕННОЕ место в кодовой базе, где строится ``GEXAnalysisOut``
        из ``OptionSnapshot``. Все 4 mixin'а (live, crypto, moex, vol_index)
        делегируют сюда.
        """
        spot = snapshot.spot

        # 1. Filter + auto-params + GEX pipeline (единый прогон движка)
        engine = self.run_gex_profile_domain(snapshot, days, pipeline)
        profile = engine.profile

        # 2. Direction
        p_up, p_down, direction, confidence = self._direction.compute(
            spot, profile, engine.horizon_years, engine.atm_vol, ticker=ticker,
        )

        # 3. Levels
        support = float(profile.put_wall) if np.isfinite(profile.put_wall) else spot * 0.95
        resistance = float(profile.call_wall) if np.isfinite(profile.call_wall) else spot * 1.05

        # 4. Output
        return GEXAnalysisOut(
            symbol=ticker,
            spot=spot,
            days=days,
            direction=direction,
            confidence=confidence,
            p_up=p_up,
            p_down=p_down,
            support=support,
            resistance=resistance,
            gamma_flip=profile.gamma_flip,
            regime=profile.regime,
            profile=profile_to_schema(profile),
            summarize=self._summary.build(ticker, spot, profile),
        )

    # ── Только профиль ────────────────────────────────────────────────

    def run_profile(
        self,
        ticker: str,
        snapshot: OptionSnapshot,
        days: float,
        pipeline: GEXPipeline,
    ) -> GEXProfileOut:
        """Только GEX-профиль: filter → params → pipeline.run → profile_to_schema."""
        engine = self.run_gex_profile_domain(snapshot, days, pipeline)
        return profile_to_schema(engine.profile)


# ═══════════════════════════════════════════════════════════════════════
# Вспомогательные функции (были staticmethod'ами GEXService)
# ═══════════════════════════════════════════════════════════════════════

def _filter_by_days(snapshot: OptionSnapshot, days: float) -> OptionSnapshot:
    """Оставить опционы с T ≤ days (в днях → годах)."""
    from dataclasses import replace

    max_T = days / 365.0
    mask = snapshot.chain["T"] <= max_T
    if mask.any():
        return replace(snapshot, chain=snapshot.chain[mask].reset_index(drop=True))
    return snapshot


def _auto_params(snapshot: OptionSnapshot, days: float):
    """Вычислить горизонт и ATM-волу автоматически.

    Returns
    -------
    _AutoParams
        dataclass с полями ``horizon_years``, ``atm_vol``.
    """
    from dataclasses import dataclass

    @dataclass
    class _AutoParams:
        horizon_years: float
        atm_vol: float

    chain = snapshot.chain
    T_requested = days / 365.0
    T_max = float(chain["T"].max())
    horizon = min(T_requested, T_max)
    horizon = max(horizon, 1e-6)

    w = chain["oi"].values
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    atm_vol = float(np.sum(chain["iv"].values * w))
    return _AutoParams(horizon_years=horizon, atm_vol=atm_vol)
