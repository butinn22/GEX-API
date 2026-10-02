"""Единый GEX-движок для главной страницы GEX и GEX-конуса (ring: application).

До этого GEX-конус считал профиль (стены, Gamma Flip, режим, Net GEX, AG)
собственным вызовом ``GEXMetrics`` по предварительно отфильтрованной цепочке,
а главная страница — через :class:`GEXPipelineRunner` по полной. Из-за этого
одинаковые метрики двух страниц расходились, а любое изменение движка
приходилось дублировать в конусе.

Теперь параметры пайплайна конуса строятся ТОЙ ЖЕ функцией
:func:`build_engine_params`, что зеркалит конструкторы live/crypto/moex
сервисов, а сам профиль считается ЕДИНСТВЕННЫМ прогоном
:meth:`GEXPipelineRunner.run_gex_profile_domain`. Конус получает доменный
``GEXProfile`` (вместе с per-strike таблицей) и только визуализирует его.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gex.assets_config import CRYPTO_ASSETS, DEFAULT_ASSETS, MOEX_ASSETS
from gex.domain.data_loader import OptionSnapshot
from gex.domain.pipeline import GEXPipeline

#: Дефолтная безрисковая ставка (совпадает с историческим значением live-сервиса).
DEFAULT_RATE = 0.045


@dataclass(frozen=True)
class GexEngineParams:
    """Параметры канонического пайплайна — как у главной страницы GEX."""

    pipeline: GEXPipeline
    r: float
    q: float
    per_contract: int
    call_sign: float
    put_sign: float


def build_engine_params(symbol: str, spot: float, source_name: str) -> GexEngineParams:
    """Построить GEXPipeline ровно с теми же параметрами, что у главной
    страницы GEX для данного источника.

    Зеркалит конструкторы сервисов (не копирует, а воспроизводит тот же
    контракт параметров):
      * ``equity``   → как :class:`LiveGEXService` (DEFAULT_ASSETS, q=0);
      * ``crypto``   → как :class:`CryptoGEXService` (CRYPTO_ASSETS,
        per_contract=1, знаки дилера из справочника);
      * ``moex``     → как :class:`MOEXGEXService` (модель Блэка, q=r);
      * ``futures``  → q=r (форвард без дивидендного сноса).
    """
    symbol = symbol.strip().upper()

    if source_name == "crypto":
        cfg = CRYPTO_ASSETS[symbol]
        call_sign = float(cfg.get("call_sign", 1.0))
        put_sign = float(cfg.get("put_sign", -1.0))
        pipeline = GEXPipeline(
            spot=spot, symbol=symbol,
            r=float(cfg["r"]), q=float(cfg["q"]),
            per_contract=int(cfg["per_contract"]),
            call_sign=call_sign, put_sign=put_sign,
        )
        return GexEngineParams(
            pipeline=pipeline,
            r=float(cfg["r"]), q=float(cfg["q"]),
            per_contract=int(cfg["per_contract"]),
            call_sign=call_sign, put_sign=put_sign,
        )

    if source_name == "moex":
        cfg = MOEX_ASSETS[symbol]
        pipeline = GEXPipeline(
            spot=spot, symbol=symbol,
            r=float(cfg["r"]), q=float(cfg["r"]),
            per_contract=int(cfg["per_contract"]),
        )
        return GexEngineParams(
            pipeline=pipeline,
            r=float(cfg["r"]), q=float(cfg["r"]),
            per_contract=int(cfg["per_contract"]),
            call_sign=1.0, put_sign=-1.0,
        )

    cfg = DEFAULT_ASSETS.get(symbol, {})
    if source_name == "futures":
        r = float(cfg.get("r", DEFAULT_RATE))
        q = float(cfg.get("q", DEFAULT_RATE))
        pipeline = GEXPipeline(spot=spot, symbol=symbol, r=r, q=q)
        return GexEngineParams(pipeline=pipeline, r=r, q=q, per_contract=100, call_sign=1.0, put_sign=-1.0)

    # equity (live): как LiveGEXService
    r = float(DEFAULT_ASSETS.get(symbol, {}).get("r", DEFAULT_RATE))
    q = float(DEFAULT_ASSETS.get(symbol, {}).get("q", 0.0))
    pipeline = GEXPipeline(spot=spot, symbol=symbol, r=r, q=q)
    return GexEngineParams(pipeline=pipeline, r=r, q=q, per_contract=100, call_sign=1.0, put_sign=-1.0)


@dataclass
class GexEngineResult:
    """Канонический профиль + цепочка, на которой он посчитан (после days-фильтра)."""

    profile: object                      # gex.domain.metrics.GEXProfile
    snapshot: OptionSnapshot             # цепочка после _filter_by_days — как у главной страницы
    r: float
    q: float
    horizon_years: float
    atm_vol: float
    stk_all: pd.DataFrame                # strike / gex_net / ag — адаптер из profile.per_strike


def stk_all_from_profile(profile) -> pd.DataFrame:
    """Адаптер канонического per-strike профиля в формат конуса.

    Конус исторически работал с колонками ``strike``/``gex_net``/``ag``
    (см. ``_strike_gex``). Из канонического профиля те же числа получаются
    без пересчёта: ``ag = |gex_call| + |gex_put|`` — знаки внутри типа
    постоянны, поэтому сумма модулей равна модулю суммы.
    """
    ps = profile.per_strike
    return pd.DataFrame({
        "strike": ps["strike"].astype(float).values,
        "gex_net": ps["gex_net"].astype(float).values,
        "ag": ps["gex_call"].abs().add(ps["gex_put"].abs()).astype(float).values,
    })
