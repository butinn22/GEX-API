"""GEX Cone: /gexcone/{ticker} — конус вероятностей цены на основе GEX-анализа.

Собирает опционную цепочку (yfinance для US-акций/ETF, Bybit для крипты),
выбирает наиболее объёмные страйки (топ-OI по каждой экспирации и глобально)
и строит конус вероятностей: границы 1σ/2σ/3σ по каждой экспирации +
ступеньки вероятностей на объёмных уровнях (см. :mod:`gex.gexcone`).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from gex.assets_config import DEFAULT_ASSETS, MOEX_ASSETS
from gex.application.cone import ComputeConeUseCase, ConeRequest
from gex.domain.gexcone import atr_14, build_gex_cone, historical_vol
from gex.adapters.cache.redis_client import deserialize_value, get_redis
from gex.adapters.cache.result_cache import cache_key, result_cache
from gex.adapters.cache.keys import PROVIDER_MOEX, PROVIDER_YFINANCE, hv_key
from gex.adapters.fetchers.yf_fetcher import YFOptionsFetcher, _YF_TICKER_MAP
from gex.deps import provide_gex_service

from ._helpers import handle

from fastapi import Depends
from gex.auth.dependencies import require_subscription
router = APIRouter(dependencies=[Depends(require_subscription("EXTENDED"))], tags=["gexcone"])

#: Криптовалюты с ликвидным опционным рынком (Bybit V5).
#: Маппинг крипты в yfinance-тикер для исторической волатильности.
_CRYPTO_HV_SYMBOL: dict[str, str] = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
    "XRP": "XRP-USD", "DOGE": "DOGE-USD",
}
#: TTL кэша исторической волатильности (HV меняется медленно).
_HV_TTL = 3600


def _vol_stats(ticker: str) -> tuple[float | None, float | None]:
    """(HV годовая, ATR14 в цене) с Redis-кэшем.

    US-тикеры → yfinance как есть (с ``_YF_TICKER_MAP``), крипта → ``*-USD``,
    MOEX-инструменты → дневные свечи ISS. При любой ошибке возвращает ``(None, None)`` —
    конус строится без этих фильтров/коррекций.

    Провайдер входит в ключ (итер. 25) и определяется **до** чтения кэша. Раньше ключ
    ``gex:hv:{T}`` строился до выбора источника, а в значение писалось либо ISS, либо
    yfinance — при одинаковой форме ``{"hv":…, "atr":…}`` подмена источника была
    неотличима. Тикер нормализуется: ``"rts"`` и ``"RTS"`` больше не дают два разных
    ключа для одних данных.
    """
    ticker_up = ticker.strip().upper()
    provider = PROVIDER_MOEX if ticker_up in MOEX_ASSETS else PROVIDER_YFINANCE
    key = hv_key(ticker_up, provider=provider)
    redis = get_redis()
    if redis is not None and redis.connected:
        try:
            raw = redis.get(key)
            if raw is not None:
                v = deserialize_value(raw)
                if isinstance(v, dict) and "hv" in v:
                    return (float(v["hv"]), float(v["atr"])) if v.get("atr") else (float(v["hv"]), None)
                if isinstance(v, (int, float)):
                    return float(v), None  # старый формат кэша
        except Exception:
            pass

    # MOEX-инструменты (фьючерсы RTS/MIX/CNY/Si и РФ-акции): HV/ATR из дневных
    # свечей ISS — yfinance для них не работает.
    if ticker.upper() in MOEX_ASSETS:
        from gex.adapters.fetchers.moex_candles_fetcher import MOEXCandlesFetcher

        hv = atr = None
        try:
            hist = MOEXCandlesFetcher().fetch_daily(ticker.upper())
            if hist is not None and not hist.empty:
                tail = hist.tail(90)
                hv = historical_vol(tail["Close"].astype(float).tolist())
                atr = atr_14(tail["High"].astype(float).tolist(),
                             tail["Low"].astype(float).tolist(),
                             tail["Close"].astype(float).tolist())
        except Exception:
            pass
        if redis is not None and redis.connected:
            try:
                redis.set(key, {"hv": hv, "atr": atr}, ex=_HV_TTL)
            except Exception:
                pass
        return hv, atr

    yf_symbol = _CRYPTO_HV_SYMBOL.get(ticker_up, _YF_TICKER_MAP.get(ticker_up, ticker_up))
    hv = atr = None
    try:
        # Через сервисный слой под дедлайном: прямой yf.Ticker(...).history не имеет
        # таймаута и при деградации провайдера держал бы запрос минутами.
        from gex.application import breadth_service

        hist = breadth_service.fetch_yf_history(yf_symbol, period="3mo", interval="1d", auto_adjust=False)
        if hist is not None and not hist.empty:
            hv = historical_vol(hist["Close"].tolist())
            atr = atr_14(hist["High"].tolist(), hist["Low"].tolist(), hist["Close"].tolist())
    except Exception:
        pass

    if redis is not None and redis.connected:
        try:
            redis.set(key, {"hv": hv, "atr": atr}, ex=_HV_TTL)
        except Exception:
            pass
    return hv, atr


@router.get("/gexcone/{ticker}")
def get_gexcone(
    ticker: str,
    expiries: int = Query(5, ge=1, le=15, description="Максимум экспираций в цепочке (как у главной страницы GEX)"),
    horizon_days: int = Query(14, ge=1, le=30, description="Горизонт прогноза, дней (1-2 недели) — он же days движка GEX"),
    wall_decay: float = Query(2.0, ge=0.0, le=20.0, description="Сила затухания вероятности за объёмными страйками (0 = выкл)"),
    top_oi: int = Query(4, ge=1, le=10, description="Сколько топ-OI страйков на экспирацию"),
    oi_quantile: float = Query(0.9, ge=0.5, le=1.0, description="Доля суммарного OI, покрываемая страйками (хвосты отбрасываются)"),
    svc=Depends(provide_gex_service),
) -> dict:
    """Рассчитать GEX-конус вероятностей для тикера с опционной цепочкой.

    Источник данных определяется по тикеру: BTC/ETH/SOL/XRP/DOGE → Bybit;
    RTS/MIX/CNY/SI и РФ-акции (см. MOEX_ASSETS: опционы FORTS на фьючерсы
    акций и еженедельные опционы на акции) → MOEX ISS; остальные → yfinance.

    GEX-ядро (стены, Gamma Flip, режим, Net GEX, Aggregate Gamma) считается
    ЕДИНСТВЕННЫМ движком (GEXPipelineRunner.run_gex_profile_domain) по той
    же цепочке и с теми же параметрами, что и главная страница GEX — конус
    лишь визуализирует канонический профиль (σ-границы, квантили, уровни,
    вероятности). Возвращает метаданные (спот, IV ATM, режим GEX),
    экспирации с границами σ/квантилями и **топ-OI уровнями каждой даты**,
    а также глобальные топ-OI уровни (для линий и таблицы вероятностей).
    """

    def _compute() -> dict:
        ticker_clean = ticker.strip().upper()

        def _fetch_snapshot(symbol: str, source, expiries: int):
            """Выбрать фетчер по **уже определённому** источнику.

            Источник решает use-case (``gex.application.cone``), а не фетчер: иначе
            «какой бирже принадлежит тикер» решалось бы дважды — здесь и в параметрах
            конуса, и расхождение было бы не видно.
            """
            if source.name == "crypto":
                from gex.adapters.fetchers.bybit_fetcher import BybitOptionsFetcher
                return BybitOptionsFetcher(
                    max_expiries=expiries, redis_client=get_redis()
                ).fetch(symbol)
            if source.name == "moex":
                from gex.adapters.fetchers.moex_fetcher import MOEXOptionsFetcher
                return MOEXOptionsFetcher(
                    max_expiries=expiries, redis_client=get_redis()
                ).fetch(symbol)
            return YFOptionsFetcher(
                max_expiries=expiries, redis_client=get_redis()
            ).fetch(symbol)

        use_case = ComputeConeUseCase(
            fetch_snapshot=_fetch_snapshot,
            vol_stats=lambda symbol: _vol_stats(symbol),
            profile_provider=lambda symbol, snapshot, source_name: svc.analyze_cone_engine(
                symbol, snapshot, source_name, days=horizon_days,
            ),
            build=build_gex_cone,
        )
        cone = use_case.execute(ConeRequest(
            ticker=ticker_clean,
            expiries=expiries,
            horizon_days=horizon_days,
            wall_decay=wall_decay,
            top_oi=top_oi,
            oi_quantile=oi_quantile,
        ))

        return {
            "ticker": cone.ticker,
            "spot": cone.spot,
            "as_of": cone.as_of,
            "r": cone.r,
            "q": cone.q,
            "iv_atm": cone.iv_atm,
            "wall_decay": cone.wall_decay,
            "regime": cone.regime,
            "net_gex": cone.net_gex,
            "total_ag": round(cone.total_ag, 2),
            "gamma_score": round(cone.gamma_score, 2),
            "vol_mult": round(cone.vol_mult, 4),
            "call_wall": cone.call_wall,
            "put_wall": cone.put_wall,
            "gamma_flip": cone.gamma_flip,
            "oi_quantile": cone.oi_quantile,
            "hv": cone.hv,
            "atr": cone.atr,
            "axis_min": cone.axis_min,
            "axis_max": cone.axis_max,
            "cone_path": cone.cone_path,
            "expirations": [_expiry_to_dict(e) for e in cone.expirations],
            "levels": [_level_to_dict(l) for l in cone.levels],
        }

    # EC-8: `expiries` влияет на результат (max_expiries уходит в build_gex_cone и в фетчер),
    # но раньше не входил в ключ кэша — запрос с другим числом экспираций 600с получал
    # конус, посчитанный для прежнего значения.
    # EC-8: `expiries` влияет на результат (max_expiries уходит и в калькулятор, и в фетчер),
    # но раньше не входил в ключ кэша — запрос с другим числом экспираций 600 с получал конус,
    # посчитанный для прежнего значения. Части ключа даёт сам запрос: один список параметров.
    _key = cache_key("res", "gexcone", *ConeRequest(
        ticker=ticker, expiries=expiries, horizon_days=horizon_days,
        wall_decay=wall_decay, top_oi=top_oi, oi_quantile=oi_quantile,
    ).cache_parts())
    return result_cache.get(_key, 600, lambda: handle(_compute, error_src="Bybit/yfinance"))


def _expiry_to_dict(e) -> dict:
    return {
        "date": e.date,
        "dte": e.dte,
        "iv_atm": round(e.iv_atm, 6),
        "vol_gex": round(e.vol_gex, 6),
        "gex_net": round(e.gex_net, 2),
        "ag": round(e.ag, 2),
        "gamma_score": round(e.gamma_score, 2),
        "ag_weight": round(e.ag_weight, 4),
        "regime": e.regime,
        "median": round(e.median, 4),
        "upper_1sd": round(e.upper_1sd, 4),
        "lower_1sd": round(e.lower_1sd, 4),
        "upper_2sd": round(e.upper_2sd, 4),
        "lower_2sd": round(e.lower_2sd, 4),
        "upper_3sd": round(e.upper_3sd, 4),
        "lower_3sd": round(e.lower_3sd, 4),
        "p10": round(e.p10, 4),
        "p25": round(e.p25, 4),
        "p75": round(e.p75, 4),
        "p90": round(e.p90, 4),
        "expected_move_1sd": round(e.expected_move_1sd, 4),
        "upper_cone": round(e.upper_cone, 4),
        "lower_cone": round(e.lower_cone, 4),
        "upper_stick": e.upper_stick,
        "lower_stick": e.lower_stick,
        "levels": [_expiry_level_to_dict(l) for l in e.levels],
    }


def _expiry_level_to_dict(l) -> dict:
    return {
        "strike": round(l.strike, 4),
        "oi": round(l.oi, 2),
        "side": l.side,
        "strength": round(l.strength, 4),
        "gex_net": round(l.gex_net, 2),
        "ag": round(l.ag, 2),
        "kind": l.kind,
        "probs": list(l.probs),
    }


def _level_to_dict(l) -> dict:
    return {
        "strike": round(l.strike, 4),
        "oi": round(l.oi, 2),
        "side": l.side,
        "strength": round(l.strength, 4),
        "kind": l.kind,
        "gex_net": round(l.gex_net, 2),
        "ag": round(l.ag, 2),
        "probs": list(l.probs),
    }
