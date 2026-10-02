"""Фоновый фетчер: consumer-обработчик задач Redis TaskQueue.

Получает ``FetchTask`` из очереди (Redis Streams за портом) и выполняет соответствующий
fetch с rate limiting'ом. Результаты сохраняются в Redis (через существующий
``gex.redis_client.RedisClient``) для мгновенного доступа из HTTP-ручек.

Watchlist инструментов для превентивного фетчинга:
  * US акции/ETF: SPY, QQQ, IWM, DIA, AAPL, NVDA, MSFT, GOOGL, META, AMZN, AVGO, SMH, RSP
  * Крипта: BTC, ETH, SOL, XRP, DOGE
  * MOEX: RTS, MIX, CNY, SI
  * Vol индексы: VIX, VVIX, MOVE, COR1M
"""
from __future__ import annotations

from gex.assets_config import CRYPTO_ASSETS

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from gex.adapters.ratelimit.rate_limiter import get_rate_limiter
from gex.adapters.providers.catalog import CANONICAL_TIMEFRAMES
from gex.adapters.cache.redis_client import get_redis
from gex.application.jobs import FetchTask

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Watchlist
# ====================================================================== #
US_TICKERS: list[str] = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "NVDA", "MSFT", "GOOGL", "META", "AMZN", "AVGO", "SMH", "RSP",
    "MAGS", "TSLA",
    "ES", "NQ",  # E-mini futures via SPX/NDX index options proxy
]

# Вселенная крипты выводится из таблицы активов: список из пяти монет был ещё одной
# копией, и добавление монеты в конфиг не попадало в прогрев (и наоборот).
CRYPTO_TICKERS: list[str] = sorted(CRYPTO_ASSETS)

MOEX_TICKERS: list[str] = ["RTS", "MIX", "CNY", "SI"]

VOL_TICKERS: list[str] = ["VIX", "VVIX", "MOVE"]  # COR1M — через CBOE CDN

# Инструменты только для OHLCV (без опционных цепочек)
OHLCV_ONLY_TICKERS: list[str] = ["DXY"]

COMMODITY_TICKERS_BG: list[str] = ["UKOIL", "GOLD", "SILVER", "NATGAS", "COPPER", "PALLAD", "PLAT", "URANIUM", "NICKEL"]

ALL_TICKERS: list[str] = US_TICKERS + CRYPTO_TICKERS + MOEX_TICKERS + VOL_TICKERS + COMMODITY_TICKERS_BG

OHLCV_TIMEFRAMES: list[str] = list(CANONICAL_TIMEFRAMES)

# Сопоставление провайдера ↔ тикеры для превентивного фетчинга задаётся ТОЛЬКО в
# `build_prewarm_tasks()` ниже: прежний словарь PREWARM_MAP никем не использовался и
# расходился с реальным набором задач (крипта: `bybit` для цепочек и `yfinance` для OHLCV) —
# единый справочник провайдеров собирается в рамках итерации 29 (P4).


# ====================================================================== #
#  Consumer handler
# ====================================================================== #
# ====================================================================== #
#  Счётчики выполнения (для админ-панели, хранятся в Redis)
# ====================================================================== #
_STATS_KEYS = {
    "total": "gex:stats:fetch:total",
    "ok": "gex:stats:fetch:ok",
    "err": "gex:stats:fetch:err",
}


def _bump_stats(task_type: str, provider: str, ok: bool) -> None:
    """Инкремент счётчиков фетчинга в Redis (best-effort)."""
    try:
        redis = get_redis()
        if redis is None or not redis.connected:
            return
        conn = redis._conn
        pipe = conn.pipeline()
        pipe.incr(_STATS_KEYS["total"])
        pipe.incr(_STATS_KEYS["ok"] if ok else _STATS_KEYS["err"])
        pipe.incr(f"gex:stats:fetch:by_type:{task_type}")
        pipe.incr(f"gex:stats:fetch:by_provider:{provider}")
        pipe.set(f"gex:stats:fetch:last:{task_type}", int(time.time()))
        pipe.expire(_STATS_KEYS["total"], 30 * 24 * 3600)  # 30 дней
        pipe.execute()
    except Exception:  # noqa: BLE001
        pass


def get_fetch_stats() -> dict:
    """Собрать статистику фетчера из Redis-счётчиков."""
    import time as _time

    stats: dict = {
        "total": 0, "ok": 0, "err": 0,
        "by_type": {}, "by_provider": {},
        "last_activity": None,
        "recent": [],
    }
    try:
        redis = get_redis()
        if redis is None or not redis.connected:
            return stats
        conn = redis._conn
        for k, name in ((_STATS_KEYS["total"], "total"), (_STATS_KEYS["ok"], "ok"), (_STATS_KEYS["err"], "err")):
            v = conn.get(k)
            stats[name] = int(v) if v else 0
        for t in ("ohlcv", "chain", "vol", "gex_profile", "sector", "breadth", "breadth_imoex", "composite"):
            v = conn.get(f"gex:stats:fetch:by_type:{t}")
            if v:
                stats["by_type"][t] = int(v)
        for p in ("yfinance", "bybit", "moex_iss", "webull"):
            v = conn.get(f"gex:stats:fetch:by_provider:{p}")
            if v:
                stats["by_provider"][p] = int(v)
        # Последняя активность: max(last:{type}) по всем типам
        last_ts = 0
        for t in ("ohlcv", "chain", "vol", "gex_profile", "sector", "breadth", "breadth_imoex", "composite"):
            v = conn.get(f"gex:stats:fetch:last:{t}")
            if v:
                last_ts = max(last_ts, int(v))
        if last_ts:
            stats["last_activity"] = datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat()
        # Последние выполненные (по логам Redis-списка, если ведём) — опционально
        recent = conn.lrange("gex:stats:fetch:recent", 0, 14)
        for raw in reversed(recent or []):
            try:
                stats["recent"].append(json.loads(raw))
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_fetch_stats failed: %s", exc)
    return stats


def _push_recent(task_type: str, provider: str, ticker: str, ok: bool) -> None:
    """Сохранить последние выполненные задачи (кольцевой список в Redis)."""
    try:
        redis = get_redis()
        if redis is None or not redis.connected:
            return
        import time as _time
        item = json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "task_type": task_type,
            "provider": provider,
            "ticker": ticker,
            "ok": ok,
        })
        conn = redis._conn
        pipe = conn.pipeline()
        pipe.lpush("gex:stats:fetch:recent", item)
        pipe.ltrim("gex:stats:fetch:recent", 0, 19)
        pipe.expire("gex:stats:fetch:recent", 30 * 24 * 3600)
        pipe.execute()
    except Exception:  # noqa: BLE001
        pass


def handle_fetch_task(task: Any) -> None:
    """Обработчик задачи фетчинга для Redis TaskQueue consumer.

    Вызывается Redis TaskQueue consumer для каждого сообщения из очереди.
    Dispatch по ``task.task_type`` → соответствующий fetcher.

    Parameters
    ----------
    task : FetchTask
        Задача от scheduler.
    """
    if not isinstance(task, FetchTask):
        logger.warning("Invalid task type: %s", type(task).__name__)
        return

    task_type = task.task_type
    provider = task.provider
    ticker = task.ticker

    logger.info("Handling fetch: %s/%s/%s", task_type, provider, ticker)

    try:
        if task_type == "ohlcv":
            _fetch_ohlcv(ticker, task.params)
        elif task_type == "chain":
            _fetch_chain(ticker, provider, task.params)
        elif task_type == "vol":
            _fetch_vol(ticker)
        elif task_type == "sector":
            _fetch_sector()
        elif task_type == "breadth":
            _fetch_breadth()
        elif task_type == "breadth_imoex":
            _fetch_breadth_imoex()
        elif task_type == "composite":
            _fetch_composite()
        elif task_type == "gex_profile":
            _fetch_gex_profile(ticker, task.params)
        else:
            logger.warning("Unknown task_type: %s", task_type)
            return
        _bump_stats(task_type, provider, ok=True)
        _push_recent(task_type, provider, ticker, ok=True)
    except Exception as exc:
        logger.error("Fetch error %s/%s/%s: %s", task_type, provider, ticker, exc)
        _bump_stats(task_type, provider, ok=False)
        _push_recent(task_type, provider, ticker, ok=False)
        if task_type == "ohlcv":
            _fetch_ohlcv(ticker, task.params)
        elif task_type == "chain":
            _fetch_chain(ticker, provider, task.params)
        elif task_type == "vol":
            _fetch_vol(ticker)
        elif task_type == "sector":
            _fetch_sector()
        elif task_type == "breadth":
            _fetch_breadth()
        elif task_type == "breadth_imoex":
            _fetch_breadth_imoex()
        elif task_type == "composite":
            _fetch_composite()
        elif task_type == "gex_profile":
            _fetch_gex_profile(ticker, task.params)
        else:
            logger.warning("Unknown task_type: %s", task_type)
    except Exception as exc:
        logger.error("Fetch error %s/%s/%s: %s", task_type, provider, ticker, exc)


def _fetch_ohlcv(ticker: str, params: dict) -> None:
    """Fetch OHLCV — US stocks only (yfinance). Crypto/MOEX fetched on-demand."""
    from gex.adapters.fetchers.bybit_fetcher import _CRYPTO_ASSETS
    from gex.commodity_assets import COMMODITY_ASSETS
    from gex.adapters.fetchers.moex_candles_fetcher import _MOEX_OHLCV_ASSETS
    from gex.adapters.fetchers.ta_fetcher import TATimeframesFetcher

    t_upper = ticker.upper()

    # Skip crypto and MOEX — yfinance doesn't support them
    if t_upper in _CRYPTO_ASSETS or t_upper in _MOEX_OHLCV_ASSETS:
        return

    redis = get_redis()
    fetch_ticker = ticker
    if t_upper in COMMODITY_ASSETS:
        fetch_ticker = COMMODITY_ASSETS[t_upper]["yf_symbol"]

    # Central orchestrator path (when enabled).
    try:
        from gex.orchestrator.sync_gateway import sync_fetch_ohlcv
        df = sync_fetch_ohlcv(fetch_ticker, "yfinance", "1d", limit=500)
        if df is not None and not df.empty:
            logger.info("  OHLCV %s via orchestrator: %d bars", ticker, len(df))
            return
    except Exception:
        pass

    fetcher = TATimeframesFetcher(redis_client=redis if redis and redis.connected else None)
    try:
        tfs = fetcher.fetch(fetch_ticker)
        logger.info("  OHLCV %s: %d TF loaded", ticker, len(tfs))
    except (ValueError, RuntimeError) as exc:
        logger.warning("  OHLCV %s failed: %s", ticker, exc)


def _fetch_chain(ticker: str, provider: str, params: dict) -> None:
    """Fetch опционной цепочки и сохранить в Redis."""
    redis = get_redis()

    # Central orchestrator path for option chains.
    try:
        from gex.orchestrator.sync_gateway import sync_fetch_option_chain
        provider_code = "iss" if provider == "moex_iss" else provider
        snapshot = sync_fetch_option_chain(
            ticker,
            provider_code,
            max_expiries=int(params.get("max_expiries", 5)),
        )
        if snapshot is not None:
            from gex.application.service import GEXService
            gex = GEXService()
            gex.ingest_chain(ticker, snapshot)
            logger.info("  Chain %s/%s via orchestrator: %d rows, spot=%.2f",
                        provider, ticker, len(snapshot.chain), snapshot.spot)
            return
    except Exception as exc:
        logger.warning("  Chain orchestrator path failed %s/%s: %s", provider, ticker, exc)

    if provider == "webull":
        from gex.adapters.fetchers.webull_fetcher import WebullOptionsFetcher
        max_exp = params.get("max_expiries", 5)
        fetcher = WebullOptionsFetcher(
            max_expiries=max_exp,
            redis_client=redis if redis and redis.connected else None,
        )
    elif provider == "yfinance":
        from gex.adapters.fetchers.yf_fetcher import YFOptionsFetcher
        max_exp = params.get("max_expiries", 3)
        fetcher = YFOptionsFetcher(
            max_expiries=max_exp,
            redis_client=redis if redis and redis.connected else None,
        )
    elif provider == "bybit":
        from gex.adapters.fetchers.bybit_fetcher import BybitOptionsFetcher
        max_exp = params.get("max_expiries", 3)
        fetcher = BybitOptionsFetcher(
            max_expiries=max_exp,
            redis_client=redis if redis and redis.connected else None,
        )
    elif provider == "moex_iss":
        from gex.adapters.fetchers.moex_fetcher import MOEXOptionsFetcher
        max_exp = params.get("max_expiries", 3)
        fetcher = MOEXOptionsFetcher(
            max_expiries=max_exp,
            redis_client=redis if redis and redis.connected else None,
        )
    else:
        logger.warning("Unknown chain provider: %s", provider)
        return

    try:
        snapshot = fetcher.fetch(ticker)
        # Сохраняем также в in-memory репозиторий GEXService
        from gex.application.service import GEXService
        gex = GEXService()
        gex.ingest_chain(ticker, snapshot)
        logger.info("  Chain %s/%s: %d rows, spot=%.2f",
                     provider, ticker, len(snapshot.chain), snapshot.spot)
    except (ValueError, RuntimeError) as exc:
        logger.warning("  Chain %s/%s failed: %s", provider, ticker, exc)


def _fetch_vol(ticker: str) -> None:
    """Fetch vol-индикатора."""
    redis = get_redis()
    from gex.adapters.fetchers.vol_fetcher import VolIndicatorsFetcher

    fetcher = VolIndicatorsFetcher(
        redis_client=redis if redis and redis.connected else None,
    )
    try:
        # Если тикер "^VIX" — он внутри VOL_INDICATORS
        snapshots = fetcher.fetch()
        logger.info("  Vol indicators: %d loaded", len(snapshots))
    except (ValueError, RuntimeError) as exc:
        logger.warning("  Vol fetch failed: %s", exc)


def _fetch_gex_profile(ticker: str, params: dict) -> None:
    """Fetch GEX-профиля (live) и сохранить в репозиторий."""
    from gex.application.service import GEXService
    gex = GEXService()

    days = params.get("days", 30)
    max_exp = params.get("max_expiries", 3)
    try:
        result = gex.analyze_live(ticker, days=days, max_expiries=max_exp)
        logger.info("  GEX profile %s: direction=%s, confidence=%.2f",
                     ticker, result.direction, result.confidence)
    except (ValueError, RuntimeError) as exc:
        logger.warning("  GEX profile %s failed: %s", ticker, exc)


# ====================================================================== #
#  Создание задач для превентивного фетчинга
# ====================================================================== #
def _fetch_sector() -> None:
    """Prewarm секторов: 12 ETF через TATimeframesFetcher (OHLCV в Redis-кэш)."""
    from gex.application.sector_service import SECTOR_TICKERS
    from gex.adapters.fetchers.ta_fetcher import TATimeframesFetcher

    limiter = get_rate_limiter()
    fetcher = TATimeframesFetcher(redis_client=get_redis())
    for ticker in SECTOR_TICKERS:
        limiter.wait("yfinance")  # не превышаем лимиты провайдера
        try:
            fetcher.fetch(ticker)
        except Exception as exc:
            logger.warning("  Sector %s failed: %s", ticker, exc)
    logger.info("  Sector prewarm: %d ETF", len(SECTOR_TICKERS))


def _fetch_breadth() -> None:
    """Prewarm рыночной ширины: старый RSP/SPY-кэш + новый breadth_service
    («структура движения» — ES/RSP/MAGS 10 мин, S&P500-широта 6 ч)."""
    from gex.adapters.fetchers.breadth_fetcher import fetch_mcclellan
    from gex.application.breadth_service import ensure_warm

    get_rate_limiter().wait("yfinance")
    try:
        data = fetch_mcclellan()
        logger.info("  Breadth prewarm (legacy RSP/SPY): %s", "OK" if data else "no data")
    except Exception as exc:
        logger.warning("  Breadth prewarm (legacy) failed: %s", exc)
    try:
        ok = ensure_warm()
        logger.info("  Breadth prewarm (v2 structure): %s", "OK" if ok else "failed")
    except Exception as exc:
        logger.warning("  Breadth prewarm (v2) failed: %s", exc)


def _fetch_breadth_imoex() -> None:
    """Prewarm ширины MOEX (IMOEX): полный ISS-парсинг через оркестратор.

    Вызывается ТОЛЬКО планировщиком в фиксированные слоты 23:00/08:00 МСК.
    Суточная квота оркестратора (3/24h) защищает ISS от лишних обращений.
    """
    from gex.application.breadth_imoex_service import ensure_warm

    try:
        ok = ensure_warm(force_refresh=True)
        logger.info("  Breadth IMOEX prewarm: %s", "OK" if ok else "failed")
    except Exception as exc:
        logger.warning("  Breadth IMOEX prewarm failed: %s", exc)


def _fetch_composite() -> None:
    """Prewarm композитной формулы (заполняет кэш /composite-formula)."""
    from gex.routers.breadth_sector_router import get_composite_formula

    get_rate_limiter().wait("yfinance")
    try:
        result = get_composite_formula()
        logger.info("  Composite prewarm: %d точек", len(result.get("dates", [])))
    except Exception as exc:
        logger.warning("  Composite prewarm failed: %s", exc)


def build_prewarm_tasks(priority: int = 0) -> list[Any]:
    """Создать список задач для превентивного фетчинга всех инструментов.

    Возвращает список ``FetchTask`` для публикации в очередь.
    """
    tasks: list[FetchTask] = []

    # 1. OHLCV для всех US тикеров + крипты + MOEX
    for ticker in US_TICKERS + CRYPTO_TICKERS + OHLCV_ONLY_TICKERS:
        tasks.append(FetchTask(
            task_type="ohlcv",
            provider="yfinance",
            ticker=ticker,
            priority=priority,
        ))
    for ticker in MOEX_TICKERS:
        tasks.append(FetchTask(
            task_type="ohlcv",
            provider="moex_iss",
            ticker=ticker,
            priority=priority,
        ))

    # 2. Option chains для US тикеров через yfinance
    for ticker in US_TICKERS:
        tasks.append(FetchTask(
            task_type="chain",
            provider="yfinance",
            ticker=ticker,
            params={"max_expiries": 3},
            priority=priority,
        ))

    # 3. Option chains для крипты через Bybit
    for ticker in CRYPTO_TICKERS:
        tasks.append(FetchTask(
            task_type="chain",
            provider="bybit",
            ticker=ticker,
            params={"max_expiries": 3},
            priority=priority,
        ))

    # 4. Option chains для MOEX
    for ticker in MOEX_TICKERS:
        tasks.append(FetchTask(
            task_type="chain",
            provider="moex_iss",
            ticker=ticker,
            params={"max_expiries": 3},
            priority=priority,
        ))

    # 5. Vol indicators
    tasks.append(FetchTask(
        task_type="vol",
        provider="yfinance",
        ticker="ALL",
        priority=priority,
    ))

    # 6. Агрегатные страницы: сектора / ширина / композит
    for tt in ("sector", "breadth", "composite"):
        tasks.append(FetchTask(task_type=tt, provider="yfinance", ticker="ALL", priority=priority))

    logger.info("Built %d prewarm tasks", len(tasks))
    return tasks
