"""TA + OHLCV: /ta/{ticker}, /ta/{ticker}/{timeframe}, /ohlcv/{ticker}."""

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Query

from gex.auth.dependencies import get_optional_user
from gex.auth.models import User, subscription_is_active
from gex.deps import provide_ta_service
from gex.application.ohlcv_service import fetch_ohlcv
from gex.adapters.cache.result_cache import cache_key, result_cache
from gex.schemas import OHLCVOut, TAAnalysisOut, TimeframeOut
from gex.adapters.notifications.telegram_sender import notify_ta_analysis, notify_ta_timeframe

from ._helpers import handle

logger = logging.getLogger(__name__)

from fastapi import Depends
from gex.auth.dependencies import require_subscription
router = APIRouter(dependencies=[Depends(require_subscription("BASIC"))], tags=["technical-analysis", "chart"])

@router.get("/ta/{ticker}", response_model=TAAnalysisOut)
def get_ta(ticker: str, n_paths: int = Query(10_000, ge=100, le=100_000),
    notify: bool = Query(False), background_tasks: BackgroundTasks = None,
    user: User | None = Depends(get_optional_user),
    svc=Depends(provide_ta_service)) -> TAAnalysisOut:
    # Уведомление в Telegram — только при активной подписке.
    chat_id = user.telegram_chat_id if (user and subscription_is_active(user)) else None
    _key = cache_key("res", "ta", ticker, n_paths)
    if notify:
        return handle(lambda: svc.analyze(ticker, n_paths=n_paths),
            notify_fn=notify_ta_analysis, notify=True, background_tasks=background_tasks,
            chat_id=chat_id, error_src="yfinance")
    return result_cache.get(_key, 600, lambda: handle(
        lambda: svc.analyze(ticker, n_paths=n_paths),
        notify_fn=notify_ta_analysis, notify=False, background_tasks=background_tasks,
        chat_id=chat_id, error_src="yfinance"))

@router.get("/ta/{ticker}/{timeframe}", response_model=TimeframeOut)
def get_ta_timeframe(ticker: str, timeframe: str,
    n_paths: int = Query(10_000, ge=100, le=100_000),
    notify: bool = Query(False), background_tasks: BackgroundTasks = None,
    user: User | None = Depends(get_optional_user),
    svc=Depends(provide_ta_service)) -> TimeframeOut:
    # Уведомление в Telegram — только при активной подписке.
    chat_id = user.telegram_chat_id if (user and subscription_is_active(user)) else None
    _key = cache_key("res", "ta_tf", ticker, timeframe, n_paths)
    if notify:
        return handle(lambda: svc.analyze_timeframe(ticker, timeframe, n_paths=n_paths),
            notify_fn=lambda r: notify_ta_timeframe(r, ticker.upper(), chat_id=chat_id),
            notify=True, background_tasks=background_tasks, error_src="yfinance")
    return result_cache.get(_key, 600, lambda: handle(
        lambda: svc.analyze_timeframe(ticker, timeframe, n_paths=n_paths),
        notify_fn=lambda r: notify_ta_timeframe(r, ticker.upper(), chat_id=chat_id),
        notify=False, background_tasks=background_tasks, error_src="yfinance"))

@router.get("/ohlcv/{ticker}", response_model=OHLCVOut)
async def get_ohlcv(ticker: str, timeframe: str = Query("1d"),
    limit: int = Query(200, ge=1, le=1000)) -> OHLCVOut:
    # Data orchestrator path (when enabled): the request passes through cache,
    # singleflight, provider queue/rate limiter and returns the same OHLCVOut.
    from fastapi import HTTPException

    # Оркестратор выключен (текущий .env: ORCHESTRATOR_ENABLED=false) — это **штатный** откат
    # на legacy-путь, а не ошибка: раньше здесь бросалось исключение, broad-except печатал полный
    # traceback ERROR на каждый /ohlcv. Теперь выключенный случай — тихий debug без исключения.
    orchestrator_enabled = False
    try:
        from gex.orchestrator.config import orchestrator_settings

        orchestrator_enabled = bool(orchestrator_settings.enabled)
    except Exception:
        logger.debug("Настройки оркестратора недоступны — legacy-путь OHLCV", exc_info=True)

    if orchestrator_enabled:
        # Импорты оркестратора отделены от «боевого» try: имена из них фигурируют в
        # `except (...)` ниже, и если бы импорт падал **внутри** того же try, Python
        # вычислял бы except-клаузы уже во время обработки исключения → NameError →
        # 500 наружу. Старый код в этом случае падал на broad-except и уходил на legacy —
        # не регрессируем: не вышло импортировать → тихий откат на legacy-путь.
        try:
            from gex.orchestrator.exceptions import (
                CircuitOpenError,
                QueueFullError,
                RateLimitedError,
                TimeoutWaitingError,
                UpstreamError,
            )
            from gex.orchestrator.router_api import get_orchestrator
        except Exception:
            logger.debug("Оркестратор недоступен — legacy-путь OHLCV", exc_info=True)
            orchestrator_enabled = False

    if orchestrator_enabled:
        try:
            orch = get_orchestrator()
            from gex.application.ohlcv_service import COMMODITY_ASSETS, detect_asset_type
            from gex.schemas import OHLCVBarOut

            asset_type = detect_asset_type(ticker)
            symbol = ticker.strip().upper()
            if asset_type == "commodity" and symbol in COMMODITY_ASSETS:
                symbol = COMMODITY_ASSETS[symbol]["yf_symbol"]
            provider = {
                "stock": "yfinance",
                "commodity": "yfinance",
                "crypto": "bybit",
                "moex": "iss",
            }.get(asset_type, "yfinance")
            result = await orch.fetch_candles(
                symbol,
                provider=provider,
                interval=timeframe,
                priority="interactive",
                params={"limit": limit},
                max_wait_ms=3000,
            )
            bars = [OHLCVBarOut(t=b["t"], o=float(b["o"]), h=float(b["h"]), l=float(b["l"]),
                                c=float(b["c"]), v=float(b["v"])) for b in (result.data or [])]
            if bars:
                return OHLCVOut(
                    symbol=ticker.strip().upper(),
                    asset_type=asset_type,
                    timeframe=timeframe,
                    spot=round(bars[-1].c, 2),
                    bars=bars[-limit:],
                )
            # Пустой набор — не сбой оркестратора: тихо уходим на legacy-путь (без исключения).
        except (RateLimitedError, CircuitOpenError, QueueFullError) as exc:
            # These are meaningful protection signals: do not bypass the orchestrator.
            raise HTTPException(
                status_code=getattr(exc, "status_code", 429),
                detail=exc.to_payload(),
            ) from exc
        except TimeoutWaitingError as exc:
            raise HTTPException(status_code=504, detail=exc.to_payload()) from exc
        except UpstreamError as exc:
            raise HTTPException(status_code=502, detail=exc.to_payload()) from exc
        except Exception:
            # Оркестратор включён, но случился настоящий сбой (Redis недоступен, адаптер ещё не
            # реализован и т.п.): WARNING с traceback, чтобы это было видно, но не спамило ERROR.
            logger.warning("Orchestrator OHLCV path failed; falling back to legacy fetch", exc_info=True)

    _key = cache_key("res", "ohlcv", ticker, timeframe, limit)
    # ВАЖНО: fetch_ohlcv — синхронный и блокирующий (yfinance/Bybit/iss). Роутер объявлен
    # `async def`, поэтому прямой вызов морозит event loop на всё время ожидания апстрима
    # (до 30 с): перестают отвечать ВСЕ маршруты, включая /health, а дашборд шлёт по
    # запросу на карточку и фризится серийно. Уносим вычисление в поток.
    value = await asyncio.to_thread(lambda: result_cache.get(_key, 60, lambda: handle(
        lambda: fetch_ohlcv(ticker, timeframe=timeframe, limit=limit),
        error_src="yfinance/Bybit")))
    if value is None:
        # Единственный путь, где single-flight мог отдать None: лидер не успел, а stale нет.
        # None до FastAPI = ResponseValidationError → 500. Отдаём честный 503 с Retry-After.
        raise HTTPException(
            status_code=503,
            detail="Данные ещё рассчитываются, повторите запрос",
            headers={"Retry-After": "2"},
        )
    return value
