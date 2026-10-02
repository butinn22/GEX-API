"""Breadth + Sector + Composite Formula routes.

Originally inline in main.py (~200 lines). Phase 2: extract into proper services.

Страницы широты рынка вынесены из запроса (итер. 38)
----------------------------------------------------
Три обработчика — ``GET /breadth``, ``GET /sector/breadth``, ``GET /breadth-imoex`` — до этой
итерации считали всё внутри запроса. Промах кэша означал ожидание внешнего провайдера:
секторный композит — 14–22 с, широта S&P 500 — ~500 бумаг и минуты, а при недоступном Redis
кэш выключался целиком и это становилось поведением **каждой** загрузки страницы. Отсюда и
«висит до таймаута»: расчёт жил ровно столько, сколько отвечал yfinance.

Теперь контракт другой:

* запрос **только читает** готовый payload (``SnapshotPort.peek``) — вычислений в запросе нет
  вообще, поэтому и подвиснуть не на чем;
* payload кладёт фоновый пересчёт (Celery, очередь ``gex_market``; аварийно — процесс API);
* свежесть добирается по расписанию Celery Beat (``market-refresh-every-5m``);
* свежих данных нет — отдаём устаревшие (до ``stale_max`` из ``gex.domain.freshness``);
* нет и устаревших — честный ``503`` с ``Retry-After`` и запрос на фоновый пересчёт.

``X-Cache``/``X-Cache-Age`` и ``meta`` в теле описывают состояние честно: страница,
показывающая вчерашние данные, обязана это говорить, а не выглядеть свежей.
"""
import asyncio
import logging
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Response

from gex.adapters.cache.keys import page_key
from gex.adapters.cache.redis_client import get_redis
from gex.application import breadth_service, market_pages
from gex.auth.dependencies import require_master_admin, require_subscription
from gex.auth.models import User
from gex.deps import provide_page_store
from gex.domain.freshness import market_open_now, policy_for_page
from gex.ports.cache import CachedPayload, CacheStatus
from gex.workers import config as workers_config
from gex.workers.dispatch import dispatch_refresh

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_subscription("BASIC"))], tags=["breadth", "sector"])

#: Пауза между проверками холодного значения. Мелкий шаг — чтобы «появилось быстро» не
#: округлялось вверх до всего окна ожидания.
_COLD_POLL_S = 0.15

#: Ответ на холодном старте: данные готовятся, повторить через столько секунд.
_WARMING_RETRY_AFTER_S = 10

#: Предел чтения хранилища сервиса IMOEX (Redis → файл-снапшот). Первый вызов после старта
#: процесса платит за подключение к Redis (клиент создаётся при импорте модуля сервиса), и
#: без предела это время уходило бы в ответ страницы. Чтение — вспомогательный путь: если не
#: успело, скажем «данные готовятся», как и при пустом хранилище.
_IMOEX_READ_TIMEOUT_S = 1.5


# ═══════════════════════════════════════════════════════════════════════
# Чтение снапшота страницы (общее для /breadth и /sector/breadth)
# ═══════════════════════════════════════════════════════════════════════
async def _peek(store: Any, key: str, page: str) -> Optional[CachedPayload]:
    """Прочитать последнее известное значение, не блокируя цикл событий.

    Redis-клиент синхронный (``redis.Redis``), поэтому чтение уходит в поток: обработчик
    ``async def`` не имеет права держать цикл на сетевом вызове — именно так один медленный
    ``/ohlcv`` однажды остановил весь сервис (инцидент 2026-09-21).
    """
    policy = policy_for_page(page, market_open=market_open_now())
    return await asyncio.to_thread(
        store.peek, key, fresh=policy.fresh, stale_max=policy.stale_max
    )


def _snapshot_response(
    snapshot: CachedPayload,
    *,
    page: str,
    mode: Optional[str],
    response: Response,
) -> dict:
    """Собрать тело ответа и заголовки состояния из снапшота."""
    stale = snapshot.status is CacheStatus.STALE
    response.headers["X-Cache"] = "stale" if stale else "hit"
    response.headers["X-Cache-Age"] = str(int(snapshot.age_s))
    response.headers["X-Cache-Page"] = page
    if snapshot.etag:
        response.headers["ETag"] = f'"{snapshot.etag}"'
    # ``max-age`` — по остатку окна свежести: браузер не должен спрашивать чаще, чем данные
    # меняются, но и не должен считать вчерашнее свежим.
    remaining = max(0, int(snapshot.fresh - snapshot.age_s))
    response.headers["Cache-Control"] = f"private, max-age={remaining}"

    body = market_pages.with_meta(
        snapshot.value,
        {
            "page": page,
            "mode": mode or "",
            "status": "stale" if stale else "hit",
            "age_s": round(float(snapshot.age_s), 1),
            "stale": stale,
            "served_from_cache": True,
        },
    )
    return body


def _request_refresh_in_background(page: str, mode: Optional[str], *, force: bool = False) -> None:
    """Попросить фоновый пересчёт, не ожидая ни его, ни ответа брокера.

    Выделено в отдельную функцию, потому что это единственный правильный способ просить
    пересчёт из обработчика: публикация в очередь сама по себе может занять секунды (мёртвый
    Redis → политика повторов Celery), и ждать её в запросе — это ровно та ошибка, из-за
    которой страницы «висели до таймаута». Исключение фоновой задачи только логируется.
    """
    task = asyncio.ensure_future(asyncio.to_thread(dispatch_refresh, page, mode, force=force))
    task.add_done_callback(_log_dispatch_failure)


async def _wait_for_snapshot(
    store: Any, key: str, page: str, timeout_s: float
) -> Optional[CachedPayload]:
    """Подождать появления годного значения не дольше ``timeout_s``.

    Ожидание короткое и обязательное по смыслу: между «двойной проверкой кэша» и «ответом
    503» должна быть хотя бы одна попытка увидеть работу воркера — иначе холодный старт при
    живом воркере давал бы ошибку там, где данные появляются через доли секунды.
    """
    deadline = asyncio.get_running_loop().time() + max(0.0, timeout_s)
    while True:
        snapshot = await _peek(store, key, page)
        if snapshot is not None and snapshot.status is not CacheStatus.EXPIRED:
            return snapshot
        if asyncio.get_running_loop().time() >= deadline:
            return None
        await asyncio.sleep(_COLD_POLL_S)


async def _snapshot_or_warming(
    store: Any,
    *,
    page: str,
    mode: Optional[str],
    response: Response,
    force: bool = False,
) -> dict:
    """Отдать снапшот страницы или честный «данные готовятся» + просьба о пересчёте.

    Порядок предпочтений неизменен и повторяет SWR-логику: свежее → устаревшее → просьба о
    пересчёте → 503. Вычислений здесь нет ни в одной ветке.
    """
    key = page_key(page, mode) if mode else page_key(page)

    snapshot = await _peek(store, key, page)
    if snapshot is not None and snapshot.status is not CacheStatus.EXPIRED:
        if snapshot.status is CacheStatus.STALE:
            # Устаревшее отдаём сразу, а обновление просим фоном — пользователь не ждёт.
            _request_refresh_in_background(page, mode, force=force)
        return _snapshot_response(snapshot, page=page, mode=mode, response=response)

    # Холодный старт: просим пересчёт (не ожидая ответа брокера) и ждём появления значения
    # не дольше окна из конфигурации. Ждать дольше смысла нет: холодный пересчёт этих
    # страниц измеряется минутами, и «подождать его в запросе» — это и есть исходный дефект.
    _request_refresh_in_background(page, mode, force=force)
    snapshot = await _wait_for_snapshot(store, key, page, workers_config.MARKET_COLD_WAIT_S)
    if snapshot is not None:
        return _snapshot_response(snapshot, page=page, mode=mode, response=response)

    # Значение есть, но протухло: об этом честно сообщаем — иначе «данных нет» выглядит как
    # «страница сломана», хотя данные были и их просто больше нельзя показывать.
    expired = await _peek(store, key, page)
    age_hint = (
        f" Последнее известное значение старше {int(expired.age_s // 60)} мин и показано не будет."
        if expired is not None else ""
    )
    logger.warning("Страница %s: данных нет (key=%s) — пересчёт запрошен фоном", page, key)
    raise HTTPException(
        status_code=503,
        detail=(
            f"Данные страницы «{page}» готовятся фоном, запрос на пересчёт отправлен. "
            f"Обновите страницу через {_WARMING_RETRY_AFTER_S} с.{age_hint}"
        ),
        headers={"Retry-After": str(_WARMING_RETRY_AFTER_S), "X-Cache": "empty"},
    )


def _log_dispatch_failure(task: "asyncio.Future") -> None:
    """Не потерять исключение фоновой просьбы о пересчёте (иначе оно молча пропадёт)."""
    try:
        task.result()
    except Exception as exc:  # noqa: BLE001 — фон: максимум warning
        logger.warning("Просьба о пересчёте завершилась ошибкой: %s", exc)


# ═══════════════════════════════════════════════════════════════════════
# /breadth
# ═══════════════════════════════════════════════════════════════════════
# «Структура движения рынка» для BASIC-подписчиков (2026-09-04):
#   market — ES/RSP/giants (MAGS или Топ-10 по кап.) + отношения к ES;
#   stocks — S&P 500: % выше EMA20/50/200 + настоящий McClellan по A/D;
#   corr — скользящие корреляции движения лидеров с базой и шириной;
#   current — 5д-сигналы и диагноз. Расчёт — gex/application/market_pages.py
#   (фоново: Celery `gex.market.refresh_page`, расписание — Beat, раз в 5 минут).
@router.get("/breadth")
async def get_breadth(
    response: Response,
    giants: Literal["mags", "top10"] = "mags",
    store=Depends(provide_page_store),
) -> dict:
    """Широта рынка США: последний готовый расчёт (вычислений в запросе нет)."""
    return await _snapshot_or_warming(
        store, page=market_pages.PAGE_BREADTH, mode=giants, response=response
    )


# ═══════════════════════════════════════════════════════════════════════
# /breadth-imoex
# ═══════════════════════════════════════════════════════════════════════
def _read_latest_sync() -> Optional[dict]:
    """Синхронное чтение хранилища сервиса IMOEX.

    Импорт стоит **внутри** этой функции, а не в асинхронной: модуль сервиса создаёт свой
    ``RedisClient`` при импорте, то есть платит за попытку подключения. В обработчике
    ``async def`` этот импорт выполнялся бы в потоке цикла событий и останавливал бы весь
    сервис на секунды (измерено: первый запрос страницы — 4.8 с при ответе 15 мс на втором).
    В рабочем потоке цена та же, но платит её только этот запрос.
    """
    from gex.application.breadth_imoex_service import get_latest

    return get_latest()


async def _read_imoex() -> Optional[dict]:
    """Последний удачный расчёт IMOEX (Redis → файл-снапшот), без обращения к ISS.

    Чтение ограничено по времени: это вспомогательный путь к снапшоту страницы, и он не
    имеет права держать ответ дольше, чем длится само обращение к хранилищу.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_read_latest_sync), timeout=_IMOEX_READ_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        logger.warning(
            "IMOEX: чтение хранилища сервиса не уложилось в %.1f c — отвечаю «данные готовятся»",
            _IMOEX_READ_TIMEOUT_S,
        )
        return None


@router.get("/breadth-imoex")
async def get_breadth_imoex(response: Response, store=Depends(provide_page_store)) -> dict:
    """Широта MOEX (IMOEX) — только чтение последнего удачного расчёта.

    Страница НИКОГДА не инициирует обращение к ISS: данные готовит фоновый расчёт
    (Celery Beat, слоты 23:00 и 08:00 МСК из ``gex.domain.schedule.PERIODIC_SLOTS``).

    Читаются **два** источника, и порядок здесь принципиален: сначала снапшот страницы
    (мгновенный, с метаданными о возрасте), затем хранилище сервиса (Redis → файл). Второй
    источник — не дублирование, а страховка: он остаётся заполненным даже когда очередь
    недоступна, и без него страница отвечала бы «данных нет» там, где они есть.
    """
    snapshot = await _peek(store, page_key(market_pages.PAGE_IMOEX), market_pages.PAGE_IMOEX)
    if snapshot is not None and snapshot.status is not CacheStatus.EXPIRED:
        if snapshot.status is CacheStatus.STALE:
            _request_refresh_in_background(market_pages.PAGE_IMOEX, None, force=False)
        return _snapshot_response(
            snapshot, page=market_pages.PAGE_IMOEX, mode=None, response=response
        )

    latest = await _read_imoex()
    if not latest:
        # Просьбу о пересчёте отправляем и **не ждём** её: ответ про «данные готовятся» нужен
        # пользователю сразу, а не через время отказа брокера.
        _request_refresh_in_background(market_pages.PAGE_IMOEX, None, force=False)
        raise HTTPException(
            status_code=503,
            detail=(
                "Данные MOEX готовятся по расписанию (23:00 и 08:00 МСК). "
                f"Обновите страницу через {_WARMING_RETRY_AFTER_S} с."
            ),
            headers={"Retry-After": str(_WARMING_RETRY_AFTER_S), "X-Cache": "empty"},
        )

    response.headers["X-Cache"] = "service"
    body = market_pages.with_meta(
        latest,
        {"page": market_pages.PAGE_IMOEX, "mode": "", "status": "hit", "served_from_cache": True},
    )
    return body


@router.post("/breadth-imoex/refresh", status_code=202)
async def refresh_breadth_imoex(user: User = Depends(require_master_admin)) -> dict:
    """Принудительное обновление IMOEX-ширины — ТОЛЬКО для master-admin.

    Раньше этот вызов парсил ISS **внутри запроса** и висел минутами (а при медленном ISS —
    до таймаута клиента). Теперь он ставит фоновую задачу и сразу отвечает ``202``: админка
    не ждёт, страница перечитывает данные по своей кнопке. Парсинг идёт в обход суточной
    квоты (``bypass_rate_limit``), потому что это осознанное действие человека, а не
    расписание.
    """
    from gex.workers.dispatch import dispatch_admin_refresh

    ticket = await asyncio.to_thread(
        dispatch_admin_refresh, market_pages.PAGE_IMOEX, None, bypass_rate_limit=True
    )
    if not ticket.started and not ticket.pending:
        raise HTTPException(
            status_code=503,
            detail=f"Не удалось запустить обновление IMOEX: {ticket.reason}",
        )
    body = {
        "status": "accepted",
        "channel": ticket.channel,
        "message": "Обновление данных IMOEX запущено в фоне",
    }
    if ticket.task_id:
        body["task_id"] = ticket.task_id
        body["status_url"] = f"/tasks/{ticket.task_id}"
    return body


CBOE_COR1M_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/COR1M_History.csv"


def _fetch_cor1m_series() -> pd.Series:
    """Загрузить COR1M (индекс корреляции 1-мес) с CBOE CDN.

    CSV приходит БЕЗ заголовка: ``MM/DD/YYYY,open,high,low,close``.
    Возвращает pd.Series Close с DatetimeIndex; при ошибке — пустой Series.
    """
    try:
        import io

        import requests as req
        resp = req.get(CBOE_COR1M_URL, timeout=15)
        resp.raise_for_status()
        # CSV с заголовком: DATE,OPEN,HIGH,LOW,CLOSE (верхний регистр, MM/DD/YYYY)
        cor_df = pd.read_csv(
            io.StringIO(resp.text),
            index_col=0,
            parse_dates=True,
        )
        close_col = "CLOSE" if "CLOSE" in cor_df.columns else "Close"
        cor_series = cor_df[close_col].astype(float).dropna()
        cor_series.name = "COR1M"
        return cor_series
    except Exception as e:
        logger.warning("CBOE COR1M: %s", e)
        return pd.Series(dtype=float, name="COR1M")


# ═══════════════════════════════════════════════════════════════════════
# /vix-cor1m-history
# ═══════════════════════════════════════════════════════════════════════
@router.get("/vix-cor1m-history")
def get_vix_cor1m_history(user: User = Depends(require_master_admin)) -> dict:
    """История VIX/COR1M + текущие значения (кэш 10 мин)."""
    from gex.adapters.cache.redis_client import deserialize_value

    _cache = get_redis()
    if _cache and _cache.connected:
        _hit = _cache.get("gex:vix:cor1m:history")
        if _hit is not None:
            return deserialize_value(_hit)

    series = breadth_service.fetch_yf_close_series({"^VIX": "VIX"}, period="1y", auto_adjust=False)
    vix_close = series.get("VIX")

    cor = _fetch_cor1m_series()

    if vix_close is None or vix_close.empty or cor.empty:
        raise HTTPException(status_code=502, detail="VIX/COR1M данные недоступны")

    combined = pd.concat([vix_close, cor], axis=1).dropna()
    if len(combined) < 2:
        raise HTTPException(status_code=502, detail="Недостаточно данных VIX/COR1M")

    ratio = (combined["VIX"] / combined["COR1M"]).dropna()
    # Последние 250 торговых дней
    if len(ratio) > 250:
        ratio = ratio.iloc[-250:]

    dates = [str(d.date()) for d in ratio.index]
    values = [round(float(v), 3) for v in ratio.values]

    result = {
        "dates": dates,
        "ratio": values,
        "current_vix": round(float(combined["VIX"].iloc[-1]), 2),
        "current_ratio": round(float(ratio.iloc[-1]), 2),
        "cor1m": round(float(combined["COR1M"].iloc[-1]), 2),
    }
    try:
        if _cache and _cache.connected:
            _cache.set("gex:vix:cor1m:history", result, ex=600)
    except Exception:
        pass
    return result


# ═══════════════════════════════════════════════════════════════════════
# /composite-formula
# ═══════════════════════════════════════════════════════════════════════
@router.get("/composite-formula")
def get_composite_formula(user: User = Depends(require_master_admin)) -> dict:
    try:
        from gex.adapters.cache.redis_client import deserialize_value
        _cache = get_redis()
        if _cache and _cache.connected:
            _hit = _cache.get("gex:composite:formula")
            if _hit is not None:
                return deserialize_value(_hit)

        # Load indicator history via yfinance (дедлайн-защищённо, через сервис широты)
        closes: dict[str, pd.Series] = breadth_service.fetch_yf_close_series(
            {"^VIX": "VIX", "^VVIX": "VVIX", "DX-Y.NYB": "DXY"},
            period="1y",
            auto_adjust=False,
        )

        # COR1M via CBOE CDN
        cor_series = _fetch_cor1m_series()
        if len(cor_series) > 1:
            closes["COR1M"] = cor_series

        # McClellan Summation Index
        mcc_summation = pd.Series(dtype=float)
        try:
            from gex.adapters.fetchers.breadth_fetcher import fetch_mcclellan
            mcc = fetch_mcclellan()
            if mcc and mcc.dates and mcc.mc_summation_index:
                mcc_summation = pd.Series(
                    mcc.mc_summation_index,
                    index=pd.to_datetime(mcc.dates),
                    name="SumIdx",
                ).dropna()
        except Exception as e:
            logger.warning("McClellan: %s", e)

        # PCR from SPY options (дедлайн-защищённо, через сервис широты)
        pcr_series = pd.Series(dtype=float)
        pcr = breadth_service.fetch_spy_put_call_ratio()
        if pcr is not None:
            pcr_series = pd.Series([pcr], name="PCR")

        # Build combined dataframe, forward-fill
        all_series = [s for s in closes.values() if len(s) > 1]
        if len(all_series) < 2:
            raise HTTPException(status_code=502, detail="Not enough data for composite formula")

        combined = pd.concat(all_series, axis=1).ffill().bfill()
        if len(mcc_summation) > 1:
            combined = combined.join(mcc_summation, how="left").ffill()

        # Formula: VIX / (VVIX * COR1M) * DXY * SumIdx / PCR
        formula_series = pd.Series(dtype=float)
        if all(k in combined.columns for k in ("VIX", "VVIX", "COR1M", "DXY")):
            denom = combined["VVIX"] * combined["COR1M"]
            denom = denom.replace(0, np.nan).ffill().bfill()
            ratio = combined["VIX"] / denom * combined["DXY"]
            if "SumIdx" in combined.columns:
                ratio = ratio * combined["SumIdx"].abs() / 1000.0
            if len(pcr_series) > 0:
                ratio = ratio / max(float(pcr_series.iloc[-1]), 0.01)
            formula_series = ratio.dropna()

        # Last 200 days
        if len(formula_series) > 200:
            formula_series = formula_series.iloc[-200:]

        values = [round(float(v), 4) for v in formula_series.values]
        dates = [str(d.date()) for d in formula_series.index]

        current = values[-1] if values else 0.0
        arr = np.array(values) if values else np.array([0.0])
        upper = round(float(np.percentile(arr, 80)), 2)
        lower = round(float(np.percentile(arr, 20)), 2)
        median = round(float(np.median(arr)), 2)

        result = {
            "dates": dates,
            "values": values,
            "current": current,
            "bands": {"upper": upper, "lower": lower, "median": median},
            "n_days": len(values),
        }
        try:
            if _cache and _cache.connected:
                _cache.set("gex:composite:formula", result, ex=600)
        except Exception:
            pass
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Composite formula error: {e}")


# ═══════════════════════════════════════════════════════════════════════
# /sector/breadth
# ═══════════════════════════════════════════════════════════════════════
@router.get("/sector/breadth")
async def get_sector_breadth(
    response: Response,
    store=Depends(provide_page_store),
) -> dict:
    """Композит секторов США — последний готовый расчёт (в запросе не считается).

    Раньше это был SWR-кэш с синхронным пересчётом на промахе: 14–22 с ожидания на первой
    загрузке и полное отсутствие кэша при недоступном Redis. Теперь промах означает «данные
    готовятся, пересчёт запущен», а расчёт делает воркер очереди ``gex_market``.
    """
    return await _snapshot_or_warming(
        store, page=market_pages.PAGE_SECTOR, mode=None, response=response
    )

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════
def _build_composite(indicators: dict, mcc) -> dict:
    """Build 0-100 composite indicator from all vol-reversal indicators."""
    score = 0.0
    n = 0
    parts = {}

    for sym in ("VIX", "VVIX", "MOVE"):
        ind = indicators.get(sym)
        if ind and np.isfinite(ind.rsi_weekly):
            contrib = 100 - ind.rsi_weekly
            score += contrib
            parts[sym] = {"rsi": round(ind.rsi_weekly, 1), "contrib": round(contrib, 1)}
            n += 1

    ind = indicators.get("COR1M")
    if ind and np.isfinite(ind.zscore):
        contrib = 50 - ind.zscore * 10
        score += contrib
        parts["COR1M"] = {"zscore": round(ind.zscore, 2), "contrib": round(contrib, 1)}
        n += 1

    ind = indicators.get("DXY")
    if ind and np.isfinite(ind.rsi_weekly):
        contrib = 100 - ind.rsi_weekly
        score += contrib
        parts["DXY"] = {"rsi": round(ind.rsi_weekly, 1), "contrib": round(contrib, 1)}
        n += 1

    ind = indicators.get("TLT")
    if ind and np.isfinite(ind.rsi_weekly):
        contrib = 100 - ind.rsi_weekly
        score += contrib
        parts["TLT"] = {"rsi": round(ind.rsi_weekly, 1), "contrib": round(contrib, 1)}
        n += 1

    ind = indicators.get("PCR")
    if ind and np.isfinite(ind.rsi_weekly):
        contrib = 100 - ind.rsi_weekly
        score += contrib
        parts["PCR"] = {"rsi": round(ind.rsi_weekly, 1), "contrib": round(contrib, 1)}
        n += 1

    if mcc and np.isfinite(mcc.rsi_14):
        contrib = mcc.rsi_14
        score += contrib
        parts["MCC"] = {"rsi": round(mcc.rsi_14, 1), "contrib": round(contrib, 1)}
        n += 1

    if n == 0:
        return {"value": 50.0, "label": "neutral (no data)", "parts": parts, "n_indicators": 0}

    final = round(score / n, 1)
    if final < 30:
        label = "oversold (potential rally)"
    elif final > 70:
        label = "overbought (correction risk)"
    else:
        label = "neutral"

    return {"value": final, "label": label, "parts": parts, "n_indicators": n}
