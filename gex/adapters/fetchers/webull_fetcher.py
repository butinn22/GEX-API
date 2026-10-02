"""Webull options data fetcher — primary source for US stocks.

Free, no auth required. Provides real OI, IV, and pre-computed greeks.
Fallback to yfinance if Webull is unavailable.

API: POST https://quotes-gw.webullfintech.com/api/quote/option/strategy/list
     Body: {"count": -1, "direction": "all", "tickerId": <id>}
     Headers: standard web + random UUID as "did" (device ID)

Returns canonical :class:`OptionSnapshot` compatible with GEX pipeline.

────────────────────────────────────────────────────────────────────────────
АУДИТ 2026-09-17 (почему модуль переписан)
────────────────────────────────────────────────────────────────────────────
Замер (``scripts/probe_webull.py``) показал: базовый POST возвращает ВСЕ
экспирации (SPY — 33, AAPL/NVDA — 24), но **только ближайшая** несёт
``openInterest > 0`` и ``impVol > 0``. Остальные приходят оболочками
(OI = 0, IV = 0) и полностью отбрасываются фильтром ``oi > 0``.

Итог до правки: GEX-профиль строился по **одной** экспирации (0DTE/1DTE),
независимо от ``max_expiries``:

    SPY  : 206 строк, 130 страйков, OI 439 368
    AAPL : 184 строки, 107 страйков
    NVDA : 152 строки,  90 страйков

Подсказку дал сам API: запрос с ``expireDate`` без ``unSymbol`` отвечает
``HTTP 417 "UnSymbol can't be null when expireDate is not null!"`` — то есть
**целевая экспирация запрашивается отдельно**. Проверено
(``scripts/probe_webull_persymbol.py``): повторный POST с
``{"expireDate": <дата>, "unSymbol": <тикер>}`` возвращает эту экспирацию
с настоящими OI и IV.

После правки (SPY, 6 экспираций): **2192 строки, 321 страйк, OI 6 075 195** —
в ~14 раз больше открытого интереса и в 2.5 раза больше страйков.

Стоимость: 1 базовый запрос + N запросов по экспирациям,并行 6 воркеров →
12 экспираций за ~2.5 с, ни одного 429/5xx (``scripts/probe_webull_limits.py``).
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from gex.domain.data_loader import GEXDataLoader, OptionSnapshot
from gex.adapters.cache.redis_client import RedisClient, deserialize_value
from gex.adapters.cache.keys import PROVIDER_WEBULL, chain_key_v2

logger = logging.getLogger(__name__)

# ── Webull API config ──────────────────────────────────────────────
_WEBULL_SEARCH = "https://quotes-gw.webullfintech.com/api/search/pc/tickers"
_WEBULL_OPTIONS = "https://quotes-gw.webullfintech.com/api/quote/option/strategy/list"

# Persistent device ID (like reference's did.bin). Reused across calls to
# avoid looking like a new device each time, which can trigger rate limits.
_DID: str = ""

# Короткая экспоненциальная задержка с джиттером для одной ретрай-попытки
# на 429/5xx/сетевых ошибках (Webull — без SLA).
_BACKOFF_BASE_SECONDS: float = 0.5

#: HTTP-коды, на которые уместен ровно один повтор.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

#: Сколько экспираций качать параллельно. Замер: 6 воркеров → 12 экспираций
#: за 2.5 с без единого 429. Больше — риск поrate-лимиту, меньше — медленнее.
_DEFAULT_MAX_WORKERS = 6

#: Жёсткий потолок запросов за один fetch (базовый + N по экспирациям).
#: Защита от «20 экспираций × каждый перебор» при нештатном конфиге.
_MAX_REQUEST_BUDGET = 24


def _default_strike_window() -> Optional[float]:
    """Страйк-окно по умолчанию из env (``None`` = не обрезать).

    ``GEX_WEBULL_STRIKE_WINDOW_PCT=50`` → оставить страйки в ±50% от спота.
    Пустое/0 → окно отключено (максимальная представительность).
    """
    raw = os.getenv("GEX_WEBULL_STRIKE_WINDOW_PCT", "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        logger.warning("GEX_WEBULL_STRIKE_WINDOW_PCT=%r не число — окно отключено", raw)
        return None
    if v <= 0:
        return None
    return v / 100.0 if v > 1 else v


def _backoff_delay(attempt: int) -> float:
    """Экспоненциальная задержка ``0.5·2^attempt`` + джиттер 0–0.3 с."""
    return _BACKOFF_BASE_SECONDS * (2 ** attempt) + random.uniform(0.0, 0.3)


def _get_did() -> str:
    global _DID
    if not _DID:
        _DID = uuid.uuid4().hex
    return _DID


def _build_headers() -> dict[str, str]:
    """Build request headers matching Webull webapp (ref: build_req_headers)."""
    return {
        **_WEBULL_HEADERS_BASE,
        "did": _get_did(),
        "reqid": uuid.uuid4().hex,
    }


_WEBULL_HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json;charset=UTF-8",
    "platform": "web",
    "hl": "en",
    "os": "web",
    "osv": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "app": "global",
    "appid": "webull-webapp",
    "ver": "4.10.0",
    "device-type": "Web",
    "locale": "en-us",
    "region": "6",
    "lzone": "dc_core_r001",
    "Origin": "https://app.webull.com",
    "Referer": "https://app.webull.com/",
    "Connection": "keep-alive",
}

# Cache tickerId → Webull internal ID
_ticker_id_cache: dict[str, int] = {}

# Один Session на поток: requests.Session не обязан быть потокобезопасным,
# а фанаут по экспирациям идёт в ThreadPoolExecutor.
_thread_local = threading.local()


def _thread_session() -> requests.Session:
    """Session, переиспользуемая внутри одного потока (keep-alive)."""
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        _thread_local.session = sess
    return sess


class WebullOptionsFetcher:
    """Fetch option chains from Webull (free, no auth).

    В отличие от предыдущей версии, цепочка собирается **по экспирациям**:
    базовый запрос даёт список дат (с OI только у ближайшей), затем каждая
    нужная экспирация дозапрашивается отдельным POST с ``expireDate`` +
    ``unSymbol``. Подробности — в docstring модуля.

    Parameters
    ----------
    max_expiries : int
        Max number of expiration dates to load.
    timeout : int
        HTTP timeout in seconds.
    max_days : float, optional
        Горизонт в днях: экспирации дальше ``max_days`` не запрашиваются
        (экономия запросов — они всё равно были бы отсечены фильтром
        ``_filter_by_days`` в анализаторе).
    strike_window_pct : float, optional
        Половина страйк-окна в долях от спота (``0.5`` = ±50%). ``None``
        (по умолчанию) — не обрезать. Сколько строк отбросило окно,
        видно в логе и в ``snapshot.meta["webull"]``.
    max_workers : int
        Число параллельных запросов по экспирациям.
    """

    def __init__(
        self,
        max_expiries: int = 5,
        timeout: int = 15,
        redis_client: Optional[RedisClient] = None,
        max_days: Optional[float] = None,
        strike_window_pct: Optional[float] = None,
        max_workers: int = _DEFAULT_MAX_WORKERS,
    ):
        self.max_expiries = max(1, int(max_expiries))
        self.timeout = timeout
        self._redis = redis_client
        self.max_days = float(max_days) if max_days else None
        self.strike_window_pct = (
            strike_window_pct if strike_window_pct is not None else _default_strike_window()
        )
        self.max_workers = max(1, min(int(max_workers), 8))
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    #  Ключ кэша (сегмент области выборки)
    # ------------------------------------------------------------------ #
    def _cache_key(self, ticker_str: str) -> str:
        """Ключ с областью выборки: горизонт дней + страйк-окно.

        ``scope`` — один сегмент без ``:`` (требование ``clean_segment``):
        ``d30w50`` = до 30 дней, окно ±50%; ``d0w0`` = без ограничений.
        """
        scope = "d{}w{}".format(
            int(self.max_days) if self.max_days else 0,
            int(round(self.strike_window_pct * 100)) if self.strike_window_pct else 0,
        )
        return chain_key_v2(ticker_str, self.max_expiries, scope, provider=PROVIDER_WEBULL)

    # ------------------------------------------------------------------ #
    #  Публичная точка входа
    # ------------------------------------------------------------------ #
    def fetch(self, ticker: str) -> OptionSnapshot:
        """Fetch fresh option chain from Webull; returns canonical OptionSnapshot."""
        ticker_str = ticker.strip().upper()

        # ── Redis cache check ──
        ck = self._cache_key(ticker_str)
        if self._redis is not None and self._redis.connected:
            cached = self._redis.get(ck)
            if cached is not None:
                try:
                    result = deserialize_value(cached)
                    if isinstance(result, OptionSnapshot):
                        logger.info("Webull CACHE HIT for %s", ticker_str)
                        return result
                except Exception:
                    pass

        # ── 1. Resolve ticker → Webull tickerId ──
        ticker_id = _resolve_ticker_id(ticker_str, self._session, self.timeout)

        # ── 2. Базовый запрос: список экспираций ──
        payload = {"count": -1, "direction": "all", "tickerId": ticker_id}
        resp = self._post_options(payload, ticker_str)
        if resp.status_code != 200:
            raise RuntimeError(f"Webull HTTP {resp.status_code} for {ticker_str}")
        data = resp.json()

        expire_list = data.get("expireDateList", [])
        if not expire_list:
            raise ValueError(f"Webull: no option expirations for {ticker_str}")

        spot = _parse_float(data.get("close"))
        now = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()

        entries = _parse_expiry_entries(expire_list, now)
        if not entries:
            raise ValueError(f"Webull: no parsable expirations for {ticker_str}")

        # ── 3. Выбор экспираций ──
        selected = self._select_expiries(entries)

        # ── 4. Фанаут: дозапрос каждой экспирации, у которой нет OI ──
        rows: list[dict] = []
        stats = {
            "expiries_total": len(entries),
            "expiries_selected": len(selected),
            "expiries_fetched": 0,
            "expiries_from_baseline": 0,
            "expiries_failed": 0,
            "requests": 1,
            "dropped_zero_oi": 0,
            "dropped_bad_strike": 0,
            "dropped_window": 0,
            "kept_no_iv": 0,
        }

        need_fetch: list[dict] = []
        for e in selected:
            if e["has_oi"]:
                # Ближайшая экспирация уже полная — не тратим на неё запрос.
                rows.extend(_rows_from_options(e["options"], e["T"]))
                stats["expiries_from_baseline"] += 1
            else:
                need_fetch.append(e)

        if need_fetch:
            fetched = self._fetch_expiries_parallel(ticker_id, need_fetch, ticker_str)
            stats["requests"] += len(need_fetch)
            for e, opts in fetched:
                if opts is None:
                    stats["expiries_failed"] += 1
                    continue
                rows.extend(_rows_from_options(opts, e["T"]))
                stats["expiries_fetched"] += 1

        if not rows:
            raise ValueError(
                f"Webull: empty chain for {ticker_str} "
                f"({len(selected)} expirations, no rows with usable data)"
            )

        raw = pd.DataFrame(rows)

        # ── 5. Фильтры ──
        before = len(raw)
        drop = raw.apply(lambda r: _classify_drop(r["oi"], r["strike"]), axis=1)
        stats["dropped_zero_oi"] = int((drop == "zero_oi").sum())
        stats["dropped_bad_strike"] = int((drop == "bad_strike").sum())
        raw = raw[drop.isna()].copy()
        stats["kept_no_iv"] = int((raw["iv"].isna() | (raw["iv"] <= 0)).sum())

        if self.strike_window_pct and spot > 0 and len(raw):
            lo = spot * (1.0 - self.strike_window_pct)
            hi = spot * (1.0 + self.strike_window_pct)
            inside = (raw["strike"] >= lo) & (raw["strike"] <= hi)
            stats["dropped_window"] = int((~inside).sum())
            raw = raw[inside].copy()

        if raw.empty:
            raise ValueError(
                f"Webull: chain for {ticker_str} empty after filters "
                f"(was {before} rows)"
            )

        logger.info(
            "Webull %s: экспираций %d/%d (запрошено %d, из базового %d, ошибок %d), "
            "%d запросов → %d строк, %d страйков, spot=%.2f; отброшено: oi<=0 %d, "
            "strike<=0 %d, окном ±%.0f%% %d; без IV (к интерполяции): %d",
            ticker_str, stats["expiries_selected"], stats["expiries_total"],
            stats["expiries_fetched"], stats["expiries_from_baseline"],
            stats["expiries_failed"], stats["requests"], len(raw),
            raw["strike"].nunique(), spot,
            stats["dropped_zero_oi"], stats["dropped_bad_strike"],
            (self.strike_window_pct or 0) * 100, stats["dropped_window"],
            stats["kept_no_iv"],
        )

        # ── 6. Clean via GEXDataLoader ──
        loader = GEXDataLoader(spot=spot, symbol=ticker_str)
        snapshot = loader.load_dataframe(
            raw, as_of=datetime.now(timezone.utc), preserve_expiry=True,
        )
        snapshot.meta["webull"] = {
            **stats,
            "strike_window_pct": self.strike_window_pct,
            "spot": spot,
            "raw_rows": int(before),
        }

        logger.info(
            "Webull %s: очистка %d → %d строк (%d страйков, %d экспираций)",
            ticker_str, len(raw), len(snapshot.chain),
            snapshot.chain["strike"].nunique(),
            int((snapshot.chain["T"] * 365).round().nunique()),
        )

        # ── Cache ──
        if self._redis is not None and self._redis.connected:
            try:
                self._redis.set(ck, snapshot, ex=600)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Webull cache write failed for %s: %s", ticker_str, exc)

        return snapshot

    # ------------------------------------------------------------------ #
    #  Выбор экспираций
    # ------------------------------------------------------------------ #
    def _select_expiries(self, entries: list[dict]) -> list[dict]:
        """Ближайшие ``max_expiries`` экспираций внутри горизонта ``max_days``.

        ``entries`` уже отсортированы по дате. Горизонт режет экспирации,
        которые анализатор всё равно отбросил бы фильтром по ``days``, —
        это чистый выигрыш в числе запросов без потери данных.
        """
        out: list[dict] = []
        for e in entries:
            if self.max_days is not None and e["days"] > self.max_days:
                break
            out.append(e)
            if len(out) >= self.max_expiries:
                break
        if not out and entries:
            # Горизонт отсёк всё → берём ближайшую, иначе анализатор останется
            # без цепочки (то же поведение, что у _filter_by_days).
            logger.warning(
                "Webull: горизонт max_days=%s не покрывает ни одной экспирации — "
                "беру ближайшую (~%d дн)", self.max_days, entries[0]["days"],
            )
            out = [entries[0]]
        budget = _MAX_REQUEST_BUDGET - 1
        if len(out) > budget:
            logger.warning(
                "Webull: %d экспираций урезано до бюджета %d", len(out), budget,
            )
            out = out[:budget]
        return out

    # ------------------------------------------------------------------ #
    #  Фанаут по экспирациям
    # ------------------------------------------------------------------ #
    def _fetch_expiries_parallel(
        self, ticker_id: int, entries: list[dict], ticker: str,
    ) -> list[tuple[dict, Optional[list]]]:
        """Дозапросить каждую экспирацию: POST с ``expireDate`` + ``unSymbol``.

        Ошибка одной экспирации не роняет всю загрузку: возвращается ``None``,
        и эта экспирация просто отсутствует в профиле. Полный отказ всех
        допустим — тогда остаётся то, что дал базовый запрос.
        """
        def _one(e: dict) -> tuple[dict, Optional[list]]:
            body = {
                "count": -1,
                "direction": "all",
                "tickerId": ticker_id,
                "expireDate": e["date"],
                "unSymbol": e["un_symbol"] or ticker,
            }
            try:
                resp = _thread_session().post(
                    _WEBULL_OPTIONS, json=body, headers=_build_headers(),
                    timeout=self.timeout,
                )
                if resp.status_code != 200:
                    logger.warning(
                        "Webull %s: экспирация %s — HTTP %d",
                        ticker, e["date"], resp.status_code,
                    )
                    return e, None
                opts = _options_for_date(resp.json(), e["date"])
                if not opts:
                    logger.warning(
                        "Webull %s: экспирация %s вернулась пустой", ticker, e["date"],
                    )
                    return e, None
                return e, opts
            except requests.RequestException as exc:
                logger.warning(
                    "Webull %s: экспирация %s — сетевая ошибка: %s",
                    ticker, e["date"], exc,
                )
                return e, None

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(_one, entries))

    # ------------------------------------------------------------------ #
    #  Базовый POST с одним повтором
    # ------------------------------------------------------------------ #
    def _post_options(self, payload: dict, ticker: str) -> requests.Response:
        """POST ``option/strategy/list`` с ровно одним повтором на транзиентных ошибках."""
        for attempt in range(2):
            try:
                resp = self._session.post(
                    _WEBULL_OPTIONS, json=payload, headers=_build_headers(),
                    timeout=self.timeout,
                )
                if resp.status_code in _RETRYABLE_STATUS and attempt == 0:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "Webull %s: HTTP %d (попытка %d) — повтор через %.2fs",
                        ticker, resp.status_code, attempt + 1, delay,
                    )
                    time.sleep(delay)
                    continue
                return resp
            except requests.RequestException as exc:
                if attempt == 0:
                    delay = _backoff_delay(attempt)
                    logger.warning(
                        "Webull %s: сетевая ошибка (попытка %d): %s — повтор через %.2fs",
                        ticker, attempt + 1, exc, delay,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Webull network error for {ticker}: {exc}") from exc

        # Недостижимо (цикл всегда возвращает или бросает); для type-checker.
        raise RuntimeError(f"Webull unexpected retry exit for {ticker}")


# ====================================================================== #
#  Разбор ответа
# ====================================================================== #
def _parse_expiry_entries(expire_list: list, now: pd.Timestamp) -> list[dict]:
    """Распарсить ``expireDateList`` → список словарей, отсортированных по дате.

    Ключи словаря: ``date``, ``days``, ``T``, ``un_symbol``, ``options``,
    ``has_oi`` (есть ли в базовом ответе строки с OI > 0).
    """
    out: list[dict] = []
    for exp in expire_list:
        if not isinstance(exp, dict):
            continue
        from_dt = exp.get("from", {})
        date_str = from_dt.get("date", "") if isinstance(from_dt, dict) else ""
        options = exp.get("data", [])
        if not date_str or not options or not isinstance(options, list):
            continue
        try:
            exp_ts = pd.Timestamp(date_str)
            days = max((exp_ts - now).total_seconds() / 86400.0, 0.0)
        except (ValueError, TypeError):
            continue
        un_symbol = ""
        for o in options:
            s = o.get("unSymbol")
            if s:
                un_symbol = str(s)
                break
        has_oi = any(_parse_float(o.get("openInterest")) > 0 for o in options)
        out.append({
            "date": date_str,
            "days": days,
            "T": max(days / 365.0, 1e-6),
            "un_symbol": un_symbol,
            "options": options,
            "has_oi": has_oi,
        })
    out.sort(key=lambda e: e["days"])
    return out


def _options_for_date(data: dict, date_str: str) -> list:
    """Достать строки запрошенной экспирации из ответа.

    Webull всё равно возвращает **весь** список экспираций (замер: 33 из 33),
    но заполненной оказывается только запрошенная — поэтому ищем по дате,
    а не берём первый элемент.
    """
    for exp in data.get("expireDateList", []) or []:
        if not isinstance(exp, dict):
            continue
        from_dt = exp.get("from", {})
        d = from_dt.get("date", "") if isinstance(from_dt, dict) else ""
        if d == date_str:
            return exp.get("data", []) or []
    # Дата не совпала (сдвиг формата?) — если вернулась ровно одна, берём её.
    lst = data.get("expireDateList", []) or []
    if len(lst) == 1 and isinstance(lst[0], dict):
        return lst[0].get("data", []) or []
    return []


def _rows_from_options(options: list, T: float) -> list[dict]:
    """Сырые строки цепочки из списка опционов одной экспирации."""
    rows: list[dict] = []
    for opt in options:
        oi = _parse_float(opt.get("openInterest"))
        iv = _parse_float(opt.get("impVol"))
        strike = _parse_float(opt.get("strikePrice"))
        direction = str(opt.get("direction", "")).lower()
        opt_type = "C" if direction == "call" else "P"
        has_iv = iv > 0
        rows.append({
            "strike": strike,
            "oi": oi,
            "iv": iv if has_iv else float("nan"),
            "type": opt_type,
            "T": T,
        })
    return rows


def _resolve_ticker_id(
    ticker: str, session: requests.Session, timeout: int,
) -> int:
    """Resolve ticker symbol to Webull internal tickerId (with cache).

    Требуется **точное** совпадение символа. Раньше при его отсутствии брался
    первый результат поиска — то есть GEX мог молча считаться по другому
    инструменту (поиск Webull подбирает похожие тикеры). Такая ошибка не
    видна ни в профиле, ни в метриках, поэтому лучше уронить загрузку и дать
    отработать fallback'у на yfinance.
    """
    if ticker in _ticker_id_cache:
        return _ticker_id_cache[ticker]

    params = {
        "keyword": ticker,
        "pageIndex": 1,
        "pageSize": 10,
        "regionId": 6,  # US
    }
    headers = _build_headers()

    last_exc: Optional[Exception] = None
    results: list = []
    for attempt in range(2):
        try:
            resp = session.get(
                _WEBULL_SEARCH, params=params, headers=headers, timeout=timeout,
            )
            if resp.status_code in _RETRYABLE_STATUS and attempt == 0:
                delay = _backoff_delay(attempt)
                logger.warning(
                    "Webull search %s: HTTP %d — повтор через %.2fs",
                    ticker, resp.status_code, delay,
                )
                time.sleep(delay)
                continue
            resp.raise_for_status()
            results = resp.json().get("data", []) or []
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == 0:
                delay = _backoff_delay(attempt)
                logger.warning(
                    "Webull search %s: сетевая ошибка (%s) — повтор через %.2fs",
                    ticker, exc, delay,
                )
                time.sleep(delay)
                continue
    else:
        raise RuntimeError(
            f"Webull ticker lookup failed for '{ticker}': {last_exc}"
        )

    wanted = ticker.upper()
    for item in results:
        symbol = str(item.get("symbol", "")).upper()
        dis_symbol = str(item.get("disSymbol", "")).upper()
        if symbol == wanted or dis_symbol == wanted:
            try:
                tid = int(item["tickerId"])
            except (KeyError, TypeError, ValueError):
                continue
            _ticker_id_cache[ticker] = tid
            return tid

    seen = [str(i.get("symbol", "")) for i in results[:5]]
    raise ValueError(
        f"Webull: точного совпадения для '{ticker}' нет "
        f"(ответ поиска: {seen}) — не подменяю тикер похожим"
    )


def _parse_float(val) -> float:
    """Parse numeric value from Webull response (can be string or number)."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val))
    except (ValueError, TypeError):
        return 0.0


def _classify_drop(oi: float, strike: float) -> Optional[str]:
    """Причина отбрасывания строки (None = оставить).

    Отбрасываются только контракты, которые физически не дают вклада в GEX:
    без открытого интереса (``oi <= 0``) или с некорректным страйком
    (``strike <= 0``). Порядок проверки — OI, затем страйк.

    Отсутствующая IV причиной отбрасывания **не** является: строки идут в
    цепочку с ``iv = NaN``, а :meth:`GEXDataLoader._interpolate_iv`
    восстанавливает волатильность по smile внутри группы ``(type, T)``.
    """
    if oi <= 0:
        return "zero_oi"
    if strike <= 0:
        return "bad_strike"
    return None
