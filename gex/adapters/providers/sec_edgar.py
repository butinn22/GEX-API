"""SEC EDGAR API client (company fundamentals).

Единая точка входа в EDGAR для модуля фундаментального анализа:

* ``get_ticker_to_cik()`` — реестр тикер → CIK (``company_tickers.json``),
  кэшируется в Redis на сутки (все ~11k тикеров, ~1 МБ pickle);
* ``get_company_facts(cik)`` — XBRL-факты компании
  (``data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json``).

Правила работы с SEC (иначе 403/бан):

* обязательный User-Agent вида ``Имя Контакт@домен`` (``SEC_USER_AGENT``);
* rate limit: SEC разрешает max 10 req/s — держим 5 req/s через глобальный
  :class:`~gex.rate_limiter.RateLimiter` (bucket ``sec``);
* ретраи с backoff на 429/5xx (до 3 попыток).

Сырые company facts (JSON до ~7 МБ на компанию) кэшируются в Redis сжатыми
(zlib, TTL ``SEC_FACTS_TTL_HOURS``): без этого каждый уникальный набор
параметров прогноза перекачивал 7 МБ из EDGAR заново (см. security-аудит
2026-09-04, публичный DoS на /companies/{ticker}/forecast). Свежесть
конечного ответа дополнительно обеспечивается SWR-кэшем в
:mod:`gex.sec_fundamentals`.
"""
from __future__ import annotations

import json
import logging
import time
import zlib
from typing import Any

import requests

from gex.auth.config import settings
from gex.adapters.ratelimit.rate_limiter import get_rate_limiter
from gex.adapters.cache.redis_client import RedisClient, deserialize_value, get_redis

logger = logging.getLogger(__name__)

#: Реестр тикеров SEC (ticker → cik_str)
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
#: XBRL-факты компании по CIK (10 цифр, с ведущими нулями)
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

#: TTL кэша реестра тикеров (сек) — ~1 МБ, обновляем раз в сутки
TICKER_MAP_TTL = 24 * 3600

#: Попытки HTTP-запроса при 429/5xx, паузы между ними (сек)
_MAX_ATTEMPTS = 3
_RETRY_SLEEP = (2, 4)


class EdgartError(RuntimeError):
    """Ошибка взаимодействия с SEC EDGAR (→ 502 в роутере)."""


def _headers() -> dict[str, str]:
    """Заголовки для data.sec.gov: обязательный UA + gzip."""
    return {
        "User-Agent": settings.SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }


def _http_get(url: str, timeout: int) -> Any:
    """GET с rate-limit'ом, ретраями и понятными ошибками.

    Raises
    ------
    EdgartError
        Сеть недоступна, 403 (плохой UA), 404, или исчерпаны ретраи.
    """
    limiter = get_rate_limiter()
    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        limiter.wait("sec")  # блокируемся до токена (5 req/s)
        try:
            resp = requests.get(url, headers=_headers(), timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            logger.warning("SEC EDGAR request failed (attempt %s): %s", attempt + 1, exc)
            time.sleep(_RETRY_SLEEP[min(attempt, len(_RETRY_SLEEP) - 1)])
            continue

        if resp.status_code == 403:
            raise EdgartError(
                "SEC EDGAR вернул 403: проверьте SEC_USER_AGENT "
                f"(формат 'Имя Контакт@домен'). URL: {url}"
            )
        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = EdgartError(f"SEC EDGAR HTTP {resp.status_code}")
            logger.warning("SEC EDGAR throttled (attempt %s): HTTP %s", attempt + 1, resp.status_code)
            time.sleep(_RETRY_SLEEP[min(attempt, len(_RETRY_SLEEP) - 1)])
            continue
        if resp.status_code != 200:
            raise EdgartError(f"SEC EDGAR HTTP {resp.status_code} для {url}")

        return resp.json()

    raise EdgartError(f"SEC EDGAR недоступен после {_MAX_ATTEMPTS} попыток: {last_error}")


# ══════════════════════════════════════════════════════════════════════ #
#  Ticker → CIK resolver
# ══════════════════════════════════════════════════════════════════════ #
def get_ticker_to_cik(redis: RedisClient | None = None) -> dict[str, str]:
    """Получить реестр тикер → CIK (10 цифр, с ведущими нулями).

    Кэшируется в Redis на сутки (``gex:sec:ticker_map``). При недоступном
    Redis или ошибке десериализации — тянем из SEC напрямую.
    """
    redis = redis if redis is not None else get_redis()
    key = "gex:sec:ticker_map"

    if redis is not None and redis.connected:
        cached = redis.get(key)
        if cached is not None:
            try:
                return deserialize_value(cached)
            except Exception as exc:
                logger.debug("Ticker map cache deserialize error: %s — refetching", exc)

    data = _http_get(TICKERS_URL, timeout=30)
    result: dict[str, str] = {}
    for item in data.values():
        ticker = str(item.get("ticker", "")).upper()
        cik = str(item.get("cik_str", "")).zfill(10)
        if ticker and cik != "0000000000":
            result[ticker] = cik

    if redis is not None and redis.connected:
        redis.set(key, result, ex=TICKER_MAP_TTL)
    logger.info("SEC ticker map loaded: %d tickers", len(result))
    return result


# ══════════════════════════════════════════════════════════════════════ #
#  Company facts
# ══════════════════════════════════════════════════════════════════════ #
def get_company_facts(cik: str) -> dict:
    """Получить XBRL-факты компании по CIK (10 цифр).

    Структура ответа: ``{"cik": ..., "entityName": ..., "facts": {"us-gaap": ...}}``.
    Кэшируется в Redis сжатым (zlib) на SEC_FACTS_TTL_HOURS — защита от
    повторных 7МБ-фетчей EDGAR при разных наборах параметров прогноза.
    """
    cik = str(cik).zfill(10)
    facts_key = f"gex:sec:facts:{cik}"

    redis: RedisClient | None = get_redis()
    if redis is not None and redis.connected:
        try:
            blob = redis.get(facts_key)
            if blob is not None:
                raw_bytes = zlib.decompress(blob)
                return json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 — битый кэш → тянем из SEC
            logger.debug("SEC facts cache miss/error for %s: %s", cik, exc)

    url = COMPANY_FACTS_URL.format(cik=cik)
    data = _http_get(url, timeout=60)
    if not isinstance(data, dict):
        raise EdgartError(f"SEC EDGAR вернул некорректные данные для CIK {cik}")

    if redis is not None and redis.connected:
        try:
            blob = zlib.compress(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            redis.set(facts_key, blob, ex=int(settings.SEC_FACTS_TTL_HOURS * 3600))
        except Exception as exc:  # noqa: BLE001 — переполнение/сбой redis не роняет запрос
            logger.warning("SEC facts cache write failed for %s: %s", cik, exc)
    return data
