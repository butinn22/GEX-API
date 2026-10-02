"""GeoIP-определение страны/языка для мультиязычной версии сайта.

Принцип работы
--------------
1. По IP клиента определяется страна через бесплатный HTTP API ip-api.com
   (без ключа, лимит ~45 запросов/мин). Результат кэшируется в Redis
   (``gex:geoip:{ip}``, TTL 24 часа).
2. Страна → язык: явные списки русскоязычных и англоязычных стран.
3. Если страну определить нельзя (локальный IP, API недоступен) — fallback
   на заголовок ``Accept-Language``, затем настройка ``GEOIP_DEFAULT_LANG``
   (по умолчанию ``ru`` — текущее поведение сайта не ломается).

Все внешние запросы — с таймаутом и graceful degradation: сбой GeoIP
никогда не роняет запрос и не блокирует пользователя.

Конфиг (.env)::

    GEOIP_ENABLED=true        # false — полностью отключить определение
    GEOIP_DEFAULT_LANG=ru     # язык по умолчанию (при невозможности определить)
"""
from __future__ import annotations

import ipaddress
import logging
from typing import Optional

from gex.auth.config import settings

logger = logging.getLogger(__name__)

# ── Страны с приоритетом русского языка ────────────────────────────────
RU_COUNTRIES = {"RU", "BY", "KZ", "UA", "AM", "AZ", "GE", "KG", "MD", "TJ", "TM", "UZ"}

# ── Страны с приоритетом английского языка (основная англоязычная аудитория) ──
EN_COUNTRIES = {
    "US", "GB", "CA", "AU", "NZ", "IE", "ZA", "SG", "IN", "PH", "MY",
    "DE", "FR", "ES", "IT", "NL", "PL", "SE", "NO", "FI", "DK", "CH",
    "AT", "BE", "PT", "GR", "CZ", "RO", "BG", "HU", "AE", "SA", "IL",
    "JP", "KR", "HK", "TW", "TH", "ID", "VN", "BR", "MX", "AR", "CL",
    "CO", "PE", "TR", "NG", "KE", "EG", "MA",
}

_PRIVATE_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
)


def _normalize_ip(ip: str | None) -> str | None:
    """Нормализовать IP; приватные/локальные вернуть как None (не запрашиваем API)."""
    if not ip:
        return None
    ip = ip.strip().lower()
    if ip.startswith("::ffff:"):
        ip = ip.split("::ffff:", 1)[1]
    if ip in ("::1", "localhost"):
        return None
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None
    if addr.is_private or addr.is_loopback or addr.is_link_local:
        return None
    return str(addr)


def _client_ip(request) -> Optional[str]:
    """IP клиента с учётом прокси-заголовков (nginx: X-Forwarded-For / X-Real-IP)."""
    for header in ("x-forwarded-for", "x-real-ip"):
        val = request.headers.get(header)
        if val:
            first = val.split(",")[0].strip()
            if first:
                return _normalize_ip(first)
    if request.client:
        return _normalize_ip(request.client.host)
    return None


def _country_from_accept_language(accept_language: str | None) -> str | None:
    """Грубая прикидка страны по Accept-Language (fallback)."""
    if not accept_language:
        return None
    lang = accept_language.split(",")[0].split(";")[0].strip().lower()
    if lang.startswith("ru"):
        return "RU"
    if lang.startswith("en"):
        return "US"
    return None


def _country_from_ip(ip: str) -> Optional[str]:
    """Запросить страну по IP через ip-api.com (с Redis-кэшем)."""
    redis = None
    cache_key = f"gex:geoip:{ip}"

    try:
        from gex.adapters.cache.redis_client import get_redis
        redis = get_redis()
        if redis and redis.connected:
            cached = redis.get(cache_key)
            if cached is not None:
                try:
                    return cached.decode("utf-8") if isinstance(cached, bytes) else str(cached)
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        redis = None

    country = None
    try:
        import json
        import urllib.request

        url = f"http://ip-api.com/json/{ip}?fields=status,countryCode&lang=en"
        req = urllib.request.Request(url, headers={"User-Agent": "GEX-Analytics/1.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("status") == "success" and data.get("countryCode"):
            country = str(data["countryCode"]).upper()
    except Exception as exc:  # noqa: BLE001
        logger.debug("GeoIP lookup failed for %s: %s", ip, exc)

    if country:
        try:
            if redis and redis.connected:
                redis.set(cache_key, country, ex=24 * 3600)
        except Exception:  # noqa: BLE001
            pass
    return country


def detect_language(ip: str | None, accept_language: str | None = None) -> str:
    """Определить язык сайта по IP/заголовкам.

    Порядок: страна по IP → Accept-Language → GEOIP_DEFAULT_LANG.
    """
    if not settings.GEOIP_ENABLED:
        return settings.GEOIP_DEFAULT_LANG or "ru"

    # 1. По IP
    norm_ip = _normalize_ip(ip)
    if norm_ip:
        country = _country_from_ip(norm_ip)
        if country:
            if country in RU_COUNTRIES:
                return "ru"
            if country in EN_COUNTRIES:
                return "en"
            # Страна определена, но не в списках: нейтральные получают EN
            return "en"

    # 2. По Accept-Language
    country_hint = _country_from_accept_language(accept_language)
    if country_hint == "RU":
        return "ru"
    if country_hint == "US":
        return "en"

    # 3. Дефолт (текущее поведение)
    return settings.GEOIP_DEFAULT_LANG or "ru"


def geoip_payload(request) -> dict:
    """Ответ для GET /geoip: страна, язык, источник определения."""
    ip = _client_ip(request)
    accept_language = request.headers.get("accept-language")
    language = detect_language(ip, accept_language)
    country = None
    norm_ip = _normalize_ip(ip)
    if norm_ip:
        country = _country_from_ip(norm_ip)
    return {
        "ip": norm_ip or None,
        "country_code": country,
        "language": language,
        "geoip_enabled": settings.GEOIP_ENABLED,
    }
