"""DemoScopeMiddleware — серверная граница demo-режима («Просмотр демо»).

Demo-токены несут claim ``demo=1`` (ставится при выпуске токена для
DEMO_EMAIL, см. gex.auth.service._demo_claim). Этот middleware — ЕДИНСТВЕННАЯ
точка ограничения: demo-пользователю разрешено только

  * служебное: GET /health, GET /auth/me, GET /auth/logout,
    GET|PUT /auth/settings/dashboard (личная видимость карточек ES);
  * чтение данных ТОЛЬКО по тикеру ES (GET):
    /ohlcv/…, /ta/…, /gex/…, /gex/{t}/profile, /gexcone/…, /live/gex/…,
    /trendlines/…, /ext/gex/…

Всё остальное (сканеры, авто-сигналы, ширина/товары, admin, payments,
любые другие тикеры) — 403. Не-demo токены и анонимные запросы проходят
прозрачно (их ограничивают штатные барьеры подписки).

ИБ-заметка: подделка demo-claim невозможна без JWT_SECRET; demo-аккаунт
нельзя зарегистрировать/сменить email на него; вход — только /auth/demo.
Параметры пути проверяются целиком (тикер — первый сегмент после
префикса), т.е. /ohlcv/NVDA?… или /ta/TSLA/… не просочатся.
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse

from gex.auth.service import decode_token

# Только этот тикер виден в demo (должен совпадать с DEMO_TICKER на фронте).
DEMO_TICKER = "es"

# Служебные пути, доступные demo-аккаунту без тикера.
_EXACT_ALLOWED = {"/health", "/auth/me", "/auth/logout", "/payment/plans"}
_SETTINGS_DASHBOARD = "/auth/settings/dashboard"

# GET-префиксы данных: разрешён только тикер DEMO_TICKER (первый сегмент).
_READ_PREFIXES = (
    "/ohlcv/",
    "/ta/",
    "/gex/",
    "/gexcone/",
    "/live/gex/",
    "/trendlines/",
    "/ext/gex/",
    "/novel-candles/",
    "/rsi-novel/",
)

_DENY_MSG = (
    "Демо-режим открывает только просмотр данных ES "
    "(тех. анализ, GEX, GEX Cone). Раздел недоступен в демо."
)


def _demo_allowed(method: str, path: str) -> bool:
    if path in _EXACT_ALLOWED:
        return True
    if path == _SETTINGS_DASHBOARD and method in ("GET", "PUT"):
        return True
    if method != "GET":
        return False
    for prefix in _READ_PREFIXES:
        if path.startswith(prefix):
            ticker = path[len(prefix):].split("/", 1)[0].strip().lower()
            return ticker == DEMO_TICKER
    return False


class DemoScopeMiddleware(BaseHTTPMiddleware):
    """Если токен demo — пускаем только по allowlist-у, иначе 403."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        auth = request.headers.get("authorization") or ""
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if not token:
            return await call_next(request)

        try:
            payload = decode_token(token)
        except Exception:  # noqa: BLE001 — невалидный токен разберёт штатная аутентификация
            return await call_next(request)

        if payload.get("demo") != 1:
            return await call_next(request)

        method = request.method.upper()
        path = request.url.path
        if method == "OPTIONS" or _demo_allowed(method, path):
            return await call_next(request)

        return JSONResponse(status_code=403, content={"detail": _DENY_MSG})
