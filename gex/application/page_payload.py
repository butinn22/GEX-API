"""Payload страницы: свежесть по классу страницы, etag и 304 (ring: application).

Задача заказчика звучала так: «кэшировать данные, которые пользователь видел на странице».
Пункт назначения — второй визит должен отдавать payload мгновенно, а свежесть добираться
фоном. Здесь собирается недостающая часть: **политика свежести** (из
:mod:`gex.domain.freshness`) + **хранилище** (:class:`gex.ports.cache.CachePort`) складываются
в одну операцию, которой может пользоваться роутер.

Почему ключ строит вызывающий
-----------------------------
Форма ключа живёт в одном месте — ``gex.adapters.cache.keys`` (правила R6/R7 это стерегут),
а слой application не имеет права импортировать adapters. Поэтому ключ передаётся функцией
(``key_factory``), которую подставляет композиционный корень: так формат ключа остаётся
единственным, а use-case не знает ни про Redis, ни про схему ключей.

Роль `market_open`
------------------
Окна свежести зависят от сессии: вне торгов часы не делают данные свежее, поэтому окна
растягиваются (``fresh_off``/``stale_off``). Проверка сессии — в домене, а здесь только
подстановка значения: тест может задать сессию явно и не зависеть от текущих часов.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from gex.domain.freshness import market_open_now, page_class, policy_for_page
from gex.ports.cache import CachedPayload, CacheStatus, CachePort

logger = logging.getLogger(__name__)

__all__ = ["PageNotFoundError", "PageResult", "PagePayloadService"]


class PageNotFoundError(KeyError):
    """Страница не описана в ``PAGE_CLASS`` — TTL нельзя задавать литералом."""


@dataclass(frozen=True)
class PageResult:
    """Результат для роутера: значение + всё, что нужно для заголовков ответа."""

    page: str
    payload: Any
    status: CacheStatus
    etag: Optional[str]
    fresh: int
    stale_max: int
    version: int = 1
    computing: bool = False

    def is_not_modified(self, client_etag: Optional[str]) -> bool:
        """Совпадает ли etag клиента с текущим (для ответа 304 без тела).

        Сравнение идёт по точному значению и только при непустом etag: пустой etag
        (например, значение не пережило запись в кэш) не должен превращаться в «не изменилось».
        """
        return bool(client_etag) and bool(self.etag) and client_etag == self.etag

    def headers(self) -> dict[str, str]:
        """Заголовки, которые честно описывают состояние ответа.

        ``X-Cache`` и ``X-Computing`` нужны фронту: по ним он рисует индикатор ревалидации
        вместо того, чтобы гадать по времени ответа.
        """
        headers = {
            "X-Cache": self.status.value,
            "X-Cache-Fresh": str(self.fresh),
            "X-Cache-Stale-Max": str(self.stale_max),
        }
        if self.etag:
            headers["ETag"] = f'"{self.etag}"'
        if self.computing:
            headers["X-Computing"] = "1"
        if self.status == CacheStatus.STALE:
            headers["Cache-Control"] = f"max-age={self.fresh}"
        return headers


class PagePayloadService:
    """Отдаёт payload страницы по политике её класса свежести."""

    def __init__(
        self,
        port: CachePort,
        key_factory: Callable[..., str],
        *,
        market_open: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._port = port
        self._key_factory = key_factory
        self._market_open = market_open or market_open_now

    def get(
        self,
        page: str,
        compute: Callable[[], Any],
        *,
        params: Iterable[Any] = (),
        timeframe: Optional[str] = None,
        market_open: Optional[bool] = None,
        max_wait_ms: Optional[int] = None,
        key_suffix: Optional[str] = None,
    ) -> PageResult:
        """Payload страницы: из кэша или вычисленный.

        Parameters
        ----------
        page : str
            Имя страницы из ``PAGE_CLASS`` (например ``"gexcone"``). Неизвестное имя —
            :class:`PageNotFoundError`: молчаливая подстановка TTL «по умолчанию» и была
            причиной того, что 10 мест держали литерал ``ttl=600``.
        compute : callable
            Как получить значение. Вызывается только при промахе или для фонового обновления.
        params :
            Части ключа (тикер, таймфрейм, параметры расчёта).
        market_open : bool, optional
            Явное состояние сессии (тесты и предпрогрев). ``None`` — определить по часам.
        """
        key = self._build_key(page, params, key_suffix=key_suffix)
        try:
            policy = policy_for_page(page, market_open=self._session(market_open))
        except KeyError as exc:
            raise PageNotFoundError(str(exc)) from None
        if not policy.session_aware and timeframe:
            logger.debug("Страница %s не зависит от сессии, timeframe=%s не влияет", page, timeframe)

        cached: CachedPayload = self._port.get(
            key,
            fresh=policy.fresh,
            stale_max=policy.stale_max,
            compute=compute,
            max_wait_ms=max_wait_ms,
        )
        return PageResult(
            page=page,
            payload=cached.value,
            status=cached.status,
            etag=cached.etag,
            fresh=cached.fresh,
            stale_max=cached.stale_max,
            version=cached.version,
            computing=cached.computing,
        )

    def page_class(self, page: str) -> str:
        """Класс свежести страницы (для метрик и для админки)."""
        try:
            return page_class(page).value
        except KeyError as exc:
            raise PageNotFoundError(str(exc)) from None

    # ------------------------------------------------------------------ #
    #  Внутреннее
    # ------------------------------------------------------------------ #
    def _session(self, market_open: Optional[bool]) -> bool:
        return self._market_open() if market_open is None else market_open

    def _build_key(self, page: str, params: Iterable[Any], *, key_suffix: Optional[str]) -> str:
        parts = [page, *params]
        if key_suffix:
            parts.append(key_suffix)
        return self._key_factory(*parts)
