"""Политика свежести данных — канон (ring: domain, чистый Python: ни I/O, ни numpy).

Зачем в домене
--------------
Требование продукта — «пользователь не видит спиннер и не ждёт провайдера» — держится на одном
правиле: **страница отдаёт последнее известное значение, а свежесть добирается фоном**. Значит
«сколько это „свежее“» должно быть одним явным решением, а не 11 литералами ``ttl=600``,
разбросанными по роутерам (аудит 07: PC-01, PC-07).

Модель
------
* :class:`Freshness` — класс данных: ``LIVE_INTRADAY`` (котировки/свечи), ``PERIODIC_SCHEDULED``
  (расписание, например ширина IMOEX в 23:00/08:00 MSK), ``COMPUTED_SLOW`` (TA/cone/extended/SEC);
* :class:`Policy` — окна: ``fresh`` (данные считаются свежими, фон не дёргаем), ``stale_max``
  (предел «можно отдать устаревшее»), ``prewarm`` (желаемый период прогрева);
* **session-aware**: вне торговой сессии окна растягиваются — данные всё равно не меняются, а
  провайдера дёргать незачем (это и есть экономия лимитов).

Модуль чистый: состояние рынка передаётся аргументом (``market_open``) либо считается
:func:`session_open` по таблице сессий — никаких обращений к времени изнутри (тестируемо).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

__all__ = [
    "Freshness",
    "Policy",
    "MarketSession",
    "SESSIONS",
    "CLASS_DEFAULTS",
    "TIMEFRAME_POLICIES",
    "PAGE_CLASS",
    "PERIODIC_PAGES",
    "MSK",
    "session_open",
    "market_open_now",
    "policy_for_timeframe",
    "policy_for_page",
    "page_class",
    "is_periodic_page",
    "all_pages",
]

MSK = timezone(timedelta(hours=3))  # Moscow, UTC+3


class Freshness(str, Enum):
    """Класс свежести данных — определяет, как страница отдаётся и обновляется."""

    LIVE_INTRADAY = "live_intraday"
    PERIODIC_SCHEDULED = "periodic_scheduled"
    COMPUTED_SLOW = "computed_slow"


@dataclass(frozen=True)
class Policy:
    """Окна свежести в секундах.

    ``fresh``      — до этого возраста данные отдаются без фонового пересчёта;
    ``stale_max``  — до этого возраста их ещё можно отдать (иначе — приоритетный пересчёт);
    ``prewarm``    — как часто прогревать страницу фоном;
    ``session_aware`` — растягивать ли окна вне торговой сессии.
    """

    fresh: int
    stale_max: int
    prewarm: int
    session_aware: bool = True

    def __post_init__(self) -> None:
        if self.fresh <= 0 or self.stale_max < self.fresh:
            raise ValueError(f"некорректные окна: fresh={self.fresh}, stale_max={self.stale_max}")


@dataclass(frozen=True)
class MarketSession:
    """Торговая сессия в минутах от полуночи MSK (``start`` включительно, ``end`` исключительно)."""

    name: str
    start_min: int
    end_min: int

    def is_open(self, when_msk: datetime) -> bool:
        if when_msk.weekday() >= 5:  # сб/вс
            return False
        minutes = when_msk.hour * 60 + when_msk.minute
        return self.start_min <= minutes < self.end_min


#: US-сессия 09:30–16:00 ET = 16:30–23:00 MSK; MOEX: 10:00–18:40 MSK (основная сессия)
SESSIONS: dict[str, MarketSession] = {
    "us": MarketSession("us", 16 * 60 + 30, 23 * 60),
    "moex": MarketSession("moex", 10 * 60, 18 * 60 + 40),
}


def session_open(session: str, when: datetime | None = None) -> bool:
    """Открыта ли сессия ``session`` в момент ``when`` (по умолчанию — сейчас)."""
    if session not in SESSIONS:
        raise KeyError(f"неизвестная сессия {session!r}; доступны {sorted(SESSIONS)}")
    moment = when or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return SESSIONS[session].is_open(moment.astimezone(MSK))


def market_open_now(markets: tuple[str, ...] = ("us", "moex"), when: datetime | None = None) -> bool:
    """Открыт ли хотя бы один из рынков (консервативно: если открыт хоть один — данные «живые»)."""
    return any(session_open(m, when) for m in markets)


class _Windows:
    """Пара окон «в сессии» / «вне сессии» для одного типа данных."""

    __slots__ = ("fresh_session", "fresh_off", "stale_session", "stale_off", "prewarm")

    def __init__(self, fresh_session: int, fresh_off: int, stale_session: int, stale_off: int, prewarm: int) -> None:
        self.fresh_session = fresh_session
        self.fresh_off = fresh_off
        self.stale_session = stale_session
        self.stale_off = stale_off
        self.prewarm = prewarm


#: Политика по таймфреймам свечей: (в сессии / вне сессии), сек.
TIMEFRAME_POLICIES: dict[str, _Windows] = {
    "1m": _Windows(30, 900, 300, 7200, 30),
    "5m": _Windows(30, 900, 300, 7200, 30),
    "15m": _Windows(120, 1800, 900, 21600, 120),
    "30m": _Windows(120, 1800, 900, 21600, 120),
    "1h": _Windows(120, 1800, 900, 21600, 120),
    "2h": _Windows(120, 1800, 900, 21600, 120),
    "4h": _Windows(600, 7200, 3600, 43200, 600),
    "1d": _Windows(300, 86400, 7200, 86400, 900),
}

#: Политика по классам (используется, когда нужен только класс, а не конкретная страница).
CLASS_DEFAULTS: dict[Freshness, _Windows] = {
    Freshness.LIVE_INTRADAY: _Windows(30, 900, 300, 7200, 30),
    Freshness.COMPUTED_SLOW: _Windows(180, 1800, 3600, 43200, 300),
    Freshness.PERIODIC_SCHEDULED: _Windows(6 * 3600, 6 * 3600, 48 * 3600, 48 * 3600, 0),
}

#: Страница (ключ из URL/меню) → класс свежести. Единая точка правды вместо литералов в роутерах.
PAGE_CLASS: dict[str, Freshness] = {
    # живые данные
    "quote": Freshness.LIVE_INTRADAY,
    "ohlcv": Freshness.LIVE_INTRADAY,
    "candles": Freshness.LIVE_INTRADAY,
    "live": Freshness.LIVE_INTRADAY,
    # тяжёлые вычисления
    "ta": Freshness.COMPUTED_SLOW,
    "ta-structure": Freshness.COMPUTED_SLOW,
    "gex": Freshness.COMPUTED_SLOW,
    "gexcone": Freshness.COMPUTED_SLOW,
    "extended": Freshness.COMPUTED_SLOW,
    "chains": Freshness.COMPUTED_SLOW,
    "cone": Freshness.COMPUTED_SLOW,
    "trendlines": Freshness.COMPUTED_SLOW,
    "macd": Freshness.COMPUTED_SLOW,
    "hybrid-trend": Freshness.COMPUTED_SLOW,
    "novel-candles": Freshness.COMPUTED_SLOW,
    "rsi-novel": Freshness.COMPUTED_SLOW,
    "signal-scanner": Freshness.COMPUTED_SLOW,
    "auto-scanner": Freshness.COMPUTED_SLOW,
    "scanner": Freshness.COMPUTED_SLOW,
    "breadth": Freshness.COMPUTED_SLOW,
    "sector": Freshness.COMPUTED_SLOW,
    "commodity": Freshness.COMPUTED_SLOW,
    "sec": Freshness.COMPUTED_SLOW,
    "sec-forecast": Freshness.COMPUTED_SLOW,
    "valuation": Freshness.COMPUTED_SLOW,
    # расписание
    "breadth-imoex": Freshness.PERIODIC_SCHEDULED,
}

#: Страницы, обновляемые строго по расписанию (23:00 / 08:00 MSK) — клиент не должен видеть «обновление».
PERIODIC_PAGES: frozenset[str] = frozenset(p for p, cls in PAGE_CLASS.items() if cls is Freshness.PERIODIC_SCHEDULED)

#: Окна конкретных страниц (переопределяют CLASS_DEFAULTS).
PAGE_WINDOWS: dict[str, _Windows] = {
    "ta": _Windows(60, 1800, 1800, 21600, 120),
    "ta-structure": _Windows(60, 1800, 1800, 21600, 120),
    "gex": _Windows(180, 1800, 3600, 43200, 300),
    "gexcone": _Windows(180, 1800, 3600, 43200, 300),
    "extended": _Windows(180, 1800, 3600, 43200, 300),
    "chains": _Windows(180, 1800, 3600, 43200, 300),
    "cone": _Windows(600, 21600, 21600, 86400, 1800),
    "trendlines": _Windows(180, 1800, 3600, 43200, 300),
    "macd": _Windows(120, 1800, 1800, 21600, 300),
    "hybrid-trend": _Windows(120, 1800, 1800, 21600, 300),
    "novel-candles": _Windows(120, 1800, 1800, 21600, 300),
    "rsi-novel": _Windows(120, 1800, 1800, 21600, 300),
    "signal-scanner": _Windows(60, 1800, 900, 21600, 60),
    "auto-scanner": _Windows(60, 1800, 900, 21600, 60),
    "scanner": _Windows(60, 1800, 900, 21600, 60),
    "breadth": _Windows(300, 3600, 3600, 43200, 600),
    "sector": _Windows(600, 3600, 7200, 43200, 1800),
    "commodity": _Windows(300, 3600, 3600, 43200, 600),
    "sec": _Windows(43200, 86400, 604800, 604800, 43200),
    "sec-forecast": _Windows(43200, 86400, 604800, 604800, 43200),
    "valuation": _Windows(43200, 86400, 604800, 604800, 43200),
    "quote": _Windows(3, 60, 30, 600, 5),
    "ohlcv": _Windows(30, 900, 300, 7200, 30),
    "candles": _Windows(30, 900, 300, 7200, 30),
    "live": _Windows(3, 60, 30, 600, 5),
    "breadth-imoex": _Windows(6 * 3600, 6 * 3600, 48 * 3600, 48 * 3600, 0),
}


def page_class(page: str) -> Freshness:
    """Класс свежести страницы (``KeyError`` с понятным текстом, если страница не описана)."""
    try:
        return PAGE_CLASS[page]
    except KeyError as exc:  # noqa: F841
        raise KeyError(
            f"страница {page!r} не описана в PAGE_CLASS — добавьте её, чтобы TTL не задавался литералом"
        ) from None


def is_periodic_page(page: str) -> bool:
    return page in PERIODIC_PAGES


def _windows_for(page: str | None, timeframe: str | None) -> _Windows:
    if page is not None:
        page_class(page)  # валидация
        if page in PAGE_WINDOWS:
            return PAGE_WINDOWS[page]
        return CLASS_DEFAULTS[PAGE_CLASS[page]]
    if timeframe is not None:
        if timeframe in TIMEFRAME_POLICIES:
            return TIMEFRAME_POLICIES[timeframe]
        return CLASS_DEFAULTS[Freshness.LIVE_INTRADAY]
    return CLASS_DEFAULTS[Freshness.COMPUTED_SLOW]


def policy_for_page(page: str, *, market_open: bool = True) -> Policy:
    """Политика для страницы: в сессии — узкие окна, вне сессии — растянутые."""
    windows = _windows_for(page, None)
    fresh = windows.fresh_session if market_open else windows.fresh_off
    stale = windows.stale_session if market_open else windows.stale_off
    stale = max(stale, fresh)
    return Policy(fresh=fresh, stale_max=stale, prewarm=windows.prewarm, session_aware=True)


def policy_for_timeframe(timeframe: str, *, market_open: bool = True) -> Policy:
    """Политика для свечей заданного таймфрейма."""
    windows = _windows_for(None, timeframe)
    fresh = windows.fresh_session if market_open else windows.fresh_off
    stale = windows.stale_session if market_open else windows.stale_off
    return Policy(fresh=fresh, stale_max=max(stale, fresh), prewarm=windows.prewarm, session_aware=True)


def all_pages() -> tuple[str, ...]:
    """Все описанные страницы (для тестов полноты и для прогрева)."""
    return tuple(sorted(PAGE_CLASS))
