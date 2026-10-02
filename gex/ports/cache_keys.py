"""Построитель ключей кэша — единственное место, где задаётся форма ключа (ring: ports).

Форма ключа — часть контракта кэша между application и адаптерами хранилища, а не
деталь реализации Redis: модуль чистый (только строки, ни сети, ни SDK, ни I/O) и
живёт в кольце портов. ``gex.adapters.cache.keys`` — совместимый реэкспорт для
уже существующих вызывающих.

Зачем модуль
------------
Ключи собирались «на месте» вызовом ``cache_key("chain", ticker, str(n))``. У такой схемы
три дефекта, и все три наблюдаются в коде до этой итерации:

1. **Провайдер не входит в ключ.** ``gex:chain:BTC:5`` пишут независимо
   :mod:`gex.bybit_fetcher` (опционы Bybit) и :mod:`gex.yf_fetcher` (``BTC-USD`` через
   yfinance) — это разные инструменты, с разными страйками и экспирациями, но один ключ.
   Кто записал первым, тот и «победил»; читатель не может отличить источник.
2. **Два провайдера в одном семействе ключей.** :func:`gex.routers.gexcone_router._vol_stats`
   строит ``gex:hv:{T}`` **до** выбора источника, а затем пишет туда либо HV/ATR из дневных
   свечей MOEX ISS, либо из yfinance. Форма значения одинаковая (``{"hv":…,"atr":…}``),
   поэтому подмена источника ничем не сигнализируется.
3. **Форма ключа размазана по вызывающим.** Опечатка на одной из сторон (писатель/читатель)
   даёт вечный промах кэша, который не виден ни в логах, ни в метриках.

Поэтому форма ключа живёт здесь, а ``provider`` — **обязательный** параметр
провайдер-зависимых ключей. Отсутствие провайдера не «по умолчанию yfinance», а
:class:`CacheKeyError`: неоднозначный ключ невозможно построить.

Схема ключа
-----------
Провайдер-зависимые::

    gex:{kind}:{provider}:{parts...}      gex:chain:bybit:BTC:5

Провайдер-независимые (состояние, очередь, метрики, результат эндпоинта)::

    gex:{kind}:{parts...}                 gex:res:ta:SPY:1000

Разбор keyspace
---------------
``POST_DEPLOY``: расположение сегмента провайдера меняет keyspace, поэтому после выката
кэш холодный. Это осознанная цена: смешивать старую и новую схемы в одном Redis значило бы
сохранить ровно тот дефект, который здесь устраняется. Прогрев — штатный механизм
(:mod:`gex.scheduler`), а не ручная миграция.
"""

from __future__ import annotations

import re
from typing import Final, Optional, Union

# ====================================================================== #
#  Провайдеры
# ====================================================================== #

PROVIDER_YFINANCE: Final = "yfinance"
PROVIDER_BYBIT: Final = "bybit"
PROVIDER_MOEX: Final = "moex_iss"
PROVIDER_WEBULL: Final = "webull"
PROVIDER_SEC: Final = "sec"
PROVIDER_FINNHUB: Final = "finnhub"

KNOWN_PROVIDERS: Final[frozenset[str]] = frozenset(
    {
        PROVIDER_YFINANCE,
        PROVIDER_BYBIT,
        PROVIDER_MOEX,
        PROVIDER_WEBULL,
        PROVIDER_SEC,
        PROVIDER_FINNHUB,
    }
)

#: Синонимы, которые уже встречаются в коде и конфигах (``"moex"``, ``"yf"``).
#: Нормализуются к каноническому имени, иначе один источник получал бы два ключа.
_PROVIDER_ALIASES: Final[dict[str, str]] = {
    "yf": PROVIDER_YFINANCE,
    "yfinance": PROVIDER_YFINANCE,
    "yahoo": PROVIDER_YFINANCE,
    "bybit": PROVIDER_BYBIT,
    "moex": PROVIDER_MOEX,
    "moex_iss": PROVIDER_MOEX,
    "iss": PROVIDER_MOEX,
    "webull": PROVIDER_WEBULL,
    "sec": PROVIDER_SEC,
    "sec_edgar": PROVIDER_SEC,
    "edgar": PROVIDER_SEC,
    "finnhub": PROVIDER_FINNHUB,
}

#: Виды ключей, данные для которых приходят от внешнего провайдера. Для них
#: ``provider`` обязателен, а прямой ``cache_key(...)`` запрещён правилом R6
#: (``scripts/quality/ast_guard.py``). ``commodity:*`` покрывается префиксом.
PROVIDER_SCOPED_KINDS: Final[frozenset[str]] = frozenset({"chain", "ohlcv", "spot", "hv", "bars"})

KEY_ROOT: Final = "gex"

# ``:`` разделяет сегменты, пробел делает ключ нечитаемым в redis-cli/логах.
_FORBIDDEN_IN_PART: Final = re.compile(r"[:\s]")


class CacheKeyError(ValueError):
    """Недопустимая часть ключа или неизвестный провайдер."""


# ====================================================================== #
#  Нормализация частей
# ====================================================================== #
def normalize_provider(provider: Optional[Union[str, object]]) -> str:
    """Каноническое имя провайдера.

    >>> normalize_provider("moex"), normalize_provider("YF"), normalize_provider("moex_iss")
    ('moex_iss', 'yfinance', 'moex_iss')
    """
    if provider is None:
        raise CacheKeyError(
            "провайдер обязателен: ключ без провайдера не различает источники данных "
            f"(известные: {sorted(KNOWN_PROVIDERS)})"
        )
    raw = str(provider).strip().lower().replace("-", "_")
    try:
        return _PROVIDER_ALIASES[raw]
    except KeyError:
        raise CacheKeyError(
            f"неизвестный провайдер {provider!r}; известные: {sorted(KNOWN_PROVIDERS)} "
            f"(синонимы: {sorted(_PROVIDER_ALIASES)})"
        ) from None


def _clean(value: object, *, name: str) -> str:
    """Часть ключа: без ``:`` и пробелов, непустая, без подмены ``None`` на строку."""
    if value is None:
        raise CacheKeyError(f"{name} = None недопустим как часть ключа")
    s = str(value).strip()
    if not s:
        raise CacheKeyError(f"{name} пуст — ключ станет неоднозначным")
    if _FORBIDDEN_IN_PART.search(s):
        raise CacheKeyError(
            f"{name} = {s!r} содержит ':' или пробел — часть ключа склеится с соседней"
        )
    return s


def symbol(value: object) -> str:
    """Тикер/инструмент: верхний регистр.

    Регистр — часть идентичности ключа: ``"rts"`` и ``"RTS"`` сегодня давали **разные**
    ключи для одних и тех же данных, то есть половину промахов кэша.
    """
    return _clean(value, name="symbol").upper()


def timeframe(value: object) -> str:
    """Таймфрейм: нижний регистр (``1H`` → ``1h``); ``all`` — служебный, не трогаем."""
    return _clean(value, name="timeframe").lower()


def clean_segment(value: object, *, name: str = "segment") -> str:
    """Произвольный сегмент ключа, который **не** является провайдером.

    Нужно там, где пространство имён задаёт приложение, а не источник данных: например
    область per-IP лимита (``"auth"``, ``"sec_forecast"``). Проверять её как провайдера
    нельзя — ``normalize_provider("auth")`` корректно отклоняет неизвестное имя, и
    лимитер падал бы при конструировании ключа.
    """
    return _clean(value, name=name).lower()


def count(value: object, *, name: str = "count") -> str:
    """Целое в каноническом виде: ``5`` и ``5.0`` дают один сегмент, а не два ключа."""
    if isinstance(value, bool):
        raise CacheKeyError(f"{name} = {value!r}: bool не является числом-сегментом")
    if isinstance(value, int):
        n = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise CacheKeyError(f"{name} = {value!r}: ожидалось целое")
        n = int(value)
    elif isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        n = int(value.strip())
    else:
        raise CacheKeyError(f"{name} = {value!r}: ожидалось целое число")
    if n < 0:
        raise CacheKeyError(f"{name} = {n}: отрицательное значение")
    return str(n)


def _join(kind: str, parts: list[str]) -> str:
    head = f"{KEY_ROOT}:{kind}"
    return f"{head}:" + ":".join(parts) if parts else head


def _scoped(kind: str, provider: object, *parts: str) -> str:
    """Провайдер-зависимый ключ: провайдер идёт сразу после вида ключа."""
    return _join(kind, [normalize_provider(provider), *parts])


# ====================================================================== #
#  Провайдер-зависимые ключи (``provider`` обязателен)
# ====================================================================== #
def chain_key(symbol_: object, max_expiries: object, *, provider: object) -> str:
    """Опционная цепочка.

    >>> chain_key("btc", 5, provider="bybit")
    'gex:chain:bybit:BTC:5'
    >>> chain_key("btc", 5, provider="yfinance")
    'gex:chain:yfinance:BTC:5'
    """
    return _scoped("chain", provider, symbol(symbol_), count(max_expiries, name="max_expiries"))


def chain_key_v2(symbol_: object, max_expiries: object, scope: object, *, provider: object) -> str:
    """Опционная цепочка с **областью выборки** в ключе (v2).

    Отличие от :func:`chain_key`: сегмент ``scope`` кодирует параметры, которые
    меняют состав цепочки при том же ``max_expiries`` — горизонт в днях и
    страйк-окно. Без него цепочка «1 экспирация с OI» (поведение Webull до
    аудита 2026-09-17) и цепочка «N экспираций, полный OI» легли бы в один
    ключ: читатель не смог бы отличить полную выборку от разреженной.

    >>> chain_key_v2("spy", 8, "d30w50", provider="webull")
    'gex:chain:webull:SPY:8:d30w50'
    """
    return _scoped(
        "chain", provider, symbol(symbol_), count(max_expiries, name="max_expiries"),
        clean_segment(scope, name="scope"),
    )


def ohlcv_key(symbol_: object, tf: object, *, provider: object) -> str:
    """Свечи (в т.ч. снапшот ``all`` — все ТФ одним значением).

    >>> ohlcv_key("spy", "1H", provider="yfinance")
    'gex:ohlcv:yfinance:SPY:1h'
    """
    return _scoped("ohlcv", provider, symbol(symbol_), timeframe(tf))


def bars_key(symbol_: object, tf: object, *, provider: object) -> str:
    """FIFO-кэш закрытых баров: ``gex:bars:{provider}:{SYMBOL}:{tf}``.

    Отличается от :func:`ohlcv_key` формой **значения**, а не только префиксом: там лежит
    произвольный снапшот (DataFrame в pickle), здесь — версионированный список баров
    ``(t, o, h, l, c, v)`` в JSON (``gex.adapters.cache.bar_store``). Одна схема под одним
    ключом — читатель старой схемы принял бы бары за кадр и наоборот.

    >>> bars_key("spy", "1H", provider="yfinance")
    'gex:bars:yfinance:SPY:1h'
    """
    return _scoped("bars", provider, symbol(symbol_), timeframe(tf))


def spot_key(symbol_: object, *, provider: object) -> str:
    """Текущая цена базового актива.

    >>> spot_key("spy", provider="yfinance")
    'gex:spot:yfinance:SPY'
    """
    return _scoped("spot", provider, symbol(symbol_))


def hv_key(symbol_: object, *, provider: object) -> str:
    """Историческая волатильность + ATR14 в цене.

    Провайдер здесь не формальность: для MOEX-инструментов HV считается из дневных свечей
    ISS, для остальных — из yfinance. Одинаковая форма значения делала подмену источника
    незаметной.
    """
    return _scoped("hv", provider, symbol(symbol_))


# ====================================================================== #
#  Прикладные ключи, привязанные к провайдеру
# ====================================================================== #
def state_key(name: str, *parts: object) -> str:
    """Ключ состояния сервиса (не данные провайдера): ``gex:{name}:{parts}``.

    Нужен, чтобы состояние фоновых сканеров было видно **из другого процесса**: API-роль
    не запускает сканеры, поэтому чтение из памяти процесса всегда давало «ещё не сканировали».

    >>> state_key("scan", "auto_us")
    'gex:scan:auto_us'
    """
    return _join(_clean(name, name="state name"), [_clean(p, name="state part") for p in parts])


def commodity_key(kind: str, *parts: object, provider: object) -> str:
    """Ключи товарного блока: ``spot`` / ``ohlcv`` / ``chain`` / ``dynamics``.

    ``provider`` обязателен, хотя сегодня весь товарный блок идёт через yfinance: значение
    по умолчанию означало бы, что при добавлении второго источника ключ молча останется
    «yfinance», то есть ровно тот дефект, который устраняет итерация 25.

    ``dynamics`` идентифицируется только глубиной истории ``bars``: композит считается
    сразу по всем товарам, поэтому ``bars`` — полная идентичность результата.
    """
    kind_ = _clean(kind, name="commodity kind")
    return _scoped(f"commodity:{kind_}", provider, *(_clean(p, name="commodity part") for p in parts))


# ====================================================================== #
#  Провайдер-независимые ключи
# ====================================================================== #
# ``res:*`` (кэш ответа эндпоинта) пока строится прежним ``gex.redis_client.cache_key``:
# это кэш контракта ответа, а не источника данных, и его заменит page-store в итер. 31.
# Добавлять сюда обёртку «на будущее» — это мёртвый код, а он в проекте запрещён (R4).
def page_key(page: str, *parts: object) -> str:
    """Ключ payload'а страницы: ``gex:page:{page}[:{parts}]``.

    Провайдер здесь **не входит** в ключ осознанно: значение — это готовый ответ страницы,
    который складывает фоновый пересчёт, а не данные одного источника. Страницы широты
    собираются из нескольких источников сразу (yfinance + CBOE + состав индекса), поэтому
    «провайдер» у такого ключа один — «pipeline страницы». Провайдер-зависимые данные
    (свечи, цепочки, спот) по-прежнему строятся только через ``*_key(..., provider=...)``.

    ``page`` — имя страницы из ``gex.domain.freshness.PAGE_CLASS`` (то же имя, что в
    ``policy_for_page``): окна свежести и форма ключа обязаны называть страницу одинаково.

    >>> page_key("breadth", "mags")
    'gex:page:breadth:mags'
    >>> page_key("breadth-imoex")
    'gex:page:breadth-imoex'
    """
    name = _clean(page, name="page").lower()
    return _join("page", [name, *(_clean(p, name="page part") for p in parts)])


__all__ = [
    "PROVIDER_YFINANCE",
    "PROVIDER_BYBIT",
    "PROVIDER_MOEX",
    "PROVIDER_WEBULL",
    "PROVIDER_SEC",
    "PROVIDER_FINNHUB",
    "KNOWN_PROVIDERS",
    "PROVIDER_SCOPED_KINDS",
    "KEY_ROOT",
    "CacheKeyError",
    "clean_segment",
    "state_key",
    "page_key",
    "normalize_provider",
    "symbol",
    "timeframe",
    "count",
    "chain_key",
    "chain_key_v2",
    "ohlcv_key",
    "bars_key",
    "spot_key",
    "hv_key",
    "commodity_key",
]
