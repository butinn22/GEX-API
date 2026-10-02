"""Use-case конуса вероятностей: выбор источника и параметров, одна точка входа (ring: application).

Что было
--------
Обработчик ``/gexcone/{ticker}`` содержал три почти одинаковых вызова калькулятора: крипта,
MOEX и «всё остальное». Отличались они только ``r``, ``q`` и множителем контракта, а сам вызов
(13 аргументов) был переписан трижды. Проверить это без сети было нельзя: обработчик сразу
шёл в фетчер, а параметры источника лежали рядом с HTTP-кодом.

Что стало
---------
Источник описывается **данными** (:class:`ConeSource`), а не ветвью кода: ``r``, ``q``,
множитель контракта, знаки дилера и окно обрезки цепочки. Резолвер
(:func:`resolve_source`) — чистая функция от тикера. Калькулятор и получение цепочки
**инжектируются**, поэтому use-case проверяется на синтетической цепочке, без сети и Redis.

Почему множитель контракта — не мелочь
--------------------------------------
``per_contract`` = 1 для крипты и 100 для акций: GEX отдельного страйка это
``Γ × OI × per_contract × 100``. Ошибка здесь масштабирует весь профиль в 100 раз и не
ломает ничего видимого — числа остаются «правдоподобными», поэтому множитель берётся из
общего справочника активов и проверяется тестом отдельно.

Модель Блэка для фьючерсов
--------------------------
У опционов на фьючерс дивидендной доходности нет, и в модели Блэка её роль играет ставка:
``q = r``. Для MOEX (опционы FORTS на фьючерсы) это ``q = r``, для индексных фьючерсов
(ES/NQ) значение ``q`` уже равно ``r`` в справочнике. Это **не** «дефолт по невнимательности»:
у фьючерсного прокси через опционы индекса форвард обязан быть без дивидендного сноса.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from gex.assets_config import CRYPTO_ASSETS, DEFAULT_ASSETS, FUTURES_TICKERS, MOEX_ASSETS

__all__ = [
    "ConeRequest",
    "ConeSource",
    "ComputeConeUseCase",
    "ProfileProvider",
    "SnapshotFetcher",
    "VolStatsProvider",
    "resolve_source",
]

#: Ставка и доходность по умолчанию для прочих инструментов (US-акции и ETF).
DEFAULT_RATE = 0.045
DEFAULT_YIELD = 0.0

#: Множитель контракта для акций: 1 контракт = 100 акций.
EQUITY_PER_CONTRACT = 100

#: Знаки дилера SqueezeMetrics (совпадают с дефолтами калькулятора).
CALL_SIGN = 1.0
PUT_SIGN = -1.0

#: Минимальное окно обрезки цепочки MOEX в днях (см. :func:`_moex_window_days`).
MOEX_MIN_WINDOW_DAYS = 45.0


@dataclass(frozen=True)
class ConeRequest:
    """Параметры запроса конуса (то, что приходит строкой запроса)."""

    ticker: str
    expiries: int = 5
    horizon_days: int = 14
    wall_decay: float = 2.0
    top_oi: int = 4
    oi_quantile: float = 0.9

    @property
    def symbol(self) -> str:
        """Тикер в каноническом виде: без пробелов, в верхнем регистре."""
        return self.ticker.strip().upper()

    def cache_parts(self) -> tuple:
        """Части ключа кэша — **все** параметры, влияющие на результат.

        ``expiries`` здесь не для симметрии: именно его отсутствие в ключе было дефектом
        EC-8 — запрос с другим числом экспираций 600 секунд получал чужой конус.
        """
        return (
            self.symbol,
            self.horizon_days,
            self.wall_decay,
            self.top_oi,
            self.oi_quantile,
            self.expiries,
        )


@dataclass(frozen=True)
class ConeSource:
    """Чем источник отличается от других — данными, а не ветвью кода."""

    name: str
    r: float
    q: float
    per_contract: int
    call_sign: float = CALL_SIGN
    put_sign: float = PUT_SIGN
    #: Окно обрезки цепочки в днях (``None`` — не обрезать).
    chain_window_days: Optional[float] = None
    note: str = ""


def _moex_window_days(horizon_days: int) -> float:
    """Окно цепочки MOEX: не меньше двух горизонтов, но не меньше 45 дней.

    Зачем обрезать вообще: у S-опционов (на акции) есть годовые серии с большим OI, и
    глобальные ``iv_atm``/стены по всей цепочке считаются по срокам, к конусу не относящимся.
    Окно — параметр политики источника, а не запроса, поэтому живёт здесь.
    """
    return max(float(horizon_days) * 2.0, MOEX_MIN_WINDOW_DAYS)


def resolve_source(ticker: str, *, horizon_days: int = 14) -> ConeSource:
    """Определить источник и его параметры по тикеру. Чистая функция.

    Порядок проверок важен: крипта → MOEX → фьючерсы → прочие. Тикер, которого нет ни в
    одной таблице, считается акцией/ETF США — это соответствует историческому поведению
    (yfinance как источник по умолчанию), поэтому в ``ConeSource`` попадают дефолтные ставка
    и доходность, а не ошибка.
    """
    symbol = ticker.strip().upper()

    if symbol in CRYPTO_ASSETS:
        cfg = CRYPTO_ASSETS[symbol]
        return ConeSource(
            name="crypto",
            r=float(cfg["r"]),
            q=float(cfg["q"]),
            per_contract=int(cfg["per_contract"]),
            call_sign=float(cfg.get("call_sign", CALL_SIGN)),
            put_sign=float(cfg.get("put_sign", PUT_SIGN)),
            note="Bybit V5: 1 контракт = 1 монета",
        )

    if symbol in MOEX_ASSETS:
        cfg = MOEX_ASSETS[symbol]
        # Опционы FORTS — на фьючерсы, поэтому модель Блэка: q = r (см. модульный docstring).
        return ConeSource(
            name="moex",
            r=float(cfg["r"]),
            q=float(cfg["r"]),
            per_contract=int(cfg.get("per_contract", EQUITY_PER_CONTRACT)),
            chain_window_days=_moex_window_days(horizon_days),
            note="MOEX ISS/FORTS: опцион на фьючерс ⇒ q = r",
        )

    cfg = DEFAULT_ASSETS.get(symbol, {})
    if symbol in FUTURES_TICKERS:
        return ConeSource(
            name="futures",
            r=float(cfg.get("r", DEFAULT_RATE)),
            q=float(cfg.get("q", DEFAULT_RATE)),
            per_contract=int(cfg.get("per_contract", EQUITY_PER_CONTRACT)),
            note="фьючерс через прокси-опционы индекса: модель Блэка, q = r",
        )

    return ConeSource(
        name="equity",
        r=float(cfg.get("r", DEFAULT_RATE)),
        q=float(cfg.get("q", DEFAULT_YIELD)),
        per_contract=int(cfg.get("per_contract", EQUITY_PER_CONTRACT)),
        note="акция/ETF США через yfinance",
    )


class SnapshotFetcher(Protocol):
    """Получение очищенной цепочки. Источник передаётся — по нему выбирается фетчер."""

    def __call__(self, ticker: str, source: ConeSource, expiries: int) -> Any: ...


class VolStatsProvider(Protocol):
    """Реализованная волатильность и ATR для тикера (``(hv, atr)``, любое может быть ``None``)."""

    def __call__(self, ticker: str) -> tuple[Optional[float], Optional[float]]: ...


class ProfileProvider(Protocol):
    """Канонический прогон GEX-движка по полученной цепочке.

    Принимает (тикер, цепочку, имя источника) и возвращает результат с
    полями ``profile`` (доменный GEXProfile), ``snapshot`` (цепочка после
    days-фильтра — та же, что у главной страницы), ``r``, ``q``,
    ``atm_vol`` (OI-взвешенная ATM-вола) и ``stk_all`` (per-strike кадр
    ``strike``/``gex_net``/``ag`` из канонического профиля).
    """

    def __call__(self, ticker: str, snapshot: Any, source_name: str) -> Any: ...


@dataclass
class ComputeConeUseCase:
    """Собрать конус: определить источник → получить цепочку → канонический
    GEX-движок → визуализация.

    ``build`` и ``profile_provider`` инжектируются, а не импортируются: слой
    приложения не должен выбирать реализации. ``profile_provider`` — единый
    прогон GEX-движка (:meth:`GEXPipelineRunner.run_gex_profile_domain`),
    тот же, что строит профиль главной страницы GEX; конус получает
    канонический ``GEXProfile`` + адаптированный per-strike кадр и больше
    не пересчитывает GEX сам. В тестах на его месте — подстановка.
    """

    fetch_snapshot: SnapshotFetcher
    vol_stats: VolStatsProvider
    profile_provider: ProfileProvider
    build: Callable[..., Any]
    sources: Callable[..., ConeSource] = resolve_source
    #: Собранные вызовы — для проверок «какой источник и какие параметры ушли в калькулятор».
    calls: list = field(default_factory=list, init=False, repr=False)

    def execute(self, request: ConeRequest) -> Any:
        """Построить конус по запросу.

        Порядок: ставки и множитель известны до сети, поэтому источник резолвится первым —
        так фетчер получает уже готовое описание (какой бирже принадлежит тикер), и
        «источник решил фетчер» перестаёт быть возможным.

        Шаг движка: ``profile_provider`` получает ту же цепочку, что ушла бы на
        главную страницу GEX (с тем же горизонтом ``horizon_days`` = ``days``),
        и возвращает профиль + отфильтрованную по days цепочку — конус строится
        уже по ним.
        """
        source = self.sources(request.symbol, horizon_days=request.horizon_days)
        snapshot = self.fetch_snapshot(request.symbol, source, request.expiries)
        snapshot = self._trim_chain(snapshot, source)
        hv, atr = self.vol_stats(request.symbol)

        engine = self.profile_provider(request.symbol, snapshot, source.name)

        kwargs = {
            "r": engine.r,
            "q": engine.q,
            "atm_vol": engine.atm_vol,
            "wall_decay": request.wall_decay,
            "max_expiries": request.expiries,
            "horizon_days": request.horizon_days,
            "top_oi_per_expiry": request.top_oi,
            "oi_quantile": request.oi_quantile,
            "hv": hv,
            "atr": atr,
            "profile": engine.profile,
            "stk_all": engine.stk_all,
        }
        self.calls.append({"source": source, "kwargs": dict(kwargs)})
        return self.build(engine.snapshot, **kwargs)

    @staticmethod
    def _trim_chain(snapshot: Any, source: ConeSource) -> Any:
        """Обрезать цепочку по окну источника (только если окно задано)."""
        if source.chain_window_days is None:
            return snapshot
        try:
            chain = snapshot.chain
        except AttributeError:
            return snapshot
        if "T" not in getattr(chain, "columns", ()):
            return snapshot
        cutoff = source.chain_window_days / 365.0
        trimmed = chain[chain["T"] <= cutoff].reset_index(drop=True)
        from dataclasses import replace

        return replace(snapshot, chain=trimmed)
