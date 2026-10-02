"""Каталог таймфреймов и торговых вселенных — один источник правды (ring: adapters/providers).

Зачем
-----
Таймфреймы приложения были описаны **восемью** независимыми литералами::

    auth/settings_router.py     ALLOWED_TIMEFRAMES = ("1h", "2h", "4h", "1d")
    background_fetcher.py       OHLCV_TIMEFRAMES   = ["1h", "2h", "4h", "1d"]
    moex_candles_fetcher.py     TIMEFRAMES         = ("1h", "2h", "4h", "1d")
    signal_scanner_service.py   SUPPORTED_TIMEFRAMES = ("1h", "2h", "4h", "1d")
    signal_service.py           _SUPPORTED_TIMEFRAMES = ("1h", "2h", "4h", "1d")
    ta.py                       TIMEFRAMES_ORDER   = ["1h", "2h", "4h", "1d"]
    ta_fetcher.py               TIMEFRAMES         = ("1h", "2h", "4h", "1d")
    auto_scanner_service.py     TIMEFRAMES         = ("4h", "1d")   ← ДРУГОЙ набор

Семь из восьми совпадали, восьмой отличался — и это не было ошибкой (сканеру нужны только
«медленные» ТФ), но отличить намеренное подмножество от опечатки по коду невозможно. Плюс
каждый провайдер хранит свой словарь интервалов отдельно, а связь «2h/4h получаются
ресемплингом из 1h» существовала только в комментариях.

Что здесь есть
--------------
* :data:`CANONICAL_TIMEFRAMES` — словарь приложения;
* :data:`RESAMPLED_TIMEFRAMES` / :data:`RESAMPLE_SOURCE` — какие ТФ производные и из чего
  (важно: у Bybit они нативные, у yfinance/MOEX — нет);
* карты интервалов провайдеров — единственное место, где в код попадают ``"60"``/``24``;
* :data:`SCANNER_TIMEFRAMES` — подмножество сканера **объявлено** и объяснено;
* каталог вселенных (CSV) с единым загрузчиком: у файлов разные заголовки
  (``Тикер`` против ``ticker,name``), и разбирать это в двух местах означало бы
  повторять ту же ошибку.

Проверка полноты — ``tests/test_provider_catalog.py``: каждый канонический ТФ обязан быть
либо нативно отображён провайдером, либо выводим ресемплингом.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Таймфреймы
# ====================================================================== #
#: Канонический словарь приложения живёт в **домене** (``gex/domain/timeframes.py``),
#: потому что набор ТФ — понятие предметной области, а не деталь площадки. Здесь он
#: реэкспортируется, чтобы каталог оставался единственной точкой входа для потребителей
#: (правило R7). Прямой импорт из ``gex.domain.timeframes`` разрешён домену и не создаёт
#: нарушения R1: ``gex/domain/ta.py`` больше не тянет адаптер ради этой одной константы.
from gex.domain.timeframes import CANONICAL_TIMEFRAMES  # noqa: F401  (реэкспорт)

#: Производные таймфреймы: ни один провайдер не отдаёт их «как есть»... кроме Bybit.
RESAMPLED_TIMEFRAMES: frozenset[str] = frozenset({"2h", "4h"})

#: Из чего получается производный ТФ (ресемплинг делается из часовых баров).
RESAMPLE_SOURCE: Mapping[str, str] = {"2h": "1h", "4h": "1h"}

#: «Медленные» ТФ для сигнальной работы. Используются в двух местах с одним смыслом:
#: авто-сканер (на 1h сигналы шумят, а полный прогон стоит вчетверо дороже) и
#: ИИ-анализ finagent (старший ТФ — приоритет в сигнале). Объявлено явно, потому что
#: раньше это подмножество было записано двумя независимыми литералами, и отличить
#: намеренное сужение от опечатки по коду было невозможно.
SIGNAL_TIMEFRAMES: tuple[str, ...] = ("4h", "1d")

#: Интервалы yfinance. 2h/4h обслуживаются часовым запросом + ресемплингом.
YFINANCE_INTERVAL: Mapping[str, str] = {"1h": "1h", "2h": "1h", "4h": "1h", "1d": "1d"}

#: Коды интервала MOEX ISS: 1=1мин, 10=10мин, 60=1ч, 24=1день.
MOEX_ISS_INTERVAL_CODE: Mapping[str, int] = {"1h": 60, "2h": 60, "4h": 60, "1d": 24}

#: Интервалы Bybit V5 (спот). Отличие от остальных: 2h/4h у него **нативные**.
BYBIT_INTERVAL: Mapping[str, str] = {"1h": "60", "2h": "120", "4h": "240", "1d": "D"}

#: Карты интервалов по имени провайдера (для полноты и для provider_interval()).
PROVIDER_INTERVALS: Mapping[str, Mapping[str, object]] = {
    "yfinance": YFINANCE_INTERVAL,
    "bybit": BYBIT_INTERVAL,
    "moex_iss": MOEX_ISS_INTERVAL_CODE,
}


class CatalogError(ValueError):
    """Ошибка каталога: неизвестный таймфрейм, провайдер или вселенная."""


def normalize_timeframe(tf: object) -> str:
    """Таймфрейм в каноническом виде (``1H`` → ``1h``); неизвестный — ошибка.

    Ошибка, а не «приведём к 1h»: молчаливая подмена ТФ отдала бы пользователю не те
    бары, которые он запросил.
    """
    key = str(tf).strip().lower()
    if key not in CANONICAL_TIMEFRAMES:
        raise CatalogError(
            f"неизвестный таймфрейм {tf!r}; известные: {list(CANONICAL_TIMEFRAMES)}"
        )
    return key


def is_timeframe(tf: object) -> bool:
    """Проверка без исключения (для валидации входных параметров ручек)."""
    try:
        normalize_timeframe(tf)
    except CatalogError:
        return False
    return True


def resample_source(tf: str) -> Optional[str]:
    """Из какого нативного ТФ получается ``tf``; ``None``, если он нативный сам по себе."""
    return RESAMPLE_SOURCE.get(normalize_timeframe(tf))


def provider_interval(provider: str, tf: str) -> object:
    """Интервал провайдера для канонического ТФ (``"1h"``/``60``/``"D"``).

    Raises
    ------
    CatalogError
        Если провайдер неизвестен или не умеет этот ТФ.
    """
    tf_key = normalize_timeframe(tf)
    table = PROVIDER_INTERVALS.get(provider)
    if table is None:
        raise CatalogError(
            f"неизвестный провайдер {provider!r}; известные: {sorted(PROVIDER_INTERVALS)}"
        )
    try:
        return table[tf_key]
    except KeyError:
        raise CatalogError(
            f"провайдер {provider!r} не поддерживает {tf_key!r}"
        ) from None


def provider_supports(provider: str, tf: str) -> bool:
    """Умеет ли провайдер отдать этот ТФ (нативно или ресемплингом)."""
    try:
        provider_interval(provider, tf)
    except CatalogError:
        return False
    return True


def timeframes_for(provider: str) -> tuple[str, ...]:
    """Канонические ТФ, доступные у провайдера (в порядке :data:`CANONICAL_TIMEFRAMES`)."""
    return tuple(tf for tf in CANONICAL_TIMEFRAMES if provider_supports(provider, tf))


def is_native(provider: str, tf: str) -> bool:
    """Отдаёт ли провайдер этот ТФ без ресемплинга.

    Важно для диагностики расхождений: у Bybit 4h нативный, у yfinance — нет.
    """
    tf_key = normalize_timeframe(tf)
    source = RESAMPLE_SOURCE.get(tf_key)
    if source is None:
        return True
    interval = provider_interval(provider, tf_key)
    # Если интервал производного ТФ совпадает с интервалом его источника, значит
    # провайдер отдаёт базовые бары и ресемплинг делает приложение.
    return interval != provider_interval(provider, source)


# ====================================================================== #
#  Вселенные (CSV)
# ====================================================================== #
#: Корень данных: CSV лежат рядом с пакетом `gex` (историческое расположение).
_DATA_DIR = Path(__file__).resolve().parents[2]

#: Имя вселенной → файл. Расширение `.csv` дописывается здесь же, чтобы не дублировать.
UNIVERSE_FILES: Mapping[str, str] = {
    "us_options": "auto_scanner_tickers.csv",
    "crypto": "auto_scanner_tickers_crypto.csv",
    "fx": "auto_scanner_tickers_fx.csv",
    "ru": "auto_scanner_tickers_ru.csv",
    # 12 секторальных ETF США + RSP — тот же набор, что читает страница
    # «Композит секторов» (gex/application/sector_service.py:SECTOR_TICKERS).
    # Несовпадение ловится тестом tests/test_provider_catalog.py.
    "sectors": "auto_scanner_tickers_sectors.csv",
    "sp500": "sp500_constituents.csv",
}

#: Возможные заголовки тикерной колонки. У файлов они **разные** («Тикер» против
#: «ticker,name»), и именно на этом ломается наивный `csv.DictReader`.
_TICKER_HEADERS = ("тикер", "ticker", "symbol", "тикеры", "tickers")

#: Разобранные вселенные: CSV читаются один раз на процесс.
_universe_cache: dict[str, "Universe"] = {}


@dataclass(frozen=True)
class Universe:
    """Разобранная вселенная: тикеры в порядке файла + подписи (если есть).

    Подписи нужны сканеру и UI: у части файлов есть вторая колонка с названием
    инструмента (``NVDA,NVIDIA``), у части — нет. Держать разбор заголовков в двух
    местах (сканер + breadth) означало бы повторить ту же ошибку, что с таймфреймами.
    """

    name: str
    path: Path
    tickers: tuple[str, ...]
    names: Mapping[str, str]

    def __len__(self) -> int:
        return len(self.tickers)


def universe_path(name: str) -> Path:
    try:
        filename = UNIVERSE_FILES[name]
    except KeyError:
        raise CatalogError(
            f"неизвестная вселенная {name!r}; известные: {sorted(UNIVERSE_FILES)}"
        ) from None
    return _DATA_DIR / filename


def load_universe_detail(name: str, *, reload: bool = False) -> Universe:
    """Прочитать вселенную по имени из :data:`UNIVERSE_FILES` (кэшируется на процесс)."""
    if not reload and name in _universe_cache:
        return _universe_cache[name]
    path = universe_path(name)
    if not path.exists():
        raise CatalogError(f"файл вселенной {name!r} не найден: {path}")
    universe = load_universe_file(path, name=name)
    _universe_cache[name] = universe
    return universe


def load_universe_file(path: Path | str, *, name: Optional[str] = None) -> Universe:
    """Прочитать произвольный CSV вселенной **тем же** разбором, что и каталожные.

    Отдельная точка входа нужна вызывающим, у которых путь приходит снаружи
    (сканер принимает ``tickers_file=``). Иначе разбор заголовков пришлось бы
    дублировать — ровно та ошибка, которую устраняет каталог.

    Raises
    ------
    CatalogError
        Если файла нет, нет тикерной колонки или список пуст. Ошибка, а не пустой
        список: пустая вселенная означает «ничего не сканируем», и это обязано быть
        видно сразу, а не как тишина в логах.
    """
    file_path = Path(path)
    label = name or file_path.name
    if not file_path.exists():
        raise CatalogError(f"файл вселенной {label!r} не найден: {file_path}")

    tickers: list[str] = []
    names: dict[str, str] = {}
    seen: set[str] = set()
    with file_path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:  # пустой файл
            raise CatalogError(f"вселенная {label!r} пуста: {file_path}") from None
        column = _ticker_column(header, name=label, path=file_path)
        # Первая строка может быть данными, если заголовка нет вовсе: проверяем и её,
        # иначе потеряли бы один тикер.
        for row in [header, *reader]:
            if column >= len(row):
                continue
            ticker = row[column].strip().upper()
            if not ticker or ticker.lower() in _TICKER_HEADERS:
                continue
            if ticker in seen:
                continue
            seen.add(ticker)
            tickers.append(ticker)
            label_col = 1 if column == 0 else column + 1
            if len(row) > label_col and row[label_col].strip():
                names[ticker] = row[label_col].strip()

    if not tickers:
        raise CatalogError(f"вселенная {label!r} не содержит тикеров: {file_path}")
    logger.debug("Вселенная %s: %d тикеров из %s", label, len(tickers), file_path.name)
    return Universe(name=label, path=file_path, tickers=tuple(tickers), names=names)


def load_universe(name: str, *, reload: bool = False) -> tuple[str, ...]:
    """Тикеры вселенной (верхний регистр, без дублей, без пустых строк)."""
    return load_universe_detail(name, reload=reload).tickers


def _ticker_column(header: Iterable[str], *, name: str, path: Path) -> int:
    """Индекс тикерной колонки по заголовку (у файлов разные схемы)."""
    cells = [str(c).strip().lower() for c in header]
    for idx, cell in enumerate(cells):
        if cell in _TICKER_HEADERS:
            return idx
    # Файл без заголовка: первая ячейка и есть тикер.
    if cells and cells[0] and cells[0] not in ("name", "имя"):
        return 0
    raise CatalogError(
        f"в {path} (вселенная {name!r}) не найдена колонка тикера; "
        f"заголовок: {cells[:5]}"
    )


def available_universes() -> tuple[str, ...]:
    return tuple(sorted(UNIVERSE_FILES))


def universe_size(name: str) -> int:
    """Размер вселенной (для диагностики/метрик) — читает через тот же кэш."""
    return len(load_universe(name))


def check_catalog() -> dict:
    """Самопроверка каталога: полнота карт провайдеров и доступность вселенных.

    Возвращает отчёт (для страж-тестов и ручной диагностики), не бросает исключений:
    ``problems`` перечисляет всё, что разъехалось.
    """
    problems: list[str] = []
    for provider, table in PROVIDER_INTERVALS.items():
        for tf in CANONICAL_TIMEFRAMES:
            if tf not in table:
                source = RESAMPLE_SOURCE.get(tf)
                if source is None or source not in table:
                    problems.append(f"{provider}: нет интервала для {tf} и нет источника ресемплинга")
        for tf in table:
            if tf not in CANONICAL_TIMEFRAMES:
                problems.append(f"{provider}: интервал для неканонического ТФ {tf}")

    sizes: dict[str, Optional[int]] = {}
    for name in available_universes():
        try:
            size = universe_size(name)
        except CatalogError as exc:
            sizes[name] = None
            problems.append(str(exc))
            continue
        sizes[name] = size
        if size == 0:
            problems.append(f"вселенная {name}: пуста")

    return {
        "canonical_timeframes": list(CANONICAL_TIMEFRAMES),
        "signal_timeframes": list(SIGNAL_TIMEFRAMES),
        "providers": sorted(PROVIDER_INTERVALS),
        "universes": sizes,
        "problems": problems,
    }


__all__ = [
    "BYBIT_INTERVAL",
    "CANONICAL_TIMEFRAMES",
    "CatalogError",
    "MOEX_ISS_INTERVAL_CODE",
    "PROVIDER_INTERVALS",
    "RESAMPLED_TIMEFRAMES",
    "RESAMPLE_SOURCE",
    "SIGNAL_TIMEFRAMES",
    "UNIVERSE_FILES",
    "YFINANCE_INTERVAL",
    "available_universes",
    "check_catalog",
    "is_native",
    "is_timeframe",
    "Universe",
    "load_universe",
    "load_universe_detail",
    "load_universe_file",
    "normalize_timeframe",
    "provider_interval",
    "provider_supports",
    "resample_source",
    "timeframes_for",
    "universe_path",
    "universe_size",
]
