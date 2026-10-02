"""Каталог таймфреймов и вселенных: один источник + проверка полноты (итерация 29).

Задача итерации звучала как «11 определений `TIMEFRAMES` и 4 CSV-вселенные». Фактически
найдено **девять** независимых списков ТФ — причём один из них отличался
(``("4h","1d")`` у сканера и у ИИ-анализа finagent против ``("1h","2h","4h","1d")``
у остальных семи) — и **четыре** независимые карты интервалов провайдеров.

Проверяется то, что делает каталог источником правды:

* **полнота:** каждый канонический ТФ либо нативно отображён провайдером, либо выводим
  ресемплингом. Пропуск («забыли 2h у MOEX») обязан падать, а не превращаться в ошибку
  у пользователя;
* **согласованность вызывающих:** ни один модуль не описывает свой список ТФ или карту
  интервалов (правило R7 — здесь оно же проверяется исполняемо);
* **вселенные:** оба варианта заголовков разбираются, дубликаты схлопываются, пустой или
  отсутствующий список даёт ошибку, а не тишину.

    python tests/test_provider_catalog.py
    pytest tests/test_provider_catalog.py -q
"""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gex.adapters.providers import catalog as C  # noqa: E402


# ====================================================================== #
# 1. Словарь таймфреймов
# ====================================================================== #
def test_canonical_timeframes_are_unique_and_ordered():
    """Канон не содержит дублей и задан в порядке вывода (порядок значим для графиков)."""
    assert len(set(C.CANONICAL_TIMEFRAMES)) == len(C.CANONICAL_TIMEFRAMES)
    assert C.CANONICAL_TIMEFRAMES == ("1h", "2h", "4h", "1d")


def test_signal_subset_is_declared_and_smaller():
    """Подмножество «медленных» ТФ объявлено явно и является подмножеством канона.

    Раньше это был отдельный литерал в двух местах (сканер и finagent), и отличить
    намеренное сужение от опечатки по коду было невозможно — теперь это объявление.
    """
    assert set(C.SIGNAL_TIMEFRAMES) < set(C.CANONICAL_TIMEFRAMES)
    assert C.SIGNAL_TIMEFRAMES == ("4h", "1d")


def test_resample_relations_are_consistent():
    """Производный ТФ должен иметь источник, источник — быть каноническим и не производным."""
    assert set(C.RESAMPLE_SOURCE) == set(C.RESAMPLED_TIMEFRAMES)
    for derived, source in C.RESAMPLE_SOURCE.items():
        assert source in C.CANONICAL_TIMEFRAMES, f"{derived}: источника {source} нет в каноне"
        assert source not in C.RESAMPLED_TIMEFRAMES, f"{derived} выводится из производного {source}"


# ====================================================================== #
# 2. Полнота карт провайдеров — главная проверка итерации
# ====================================================================== #
def test_every_provider_covers_every_canonical_timeframe():
    """Ни один ТФ не «теряется»: либо нативный интервал, либо ресемплинг из нативного."""
    for provider, table in C.PROVIDER_INTERVALS.items():
        for tf in C.CANONICAL_TIMEFRAMES:
            assert C.provider_supports(provider, tf), f"{provider} не умеет {tf}"
            if tf not in table:
                source = C.RESAMPLE_SOURCE.get(tf)
                assert source and source in table, (
                    f"{provider}: {tf} нет в карте и нет источника ресемплинга"
                )


def test_provider_maps_have_no_extra_timeframes():
    """В картах нет ТФ, которых нет в каноне: иначе карта описывает несуществующее."""
    for provider, table in C.PROVIDER_INTERVALS.items():
        extra = set(table) - set(C.CANONICAL_TIMEFRAMES)
        assert not extra, f"{provider}: лишние ТФ {sorted(extra)}"


def test_resampled_timeframes_point_at_their_source_interval():
    """У производного ТФ интервал провайдера равен интервалу источника — иначе он нативный."""
    for provider in C.PROVIDER_INTERVALS:
        for tf, source in C.RESAMPLE_SOURCE.items():
            if not C.is_native(provider, tf):
                assert C.provider_interval(provider, tf) == C.provider_interval(provider, source), (
                    f"{provider}: {tf} считается производным, но интервал отличается от {source}"
                )


def test_bybit_has_native_2h_and_4h():
    """Различие, которое раньше жило только в комментариях: у Bybit 2h/4h нативные."""
    assert C.is_native("bybit", "2h") and C.is_native("bybit", "4h")
    assert not C.is_native("yfinance", "2h") and not C.is_native("yfinance", "4h")
    assert not C.is_native("moex_iss", "4h")
    # 1d у всех нативный: ресемплингом день не получить из часа без потери сессии
    for provider in C.PROVIDER_INTERVALS:
        assert C.is_native(provider, "1d")


def test_interval_values_match_real_provider_apis():
    """Значения интервалов — те, что понимают биржи (Bybit V5, ISS, yfinance)."""
    assert C.provider_interval("bybit", "1h") == "60"
    assert C.provider_interval("bybit", "4h") == "240"
    assert C.provider_interval("bybit", "1d") == "D"
    assert C.provider_interval("moex_iss", "1h") == 60 and C.provider_interval("moex_iss", "1d") == 24
    assert C.provider_interval("yfinance", "2h") == "1h", "2h у yfinance берётся часовым запросом"


def test_unknown_provider_and_timeframe_are_rejected():
    """Ошибка, а не подстановка по умолчанию: молчаливый fallback отдал бы не те бары."""
    assert _rejects(lambda: C.provider_interval("nexus", "1h")), "неизвестный провайдер принят"
    assert _rejects(lambda: C.provider_interval("yfinance", "5m")), "неизвестный ТФ принят"
    assert _rejects(lambda: C.normalize_timeframe("2h30")), "мусорный ТФ принят"
    assert _rejects(lambda: C.normalize_timeframe(None)), "None принят как ТФ"


def test_normalize_timeframe_is_case_insensitive():
    assert C.normalize_timeframe("1H") == C.normalize_timeframe(" 1h ") == "1h"
    assert C.is_timeframe("4H") and not C.is_timeframe("3h")


def test_catalog_selfcheck_reports_no_problems():
    """`check_catalog()` — самопроверка: полнота карт + доступность всех вселенных."""
    report = C.check_catalog()
    assert report["problems"] == [], report["problems"]
    assert report["providers"] == ["bybit", "moex_iss", "yfinance"]
    assert all(size and size > 0 for size in report["universes"].values()), report["universes"]


# ====================================================================== #
# 3. Вселенные (CSV): разные заголовки, дубликаты, ошибки
# ====================================================================== #
def _rejects(call) -> bool:
    """Подняла ли операция ``CatalogError`` (без `except: continue` в теле теста)."""
    try:
        call()
    except C.CatalogError:
        return True
    return False


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)


def test_catalog_universes_load_with_both_header_styles():
    """Файлы вселенных имеют разные заголовки (`Тикер` против `ticker,name`) — оба разбираются."""
    us = C.load_universe_detail("us_options")     # заголовок «Тикер»
    crypto = C.load_universe_detail("crypto")     # заголовок «ticker,name»
    assert us.tickers and crypto.tickers
    assert us.tickers[0].isupper() and crypto.tickers[0].isupper()
    assert not us.names, "у файла без колонки имени подписей быть не должно"
    assert crypto.names.get("BTC"), "подпись BTC потеряна"


def test_universe_tickers_are_unique_and_uppercased():
    """Дубликаты в состоянии сканера означали бы повторные запросы к провайдеру."""
    for name in C.available_universes():
        tickers = C.load_universe(name)
        assert len(set(tickers)) == len(tickers), f"{name}: дубликаты тикеров"
        assert all(t == t.upper() and t.strip() for t in tickers), f"{name}: регистр/пробелы"


def test_sectors_universe_matches_sector_service():
    """Сканер секторов и страница «Композит секторов» обязаны сканировать один набор ETF.

    Список живёт в двух местах (CSV каталога и ``sector_service.SECTOR_TICKERS``):
    расхождение означало бы, что сканер молча игнорирует часть секторов или
    тянет тикеры, которых нет на странице композита.
    """
    from gex.application import sector_service

    sectors = set(C.load_universe("sectors"))
    assert sectors == set(sector_service.SECTOR_TICKERS), (
        f"рассогласование: только в CSV {sorted(sectors - set(sector_service.SECTOR_TICKERS))}, "
        f"только в sector_service {sorted(set(sector_service.SECTOR_TICKERS) - sectors)}"
    )
    # Названия секторов нужны UI (XLK → Technology).
    assert C.load_universe_detail("sectors").names, "у вселенной sectors потеряны подписи"


def test_universe_is_cached_per_process():
    """CSV читаются один раз: сканер зовёт загрузчик часто."""
    assert C.load_universe_detail("crypto") is C.load_universe_detail("crypto")


def test_arbitrary_csv_file_is_parsed_by_the_same_rules():
    """Сканер принимает путь снаружи — разбор должен быть тем же, а не вторым."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "custom.csv"
        _write_csv(path, [["Тикер"], ["aapl"], ["AAPL"], ["", "мусор"], ["msft", "Microsoft"]])
        universe = C.load_universe_file(path)
        assert universe.tickers == ("AAPL", "MSFT"), universe.tickers
        assert universe.names == {"MSFT": "Microsoft"}


def test_headerless_csv_does_not_lose_first_ticker():
    """Файл без заголовка: первая строка — данные, а не заголовок."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "plain.csv"
        _write_csv(path, [["SPY"], ["QQQ"]])
        assert C.load_universe_file(path).tickers == ("SPY", "QQQ")


def test_missing_or_empty_universe_raises():
    """Пустая вселенная = «ничего не сканируем»: это ошибка, а не тихий пустой список."""
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "empty.csv"
        _write_csv(empty, [])
        assert _rejects(lambda: C.load_universe_file(Path(tmp) / "нет.csv")), "нет файла — нет ошибки"
        assert _rejects(lambda: C.load_universe_file(empty)), "пустой файл принят"
        assert _rejects(lambda: C.load_universe("нет-такой")), "неизвестная вселенная принята"


def test_header_without_ticker_column_raises():
    """Файл с непонятным заголовком не должен молча превратиться в мусорные тикеры."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weird.csv"
        _write_csv(path, [["имя", "значение"], ["что-то", "1"]])
        assert _rejects(lambda: C.load_universe_file(path)), "файл без колонки тикера принят"


# ====================================================================== #
# 4. Правило R7: словарь ТФ описывает только каталог
# ====================================================================== #
def test_r7_gate_is_clean():
    """Ни один модуль не задаёт свой список ТФ или карту интервалов."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "quality" / "ast_guard.py"), "--json"],
        capture_output=True, text=True, cwd=str(ROOT), check=False,
    )
    report = __import__("json").loads(out.stdout)
    assert report["timeframe_violations"] == [], report["timeframe_violations"]


def test_callers_use_the_catalog_instead_of_literals():
    """Расписка миграции: каждый модуль обязан брать ТФ/интервалы из каталога.

    Часть модулей импортирует каталог напрямую, часть — через посредника, который уже
    берёт значения из каталога (``ohlcv_service`` → ``ta_fetcher.TIMEFRAMES``). Второй путь
    тоже допустим: источник по-прежнему один, но проверять его надо явно, иначе «расписка»
    пройдёт на модуле, вернувшем себе литерал.
    """
    direct = {
        "gex/adapters/fetchers/ta_fetcher.py": "CANONICAL_TIMEFRAMES",
        "gex/application/background_fetcher.py": "CANONICAL_TIMEFRAMES",
        "gex/application/signal_service.py": "CANONICAL_TIMEFRAMES",
        "gex/application/signal_scanner_service.py": "CANONICAL_TIMEFRAMES",
        "gex/adapters/fetchers/moex_candles_fetcher.py": "MOEX_ISS_INTERVAL_CODE",
        "gex/auth/settings_router.py": "CANONICAL_TIMEFRAMES",
        "gex/application/auto_scanner_service.py": "SIGNAL_TIMEFRAMES",
        "finagent/router.py": "SIGNAL_TIMEFRAMES",
        "gex/adapters/providers/bybit.py": "BYBIT_INTERVAL",
        "gex/application/breadth_service.py": "universe_path",
    }
    for rel, symbol in direct.items():
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "catalog import" in src, f"{rel}: не импортирует каталог"
        assert symbol in src, f"{rel}: не использует {symbol}"

    # ``gex/domain/ta.py`` — особый случай: он берёт словарь ТФ **из домена**
    # (``gex/domain/timeframes.py``), а не из каталога. Так и должно быть: домен не имеет
    # права импортировать адаптер (правило R1), а сам словарь — понятие предметной области.
    # Каталог реэкспортирует то же значение, поэтому источник по-прежнему один; проверяем
    # именно это — что ta.py не завёл собственную копию литералом.
    ta_src = (ROOT / "gex/domain/ta.py").read_text(encoding="utf-8")
    assert "from gex.domain.timeframes import CANONICAL_TIMEFRAMES" in ta_src, (
        "gex/domain/ta.py: словарь ТФ обязан приходить из gex.domain.timeframes"
    )
    assert "gex.adapters" not in ta_src, (
        "gex/domain/ta.py: домен не импортирует адаптер (R1)"
    )

    # Косвенные потребители: значение приходит из модуля, который берёт его из каталога.
    # Модуль переехал в application, а ta_fetcher — в adapters, поэтому импорт стал
    # абсолютным. Проверяем смысл (значение приходит посредником из каталога), а не
    # форму записи импорта: относительная она или абсолютная — деталь раскладки.
    indirect = {
        "gex/application/ohlcv_service.py": ("from gex.adapters.fetchers.ta_fetcher import", "TIMEFRAMES"),
    }
    for rel, (imp, symbol) in indirect.items():
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert imp in src and symbol in src, f"{rel}: не берёт {symbol} из {imp}"
    # …и сам посредник обязан быть в списке прямых потребителей
    assert "gex/adapters/fetchers/ta_fetcher.py" in direct


def test_every_canonical_timeframe_has_freshness_policy():
    """Обратная сторона R7-исключения: политика обязана покрывать весь канон.

    `freshness.TIMEFRAME_POLICIES` шире канона (там есть 1m/5m/15m/30m) — это допустимо,
    но «забыть» в ней поддерживаемый ТФ нельзя: страница получила бы окна по умолчанию.
    """
    from gex.domain.freshness import TIMEFRAME_POLICIES

    missing = set(C.CANONICAL_TIMEFRAMES) - set(TIMEFRAME_POLICIES)
    assert not missing, f"нет политики свежести для {sorted(missing)}"


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            print(f"FAIL {fn.__name__}: {str(exc)[:300]}")
            failed += 1
        except Exception as exc:  # НЕОЖИДАННОЕ: иначе прогон обрывался, и «0 FAIL» врало
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            failed += 1
    print(f"--- provider catalog: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
