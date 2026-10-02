"""Адаптер yfinance: нормализация и деградация (итерация 30).

Что проверяется и почему
------------------------
Адаптер собрал в одно место то, что было скопировано по 11 файлам: вызов SDK, обработку
сбоя, приведение колонок. Ошибка здесь не падает — она **портит данные**: неверный уровень
``MultiIndex`` отдаёт колонку ``SPY`` вместо ``Close``, а ``capitalize()`` превращает
``Adj Close`` в ``Adj close`` и ``SPY`` в ``Spy``. Поэтому проверяются именно формы кадров
и различимость «сбой источника» / «данных нет».

Сеть не нужна: транспорт подменяется фейком (он и есть граница внешнего мира).

    python tests/test_yfinance_adapter.py
    pytest tests/test_yfinance_adapter.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Skipped(Exception):
    """Проверка требует pandas (её нет в stdlib-прогоне) — не «зелёная», а пропущенная."""


_HAS_PANDAS = True
_IMPORT_ERROR: Exception | None = None
pd = None
yfa = None

try:
    import pandas as pd

    from gex.adapters.providers import yfinance as yfa
    from gex.adapters.transport.yf_transport import YfDeadlineError
except ImportError as exc:  # pandas отсутствует — проверки пропускаются, а не «зеленеют»
    _HAS_PANDAS = False
    _IMPORT_ERROR = exc


def _require():
    if not _HAS_PANDAS:
        raise Skipped(f"нужен pandas ({_IMPORT_ERROR})")


class FakeTransport:
    """Транспорт-заглушка: отдаёт заранее заданные значения и умеет «падать»."""

    def __init__(self, **responses):
        self.responses = responses
        self.calls: list[tuple[str, tuple, dict]] = []

    def _reply(self, name: str):
        value = self.responses.get(name)
        if isinstance(value, BaseException):
            raise value
        return value

    def history(self, symbol, seconds=None, **kwargs):
        self.calls.append(("history", (symbol,), {"seconds": seconds, **kwargs}))
        return self._reply("history")

    def download(self, *args, seconds=None, **kwargs):
        self.calls.append(("download", args, {"seconds": seconds, **kwargs}))
        return self._reply("download")

    def fast_info(self, symbol, seconds=None):
        self.calls.append(("fast_info", (symbol,), {}))
        return self._reply("fast_info")

    def options(self, symbol, seconds=None):
        self.calls.append(("options", (symbol,), {}))
        return self._reply("options")

    def option_chain(self, symbol, date=None, seconds=None):
        self.calls.append(("option_chain", (symbol, date), {}))
        return self._reply("option_chain")


class FakeChain:
    def __init__(self, calls=None, puts=None):
        if calls is not None:
            self.calls = calls
        if puts is not None:
            self.puts = puts


def _use(transport: FakeTransport) -> FakeTransport:
    """Подменить транспорт адаптера (единственная внешняя граница).

    Присваивание идёт в ``__dict__`` модуля: это точка внедрения, объявленная самим
    адаптером (``_transport``), и тесту незачем притворяться, что он меняет тип.
    """
    yfa.__dict__["_transport"] = lambda: transport
    return transport


def _df(data: dict, index=None):
    return pd.DataFrame(data, index=index or pd.date_range("2026-01-01", periods=len(next(iter(data.values())))))


# ====================================================================== #
# 1. Нормализация колонок — место, где ошибка портит данные молча
# ====================================================================== #
def test_normalize_collapses_multiindex_by_field():
    """``('Close','SPY')`` → ``Close``: первый уровень это поле, второй — тикер."""
    _require()
    raw = pd.DataFrame({("Close", "SPY"): [1.0, 2.0], ("Volume", "SPY"): [10, 20]})
    out = yfa.normalize_columns(raw)
    assert list(out.columns) == ["Close", "Volume"], list(out.columns)


def test_normalize_keeps_ticker_when_fields_repeat():
    """Несколько тикеров: поля повторяются, поэтому имя склеивается, а не теряется."""
    _require()
    raw = pd.DataFrame({
        ("Close", "SPY"): [1.0], ("Close", "QQQ"): [2.0],
        ("Volume", "SPY"): [10], ("Volume", "QQQ"): [20],
    })
    assert list(yfa.normalize_columns(raw).columns) == [
        "Close_SPY", "Close_QQQ", "Volume_SPY", "Volume_QQQ"
    ]


def test_normalize_fixes_case_without_mangling_names():
    """``close`` → ``Close``, но ``Adj Close`` не превращается в ``Adj close``."""
    _require()
    raw = pd.DataFrame({"close": [1.0], "OPEN": [2.0], "Adj Close": [3.0], "SPY": [4.0]})
    assert list(yfa.normalize_columns(raw).columns) == ["Close", "Open", "Adj Close", "SPY"]


def test_normalize_handles_empty_and_none():
    _require()
    assert yfa.normalize_columns(None).empty
    assert yfa.normalize_columns(pd.DataFrame()).empty


def test_normalize_sorts_index_and_parses_dates():
    _require()
    raw = pd.DataFrame({"Close": [2.0, 1.0]}, index=["2026-02-01", "2026-01-01"])
    out = yfa.normalize_columns(raw)
    assert out.index.is_monotonic_increasing
    assert str(out.index.dtype).startswith("datetime64")


# ====================================================================== #
# 2. history: сбой отличается от «данных нет»
# ====================================================================== #
def test_history_returns_none_on_source_failure():
    """Сбой источника — это ``None``, а не исключение (так делали все 11 вызывающих)."""
    _require()
    _use(FakeTransport(history=RuntimeError("yahoo недоступен")))
    assert yfa.history("SPY") is None


def test_history_returns_none_on_deadline():
    """Зависший источник: тоже ``None``, но причина видна в логе (дедлайн ≠ пустые данные)."""
    _require()
    _use(FakeTransport(history=YfDeadlineError("не ответил за 30 с")))
    assert yfa.history("SPY") is None


def test_history_returns_none_on_empty_frame():
    _require()
    _use(FakeTransport(history=pd.DataFrame()))
    assert yfa.history("SPY") is None


def test_history_normalizes_and_forwards_parameters():
    _require()
    transport = _use(FakeTransport(history=_df({"Close": [1.0, 2.0]})))
    out = yfa.history("SPY", period="5y", interval="1mo", auto_adjust=False)
    assert out is not None and list(out.columns) == ["Close"]
    method, args, kwargs = transport.calls[0]
    assert (method, args[0]) == ("history", "SPY")
    assert kwargs["period"] == "5y" and kwargs["interval"] == "1mo" and kwargs["auto_adjust"] is False


def test_history_or_raise():
    _require()
    _use(FakeTransport(history=RuntimeError("нет")))
    try:
        yfa.history_or_raise("SPY")
    except yfa.YFinanceError:
        return
    raise AssertionError("history_or_raise не поднял YFinanceError")


# ====================================================================== #
# 3. Спот: два источника и порядок
# ====================================================================== #
def test_spot_prefers_fast_info():
    """``fast_info`` дешевле: если он дал цену, свечи не запрашиваются."""
    _require()
    transport = _use(FakeTransport(fast_info={"lastPrice": 123.45}, history=_df({"Close": [1.0]})))
    assert yfa.spot("SPY") == 123.45
    assert [c[0] for c in transport.calls] == ["fast_info"], "история запрошена зря"


def test_spot_falls_back_to_last_close():
    """У индексов ``fast_info`` часто пуст — берём последний Close."""
    _require()
    transport = _use(FakeTransport(fast_info={}, history=_df({"Close": [10.0, 20.0]})))
    assert yfa.spot("^VIX") == 20.0
    assert [c[0] for c in transport.calls] == ["fast_info", "history"]


def test_spot_returns_none_when_nothing_available():
    _require()
    _use(FakeTransport(fast_info={}, history=pd.DataFrame()))
    assert yfa.spot("НЕТ") is None


def test_spot_ignores_non_positive_prices():
    _require()
    _use(FakeTransport(fast_info={"lastPrice": 0}, history=_df({"Close": [0.0]})))
    assert yfa.spot("SPY") is None


def test_spot_handles_dead_fast_info():
    """``fast_info`` падает — это не повод не спросить историю."""
    _require()
    _use(FakeTransport(fast_info=RuntimeError("нет связи"), history=_df({"Close": [7.0]})))
    assert yfa.spot("SPY") == 7.0


# ====================================================================== #
# 4. Опционы
# ====================================================================== #
def test_option_expiries_returns_tuple_and_handles_failure():
    _require()
    _use(FakeTransport(options=["2026-09-18", "2026-10-16"]))
    assert yfa.option_expiries("SPY") == ("2026-09-18", "2026-10-16")

    _use(FakeTransport(options=RuntimeError("нет")))
    assert yfa.option_expiries("SPY") == ()


def test_option_chain_returns_calls_and_puts():
    """Возвращаются кадры, а не объект SDK: вызывающему нужны колонки."""
    _require()
    calls = pd.DataFrame({"openInterest": [10, 20]})
    puts = pd.DataFrame({"openInterest": [5]})
    _use(FakeTransport(option_chain=FakeChain(calls=calls, puts=puts)))

    result = yfa.option_chain("SPY", "2026-09-18")
    assert result is not None
    got_calls, got_puts = result
    assert float(got_calls["openInterest"].sum()) == 30
    assert float(got_puts["openInterest"].sum()) == 5


def test_option_chain_none_when_parts_missing():
    _require()
    _use(FakeTransport(option_chain=FakeChain(calls=pd.DataFrame())))  # нет puts
    assert yfa.option_chain("SPY", "2026-09-18") is None

    _use(FakeTransport(option_chain=RuntimeError("нет")))
    assert yfa.option_chain("SPY", "2026-09-18") is None


# ====================================================================== #
# 5. download и ряды закрытий
# ====================================================================== #
def test_download_normalizes_by_default():
    _require()
    _use(FakeTransport(download=pd.DataFrame({("Close", "SPY"): [1.0], ("Volume", "SPY"): [2]})))
    out = yfa.download("SPY")
    assert out is not None and list(out.columns) == ["Close", "Volume"]


def test_download_can_keep_raw_multiindex():
    """``normalize=False`` нужен там, где вызывающий разбирает уровни сам (group_by="ticker")."""
    _require()
    transport = _use(FakeTransport(download=pd.DataFrame({("SPY", "Close"): [1.0]})))
    out = yfa.download(["SPY"], group_by="ticker", normalize=False)
    assert out is not None and isinstance(out.columns, pd.MultiIndex)
    assert transport.calls[0][2]["group_by"] == "ticker"


def test_download_returns_none_on_failure():
    _require()
    _use(FakeTransport(download=RuntimeError("нет")))
    assert yfa.download("SPY") is None


def test_close_series_and_closes_for():
    _require()
    _use(FakeTransport(history=_df({"Close": [1.0, None, 3.0]})))
    series = yfa.close_series("SPY")
    assert series is not None and len(series) == 2, "NaN не отброшены"

    _use(FakeTransport(history=_df({"Close": [1.0, 2.0]})))
    assert set(yfa.closes_for(["SPY", "QQQ"])) == {"SPY", "QQQ"}


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_") and callable(v)]
    failed = skipped = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Skipped as exc:
            skipped += 1
            print(f"SKIP {fn.__name__}: {exc}")
        except AssertionError as exc:
            print(f"FAIL {fn.__name__}: {str(exc)[:300]}")
            failed += 1
        except Exception as exc:  # НЕОЖИДАННОЕ: иначе прогон обрывался, и «0 FAIL» врало
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            failed += 1
    print(f"--- yfinance adapter: {len(tests) - failed - skipped} PASS / {failed} FAIL / {skipped} SKIP ---")
    sys.exit(1 if failed else 0)
