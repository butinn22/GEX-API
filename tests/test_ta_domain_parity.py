"""Паритет `ta.py` и канонических ядер домена (итерация 38).

Зачем этот набор
----------------
`domain/indicators/*` писались как точная транскрипция индикаторов, но **никем не
использовались**: их импортировали только собственные тесты, то есть ядро сравнивалось само
с собой (модули числились сиротами). Пока есть две реализации одного индикатора, вопрос не
в дублировании строк, а в расхождении: правку в одной вторая не увидит.

Набор фиксирует результат разбора по каждому индикатору отдельно, и результат разный:

* **RSI** — расхождение ровно ``0.0`` для всей длины ряда и обеих политик, поэтому `ta._wilder_rsi`
  делегирует домену. Это доказывается здесь, а не предполагается: реализация RSI из `ta`
  выписана в тесте как **эталон** (транскрипция), и делегирование сверяется с ней.
* **EMA** — варианты **разные**: `ta` использует EMA pandas (``adjust=False``, без SMA-сида),
  домен — канонический ``ta.ema`` со SMA-сидом. На 300 барах EMA200 расходится на ~0.45 и на
  последнем баре расхождение не исчезает, а именно последний бар `compute_indicators` и отдаёт.
  Делегирование здесь изменило бы числа (и `ema_bull_stack` при borderline-значениях), поэтому
  оно **не сделано**, а расхождение закреплено проверкой: заменить один вариант другим молча
  больше нельзя.

    python tests/test_ta_domain_parity.py
    pytest tests/test_ta_domain_parity.py -q
"""
from __future__ import annotations

import inspect
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Skipped(Exception):
    """Нужны pandas/numpy (нет в stdlib-прогоне) — не «зелёная», а пропущенная."""


_HAS_DEPS = True
_IMPORT_ERROR: Exception | None = None
pd = None
np = None
ta = None
domain_rsi = None
domain_primitives = None
try:
    import numpy as np
    import pandas as pd

    import gex.domain.ta as ta
    from gex.domain.indicators import primitives as domain_primitives
    from gex.domain.indicators import rsi as domain_rsi
except ImportError as exc:
    _HAS_DEPS = False
    _IMPORT_ERROR = exc


def _require():
    if not _HAS_DEPS:
        raise Skipped(f"нужны pandas/numpy ({_IMPORT_ERROR})")


# ====================================================================== #
#  Вход
# ====================================================================== #
def walk(seed: int = 1, bars: int = 300, drift: float = 0.0005, vol: float = 0.01) -> "pd.Series":
    rnd = random.Random(seed)
    prices = [100.0]
    for _ in range(bars):
        prices.append(max(1.0, prices[-1] * (1.0 + rnd.gauss(drift, vol))))
    return pd.Series(prices)


def ohlc(seed: int = 1, bars: int = 300) -> "pd.DataFrame":
    close = walk(seed, bars)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close * 1.004,
        "Low": close * 0.996,
        "Close": close,
        "Volume": [1_000_000.0] * len(close),
    })


# ====================================================================== #
#  Эталон: прежняя реализация RSI из ta.py (транскрипция, не делегирование)
# ====================================================================== #
def reference_wilder_rsi(close: "pd.Series", period: int) -> "pd.Series":
    """RSI по Уайлдеру так, как он был реализован в ``ta.py`` до итерации 38.

    Это **эталон для сравнения**, а не второй продакшн-код: он выписан здесь намеренно,
    чтобы делегирование домену проверялось против прежнего поведения, а не против самого
    домена (иначе проверка была бы тавтологией).
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = pd.Series(np.nan, index=close.index, dtype=float)
    avg_loss = pd.Series(np.nan, index=close.index, dtype=float)
    if len(close) <= period:
        return pd.Series(50.0, index=close.index, dtype=float)

    avg_gain.iloc[period] = gain.iloc[1: period + 1].mean()
    avg_loss.iloc[period] = loss.iloc[1: period + 1].mean()
    for i in range(period + 1, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    warmup = avg_loss.isna() & avg_gain.isna()
    rsi[warmup] = 50.0
    flat = (avg_gain == 0.0) & (avg_loss == 0.0)
    rsi = rsi.fillna(100.0)
    rsi[flat] = 50.0
    return rsi.clip(0.0, 100.0)


# ====================================================================== #
#  RSI: делегирование не изменило числа
# ====================================================================== #
def test_rsi_delegation_matches_the_old_implementation():
    """Главное: после делегирования RSI совпадает с прежней реализацией побитово."""
    _require()
    for seed, period, bars in ((1, 14, 300), (2, 14, 60), (3, 7, 150), (4, 21, 400)):
        close = walk(seed, bars)
        now = ta._wilder_rsi(close, period)
        before = reference_wilder_rsi(close, period)
        assert len(now) == len(close), "потеряна длина ряда"
        assert now.index.equals(close.index), "потерян индекс: вызывающий код работает по нему"
        assert (now - before).abs().max() == 0.0, (
            f"RSI разошёлся с прежней реализацией (seed={seed}, period={period}): "
            f"максимум {float((now - before).abs().max())}"
        )


def test_rsi_matches_the_domain_kernel_directly():
    """И делегирование действительно в домен (а не в третью копию внутри ta)."""
    _require()
    close = walk(1, 300)
    mine = ta._wilder_rsi(close, 14)
    kernel = domain_rsi.wilder_rsi(close.to_numpy(dtype=float), 14)
    assert (mine.to_numpy() - kernel).max() == 0.0, "ta._wilder_rsi не совпадает с ядром домена"


def test_rsi_edge_cases_keep_their_policies():
    """Политики разогрева и плоского ряда сохранены — иначе делегирование поменяло бы смысл.

    «Серия короче периода → сплошные 50» и «плоский ряд → 50, а не 100» это разные политики,
    и обе заданы в домене явно (``warmup``/``flat``). Проверяем, что ta получает именно их.
    """
    _require()
    short = walk(5, 10)
    out = ta._wilder_rsi(short, 14)
    assert (out == 50.0).all(), f"короткая серия должна давать нейтральные 50, получено {out.unique()[:3]}"

    flat = pd.Series([100.0] * 60)
    out = ta._wilder_rsi(flat, 14)
    assert (out == 50.0).all(), f"плоский ряд должен давать 50, а не 100: {out.unique()[:3]}"

    up_only = pd.Series([float(i) for i in range(1, 61)])
    out = ta._wilder_rsi(up_only, 14)
    assert float(out.iloc[-1]) > 99.0, f"ряд только из роста должен давать RSI→100, получено {out.iloc[-1]}"


# ====================================================================== #
#  EMA: варианты разные, и это закреплено
# ====================================================================== #
def test_ema_variants_differ_and_are_not_interchangeable():
    """EMA pandas и канонический ``pine_ema`` — **разные** показатели. Не взаимозаменяемы.

    Проверка фиксирует расхождение как факт: если кто-то «унифицирует» EMA в `ta.py`,
    подставив доменную, числа изменятся, и это обнаружится здесь, а не в чужом сигнале.
    """
    _require()
    close = walk(1, 300)
    pandas_ema = close.ewm(span=200, adjust=False).mean()
    pine = pd.Series(domain_primitives.pine_ema(close, 200), index=close.index)

    diff = (pandas_ema - pine).abs()
    assert float(diff.max()) > 0.1, (
        f"расхождение исчезло ({float(diff.max())}) — значит одна из реализаций изменилась, "
        "и утверждение «варианты разные» больше не верно; нужен пересмотр решения"
    )
    # Ключевое: на ПОСЛЕДНЕМ баре расхождение не ноль, а именно последний бар отдаёт
    # compute_indicators. Если бы оно затухало в ноль, делегирование было бы безопасным.
    assert float(diff.iloc[-1]) > 0.01, (
        "на последнем баре расхождение исчезло — пересмотрите вывод: делегирование EMA "
        "могло бы стать безопасным"
    )


def test_compute_indicators_uses_the_local_ema_variant():
    """`compute_indicators` считает EMA локальным вариантом — то самое закрепление.

    Проверяется по числам: значение EMA200 из отчёта обязано совпасть с локальной EMA
    (``ewm(adjust=False)``) и отличаться от доменной там, где расхождение ожидается.
    """
    _require()
    df = ohlc(1, 300)
    got = ta.compute_indicators(df)

    local = float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
    # .iloc, а не [-1]: pine_ema возвращает ряд с исходным индексом, и [-1] был бы
    # обращением по метке (KeyError), а не по позиции.
    pine = float(domain_primitives.pine_ema(df["Close"], 200).iloc[-1])

    assert got.ema200 == local, f"EMA200 в отчёте ({got.ema200}) больше не локальный вариант ({local})"
    if abs(local - pine) > 0.01:
        assert got.ema200 != pine, "отчёт совпал с доменной EMA — значит вариант подменили"


def test_rsi_in_indicators_comes_from_the_kernel():
    """RSI в отчёте — из ядра домена (через делегирование), а не из локальной копии."""
    _require()
    df = ohlc(1, 300)
    got = ta.compute_indicators(df)
    kernel = float(domain_rsi.wilder_rsi(df["Close"].to_numpy(dtype=float), 14)[-1])
    assert got.rsi == kernel, f"RSI в отчёте ({got.rsi}) не из ядра домена ({kernel})"


# ====================================================================== #
#  Шим: дублирующей реализации больше нет, примитивы — из домена
# ====================================================================== #
def test_rsi_has_no_local_implementation_left():
    """Структурная проверка: тело ``_wilder_rsi`` больше не содержит своей реализации.

    Иначе «делегирование» могло бы вернуться к копии, а тест паритета продолжал бы
    проходить (он сравнивает числа, а не устройство).
    """
    _require()
    src = inspect.getsource(ta._wilder_rsi)
    for marker in ("avg_gain.iloc[", "for i in range(period + 1", "clip(lower=0.0)"):
        assert marker not in src, f"в _wilder_rsi вернулась своя реализация: найдено {marker!r}"
    assert "domain.indicators" in src, "делегирование в домен пропало"


def test_ta_reexports_the_canonical_primitives():
    """`ta` — шим: примитивы это **те же объекты** домена, а не копии.

    Сравнение по идентичности (`is`), а не по имени: копия с тем же именем прошла бы
    проверку по имени и разошлась бы при первой же правке.
    """
    _require()
    pairs = (
        ("pine_ema", domain_primitives.pine_ema),
        ("pine_sma", domain_primitives.pine_sma),
        ("pine_rma", domain_primitives.pine_rma),
        ("pine_stdev", domain_primitives.pine_stdev),
    )
    for name, original in pairs:
        assert getattr(ta, name) is original, f"ta.{name} — не объект домена (копия или подмена)"

    from gex.domain.indicators import frames as domain_frames
    for name in ("compute_atr", "atr_series_wilder"):
        assert getattr(ta, name) is getattr(domain_frames, name), f"ta.{name} — не объект домена"


def test_compute_indicators_contract_is_unchanged():
    """Внешний контракт `compute_indicators` не пострадал от выноса."""
    _require()
    df = ohlc(1, 300)
    got = ta.compute_indicators(df)
    assert got.ema20 and got.ema50 and got.ema200, "EMA не посчитаны"
    assert 0.0 <= got.rsi <= 100.0, f"RSI вне диапазона: {got.rsi}"
    assert isinstance(got.macd_bull_cross, bool) and isinstance(got.ema_bull_stack, bool)

    for bad, message in ((None, "пустой вход"), (pd.DataFrame({"X": [1.0]}), "нет колонки Close")):
        try:
            ta.compute_indicators(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ожидалась ошибка: {message}")


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
    print(f"--- ta/domain parity: {len(tests) - failed - skipped} PASS / {failed} FAIL / {skipped} SKIP ---")
    sys.exit(1 if failed else 0)
