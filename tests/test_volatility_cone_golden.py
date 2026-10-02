"""Golden конуса волатильности и VPC: числа до и после разбора (итерация 41).

Зачем golden
------------
``volatility_cone.py`` (954 строки) разбирается на ``domain/indicators/{quarterly_cone,vpc}``.
Функции выдают длинные кадры (десятки колонок, включая границы σ и коррекции), и проверить
«не поменяли ли числа» можно только сравнением с замороженным эталоном: тест вида «колонка
появилась» пропустит и сдвиг окна, и смену множителя σ.

Эталон снят с **исходного** файла: сначала ``--write-golden``, затем разбор, затем сверка.
Сверяются три точки входа, потому что разбор их касается по-разному:

* ``compute_all`` — то, что вызывает роутер (полный кадр: конус + VPC);
* ``compute_volatility_cone`` с **не-дефолтными** параметрами (включая выключенную
  коррекцию) — иначе ветки параметров не были бы покрыты;
* ``compute_vpc`` с нестандартной длиной окна.

Вход детерминирован (сид зафиксирован) и имеет дневной DatetimeIndex: ``detect_quarters``
работает по кварталам календаря, и на RangeIndex он не имеет смысла.

    python tests/test_volatility_cone_golden.py
    python tests/test_volatility_cone_golden.py --write-golden
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "tests" / "fixtures" / "volatility_cone_golden.json"

#: Сколько последних строк кадра попадает в эталон (полный кадр слишком велик для диффа).
TAIL = 40
NDIGITS = 8


class Skipped(Exception):
    """Нужны pandas/numpy (нет в stdlib-прогоне) — не «зелёная», а пропущенная."""


_HAS_DEPS = True
_IMPORT_ERROR: Exception | None = None
pd = None
vc = None
try:
    import pandas as pd

    import gex.domain.volatility_cone as vc
except ImportError as exc:
    _HAS_DEPS = False
    _IMPORT_ERROR = exc


def _require():
    if not _HAS_DEPS:
        raise Skipped(f"нужны pandas/numpy ({_IMPORT_ERROR})")


def _num(value):
    if value is None or isinstance(value, bool):
        return None if value is None else float(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, NDIGITS)


def ohlc(bars: int = 700, seed: int = 20260916) -> "pd.DataFrame":
    """Дневной ряд с трендовыми участками и объёмом: конусу нужна история за lookback."""
    rnd = random.Random(seed)
    index = pd.date_range("2024-01-01", periods=bars, freq="D")
    price = 100.0
    rows = []
    for i in range(bars):
        drift = 0.0012 if i % 180 < 120 else -0.0009
        close = max(1.0, price * (1.0 + drift + rnd.gauss(0, 0.011)))
        open_ = price
        high = max(open_, close) * (1.0 + abs(rnd.gauss(0, 0.005)))
        low = min(open_, close) * (1.0 - abs(rnd.gauss(0, 0.005)))
        # Колонки СТРОЧНЫМИ буквами: этот модуль (в отличие от ta.py) ждёт open/high/low/close
        # и падает с KeyError('open') на кадрах с заглавными.
        rows.append({"open": open_, "high": high, "low": low, "close": close,
                     "volume": 1_000_000.0 * (1.0 + abs(rnd.gauss(0, 0.4)))})
        price = close
    return pd.DataFrame(rows, index=index)


def frame_signature(df: "pd.DataFrame", tail: int = TAIL) -> dict:
    """Числовая подпись кадра: индекс + последние строки всех числовых колонок."""
    out: dict = {"index": [str(x) for x in df.index[-tail:]], "columns": {}}
    for name in df.columns:
        col = df[name].tail(tail)
        if col.dtype == bool:
            out["columns"][str(name)] = [bool(v) for v in col]
        elif str(col.dtype).startswith(("float", "int")):
            out["columns"][str(name)] = [_num(v) for v in col]
    return out


def collect() -> dict:
    data = ohlc()
    return {
        "compute_all": frame_signature(vc.compute_all(data)),
        "compute_all_custom": frame_signature(vc.compute_all(
            data, cone_params=dict(lookback_days=300, sd1_mult=1.5, sd2_mult=2.5,
                                   use_correction=False, rsi_influence=0.0),
            vpc_length=30)),
        "cone_only": frame_signature(vc.compute_volatility_cone(
            data, lookback_days=400, ema_len=34, bb_mult=2.5, carry_weight=0.5)),
        "cone_no_correction": frame_signature(vc.compute_volatility_cone(
            data, lookback_days=250, use_correction=False, correction_pct=0.0)),
        "vpc": frame_signature(vc.compute_vpc(data, length=14)),
    }


def _first_difference(a, b, path="") -> str:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                return _first_difference(a.get(key), b.get(key), f"{path}.{key}")
        return f"{path}: словари равны"
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: длина {len(a)} → {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return _first_difference(x, y, f"{path}[{i}]")
        return f"{path}: списки равны"
    return f"{path}: {a!r} → {b!r}"


def test_golden_matches_fixture():
    """Числа конуса и VPC совпадают с эталоном, снятым до разбора файла."""
    _require()
    assert FIXTURE.exists(), f"нет эталона: {FIXTURE}"
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    got = collect()
    for section in expected:
        if expected[section] != got.get(section):
            raise AssertionError(f"расхождение в «{section}»: "
                                 f"{_first_difference(expected[section], got.get(section))}")


def test_golden_is_not_vacuous():
    """Эталон должен содержать посчитанные колонки с разными значениями."""
    _require()
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    all_frame = expected["compute_all"]
    assert len(all_frame["columns"]) >= 15, f"колонок всего {len(all_frame['columns'])}"
    assert all_frame["index"], "индекс не сохранён"

    for name, values in all_frame["columns"].items():
        filled = [v for v in values if v is not None]
        if not filled or all(isinstance(v, bool) for v in filled):
            continue  # флаговые колонки (is_new_quarter) законно постоянны в окне
        assert len(set(filled)) > 1, f"колонка {name} постоянна — расчёт не выполнен"

    # Ключевые колонки конуса и канала: имена взяты из фактического вывода, а не по памяти
    # (в этом модуле медиана называется median_price, а не basis).
    for required in ("median_price", "upper_1sd", "lower_1sd", "upper_2sd", "lower_2sd",
                     "vpc_upper", "vpc_lower", "vpc_mid", "qema21", "daily_volatility"):
        assert required in all_frame["columns"], f"нет колонки {required}"
        assert any(v is not None for v in all_frame["columns"][required]), f"{required} пуста"

    # Полосы обязаны быть упорядочены: это свойство самого расчёта, и его сдвиг означает
    # поломку, которую по отдельным числам можно не заметить.
    for upper, lower in (("upper_1sd", "lower_1sd"), ("upper_2sd", "lower_2sd"),
                         ("vpc_upper", "vpc_lower")):
        pairs = [(u, l) for u, l in zip(all_frame["columns"][upper], all_frame["columns"][lower])
                 if u is not None and l is not None]
        assert pairs, f"нет строк для сравнения {upper}/{lower}"
        assert all(u >= l for u, l in pairs), f"{upper} ниже {lower} — поломка порядка полос"


def test_regenerate_when_asked():
    """`--write-golden` перезаписывает эталон (осознанное действие)."""
    _require()
    if "--write-golden" not in sys.argv:
        _skip_unless_regenerating("эталон перезаписывается только по флагу --write-golden")
    data = collect()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
    print(f"эталон записан: {FIXTURE}")


def _skip_unless_regenerating(message: str) -> None:
    """Пропустить тест так, чтобы это понял и pytest, и собственный раннер.

    Различие обязательное: набор запускается и скриптом, и через pytest. Под pytest нужен
    его собственный ``skip`` (иначе он видит FAILED), вне pytest — наш ``Skipped`` (иначе
    исключение pytest не поймает раннер и запуск скриптом упадёт вместо SKIP).
    Признак «идёт pytest» — сам модуль ``pytest`` в ``sys.modules``: он появляется там
    только когда тесты запускает pytest, а не когда модуль просто установлен.
    """
    if "pytest" in sys.modules:
        sys.modules["pytest"].skip(message)
    raise Skipped(message)


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
            print(f"FAIL {fn.__name__}: {str(exc)[:400]}")
            failed += 1
        except Exception as exc:  # НЕОЖИДАННОЕ
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:400]}")
            failed += 1
    print(f"--- volatility cone golden: {len(tests) - failed - skipped} PASS / {failed} FAIL / {skipped} SKIP ---")
    sys.exit(1 if failed else 0)
