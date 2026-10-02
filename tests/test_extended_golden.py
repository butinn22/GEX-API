"""Golden расширенного GEX-профиля: per-strike до и после выноса (итерация 39).

Зачем golden
------------
`extended.py` (993 строки: 7 dataclass-ов + 17 методов) разбирается на чистую математику
и оркестрацию. Проверить «не поменяли ли числа» иначе как сравнением всего отчёта нельзя:
тесты вида «профиль построился» пропускают и сдвиг страйка, и смену знака гаммы.

Эталон (`tests/fixtures/extended_golden.json`) снят с исходного файла: сначала
``--write-golden``, затем вынос, затем сверка.

Что попадает в эталон
---------------------
Отчёт целиком: скаляры (net/total GEX, AG, режим, смещение, нулевая гамма, стены,
max pain, put/call, метрики), **каждый страйк** (``per_strike``) со всеми полями, зоны
питания, ключевые уровни и сценарии хеджа. Вход детерминированный (сид фиксирован),
снапшот передаётся снаружи — сети нет.

Оговорка, которая важнее удобства
---------------------------------
Профиль объёма и ATR считаются по истории, а её в тестовом окружении нет (``yfinance``
не установлен), поэтому `analyze` деградирует: в отчёте видно предупреждение
``history(...) упал``. Эталон снимает **именно эту** (деградированную) ветку, и это
нормально для golden рефакторинга — обе стороны сравнения деградируют одинаково.
Но объёмный профиль этим набором **не покрыт**; он проверяется отдельно, когда появится
окружение с yfinance. Так честнее, чем считать покрытым то, что не исполнялось.

    python tests/test_extended_golden.py
    python tests/test_extended_golden.py --write-golden
"""
from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "tests" / "fixtures" / "extended_golden.json"

NDIGITS = 6


class Skipped(Exception):
    """Нужны pandas/numpy/scipy (нет в stdlib-прогоне) — не «зелёная», а пропущенная."""


_HAS_DEPS = True
_IMPORT_ERROR: Exception | None = None
pd = None
analyzer_cls = None
OptionSnapshot = None
try:
    import pandas as pd

    from gex.domain.data_loader import OptionSnapshot
    from gex.application.extended import ExtendedGEXAnalyzer as analyzer_cls
except ImportError as exc:
    _HAS_DEPS = False
    _IMPORT_ERROR = exc


def _require():
    if not _HAS_DEPS:
        raise Skipped(f"нужны pandas/numpy/scipy ({_IMPORT_ERROR})")


def _num(value):
    """JSON-совместимое число: NaN/бесконечность → None, чтобы дифф был читаемым."""
    if value is None or isinstance(value, bool):
        return None if value is None else float(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, NDIGITS)


def chain(spot: float = 100.0) -> "pd.DataFrame":
    """Цепочка с двумя выраженными страйками по OI, тремя сроками и улыбкой по IV."""
    rnd = random.Random(20260916)
    rows = []
    for T in (0.04, 0.10, 0.30):
        for strike in range(85, 116, 5):
            moneyness = abs(strike - spot) / spot
            iv = 0.20 + 0.9 * moneyness ** 2 + rnd.gauss(0, 0.002)
            # type = "C"/"P", а не "call"/"put": анализатор сравнивает именно с "C"
            # (extended.py: is_call = df["type"] == "C"). При другой конвенции профиль
            # получается ВЕСЬ ИЗ НУЛЕЙ и без единой ошибки — это отдельная находка,
            # закреплённая ниже проверкой, а не только этим комментарием.
            for kind in ("C", "P"):
                oi = 8000.0 if strike in (100.0, 105.0) else 400.0 + 30.0 * (15 - abs(strike - spot))
                # Асимметрия call/put обязательна: на симметричной цепочке net GEX = 0 при
                # ЛЮБЫХ знаках дилера, поэтому перепутанные call_sign/put_sign остались бы
                # незамеченными (AG и страйки не изменились бы). Здесь puts намеренно
                # легче call-ов — тогда net GEX отличен от нуля и знак наблюдаем.
                if kind == "P":
                    oi *= 0.7
                rows.append({"strike": float(strike), "type": kind, "oi": float(oi),
                             "iv": round(iv, 5), "T": T})
    return pd.DataFrame(rows)


def snapshot(spot: float = 100.0, symbol: str = "TEST"):
    return OptionSnapshot(symbol=symbol, spot=spot,
                          as_of=datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc),
                          chain=chain(spot))


def collect() -> dict:
    """Снять полный числовой отпечаток отчёта текущей реализацией."""
    analyzer = analyzer_cls()
    report = analyzer.analyze("TEST", days=30.0, max_expiries=3,
                              snapshot=snapshot(), source="yfinance")

    scalars = {}
    for name, value in vars(report).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scalars[name] = _num(value)
        elif isinstance(value, bool):
            scalars[name] = value
        elif isinstance(value, str):
            scalars[name] = value

    def row(obj):
        out = {}
        for name, value in vars(obj).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[name] = _num(value)
            elif isinstance(value, bool):
                out[name] = value
            elif isinstance(value, (str, type(None))):
                out[name] = value
        return out

    return {
        "scalars": scalars,
        "per_strike": [row(x) for x in (getattr(report, "per_strike", None) or [])],
        "power_zones": [row(x) for x in (getattr(report, "power_zones", None) or [])],
        "key_levels": [row(x) for x in (getattr(report, "key_levels", None) or [])],
        "hedge_scenarios": [row(x) for x in (getattr(report, "hedge_scenarios", None) or [])],
        "volume_zones": [row(x) for x in (getattr(report, "volume_zones", None) or [])],
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
    """Числа профиля совпадают с эталоном, снятым до разбора файла."""
    _require()
    assert FIXTURE.exists(), f"нет эталона: {FIXTURE}"
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    got = collect()
    for section in ("scalars", "per_strike", "power_zones", "key_levels",
                    "hedge_scenarios", "volume_zones"):
        if expected.get(section) != got.get(section):
            raise AssertionError(f"расхождение в «{section}»: {_first_difference(expected.get(section), got.get(section))}")


def test_golden_is_not_vacuous():
    """Эталон должен что-то содержать: иначе совпадение ничего не доказывает."""
    _require()
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(expected["per_strike"]) >= 5, f"страйков в эталоне: {len(expected['per_strike'])}"
    assert expected["power_zones"], "зоны питания пусты"
    assert expected["key_levels"], "ключевые уровни пусты"
    assert expected["scalars"].get("net_gex") not in (None, 0.0), (
        "net GEX равен нулю — цепочка симметрична, и знак дилера по эталону не проверить"
    )
    assert expected["scalars"].get("regime"), "режим не определён"
    # Страйки должны различаться: одинаковые значения означали бы, что профиль не посчитан.
    ags = {row.get("ag") for row in expected["per_strike"]}
    assert len(ags) > 1, f"все страйки одинаковы: {ags}"


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




def test_wrong_type_convention_gives_all_zero_profile_without_an_error():
    """Конвенция ``type`` не проверяется: ``call/put`` вместо ``C/P`` даёт нули молча.

    Находка итерации 39. Анализатор сравнивает тип с ``"C"``; любой другой ярлык
    превращает все опционы в путы, а профиль — в нули. Ни ошибки, ни предупреждения:
    отчёт выглядит корректным, числа правдоподобными (ноль), и отличить «профиль посчитан
    и он нулевой» от «конвенция не та» по ответу нельзя. Первый вариант эталона этой
    итерации именно на это и попался — golden оказался вырожденным (все AG = 0), и
    проверка вырожденности его отвергла.

    Закрепление: если поведение изменят на явную ошибку, тест упадёт и потребует
    осознанного переписывания — это и нужно.
    """
    _require()
    rows = []
    for T in (0.04, 0.10):
        for strike in range(90, 111, 5):
            for kind in ("call", "put"):
                rows.append({"strike": float(strike), "type": kind, "oi": 5000.0,
                             "iv": 0.22, "T": T})
    snap = OptionSnapshot(symbol="TEST", spot=100.0,
                          as_of=datetime(2026, 9, 16, tzinfo=timezone.utc),
                          chain=pd.DataFrame(rows))
    report = analyzer_cls().analyze("TEST", days=30.0, max_expiries=2,
                                    snapshot=snap, source="yfinance")
    assert float(report.total_ag or 0.0) == 0.0, (
        "поведение изменилось: профиль больше не пустеет на call/put — снимите это "
        "закрепление и обновите комментарий"
    )

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
    print(f"--- extended golden: {len(tests) - failed - skipped} PASS / {failed} FAIL / {skipped} SKIP ---")
    sys.exit(1 if failed else 0)
