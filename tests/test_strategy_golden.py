"""Golden конвейера стратегии: числа до и после декомпозиции (итерации 37–40).

Зачем golden именно здесь
-------------------------
`EMAFilterTrendStrategy` — 1102 строки и 46 методов: вынести их в модули механически можно,
но проверить «не поменяли ли поведение» иначе как сравнением чисел нельзя. Тесты вида
«сигнал вообще появился» такое не ловят: перепутанный порядок колонок или сдвиг EMA на бар
даёт ровно те же «BUY/SELL», но другие цены.

Как это устроено
----------------
Вход детерминированный (сид зафиксирован), поэтому фикстура воспроизводима. Эталон
(`tests/fixtures/strategy_golden.json`) снят с **исходного** кода до выноса: сначала
запускается `python tests/test_strategy_golden.py --write-golden`, потом делается вынос,
потом тот же файл сверяется с фикстурой. Так «переписал и стало так же» превращается
в проверяемое утверждение.

Что попадает в эталон
---------------------
* все числовые колонки конвейера за последние бары (полный набор фич и сигналов);
* решения `evaluate` в четырёх состояниях позиции (в том числе с добавлением) — для
  нескольких баров;
* параметры режима, метрики режима, оценка входа, класс уверенности;
* цены тейк-профита и трейлинг-стопа.

Отдельно проверяется, что эталон **не вырожден**: в нём есть не-`HOLD` решения и заполненные
колонки. Иначе «всё совпало» означало бы «всё равно пусто».

    python tests/test_strategy_golden.py                # сверить
    python tests/test_strategy_golden.py --write-golden # перезаписать эталон (осознанно!)
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "tests" / "fixtures" / "strategy_golden.json"

#: Сколько последних баров попадает в эталон (полный кадр слишком велик для диффа).
TAIL_BARS = 60

#: Округление: сравнение должно ловить смысловые сдвиги, а не последний бит float.
NDIGITS = 6


class Skipped(Exception):
    """Нужны pandas/numpy (нет в stdlib-прогоне) — не «зелёная», а пропущенная."""


_HAS_DEPS = True
_IMPORT_ERROR: Exception | None = None
pd = None
ta = None
try:
    import pandas as pd

    from gex.strategy.trading_algorithm import (
        EMAFilterTrendStrategy,
        GEXContext,
        SignalAction,
        StrategySettings,
        TradingState,
    )
except ImportError as exc:
    _HAS_DEPS = False
    _IMPORT_ERROR = exc


def _require():
    if not _HAS_DEPS:
        raise Skipped(f"нужны pandas/numpy ({_IMPORT_ERROR})")


# ====================================================================== #
#  Детерминированный вход
# ====================================================================== #
def make_ohlc(bars: int = 400) -> "pd.DataFrame":
    """Ценовой ряд с тремя режимами (рост, падение, боковик) — чтобы сработали сигналы.

    Сид фиксирован: без этого эталон нельзя воспроизвести, а значит нельзя и сверить.
    Смена режима нужна, чтобы в кадре были и входы, и выходы, а не один режим подряд.
    """
    rnd = random.Random(20260916)
    price = 100.0
    rows = []
    for i in range(bars):
        if i < 140:
            drift = 0.0016      # рост
        elif i < 290:
            drift = -0.0013     # падение
        else:
            drift = 0.0022      # снова рост — чтобы в последних барах сработали ОБА входа
        # Третий участок подобран по факту: при боковике (+0.0004) в хвосте срабатывали
        # только короткие входы, и эталон хвоста покрывал половину логики входов.
        shock = rnd.gauss(0.0, 0.009)
        close = max(1.0, price * (1.0 + drift + shock))
        open_ = price
        high = max(open_, close) * (1.0 + abs(rnd.gauss(0.0, 0.004)))
        low = min(open_, close) * (1.0 - abs(rnd.gauss(0.0, 0.004)))
        volume = 1_000_000.0 * (1.0 + abs(rnd.gauss(0.0, 0.3)))
        rows.append({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
        price = close
    return pd.DataFrame(rows)


def _num(value) -> float | None:
    """Привести значение к JSON-совместимому числу (NaN → None, чтобы дифф читался)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, NDIGITS)


def collect() -> dict:
    """Снять весь числовой отпечаток конвейера текущей реализацией."""
    strategy = EMAFilterTrendStrategy(StrategySettings())
    ohlc = make_ohlc()
    features = strategy.calculate(ohlc)

    # 1) числовые колонки кадра за последние бары
    tail = features.tail(TAIL_BARS)
    columns: dict[str, list] = {}
    for name in sorted(features.columns):
        col = tail[name]
        if col.dtype == bool:
            columns[name] = [bool(v) for v in col]
        elif str(col.dtype).startswith(("float", "int")):
            columns[name] = [_num(v) for v in col]

    # 2) решения evaluate в разных состояниях позиции — по нескольким барам
    decisions = []
    close = float(ohlc["close"].iloc[-1])
    states = [
        ("flat", TradingState(position_side="flat", position_qty=0.0)),
        ("long", TradingState(position_side="long", position_qty=1.0,
                              long_entry_price=close * 0.98)),
        ("long_add", TradingState(position_side="long", position_qty=1.0,
                                  long_entry_price=close * 0.98, last_add_bar=0)),
        ("short", TradingState(position_side="short", position_qty=1.0,
                               short_entry_price=close * 1.02)),
        ("short_add", TradingState(position_side="short", position_qty=1.0,
                                   short_entry_price=close * 1.02, last_add_bar=0)),
    ]
    for cut in (0, 1, 5, 20):
        window = features.iloc[: len(features) - cut] if cut else features
        for label, state in states:
            sig = strategy.evaluate(window, state=state)
            decisions.append({
                "bars_back": cut, "state": label,
                "action": str(sig.action),
                "reason": sig.reason,
                "qty_fraction": _num(sig.quantity_fraction),
                # order_type и close_price сигнал кладёт в metadata — атрибутов с такими
                # именами нет, и обращение к ним дало бы None в эталоне (проверка ослабла бы)
                "close_price": _num(sig.metadata.get("close_price")),
                "order_type": sig.metadata.get("order_type"),
            })

    # 3) режим и метрики
    params = strategy.regime_params()
    sliders = strategy.regime_sliders()
    metrics = strategy.trend_regime_metrics(features) or {}
    regime = {
        "params": {k: _num(v) for k, v in vars(params).items()},
        "sliders": {k: _num(v) for k, v in vars(sliders).items()},
        "metrics": {k: _num(v) for k, v in metrics.items()},
    }

    # 4) оценка входа и класс уверенности.
    #    confidence_class — статический: (оценка входа, множитель GEX, v-оценка, порог).
    #    Перебираем и «v-оценка неизвестна», и граничные комбинации: класс — это решение,
    #    и он должен совпасть побайтово, а не «примерно».
    row_scores = []
    last = features.iloc[-1]
    for direction in ("long", "short"):
        score = strategy.score_entry_quality(last, direction)
        row_scores.append({
            "direction": direction,
            "score": _num(score),
            "confidence": strategy.confidence_class(score, 1.0, 60.0, 60.0),
        })
    for score, mult, v_score, threshold in (
        (0.9, 1.0, 60.0, 60.0), (0.7, 1.0, 60.0, 60.0), (0.5, 1.4, 60.0, 60.0),
        (0.9, 1.0, 60.0, 80.0), (0.9, 1.0, None, 60.0), (0.0, 0.5, 0.0, 100.0),
        (0.8, 2.0, 200.0, 0.0),
    ):
        row_scores.append({
            "direction": f"grid({score},{mult},{v_score},{threshold})",
            "score": _num(score),
            "confidence": strategy.confidence_class(score, mult, v_score, threshold),
        })

    # 4б) вердикт режима и разрешение сигнала
    regime_verdict = strategy.trend_regime(features) or {}
    verdict = {k: _num(v) for k, v in regime_verdict.items() if k != "metrics"}

    # 5) риск: тейк-профит и трейлинг-стоп
    risk = []
    for direction in ("long", "short"):
        for atr_value in (None, 2.5):
            risk.append({
                "direction": direction,
                "atr": _num(atr_value),
                "take_profit": _num(strategy.take_profit_price(close, direction, atr_value)),
                "trailing": _num(strategy.trailing_stop_price(
                    close, direction,
                    highest_price=close * 1.03, lowest_price=close * 0.97,
                    atr_value=atr_value,
                )),
            })

    # 6) GEX-фильтр: контекст задаётся синтетически
    gex = []
    for action in ("BUY", "SELL", "HOLD"):
        for z in (None, 2.0):
            ctx = GEXContext(regime="positive", net_gex=1.2e9, gamma_flip=close * 0.995,
                             z_score=z, call_wall=close * 1.03, put_wall=close * 0.97)
            sig = strategy._signal(SignalAction[action], "golden", close_price=close)
            filtered, mult, why = strategy.apply_gex_filter(sig, ctx)
            gex.append({"action": action, "z": _num(z), "out": str(filtered.action),
                        "mult": _num(mult), "why": why})
    # контекст недоступен — фильтр обязан пропустить сигнал без изменений
    sig = strategy._signal(SignalAction.BUY, "golden", close_price=close)
    filtered, mult, why = strategy.apply_gex_filter(sig, None)
    gex.append({"action": "BUY", "z": None, "out": str(filtered.action),
                "mult": _num(mult), "why": why})

    return {
        "bars": int(len(ohlc)),
        "columns": columns,
        "decisions": decisions,
        "regime": regime,
        "regime_verdict": verdict,
        "scoring": row_scores,
        "risk": risk,
        "gex_filter": gex,
        "ohlc_tail": {c: [_num(v) for v in ohlc[c].tail(TAIL_BARS)] for c in ohlc.columns},
    }


# ====================================================================== #
#  Сверка
# ====================================================================== #
def golden_hash(data: dict) -> str:
    """Устойчивая подпись всего эталона — для быстрой сверки в отчёте."""
    import hashlib

    blob = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def test_golden_matches_fixture():
    """Главная проверка: числа конвейера совпадают с эталоном, снятым до декомпозиции."""
    _require()
    assert FIXTURE.exists(), f"нет эталона: {FIXTURE} — снимите его на исходном коде"
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    got = collect()

    for section in ("columns", "decisions", "regime", "regime_verdict", "scoring",
                    "risk", "gex_filter"):
        if expected.get(section) != got.get(section):
            diff = _first_difference(expected.get(section), got.get(section))
            raise AssertionError(f"расхождение в «{section}»: {diff}")

    assert expected["bars"] == got["bars"], "изменилось число баров входа"


def _first_difference(a, b, path="") -> str:
    """Показать первое расхождение — иначе дифф на тысячи чисел нечитаем."""
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


def test_golden_is_not_vacuous():
    """Эталон не должен быть вырожденным: иначе «всё совпало» ничего не значит."""
    _require()
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))

    signals = {d["action"] for d in expected["decisions"]}
    assert signals - {"SignalAction.HOLD"}, "в эталоне нет ни одного не-HOLD решения"
    assert len(expected["columns"]) > 50, f"колонок в эталоне всего {len(expected['columns'])}"

    filled = {name for name, values in expected["columns"].items()
              if any(v is not None for v in values)}
    assert len(filled) >= len(expected["columns"]) - 2, (
        f"пустых колонок слишком много: {sorted(set(expected['columns']) - filled)[:6]}"
    )
    tail_entries = {
        name: sum(1 for v in expected["columns"].get(name, []) if v)
        for name in ("combined_long_entry", "combined_short_entry",
                     "combined_long_exit", "combined_short_exit")
    }
    assert any(tail_entries.values()), (
        f"в хвосте эталона нет ни одного срабатывания — проверять нечего: {tail_entries}"
    )
    # Оба направления входа должны встречаться в эталоне: иначе сравнение покрывает
    # только половину логики (на этом и споткнулся первый вариант входа — см. make_ohlc).
    sample = collect()
    for name in ("combined_long_entry", "combined_short_entry"):
        total = sum(1 for v in sample["columns"].get(name, []) if v)
        assert total >= 0, name
    entry_kinds = {
        name for name in ("combined_long_entry", "combined_short_entry")
        if any(v for v in sample["columns"].get(name, []))
    }
    assert len(entry_kinds) == 2, f"в эталонном входе сработал только один вид входа: {entry_kinds}"
    assert expected["regime"]["params"], "параметры режима пусты"
    classes = {r["confidence"] for r in expected["scoring"]}
    assert len(classes) > 1, f"класс уверенности вырожден: {classes}"
    assert expected["risk"] and expected["risk"][0]["take_profit"] is not None, "риск не считается"


def test_regenerate_when_asked():
    """`--write-golden` перезаписывает эталон (осознанное действие, а не побочный эффект)."""
    _require()
    if "--write-golden" not in sys.argv:
        _skip_unless_regenerating("эталон перезаписывается только по флагу --write-golden")
    data = collect()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
    print(f"эталон записан: {FIXTURE} (подпись {golden_hash(data)})")


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
        except Exception as exc:  # НЕОЖИДАННОЕ: иначе прогон обрывался, и «0 FAIL» врало
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:400]}")
            failed += 1
    print(f"--- strategy golden: {len(tests) - failed - skipped} PASS / {failed} FAIL / {skipped} SKIP ---")
    sys.exit(1 if failed else 0)
