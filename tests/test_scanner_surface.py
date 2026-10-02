"""Поверхность сканеров: маршруты не пропали и формат сообщений не изменился (итерация 43).

Две независимые проверки в одном наборе, потому что обе отвечают на один вопрос заказчика
(«ручки signal scanner и auto-scanner не должны пострадать»):

1. **Маршруты.** Инвентарь по AST: путь, метод, обработчик. Статически — потому что fastapi
   в окружении аудита нет, а разбор исходников видит и состояние до выноса, и после.
2. **Формат.** Golden по выводу форматтеров: цены, время, суть сигнала, подпись и HTML-строка.
   Это то, что видит пользователь в Telegram, и «переписал форматирование» без эталона
   означает «поменял текст сообщений» — при этом числа в сигналах остаются верными, и
   никакая числовая проверка этого не заметит.

    python tests/test_scanner_surface.py
    python tests/test_scanner_surface.py --write-golden
"""
from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ROUTES_FIXTURE = ROOT / "tests" / "fixtures" / "scanner_routes.json"
FORMAT_FIXTURE = ROOT / "tests" / "fixtures" / "signal_formatters.json"

#: Файлы, чьи маршруты образуют «поверхность сканеров».
ROUTE_FILES = ["gex/routers/scanner_router.py", "gex/routers/signal_scanner_router.py",
               "gex/routers/auto_scanner_router.py"]

METHODS = {"get", "post", "put", "patch", "delete"}


class Skipped(Exception):
    """Проверка требует pandas/numpy (нет в stdlib-прогоне)."""


def _has_deps() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


# ====================================================================== #
#  1. Маршруты
# ====================================================================== #
def _router_prefix(tree: ast.AST) -> str:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "router" for t in node.targets):
            if isinstance(node.value, ast.Call):
                for kw in node.value.keywords:
                    if kw.arg == "prefix":
                        return str(ast.literal_eval(kw.value))
    return ""


def routes_in(path: Path) -> list[dict]:
    tree = ast.parse(path.read_text(encoding="utf-8").lstrip("\ufeff"))
    prefix = _router_prefix(tree)
    found = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            if not (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute)
                    and isinstance(deco.func.value, ast.Name) and deco.func.value.id == "router"
                    and deco.args and deco.func.attr in METHODS):
                continue
            route = ast.literal_eval(deco.args[0])
            found.append({"method": deco.func.attr.upper(), "path": route,
                          "handler": node.name, "full": prefix + route, "file": path.name})
    return found


def route_inventory() -> list[dict]:
    out = []
    for entry in ROUTE_FILES:
        path = ROOT / entry
        out += routes_in(path)
    return out


def test_scanner_routes_match_fixture():
    """Маршруты сканеров совпадают с эталоном: ни один не потерян и не переименован."""
    assert ROUTES_FIXTURE.exists(), f"нет эталона: {ROUTES_FIXTURE}"
    expected = json.loads(ROUTES_FIXTURE.read_text(encoding="utf-8"))
    got = route_inventory()
    keys = ("method", "path", "handler", "full")
    exp = {tuple(r[k] for k in keys) for r in expected}
    act = {tuple(r[k] for k in keys) for r in got}
    assert not exp - act, f"маршруты потеряны или изменены: {sorted(exp - act)}"
    assert not act - exp, f"появились новые маршруты: {sorted(act - exp)}"


def test_scanner_routes_are_not_vacuous():
    """Эталон покрывает оба сканера: и персональный, и авто."""
    expected = json.loads(ROUTES_FIXTURE.read_text(encoding="utf-8"))
    files = {r["file"] for r in expected}
    assert "signal_scanner_router.py" in files, "нет маршрутов персонального сканера"
    assert "auto_scanner_router.py" in files, "нет маршрутов авто-сканера"
    assert len(expected) >= 9, f"маршрутов всего {len(expected)}"

    handlers = {r["handler"] for r in expected}
    for required in ("get_signals", "run_signals_now", "run_scan", "reset_scanner"):
        assert required in handlers, f"пропала ручка {required}"
    assert {"GET", "POST"} <= {r["method"] for r in expected}, "нет хотя бы одного метода из GET/POST"


# ====================================================================== #
#  2. Формат сообщений
# ====================================================================== #
class FakeSignal:
    """Сигнал-заглушка: форматтеры работают через ``_field``, поэтому подходит любой носитель."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _signals() -> list:
    base = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    return [
        FakeSignal(action="BUY", direction="long", confidence=0.82, price=123.4567,
                   entry_price=123.4567, target=130.0, stop=118.0, timeframe="1h",
                   rationale="пробой сопротивления", ticker="SPY", score=0.77,
                   order_type="entry_long", reason="long_entry",
                   detected_at=base, created_at=base, timestamp=base),
        FakeSignal(action="SELL", direction="short", confidence=0.4, price=0.12345678,
                   entry_price=0.12345678, target=0.11, stop=0.13, timeframe="15m",
                   rationale="", ticker="DOGEUSDT", score=0.31,
                   order_type="exit_long", reason="long_exit",
                   detected_at=base - timedelta(hours=3), created_at=base, timestamp=base),
        FakeSignal(action="HOLD", direction="flat", confidence=0.1, price=100000.0,
                   ticker="BTC", timeframe="1d", rationale="боковик",
                   order_type="", reason="short_add",
                   detected_at=base - timedelta(days=1), created_at=base, timestamp=base),
        FakeSignal(action="BUY", direction="long", confidence=0.95, price=None,
                   ticker="TEST", timeframe="4h", rationale=None, score=None,
                   order_type="add_short", reason="",
                   detected_at=None, created_at=base, timestamp=base),
    ]


#: Сигналы, отличающиеся ТОЛЬКО ценой и временем бара. Суть обязана совпасть: форматтер
#: документирован как «без цены и времени бара», чтобы уведомление не уходило на каждый тик.
def _price_only_variants() -> list:
    base = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    return [
        FakeSignal(action="BUY", order_type="entry_long", price=100.0, detected_at=base),
        FakeSignal(action="BUY", order_type="entry_long", price=101.5,
                   detected_at=base + timedelta(minutes=5)),
        FakeSignal(action="BUY", order_type="entry_long", price=99.0,
                   detected_at=base + timedelta(hours=4)),
    ]


def _fmt_module():
    import gex.application.signal_scanner_service as svc
    return svc


def format_signature() -> dict:
    """Отпечаток форматтеров: все публичные функции форматирования на наборе сигналов."""
    svc = _fmt_module()
    when = datetime(2026, 9, 16, 15, 30, tzinfo=timezone.utc)
    out: dict = {"prices": [], "times": [], "essence": [], "chip": [], "html": [], "esc": []}
    for sig in _signals():
        out["prices"].append(svc.fmt_signal_price(getattr(sig, "price", None)))
        value = getattr(sig, "detected_at", None)
        out["times"].append(svc.fmt_signal_time(value))
        out["essence"].append(svc._signal_essence(sig))
        out["chip"].append(svc._signal_chip_label(sig))
        out["html"].append(svc.signal_line_html(
            getattr(sig, "ticker", "X"), getattr(sig, "timeframe", "1h"), sig, when))
    for raw in ("<b>x</b>", "a & b", 'кавычки "тут"', "уже 'так'", "", None):
        out["esc"].append(svc._esc_html(raw))
    return out


def test_formatters_match_fixture():
    """Текст сообщений совпадает с эталоном, снятым до выноса форматтеров."""
    if not _has_deps():
        raise Skipped("нужен numpy")
    assert FORMAT_FIXTURE.exists(), f"нет эталона: {FORMAT_FIXTURE}"
    expected = json.loads(FORMAT_FIXTURE.read_text(encoding="utf-8"))
    got = format_signature()
    for section in expected:
        assert expected[section] == got.get(section), (
            f"формат изменился в «{section}»:\n  было: {expected[section]}\n  стало: {got.get(section)}"
        )


def test_formatters_are_not_vacuous():
    """Эталон должен содержать содержательный текст, а не пустые строки.

    Каждая проверка здесь выведена из **контракта** форматтера, а не из ожидания автора теста.
    Первый вариант требовал, чтобы в «сути» сигнала была цена, и упал: ``_signal_essence``
    документирован как «БЕЗ цены и времени бара» — именно чтобы уведомление не уходило на
    каждый тик. Утверждение противоречило замыслу, и правильным оказался не код, а тест.
    """
    if not _has_deps():
        raise Skipped("нужен numpy")
    expected = json.loads(FORMAT_FIXTURE.read_text(encoding="utf-8"))
    non_empty_html = [x for x in expected["html"] if x]
    assert non_empty_html, "ни одна HTML-строка не сформирована"
    assert any("SPY" in x for x in non_empty_html), "в строках нет тикера"
    assert any("покупка" in x or "продажа" in x for x in non_empty_html), (
        "в строках нет направления — человеку нечего прочитать"
    )

    # Суть: формат «ACTION|order_type» и карта подписей покрыты обоими путями
    assert all("|" in x for x in expected["essence"]), expected["essence"]
    assert any(x.startswith("BUY|entry_long") for x in expected["essence"]), (
        "order_type не попадает в суть сигнала"
    )
    chips = [c for c in expected["chip"] if c]
    assert "LONG ENTRY" in chips, f"нет подписи по order_type: {chips}"
    assert "SHORT ADD" in chips, f"нет подписи по фолбэку reason: {chips}"

    assert any(x is None for x in expected["times"]), "не покрыта ветка отсутствующего времени"
    assert expected["prices"][-1] == "?", (
        f"цена None должна давать заглушку '?', получено {expected['prices'][-1]!r}"
    )
    assert any("&amp;" in str(x) for x in expected["esc"]), "экранирование HTML не сработало"


def test_essence_ignores_price_and_bar_time():
    """Свойство, ради которого «суть» и введена: смена цены не должна её менять.

    Проверяется на сигналах, различающихся ТОЛЬКО ценой и временем бара: если бы форматтер
    включал их в суть, каждое обновление цены считалось бы новым сигналом и пользователю
    уходило бы уведомление на каждый тик — то есть дефект был бы не в тексте, а в спаме.
    """
    if not _has_deps():
        raise Skipped("нужен numpy")
    svc = _fmt_module()
    essences = {svc._signal_essence(s) for s in _price_only_variants()}
    assert len(essences) == 1, f"цена/время бара попали в суть сигнала: {essences}"
    assert essences != {""}, "суть сигнала пуста — сравнение ничего не значит"


def test_escaping_pins_captured_behaviour():
    """Экранирование: спецсимволы уходят, ``None`` превращается в строку «None».

    Последнее — наблюдение, а не одобрение: ``_esc_html(None)`` возвращает текст ``None``,
    и если пропущенное поле попадёт в сообщение, пользователь увидит слово None. Здесь это
    зафиксировано, чтобы поведение не менялось незаметно; исправление — отдельное решение.
    """
    if not _has_deps():
        raise Skipped("нужен numpy")
    svc = _fmt_module()
    assert svc._esc_html("<b>x</b>") == "&lt;b&gt;x&lt;/b&gt;"
    assert svc._esc_html("a & b") == "a &amp; b"
    assert svc._esc_html(None) == "None", (
        "поведение _esc_html(None) изменилось — обновите закрепление и решите, что правильно"
    )


def _skip_unless_regenerating(message: str) -> None:
    if "pytest" in sys.modules:
        sys.modules["pytest"].skip(message)
    raise Skipped(message)


def test_regenerate_when_asked():
    """`--write-golden` перезаписывает оба эталона (осознанное действие)."""
    if "--write-golden" not in sys.argv:
        _skip_unless_regenerating("эталоны перезаписываются только по флагу --write-golden")
    ROUTES_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    routes = sorted(route_inventory(), key=lambda r: (r["full"], r["method"]))
    ROUTES_FIXTURE.write_text(json.dumps(routes, ensure_ascii=False, indent=1) + "\n",
                              encoding="utf-8")
    FORMAT_FIXTURE.write_text(
        json.dumps(format_signature(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"эталоны записаны: {ROUTES_FIXTURE.name} ({len(routes)} маршрутов), {FORMAT_FIXTURE.name}")


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
    print(f"--- scanner surface: {len(tests) - failed - skipped} PASS / {failed} FAIL / {skipped} SKIP ---")
    sys.exit(1 if failed else 0)
