"""Стражи транспортного слоя (итерация 22).

Три правила, которые нельзя нарушать дальше:

1. **Сеть — только из ``gex/adapters/``.** Список оставшихся «голых» вызовов зафиксирован в
   ``quality-baseline/http_egress.json`` и может только уменьшаться (ратчет).
2. **``requests`` импортируется ровно в одном месте** — в ``adapters/transport/http.py``, и только
   внутри функции: иначе транспорт нельзя импортировать без установленных зависимостей.
3. **Консолидированные точки не regress:** пять модулей, которые раньше по-своему ходили в Bybit,
   больше не содержат сетевых вызовов.

Проверка идёт по AST, а не по тексту: в ``signal_service.py`` слово ``requests.get`` осталось
в докстринге, и текстовый grep дал бы ложное срабатывание (STATUS §2.3).

    python tests/test_transport_guard.py
    pytest tests/test_transport_guard.py -q
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "quality"))

from http_egress import scan  # noqa: E402

BASELINE = ROOT / "quality-baseline" / "http_egress.json"

#: Сетевые библиотеки, которые нельзя импортировать на уровне модуля (проверяется в
#: ``test_network_libraries_are_imported_lazily``). Держим одним множеством, чтобы правило
#: и бейзлайн не разошлись.
NETWORK_LIBS = frozenset({"requests", "httpx", "urllib3", "yfinance"})

#: Модули, из которых убрана собственная загрузка свечей Bybit (итерация 22).
BYBIT_MIGRATED = (
    "gex/application/ohlcv_service.py",
    "gex/application/novel_candles.py",
    "gex/application/signal_service.py",
    "gex/application/trendline_service.py",
    "gex/application/macd_trend_service.py",
)

#: Модули, переведённые на deadline-обёртку yfinance (итерация 23).
YF_MIGRATED = (
    "gex/adapters/fetchers/ta_fetcher.py",
    "gex/adapters/fetchers/yf_fetcher.py",
)


def _baseline() -> dict:
    assert BASELINE.exists(), (
        f"нет бейзлайна {BASELINE} — выполните: python scripts/quality/http_egress.py --write"
    )
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_no_new_files_with_direct_egress():
    """Новые «голые» вызовы запрещены: файл, которого нет в бейзлайне канала, — провал."""
    current = scan()
    baseline = _baseline()

    for channel, files in current.items():
        known = baseline.get(channel, {})
        new_files = sorted(set(files) - set(known))
        assert not new_files, (
            f"канал {channel}: появились прямые вызовы вне разрешённой зоны — "
            + ", ".join(new_files)
        )


def test_egress_count_does_not_grow():
    """Ратчет по каждому каналу: миграция только уменьшает список."""
    current = scan()
    baseline = _baseline()

    for channel, files in current.items():
        known = baseline.get(channel, {})
        total_now = sum(len(v) for v in files.values())
        total_base = sum(len(v) for v in known.values())
        assert total_now <= total_base, (
            f"канал {channel}: вызовов больше, чем в бейзлайне: {total_now} > {total_base}"
        )


def test_bybit_duplicates_stay_removed():
    """Пять копий загрузчика Bybit удалены — следим, чтобы они не вернулись."""
    current = scan()["http"]
    for path in BYBIT_MIGRATED:
        assert path not in current, f"{path} снова ходит в сеть напрямую"


def test_yfinance_migrated_sites_stay_removed():
    """Модули, переведённые на deadline-обёртку, не должны вернуться к ``yf.Ticker``."""
    current = scan()["yfinance"]
    for path in YF_MIGRATED:
        assert path not in current, f"{path} снова вызывает yfinance напрямую"


#: Точки входа CLI: там ``asyncio.run`` уместен — это старт процесса, а не вызов из сервиса.
ASYNCIO_RUN_ALLOWED = {"finagent/__main__.py"}


def test_no_asyncio_run_in_runtime_code():
    """``asyncio.run`` запрещён в рантайме: он убивает loop-bound объекты (аудит 05: F-10).

    Разрешённые места: ``gex/adapters/transport/loop.py`` (сам мост) и CLI-точки входа,
    где ``asyncio.run`` — это старт процесса, а не вызов на каждый запрос.
    """
    offenders = []
    for package in ("gex", "finagent"):
        for path in (ROOT / package).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel_path = str(path.relative_to(ROOT)).replace("\\", "/")
            if rel_path in ASYNCIO_RUN_ALLOWED or rel_path.endswith("transport/loop.py"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _dotted_name(node.func) == "asyncio.run":
                    offenders.append(f"{rel_path}:{node.lineno}")
    assert not offenders, (
        "asyncio.run создаёт новый loop на каждый вызов и убивает привязанные к нему объекты — "
        f"используйте AsyncBridge: {offenders}"
    )


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def test_yfinance_is_called_only_from_transport():
    """yfinance — только из ``adapters/transport/yf_transport.py``.

    У библиотеки нет timeout, поэтому вызывать её из сервисов нельзя ни при каких условиях:
    иначе воркер снова повиснет на неопределённое время.
    """
    offenders = []
    for path in (ROOT / "gex" / "adapters").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel_path = str(path.relative_to(ROOT)).replace("\\", "/")
        if rel_path == "gex/adapters/transport/yf_transport.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] == "yfinance" for a in node.names):
                    offenders.append(rel_path)
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("yfinance"):
                offenders.append(rel_path)
    assert not offenders, f"yfinance вызывается вне транспорта: {sorted(set(offenders))}"


def test_network_libraries_are_imported_lazily():
    """``requests``/``yfinance`` — только внутри функций транспорта.

    Модули обязаны импортироваться без сетевых библиотек: иначе логику ретраев и дедлайнов
    нельзя проверить в изолированном окружении (STATUS §4.5).

    Раньше здесь требовалось ``offenders == []`` по всему ``gex/adapters/**``. После
    раскладки плоских файлов по кольцам в это дерево попали модули, которые импортировали
    ``requests`` на уровне модуля **и до переноса** (``imoex_breadth_fetcher``,
    ``telegram_sender``, ``sec_edgar``, ``webull_fetcher``, ``finnhub_client``,
    ``telegram_polling``) — проверка стала наказывать за существующий долг, а не за
    регрессию. Поэтому:

    * требование «нет импорта на уровне модуля» действует **жёстко** на транспорт;
    * по остальным адаптерам ведётся тот же ратчет, что у кольцевых правил: долг
      зафиксирован в ``quality-baseline/lazy-imports.json`` и может только уменьшаться.
    """
    offenders = []
    for path in (ROOT / "gex" / "adapters").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        for node in tree.body:  # только верхний уровень: вложенный импорт — это lazily import
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if any(n.split(".")[0] in NETWORK_LIBS for n in names):
                offenders.append(rel)

    transport = [o for o in offenders if "/transport/" in o]
    assert not transport, f"транспорт импортирует сеть на уровне модуля: {transport}"

    baseline_path = ROOT / "quality-baseline" / "lazy-imports.json"
    if baseline_path.exists():
        allowed = set(json.loads(baseline_path.read_text(encoding="utf-8")).get("allowed", []))
    else:
        allowed = set()
    new = sorted(set(offenders) - allowed)
    assert not new, f"сетевые библиотеки импортируются на уровне модуля (новые): {new}"


def test_transport_does_not_know_about_providers():
    """Транспорт не знает про провайдеров: это правило разделения ответственности."""
    src = (ROOT / "gex" / "adapters" / "transport" / "http.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)

    # gex.settings — фасад конфигурации (кольцо frameworks): транспорт обязан брать таймауты
    # и число попыток оттуда, а не из собственных констант. Всё остальное в gex.* — нарушение.
    allowed = {"gex.settings"}
    leaking = {m for m in imported if m.startswith("gex.") and m not in allowed}
    assert not leaking, f"транспорт импортирует внутренности приложения: {sorted(leaking)}"


def _is_exception_class(cls: ast.ClassDef) -> bool:
    """Класс-исключение (по имени или по базе): его поля читаются снаружи."""
    if cls.name.endswith(("Error", "Exception")):
        return True
    for base in cls.bases:
        name = _dotted_name(base) or ""
        if name.split(".")[-1].endswith(("Error", "Exception")):
            return True
    return False


def test_transport_classes_have_no_write_only_attributes():
    """Параметр, сохранённый в ``self._x`` и нигде не прочитанный, — обманка для вызывающего.

    Реальный случай из итерации 22: ``HttpTransport(connect_timeout=...)`` присваивался полю и
    никогда не использовался — таймаут соединения живёт в отправителе. Код выглядел
    настраиваемым, а настройка ни на что не влияла.
    """
    offenders = []
    for path in sorted((ROOT / "gex" / "adapters" / "transport").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            if _is_exception_class(cls):
                # У исключений поля (url/status/retry_after/seconds) читает вызывающий код
                # снаружи класса — по AST это не видно, поэтому такие классы не проверяем.
                continue
            init = next(
                (f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"),
                None,
            )
            if init is None:
                continue

            params = {a.arg for a in init.args.args + init.args.kwonlyargs if a.arg != "self"}
            read = {
                node.attr
                for node in ast.walk(cls)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, ast.Load)
            }

            # Смотрим не на имя параметра, а на атрибут, в который он сохраняется:
            # `self._connect_timeout = connect_timeout` — параметр и поле называются по-разному,
            # и именно на этом первая версия проверки «прощала» реальный дефект.
            for node in ast.walk(init):
                if not isinstance(node, ast.Assign):
                    continue
                sources = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
                for target in node.targets:
                    if not (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        continue
                    if target.attr in read:
                        continue
                    for param in sources & params:
                        offenders.append(f"{path.name}::{cls.name}.{param} → self.{target.attr}")

    assert not offenders, (
        "параметры конструктора, которые сохраняются и никогда не читаются "
        f"(настройка ни на что не влияет): {offenders}"
    )


def test_provider_adapter_uses_transport_not_requests():
    """Адаптер Bybit обязан ходить в сеть через транспорт, а не через ``requests``."""
    path = ROOT / "gex" / "adapters" / "providers" / "bybit.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    targets = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                targets.update(a.name for a in node.names)
            elif node.module:
                targets.add(node.module)

    assert not {"requests", "httpx", "urllib.request"} & targets, (
        "адаптер Bybit импортирует сеть напрямую"
    )
    assert "gex.adapters.transport.http" in targets, "адаптер обязан использовать транспорт"


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
    print(f"--- transport guard: {len(tests) - failed} PASS / {failed} FAIL ---")
    sys.exit(1 if failed else 0)
