"""Инвентарь маршрутов админки: пути не должны измениться (итерация 42).

Почему инвентарь, а не golden по числам
---------------------------------------
Критерий итерации — «пути OpenAPI не изменились». Проверять его нужно так же строго, как
числа в других итерациях, но golden по значениям здесь не подходит: разбор роутера не меняет
вычислений, он меняет **адреса**. Поэтому эталон — список маршрутов, снятый с исходного файла.

Почему статически (AST), а не через `app.openapi()`
---------------------------------------------------
Импорт роутера требует `fastapi`, которого в окружении аудита нет: тесты, тянущие fastapi,
там просто не запускаются. Разбор по AST работает в любом окружении и, что важнее, видит
исходники **обоих** состояний — до разбора (один файл) и после (фасад + подмодули).

Что проверяется
---------------
1. Множество ``(метод, путь, обработчик)`` совпадает с эталоном — ни один путь не потерян,
   не переименован и не получил другой метод.
2. Полные пути (префикс роутера + путь) совпадают: если бы подмодуль объявил свой префикс,
   адреса удвоились бы, и это видно здесь.
3. Порядок внутри группы пользователей сохранён: ``/users/export`` обязан идти до
   ``/users/{user_id}``, иначе FastAPI начнёт трактовать ``export`` как идентификатор.
4. Эталон не вырожден: 35 маршрутов, все четыре группы непусты, присутствуют все методы.

    python tests/test_admin_routes.py
    python tests/test_admin_routes.py --write-golden
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "tests" / "fixtures" / "admin_routes.json"

#: Где искать маршруты: фасад и (после разбора) подмодули пакета.
SOURCE_FILES = ["gex/auth/admin_router.py", "gex/auth/admin"]

METHODS = {"get", "post", "put", "patch", "delete"}


class Skipped(Exception):
    """Проверка не применима к текущему состоянию файлов."""


def _router_prefix(tree: ast.AST) -> str:
    """Префикс, с которым объявлен `router = APIRouter(...)` в этом файле."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "router" for t in node.targets):
            continue
        if isinstance(node.value, ast.Call):
            for kw in node.value.keywords:
                if kw.arg == "prefix":
                    return str(ast.literal_eval(kw.value))
    return ""


def _files() -> list[Path]:
    out = []
    for entry in SOURCE_FILES:
        path = ROOT / entry
        if path.is_dir():
            out += sorted(p for p in path.glob("*.py") if p.name != "__init__.py")
        elif path.exists():
            out.append(path)
    return out


def routes_in(path: Path) -> list[dict]:
    """Все маршруты файла: метод, путь, обработчик, полный путь и файл."""
    tree = ast.parse(path.read_text(encoding="utf-8").lstrip("\ufeff"))
    prefix = _router_prefix(tree)
    found = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            if not (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute)):
                continue
            fn = deco.func
            if not (isinstance(fn.value, ast.Name) and fn.value.id == "router" and fn.attr in METHODS):
                continue
            if not deco.args:
                continue
            route = ast.literal_eval(deco.args[0])
            found.append({"method": fn.attr.upper(), "path": route, "handler": node.name,
                          "full": prefix + route, "file": path.name})
    return found


FACADE = "gex/auth/admin_router.py"


def _root_prefix() -> str:
    """Префикс точки входа: именно он применяется ко всем подроутерам пакета.

    Считать полный путь как «префикс своего файла + путь» было бы неверно: после разбора
    маршруты лежат в подмодулях без префикса, а монтирует их фасад с префиксом `/auth/admin`.
    Композиция FastAPI работает так же, поэтому и проверка должна.
    """
    tree = ast.parse((ROOT / FACADE).read_text(encoding="utf-8").lstrip("\ufeff"))
    return _router_prefix(tree)


def inventory() -> list[dict]:
    prefix = _root_prefix()
    out = []
    for path in _files():
        for route in routes_in(path):
            route = dict(route, full=prefix + route["path"])
            out.append(route)
    return out


def key(route: dict) -> tuple:
    return (route["method"], route["path"], route["handler"], route["full"])


def test_routes_match_fixture():
    """Множество маршрутов совпадает с эталоном, снятым до разбора."""
    assert FIXTURE.exists(), f"нет эталона: {FIXTURE}"
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    got = inventory()

    expected_keys = {tuple(r[k] for k in ("method", "path", "handler", "full")) for r in expected}
    got_keys = {key(r) for r in got}
    missing = expected_keys - got_keys
    added = got_keys - expected_keys
    assert not missing, f"маршруты потеряны или изменены: {sorted(missing)}"
    assert not added, f"появились маршруты, которых не было: {sorted(added)}"
    assert len(got) == len(got_keys), "дубли маршрутов — один путь зарегистрирован дважды"


def test_ordering_constraints_hold():
    """`/users/export` обязан регистрироваться до `/users/{user_id}`.

    В исходном файле это отмечено комментарием «MUST be before /users/{user_id}»: при обратном
    порядке FastAPI примет слово ``export`` за идентификатор пользователя, и выгрузка сломается
    (а обычные запросы — нет, поэтому поломку легко не заметить при беглом прогоне).
    """
    got = inventory()
    users = [r for r in got if r["full"].startswith("/auth/admin/users")]
    assert users, "маршруты пользователей не найдены"

    literal = [i for i, r in enumerate(users) if r["full"] == "/auth/admin/users/export"]
    dynamic = [i for i, r in enumerate(users) if "{" in r["full"]]
    assert literal and dynamic, (literal, dynamic)
    assert literal[0] < dynamic[0], (
        "порядок нарушен: параметризованный путь зарегистрирован раньше /users/export"
    )


def test_inventory_is_not_vacuous():
    """Эталон должен содержать все группы и методы: иначе совпадение ничего не значит."""
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(expected) >= 35, f"маршрутов в эталоне всего {len(expected)}"

    groups = {
        "users": [r for r in expected if r["path"].startswith("/users")],
        "payments": [r for r in expected if r["path"].startswith(("/payments", "/payment-requisites"))],
        "system": [r for r in expected if r["path"].startswith(("/system", "/metrics", "/stats",
                                                               "/http-stats", "/logs", "/db-stats",
                                                               "/cache", "/db", "/redis"))],
        "notify": [r for r in expected if r["path"].startswith(("/email-config", "/telegram-config",
                                                                "/finagent-key", "/test-email"))],
    }
    for name, routes in groups.items():
        assert routes, f"группа «{name}» пуста в эталоне"
    assert sum(len(v) for v in groups.values()) == len(expected), (
        "есть маршруты вне четырёх групп — раскладка не соответствует плану"
    )

    method_set = {r["method"] for r in expected}
    assert {"GET", "POST", "PUT", "PATCH", "DELETE"} <= method_set, (
        f"не все методы представлены: {sorted(method_set)}"
    )
    # Уникальность проверяется по паре (метод, путь), а не по пути: один и тот же адрес
    # законно обслуживает несколько глаголов (``/email-config`` — GET и PUT), и требование
    # уникальности только по пути было бы неверным.
    pairs = {(r["method"], r["full"]) for r in expected}
    assert len(pairs) == len(expected), "в эталоне повторяются пары (метод, путь)"


def test_source_files_are_intact():
    """Фасад и подмодули существуют, и префикс объявлен ровно в одном месте.

    Подроутер со своим префиксом удвоил бы адрес (`/auth/admin/auth/admin/...`), и это
    не поймать ни одним инвентарём, если считать префиксы по файлам. Поэтому проверка явная:
    префикс — только у точки входа.
    """
    files = _files()
    assert files, "не найдено ни одного файла с маршрутами"
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8").lstrip("\ufeff"))
        routers = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "router" for t in n.targets)]
        assert len(routers) <= 1, f"{path.name}: объявлено несколько роутеров"
        if path.name != Path(FACADE).name:
            assert _router_prefix(tree) == "", (
                f"{path.name}: подроутер объявил префикс — адреса удвоятся"
            )
    facade = ROOT / FACADE
    assert facade.exists(), "точка входа gex/auth/admin_router.py обязана остаться"
    assert _root_prefix() == "/auth/admin", f"префикс точки входа изменился: {_root_prefix()!r}"


def _skip_unless_regenerating(message: str) -> None:
    """Пропустить так, чтобы это понял и pytest, и собственный раннер."""
    if "pytest" in sys.modules:
        sys.modules["pytest"].skip(message)
    raise Skipped(message)


def test_regenerate_when_asked():
    """`--write-golden` перезаписывает эталон (осознанное действие)."""
    if "--write-golden" not in sys.argv:
        _skip_unless_regenerating("эталон перезаписывается только по флагу --write-golden")
    data = sorted(inventory(), key=lambda r: (r["full"], r["method"]))
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"эталон записан: {FIXTURE} ({len(data)} маршрутов)")


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
    print(f"--- admin routes: {len(tests) - failed - skipped} PASS / {failed} FAIL / {skipped} SKIP ---")
    sys.exit(1 if failed else 0)
