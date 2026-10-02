"""Регрессионные тесты на исправленные баги (багфикс-проход после итераций 1-21).

Каждый тест — статическая проверка (AST), а не «поиск подстроки»: комментарии и докстринги,
упоминающие проблему, не должны считаться кодом. Именно на этом ловились ложные срабатывания.

Все проверки исполняются без внешних зависимостей (нет импорта numpy/sqlalchemy/redis):
разбирается только исходный текст.

Запуск:  python tests/test_bugfix_regressions.py
"""
from __future__ import annotations

import ast
import io
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SECRET_FIELDS = {"password_hash", "telegram_connect_token", "verification_token"}
FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def src(rel: str) -> str:
    return io.open(ROOT / rel, encoding="utf-8-sig").read()


def tree(rel: str) -> ast.Module:
    return ast.parse(src(rel))


def find(tree_: ast.Module, name: str) -> ast.AST | None:
    """Найти функцию по имени — и обычную, и async (async def — это AsyncFunctionDef)."""
    return next((n for n in ast.walk(tree_) if isinstance(n, FUNC_NODES) and n.name == name), None)


def module_funcs(tree_: ast.Module) -> set[str]:
    return {n.name for n in tree_.body if isinstance(n, FUNC_NODES)}


def called_names(node: ast.AST) -> set[str]:
    """Имена вызываемых функций внутри узла: `f()` и `obj.f()` → {f, obj.f}."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                out.add(n.func.id)
            elif isinstance(n.func, ast.Attribute):
                out.add(n.func.attr)
    return out


def attr_names(node: ast.AST) -> set[str]:
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


# ── BUG-STATIC-01: импорт несуществующего имени глушил квартальный рост выручки ──
def test_bug_static_01_read_metrics_is_module_level() -> None:
    assert "read_metrics" in module_funcs(tree("gex/application/sec/sec_fundamentals.py")), \
        "gex/application/sec/sec_fundamentals.py: нет модульной функции read_metrics"
    t = tree("gex/application/sec/sec_forecast.py")
    imports = [n for n in ast.walk(t)
               if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("sec_fundamentals")]
    assert imports, "sec_forecast: не найден импорт из sec_fundamentals"
    names = {a.name for imp in imports for a in imp.names}
    assert "read_metrics" in names, f"sec_forecast импортирует {names}"
    assert not {n for n in names if n.startswith("_")}, \
        f"sec_forecast импортирует приватные имена модуля: {names}"


# ── AUTH-04: секреты не покидают систему через экспорт ─────────────────────────
def test_auth_04_export_has_no_secrets() -> None:
    # Список колонок переехал из `admin_router.py` в `admin/_shared.py` при раскладке
    # админки по владельцам (итерация 42): фасад стал тонким, объявления — в `_shared`.
    t = tree("gex/auth/admin/_shared.py")
    fields = None
    for n in t.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "USER_EXPORT_FIELDS":
            fields = {e.value for e in n.value.elts if isinstance(e, ast.Constant)}
    assert fields is not None, "USER_EXPORT_FIELDS не найдено"
    assert not (fields & SECRET_FIELDS), f"экспорт содержит секреты: {fields & SECRET_FIELDS}"

    exporter = find(t, "_user_export_row")
    assert exporter, "_user_export_row не найдено"
    row_keys = {k.value for sub in ast.walk(exporter) if isinstance(sub, ast.Dict)
                for k in sub.keys if isinstance(k, ast.Constant)}
    assert not (row_keys & SECRET_FIELDS), f"строка экспорта содержит секреты: {row_keys & SECRET_FIELDS}"

    # `import_users` переехал в `admin/users.py` (раскладка админки по владельцам,
    # итерация 42): фасад и объявления — в `_shared`, обработчик — у своего ресурса.
    users_tree = tree("gex/auth/admin/users.py")
    importer = find(users_tree, "import_users")
    assert importer, "import_users не найдено"
    refs = attr_names(importer)
    refs |= {s.value for s in ast.walk(importer) if isinstance(s, ast.Constant) and isinstance(s.value, str)}
    missing = SECRET_FIELDS - refs
    assert not missing, f"импорт перестал читать секреты старых бэкапов: {missing}"


# ── AUTH-08: сравнение пароля Master в постоянном времени + fail-closed ────────
def test_auth_08_master_login_constant_time() -> None:
    t = tree("gex/auth/service.py")
    assert any(isinstance(n, ast.Import) and any(a.name == "hmac" for a in n.names)
               for n in t.body), "нет импорта hmac"
    fn = find(t, "master_admin_login")
    assert fn, "master_admin_login не найдено"
    calls = called_names(fn)
    assert "compare_digest" in calls, f"нет сравнения в постоянном времени (вызовы: {sorted(calls)})"
    assert "MASTER_PASSWORD" in attr_names(fn), "нет сверки с settings.MASTER_PASSWORD"
    # Сравнения пароля не должны идти через ==/!= (сравнение enum — это другой случай).
    for cmp_node in (n for n in ast.walk(fn) if isinstance(n, ast.Compare)):
        touched = attr_names(cmp_node) | {x.id for x in ast.walk(cmp_node) if isinstance(x, ast.Name)}
        if any("password" in name.lower() for name in touched):
            ops = [type(o).__name__ for o in cmp_node.ops]
            raise AssertionError(f"пароль сравнивается напрямую: {ast.unparse(cmp_node)} → {ops}")


# ── BUG-CACHE-TTL: commodity-кэш писался с `ttl=` → TypeError глотался ─────────
def test_bug_cache_ttl_writes_use_ex_and_raw_value() -> None:
    for rel in ("gex/adapters/fetchers/commodity_fetcher.py", "gex/application/commodity_dynamics.py"):
        t = tree(rel)
        writes = [n for n in ast.walk(t)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "set"]
        assert writes, f"{rel}: не найдено ни одного вызова .set()"
        for call in writes:
            kw = {k.arg for k in call.keywords}
            assert "ttl" not in kw, f"{rel}: вызов .set() с неподдерживаемым ttl= (строка {call.lineno})"
            assert "ex" in kw or not kw, f"{rel}: .set() без ex= ({sorted(kw)}, строка {call.lineno})"
            for arg in call.args:
                assert not (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name)
                            and arg.func.id == "serialize_value"), \
                    f"{rel}: значение сериализуется дважды (RedisClient.set сериализует сам)"


# ── EC-8: параметр expiries не входил в ключ кэша /gexcone → чужой конус ───────
def test_ec8_gexcone_cache_key_includes_expiries() -> None:
    """EC-8: ``expiries`` обязан влиять на ключ кэша /gexcone.

    Сайт вызова с итерации 36 собирает части ключа из запроса —
    ``cache_key("res", "gexcone", *ConeRequest(...).cache_parts())``. Поэтому проверяем
    сам инвариант, а не прежнюю форму AST: а) роутер зовёт ``ConeRequest`` и
    ``cache_parts``, б) ``cache_parts()`` **содержит** ``expiries``, в) значение реально
    различается при разном числе экспираций (это и есть суть дефекта).
    """
    t = tree("gex/routers/gexcone_router.py")
    calls = [n for n in ast.walk(t)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "cache_key"
             and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "res"]
    assert calls, "ключ кэша /gexcone не найден"
    uses_request_parts = any(
        isinstance(a, ast.Starred) and isinstance(a.value, ast.Call)
        and getattr(a.value.func, "attr", "") == "cache_parts"
        for c in calls for a in c.args
    )
    assert uses_request_parts, (
        "ключ /gexcone не собирается из ConeRequest.cache_parts() — "
        "часть параметров может снова выпасть из ключа"
    )

    cone = tree("gex/application/cone.py")
    cache_parts = find(cone, "cache_parts")
    assert cache_parts, "ConeRequest.cache_parts не найдено"
    returned = [n for n in ast.walk(cache_parts) if isinstance(n, ast.Return)][0]
    attrs = {n.attr for n in ast.walk(returned) if isinstance(n, ast.Attribute)}
    assert "expiries" in attrs, f"cache_parts не учитывает expiries: {sorted(attrs)}"

    # Свойство, ради которого правило и заведено: разные expiries → разные ключи.
    from gex.application.cone import ConeRequest

    def parts(exp):
        return ConeRequest(ticker="AAPL", expiries=exp).cache_parts()

    assert parts(3) != parts(5), "expiries не влияет на ключ — дефект EC-8 вернулся"


# ── BUG-AGG: удалённая ветка не вернулась ─────────────────────────────────────
def test_bug_agg_extended_has_no_aggregated_branch() -> None:
    t = tree("gex/application/extended.py")
    gone = {"_analyze_aggregated", "_build_report", "_avg_opt"}
    present = {n.name for n in ast.walk(t) if isinstance(n, FUNC_NODES)} & gone
    assert not present, f"вернулись сломанные методы: {present}"
    dead = {n.id for n in ast.walk(t) if isinstance(n, ast.Name)} | attr_names(t)
    assert "dhr_moneyness" not in dead, "вернулось поле dhr_moneyness (его нет в датаклассе)"
    for bad in ("_average_strikes", "_merge_key_levels", "_average_hedge_scenarios"):
        assert bad not in dead, f"остался вызов несуществующего метода {bad}"
    assert "|aggregated" not in src("gex/routers/extended_router.py"), \
        "роутер снова рекламирует несуществующий source=aggregated"


# ── B-04: сетевой вызов EDGAR не держит открытую сессию БД ─────────────────────
def test_b04_ensure_fresh_does_not_hold_db_session() -> None:
    t = tree("gex/application/sec/sec_fundamentals.py")
    fn = find(t, "_ensure_fresh")
    assert fn, "_ensure_fresh не найдено"
    args = [a.arg for a in fn.args.args]
    assert args == ["self", "ticker"], f"сессия БД снова передаётся в _ensure_fresh: {args}"
    calls = called_names(fn)
    assert "get_company_facts" in calls, "сетевой вызов EDGAR исчез из _ensure_fresh?"
    assert "SessionLocal" in calls, "upsert должен идти в собственной короткой сессии"
    assert find(t, "_latest_updated_at"), "нет отдельного чтения TTL"


# ── Чужие пути: `D:\gex app` не существует на этой машине ────────────────────
def test_no_foreign_project_paths() -> None:
    bad_literals, bad_depth = [], []
    root = os.path.normcase(str(ROOT))
    candidates = sorted([*ROOT.glob("finagent/tests/**/*.py"), *ROOT.glob("scripts/**/*.py")])
    for p in candidates:
        text = io.open(p, encoding="utf-8-sig", errors="replace").read()
        try:
            t = ast.parse(text)
        except SyntaxError as exc:
            bad_literals.append(f"{p.relative_to(ROOT)}: SyntaxError {exc}")
            continue
        for n in ast.walk(t):
            if (isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and "gex app" in n.value and n.value.strip()[:2].upper() == "D:"):
                bad_literals.append(str(p.relative_to(ROOT)))
        if "sys.path.insert" in text:
            lines = text.splitlines()
            upto = next(i for i, ln in enumerate(lines) if "sys.path.insert" in ln)
            ns = {"__file__": str(p)}
            exec(compile("\n".join(lines[:upto + 1]), str(p), "exec"), ns)
            if os.path.normcase(ns["sys"].path[0]) != root:
                bad_depth.append(f"{p.relative_to(ROOT)} -> {ns['sys'].path[0]}")
    assert not bad_literals, f"ссылки на несуществующий проект: {sorted(set(bad_literals))}"
    assert not bad_depth, f"sys.path ведёт не в корень репозитория: {bad_depth}"


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed.append((name, str(exc)))
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # НЕОЖИДАННОЕ: иначе прогон обрывался, и «0 FAIL» врало
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {str(exc)[:300]}")
            failed += 1
        except Exception as exc:  # неожиданная ошибка самого теста
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n--- bugfix regressions: {passed} PASS / {len(failed)} FAIL ---")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
