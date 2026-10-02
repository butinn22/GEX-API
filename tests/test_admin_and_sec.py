"""Статистика админки и чистый SEC-слой: юнит-тесты без БД и сети (итерации 34–35).

Итерация 34 — «нет SQL в роутерах»: запросы вынесены в репозитории, решения («что считать»)
в use-case. Здесь проверяется именно use-case: с подставными репозиториями и подставными
часами, поэтому тест не зависит от БД и от текущего времени.

Итерация 35 — «чистый XBRL и метрики»: разбор фактов и арифметика отчётности проверяются
синтетическими payload'ами. Обе ловушки XBRL (показатель в разных тегах; годовые и
квартальные факты в одном массиве) проверяются явно.

    python tests/test_admin_and_sec.py
    pytest tests/test_admin_and_sec.py -q
"""
from __future__ import annotations

import ast
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gex.application.auth.admin_stats import AdminStatsService  # noqa: E402
from gex.application.sec.ratios import (  # noqa: E402
    cagr,
    debt_to_equity,
    free_cash_flow,
    growth,
    margin,
    pe_ratio,
    safe_div,
)
from gex.application.sec.xbrl import extract_first, extract_series, latest_value  # noqa: E402

class Skipped(Exception):
    """Проверка требует sqlalchemy и модель БД (их нет в stdlib-прогоне).

    Пропуск, а не «зелено»: часть проверок итерации 34 работает с настоящими
    репозиториями, и объявлять их пройденными без выполнения нельзя.
    """


NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


# ====================================================================== #
#  Двойники репозиториев
# ====================================================================== #
class FakeUsers:
    def __init__(self, *, total=0, active=0, by_status=None, by_provider=None,
                 new=None, expiring=0, by_day=None):
        self._total = total
        self._active = active
        self._by_status = by_status or {}
        self._by_provider = by_provider or {}
        self._new = new or {}
        self._expiring = expiring
        self._by_day = by_day or {}
        self.calls: dict = {}

    def total(self):
        return self._total

    def active_count(self):
        return self._active

    def by_status(self):
        return dict(self._by_status)

    def by_provider(self):
        return dict(self._by_provider)

    def new_since(self, since):
        # Ключ — окно в днях: тест задаёт «сколько новых за 1 и за 7 дней».
        days = round((NOW - since).total_seconds() / 86400)
        self.calls.setdefault("windows", []).append(days)
        return self._new.get(days, 0)

    def expiring_within(self, horizon, *, now=None):
        self.calls["expiring_horizon_days"] = round(horizon.total_seconds() / 86400)
        self.calls["expiring_now"] = now
        return self._expiring

    def registrations_by_day(self, days, *, now=None):
        self.calls["registration_days"] = days
        return dict(self._by_day)


class FakePayments:
    def __init__(self, *, total=0, pending=0, by_status=None, revenue=0.0, revenue_30d=0.0):
        self._total, self._pending = total, pending
        self._by_status = by_status or {}
        self._revenue, self._revenue_30d = revenue, revenue_30d
        self.since: list = []

    def total(self):
        return self._total

    def pending_count(self):
        return self._pending

    def by_status(self):
        return dict(self._by_status)

    def revenue(self, *, since=None):
        self.since.append(since)
        return self._revenue_30d if since is not None else self._revenue


def _service(users=None, payments=None, *, rate=lambda: 92.345, clock=lambda: NOW):
    return AdminStatsService(users or FakeUsers(), payments or FakePayments(),
                            usd_rub_rate=rate, clock=clock)


# ====================================================================== #
#  Итерация 34: сборка статистики
# ====================================================================== #
def test_stats_assembly_passes_numbers_through():
    users = FakeUsers(total=120, active=42, by_status={"BASIC": 30, "EXTENDED": 12},
                      by_provider={"google": 7}, new={1: 5, 7: 21}, expiring=3)
    payments = FakePayments(total=9, pending=2, by_status={"CONFIRMED": 6}, revenue=1234.567,
                            revenue_30d=99.994)
    stats = _service(users, payments).collect()

    assert stats.total_users == 120 and stats.active_subscriptions == 42
    assert stats.by_status == {"BASIC": 30, "EXTENDED": 12}
    assert stats.by_provider == {"google": 7}
    assert stats.new_users_24h == 5 and stats.new_users_7d == 21
    assert stats.expiring_7d == 3
    assert stats.payments_total == 9 and stats.payments_pending == 2
    assert stats.revenue_total_usd == 1234.57, "выручка не округлена до центов"
    assert stats.revenue_30d_usd == 99.99


def test_windows_are_computed_by_the_use_case():
    """Окна времени — продуктовое решение, а не свойство SQL: проверяем переданные значения."""
    users = FakeUsers()
    payments = FakePayments()
    _service(users, payments).collect()

    assert users.calls["windows"] == [1, 7], users.calls["windows"]
    assert users.calls["expiring_horizon_days"] == 7
    assert users.calls["expiring_now"] == NOW, "часы не инжектированы в горизонт"
    assert users.calls["registration_days"] == 14
    # Выручка «за всё время» и «за 30 дней» — два разных запроса; второй с моментом времени.
    assert payments.since == [None, NOW - timedelta(days=30)]


def test_registrations_series_is_zero_filled():
    """Пустые дни обязательны: график, схлопывающий их, врёт о динамике."""
    users = FakeUsers(by_day={NOW.date().isoformat(): 4})
    stats = _service(users).collect()

    assert len(stats.registrations_14d) == 14
    assert stats.registrations_14d[-1] == {"date": NOW.date().isoformat(), "count": 4}
    assert stats.registrations_14d[0]["count"] == 0
    days = [row["date"] for row in stats.registrations_14d]
    assert days == sorted(days), "серия не отсортирована по дате"


def test_missing_days_give_zero_not_gap():
    stats = _service(FakeUsers(by_day={})).collect()
    assert all(row["count"] == 0 for row in stats.registrations_14d)


def test_empty_database_is_safe():
    """Пустая база — не ошибка: админка открывается на новом развёртывании."""
    stats = _service().collect()
    assert stats.total_users == 0 and stats.revenue_total_usd == 0.0
    assert len(stats.registrations_14d) == 14


def test_unavailable_rate_does_not_break_stats():
    """Курс валюты недоступен — статистика всё равно отдаётся (админка нужна именно тогда)."""

    def boom():
        raise RuntimeError("провайдер курса недоступен")

    stats = _service(rate=boom).collect()
    assert stats.usd_rub_rate == 0.0
    assert len(stats.registrations_14d) == 14


def test_rate_is_rounded():
    assert _service(rate=lambda: 92.349).collect().usd_rub_rate == 92.35


def test_stats_serialisation_is_plain_data():
    """Ответ должен быть сериализуемым без ORM: иначе SQL снова протечёт наружу."""
    payload = _service(FakeUsers(total=1)).collect().as_dict()
    assert isinstance(payload, dict) and isinstance(payload["by_status"], dict)
    assert set(payload) >= {"total_users", "registrations_14d", "revenue_total_usd"}


def test_routers_have_no_more_sql_than_before():
    """Ратчет «нет SQL в роутерах»: в админке запросов не осталось — и это теперь проверяется.

    Итог итераций 34 и 42: запросы уехали в ``adapters/persistence/auth_repositories.py``,
    а обработчики разложены по ``gex/auth/admin/*``. Поэтому бейзлайн по админке нулевой, и
    сканируется **весь пакет**, а не только фасад: иначе запрос, вернувшийся в подмодуль,
    прошёл бы мимо проверки (фасад его не содержит).

    Остальные роутеры по-прежнему обязаны держать ноль: у них бейзлайн не задан.
    """
    baseline: dict[str, int] = {}  # в админке SQL не осталось — ноль для всех файлов
    targets = [
        *sorted((ROOT / "gex" / "routers").glob("*.py")),
        ROOT / "gex" / "auth" / "admin_router.py",
        *sorted((ROOT / "gex" / "auth" / "admin").glob("*.py")),
    ]
    counts = {}
    for path in targets:
        n = _sql_calls(ast.parse(path.read_text(encoding="utf-8-sig")))
        if n:
            counts[str(path.relative_to(ROOT))] = n
    offenders = {name: (baseline.get(name, 0), cnt)
                 for name, cnt in counts.items() if cnt > baseline.get(name, 0)}
    assert not offenders, f"SQL в роутерах вырос (было, стало): {offenders}"


#: Имена, которые в этом проекте обозначают сессию БД. Проверка нужна потому, что первая
#: версия детектора ловила ЛЮБОЙ `.execute(...)` и сработала на `use_case.execute(ConeRequest)`
#: (итерация 36) — то есть на обычном вызове use-case, а не на SQL. Ложное срабатывание здесь
#: опаснее пропуска: оно заставляет переименовывать прикладные методы под гейт.
_SESSION_NAMES = {"db", "session", "sess", "conn", "connection", "engine", "cursor", "cur"}


def _looks_like_session(node: ast.expr) -> bool:
    """Похоже ли, что получатель вызова — сессия/соединение с БД."""
    if isinstance(node, ast.Name):
        return node.id.lower() in _SESSION_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr.lower().strip("_") in _SESSION_NAMES
    return False


def _sql_calls(tree: ast.AST) -> int:
    """Сколько вызовов `query`/`execute` делается на сессии (а не на чём-то ещё)."""
    n = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("query", "execute")
                and _looks_like_session(node.func.value)):
            n += 1
    return n


# ====================================================================== #
#  Итерация 35: чистый XBRL
# ====================================================================== #
def _fact(start, end, val, *, form="10-K", filed="2026-02-01", fy=2025):
    return {"start": start, "end": end, "val": val, "form": form, "filed": filed, "fy": fy}


def _companyfacts(tag="Revenues", annual=(), quarterly=(), unit="USD"):
    return {"facts": {"us-gaap": {tag: {"units": {unit: [*annual, *quarterly]}}}}}


def test_extract_series_takes_only_annual_windows():
    """Годовые и квартальные факты лежат в одном массиве: смешать их значит принять квартал за год."""
    payload = _companyfacts(
        annual=[
            _fact("2024-01-01", "2024-12-31", 100.0),
            _fact("2025-01-01", "2025-12-31", 130.0),
        ],
        quarterly=[_fact("2026-01-01", "2026-03-31", 40.0, form="10-Q")],
    )
    series = extract_series(payload, "Revenues")
    assert [f.value for f in series] == [100.0, 130.0], "в ряд попал квартальный факт"
    assert series[0].end.isoformat() == "2024-12-31"
    assert series[-1].value == 130.0


def test_extract_series_quarters_are_separate():
    payload = _companyfacts(
        annual=[_fact("2025-01-01", "2025-12-31", 130.0)],
        quarterly=[_fact("2026-01-01", "2026-03-31", 40.0, form="10-Q")],
    )
    quarters = extract_series(payload, "Revenues", annual=False, forms=("10-Q",))
    assert [f.value for f in quarters] == [40.0]


def test_refiled_period_keeps_latest_filing():
    """Пересдача формы: за один период берём последнюю по дате подачи, а не первую."""
    payload = _companyfacts(annual=[
        _fact("2025-01-01", "2025-12-31", 100.0, filed="2026-02-01"),
        _fact("2025-01-01", "2025-12-31", 111.0, filed="2026-08-01"),
    ])
    series = extract_series(payload, "Revenues")
    assert len(series) == 1 and series[0].value == 111.0


def test_extract_first_falls_back_between_tags():
    """Компании называют выручку по-разному: тег перебирается по порядку."""
    payload = {"facts": {"us-gaap": {
        "SalesRevenueNet": {"units": {"USD": [_fact("2025-01-01", "2025-12-31", 55.0)]}},
    }}}
    fact, series, tag = extract_first(payload, ("Revenues", "SalesRevenueNet"))
    assert fact is not None and fact.value == 55.0 and tag == "SalesRevenueNet"
    assert latest_value(payload, ("Revenues", "SalesRevenueNet")) == 55.0


def test_missing_tag_gives_none_not_error():
    assert extract_series({}, "НетТакого") == []
    fact, series, tag = extract_first({}, ("Revenues",))
    assert fact is None and series == [] and tag == ""
    assert latest_value({}, ("Revenues",)) is None


def test_broken_facts_are_skipped():
    """Мусор в фактах не должен ронять разбор: нечитаемый факт просто не попадает в ряд."""
    payload = _companyfacts(annual=[
        {"start": None, "end": "не дата", "val": 5, "form": "10-K"},
        {"start": "2025-01-01", "end": "2025-12-31", "val": "много", "form": "10-K"},
        _fact("2024-01-01", "2024-12-31", 100.0),
    ])
    assert [f.value for f in extract_series(payload, "Revenues")] == [100.0]


def test_quarterly_bounds_reject_odd_periods():
    """Слишком короткий или длинный период не считается кварталом (это «заглушка», не отчёт)."""
    payload = _companyfacts(quarterly=[
        _fact("2026-03-01", "2026-03-31", 10.0, form="10-Q"),   # 30 дней — не квартал
        _fact("2024-01-01", "2025-12-31", 20.0, form="10-Q"),   # 2 года — не квартал
        _fact("2026-01-01", "2026-03-31", 40.0, form="10-Q"),   # 90 дней — квартал
    ])
    assert [f.value for f in extract_series(payload, "Revenues", annual=False, forms=("10-Q",))] == [40.0]


# ====================================================================== #
#  Итерация 35: метрики отчётности
# ====================================================================== #
def test_margin_and_growth():
    assert margin(25, 100) == 25.0
    assert margin(-10, 100) == -10.0, "убыток — это факт, знак не отбрасывается"
    assert margin(25, 0) is None and margin(None, 100) is None
    assert growth(110, 100) == 10.0
    assert growth(100, 0) is None, "рост к нулю не определён"
    assert growth(100, -50) is None, "рост к убытку читался бы наоборот"
    assert growth(None, 100) is None


def test_cagr_requires_positive_ends():
    assert cagr([100, 121], 2) == 10.0
    assert cagr([100, 121, 144], 2) == 20.0
    assert cagr([100], 1) is None
    assert cagr([100, -50], 1) is None, "корень из отрицательного роста не определён"
    assert cagr([0, 100], 1) is None


def test_safe_div_and_relations():
    assert safe_div(10, 4) == 2.5
    assert safe_div(10, 0) is None, "деление на ноль не даёт «бесконечность»"
    assert safe_div(None, 4) is None
    assert debt_to_equity(50, 100) == 0.5
    assert debt_to_equity(50, -100) is None, "отрицательный капитал делает отношение бессмысленным"
    assert pe_ratio(200, 10) == 20.0
    assert pe_ratio(200, -3) is None, "отрицательный P/E не «очень дешёвый»"
    assert pe_ratio(200, 0) is None


def test_free_cash_flow_sign():
    """Капзатраты у SEC положительные (расход) — вычитаются как есть."""
    assert free_cash_flow(100, 30) == 70.0
    assert free_cash_flow(100, None) == 100.0
    assert free_cash_flow(None, 30) is None


# ====================================================================== #
#  Итерация 34: контракт между application и адаптером
# ====================================================================== #
def test_adapters_satisfy_the_ports():
    """Реализации из адаптера обязаны выполнять протоколы, на которые смотрит use-case.

    Зачем: контракт между ``application`` и ``adapters`` держался на памяти автора —
    use-case вызывал методы у duck-typed объектов. Теперь контракт записан
    (:mod:`gex.ports.persistence`), и эта проверка не даёт ему разойтись с реализацией:
    добавили метод в адаптер и забыли в протоколе — видно здесь, а не в рантайме у админа.
    """
    try:
        from gex.adapters.persistence.auth_repositories import PaymentRepository, UserRepository
        from gex.ports.persistence import PaymentReader, UserReader
    except ImportError as exc:
        raise Skipped(f"нужны sqlalchemy и модель БД ({exc})") from exc

    # runtime_checkable-протокол сверяет наличие методов — ровно то, что нужно.
    assert isinstance(UserRepository(None), UserReader), (
        "UserRepository не выполняет UserReader: "
        f"не хватает {sorted(set(dir(UserReader)) - set(dir(UserRepository(None))))}"
    )
    assert isinstance(PaymentRepository(None), PaymentReader), (
        "PaymentRepository не выполняет PaymentReader: "
        f"не хватает {sorted(set(dir(PaymentReader)) - set(dir(PaymentRepository(None))))}"
    )


def test_use_case_lives_without_infrastructure():
    """``application`` не тянет инфраструктуру в граф импортов — иначе R3 нарушен.

    Проверка статическая: она срабатывает даже там, где sqlalchemy не установлена
    (а значит, не может «пройти по случайности» из-за уже подгруженного модуля).

    Раньше здесь требовалось ``offenders == []`` — и это стало неверно после раскладки
    плоских файлов по кольцам: ``gex/application/**`` наполнялся переносом сервисов,
    которые **и до переноса** импортировали ``gex.adapters.*`` напрямую. До переноса
    они лежали в корне ``gex/`` и под R3 не попадали, поэтому «пусто» ничего не говорило
    о фактическом долге (см. ``docs/clean-arch-refactor-plan.md`` §6.9).

    Теперь проверяется то, что действительно инвариант: **новых** нарушений нет.
    Долг зафиксирован в ``quality-baseline/layering-r3.json`` и может только уменьшаться;
    вернуться к нулю он обязан за счёт портов и DI, а не ослаблением правила.
    """
    import ast as _ast
    import json as _json
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    offenders = []  # (полный относительный путь, модуль) — ключ совпадает с бейзлайном
    for path in sorted((root / "gex" / "application").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        tree = _ast.parse(path.read_text(encoding="utf-8").lstrip("\ufeff"))
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in {"sqlalchemy", "fastapi", "redis", "requests"}:
                        offenders.append((rel, a.name))
            elif isinstance(node, _ast.ImportFrom):
                mod = node.module or ""
                if mod.split(".")[0] in {"sqlalchemy", "fastapi", "redis", "requests"} \
                        or mod.startswith("gex.adapters"):
                    offenders.append((rel, mod))

    baseline_path = root / "quality-baseline" / "layering-r3.json"
    if baseline_path.exists():
        data = _json.loads(baseline_path.read_text(encoding="utf-8"))
        allowed = {(e["path"], e["import"]) for e in data.get("allowed", [])}
    else:
        allowed = set()

    new = [f"{rel}:{mod}" for rel, mod in offenders if (rel, mod) not in allowed]
    assert not new, (
        "application импортирует инфраструктуру (новые, вне бейзлайна R3): "
        + "; ".join(new[:20])
    )


def test_repositories_are_reachable_from_the_adapter_ring():
    """Репозитории живут в адаптере, а не в application: это инфраструктура.

    Первая версия модуля лежала в ``gex/application/auth/`` и нарушала R3 (SQLAlchemy).
    Проверка фиксирует раскладку, чтобы модуль не «вернулся назад» при следующей правке.
    """
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    assert (root / "gex" / "adapters" / "persistence" / "auth_repositories.py").exists()
    assert not (root / "gex" / "application" / "auth" / "repositories.py").exists(), (
        "репозитории вернулись в application — это нарушение R3"
    )


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
    print(f"--- admin+sec: {len(tests) - failed - skipped} PASS / {failed} FAIL / {skipped} SKIP ---")
    sys.exit(1 if failed else 0)
