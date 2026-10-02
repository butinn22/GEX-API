"""Guard-тесты архитектуры трека `arch-refactor_20260916`.

Назначение: превратить инварианты рефакторинга в исполняемые проверки, чтобы результат не
расползся. Специально написаны на stdlib (`ast`, `subprocess`, `pathlib`) и опираются на те же
страж-скрипты, что использовались во время рефакторинга (`scripts/quality/*`), — значит, тесты и
инструменты не могут «разъехаться».

Запуск (на ФИНАЛЬНОМ этапе, после появления venv):
    pytest tests/test_architecture_guards.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUALITY = ROOT / "scripts" / "quality"

# Известные до-рефакторные долги, которые чинятся на финальном этапе (см.
# quality-baseline/static-findings.md). Список обязан только сокращаться.
KNOWN_MISSING_IMPORT_NAMES = {
    "gex/application/sec/sec_forecast.py:1287 gex.application.sec.sec_fundamentals._read_metrics",  # BUG-STATIC-01
}


def _run_guard(script: str) -> dict:
    out = subprocess.run(
        [sys.executable, str(QUALITY / script), "--json"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    assert out.stdout, f"{script} не вернул JSON: {out.stderr[:400]}"
    return json.loads(out.stdout)


# ── Кольца и импорты ─────────────────────────────────────────────────────────

def test_dependency_rings_respected():
    """Кольца: R1–R3 жёстко, R5 — по ратчету.

    Раньше здесь проверялось ``ring_violations == []``, и это стало неверно, когда R5
    (роутеры → фетчеры) получило **ратчет-бейзлайн**: 35 унаследованных нарушений
    зафиксированы в ``quality-baseline/layering.json`` и должны уменьшаться, а не
    блокировать работу. Проверяем то, что действительно инвариант: **новых** нет.
    """
    report = _run_guard("ast_guard.py")
    assert report["new_violations"] == [], f"новые нарушения колец: {report['new_violations']}"
    # У R6/R7 бейзлайна нет вовсе: любое срабатывание — регрессия.
    assert report["provider_key_violations"] == [], report["provider_key_violations"]
    assert report["timeframe_violations"] == [], report["timeframe_violations"]


def test_no_orphan_modules():
    """Ни одного модуля без потребителей (сироты появлялись трижды: scanner_*, gex/strategy).

    Исключение — только новые кольца (`gex/domain|ports|application|adapters`): их модули
    наполняются по итерациям и получают потребителей в волнах P4–P6. Модули `gex/*.py`
    (старый плоский код) проверяются всегда: именно там сироты и заводились.
    """
    report = _run_guard("ast_guard.py")
    new_rings = ("gex/domain/", "gex/ports/", "gex/application/", "gex/adapters/")
    orphans = [o for o in report["orphan_modules"] if not o["path"].startswith(new_rings)]
    assert orphans == [], f"модули-сироты: {orphans}"


def test_imports_resolve():
    """Все внутренние импорты указывают на существующие модули/имена."""
    report = _run_guard("check_imports.py")
    assert report["missing_modules"] == [], f"битые модули: {report['missing_modules']}"
    names = {f"{m['path']}:{m['line']} {m['module']}.{m['name']}" for m in report["missing_names"]}
    assert names <= KNOWN_MISSING_IMPORT_NAMES, f"новые битые импорты: {names - KNOWN_MISSING_IMPORT_NAMES}"


# ── Backtest-контур удалён (итерации 5–9) ────────────────────────────────────

def test_backtest_contour_removed():
    """Мёртвый backtest-контур и его обвязка отсутствуют, а живой движок сигналов — на месте.

    Итерация 37 переиспользовала путь ``gex/strategy/`` под разбор стратегии по предмету
    (``settings``, ``ports``, ``indicators``, ``features``, ``signals``, ``regime``,
    ``risk``, ``decision``). Старая проверка требовала отсутствия ``gex/strategy/__init__.py``
    и потому срабатывала на новом пакете. Здесь перечислены **конкретные модули** удалённого
    контура, а содержимое каталога ограничено списком разрешённых: возврат любого старого
    файла (или появление постороннего) по-прежнему ломает проверку.
    """
    removed = [
        "gex/scanner_backtest.py",
        "gex/scanner_crypto.py",
        "gex/strategy_backtest.py",
        "gex/schemas/strategy_backtest.py",
        "gex/adl_backtest.py",
        "gex/routers/strategy_backtest_router.py",
        # модули удалённого мёртвого контура gex/strategy/** (итерации 5–9)
        "gex/strategy/_shared.py",
        "gex/strategy/config.py",
        "gex/strategy/engine.py",
        "gex/strategy/gex_proxy.py",
        "gex/strategy/optimizers.py",
        "gex/strategy/service.py",
        "gex/strategy/strategy.py",
        "gex/strategy/trend.py",
        "frontend-react/src/pages/BacktestPage.tsx",
    ]
    present = [p for p in removed if (ROOT / p).exists()]
    assert present == [], f"не удалено: {present}"

    # Содержимое gex/strategy — только разбор стратегии (итерация 37). Новый файл здесь
    # означает либо возврат мёртвого контура, либо несогласованную раскладку.
    # ``trading_algorithm.py`` переехал сюда при раскладке плоских файлов по кольцам
    # (clean-arch-refactor-plan §2, Strategy Layer) — это по-прежнему живой движок,
    # поэтому он в списке разрешённых, а не в списке удалённых контуров.
    allowed = {
        "__init__.py", "settings.py", "ports.py", "indicators.py",
        "features.py", "signals.py", "regime.py", "risk.py", "decision.py",
        "trading_algorithm.py",
    }
    strategy_dir = ROOT / "gex" / "strategy"
    if strategy_dir.exists():
        actual = {p.name for p in strategy_dir.glob("*.py")}
        unexpected = actual - allowed
        assert not unexpected, f"в gex/strategy появились посторонние модули: {sorted(unexpected)}"

    # Живой движок сигналов обязан существовать — искали по обоим местам, потому что
    # раскладка по кольцам переносит файлы, а «движок удалён» и «движок переехал»
    # обязаны различаться.
    engine_candidates = [ROOT / "gex" / "trading_algorithm.py",
                         ROOT / "gex" / "strategy" / "trading_algorithm.py"]
    assert any(p.exists() for p in engine_candidates), \
        "trading_algorithm.py — живой движок сигналов, удалять нельзя (искали: gex/ и gex/strategy/)"


def test_scanner_surface_intact():
    """Импортный граф сканерных роутеров не содержит удалённых модулей (AST, не grep)."""
    removed_modules = {
        "gex.adl_backtest",
        "gex.scanner_backtest",
        "gex.scanner_crypto",
        "gex.strategy",
        "gex.strategy_backtest",
        "gex.schemas.strategy_backtest",
        "gex.routers.strategy_backtest_router",
    }
    targets = [
        "gex/routers/auto_scanner_router.py",
        "gex/routers/signal_scanner_router.py",
        "gex/routers/scanner_router.py",
        "gex/application/auto_scanner_service.py",
        "gex/application/signal_scanner_service.py",
        "gex/application/signal_service.py",
    ]
    import ast

    for rel in targets:
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8-sig"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in removed_modules, f"{rel}:{node.lineno} импортирует удалённый {node.module}"
            elif isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name not in removed_modules, f"{rel}:{node.lineno} импортирует удалённый {a.name}"


def test_legacy_getters_removed():
    """13 мёртвых геттеров deps.py удалены; четыре живых оставлены."""
    src = (ROOT / "gex" / "deps.py").read_text(encoding="utf-8-sig")
    dead = [
        "get_gex_service", "get_ta_service", "get_trendline_service", "get_macd_trend_service",
        "get_signal_service", "get_signal_scanner_service", "get_auto_scanner_ru_service",
        "get_auto_scanner_crypto_service", "get_auto_scanner_fx_service", "get_sector_service",
        "get_extended_gex_service", "get_commodity_dynamics_service", "get_novel_candles_service",
    ]
    still_defined = [n for n in dead if f"def {n}(" in src]
    assert still_defined == [], f"мёртвые геттеры вернулись: {still_defined}"
    for keep in ("get_auto_scanner_service", "get_scan_service", "get_redis_client", "get_task_queue"):
        assert f"def {keep}(" in src, f"потерян живой геттер {keep}"


# ── Гигиена окружения и документации ─────────────────────────────────────────

def test_no_foreign_project_paths():
    """В коде не осталось путей чужой машины (`D:\\gex app`) — они ломают запуск скриптов."""
    offenders = []
    for path in (ROOT / "scripts").rglob("*.py"):
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if "D:\\gex app" in text or "D:/gex app" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"скрипты с чужим путём: {offenders}"


def test_no_amqp_broker_in_project():
    """В проекте нет брокера RabbitMQ/Kafka: ни зависимости, ни импортов, ни новых утверждений.

    Исторические пометки «раньше здесь был RabbitMQ» допустимы и перечислены в allowlist —
    они объясняют происхождение Redis TaskQueue (`gex/task_queue.py:3`). Всё остальное —
    дезинформация, которую ловил аудит.

    Уточнение 2026-09-22 (сознанный пересмотр A6)
    --------------------------------------------
    A6 отменял не Celery как таковой, а **AMQP-брокер**: RabbitMQ несовместим с
    Windows-разработкой и не был нужен объёму задач. Слой фоновых задач на Celery
    введён поверх **уже развёрнутого Redis** (см. ``gex/workers/``), поэтому:

    * ``celery`` / ``kombu`` в зависимостях разрешены — это библиотеки задач, не брокер;
    * ``pika`` / ``amqp`` / ``aiokafka`` по-прежнему запрещены целиком;
    * брокер обязан быть Redis — проверяется ниже по факту (схема URL), а не по тексту.
    """
    allowed_mentions = {
        "gex/orchestrator/queue.py",
        "gex/adapters/queue/__init__.py",  # docstring: RabbitMQ как будущий адаптер порта
        "gex/ports/job_queue.py",          # docstring порта: RabbitMQ — будущая реализация
    }
    #: Слой Celery: единственное место, где слово «celery» допустимо в gex/**.
    celery_layer_prefix = "gex/workers/"
    celery_touchpoints = {
        "gex/routers/task_router.py",            # опрос статуса фоновой задачи
        "gex/routers/auto_scanner_router.py",    # передача сканирования в воркер
        # Страницы широты рынка: запрос только читает снапшот и делегирует пересчёт очереди
        # (см. gex/workers/dispatch.py). Роутеру здесь принадлежит ровно один вопрос — каким
        # маршрутом просить пересчёт, поэтому имя очереди обязано быть видно в коде.
        "gex/routers/breadth_sector_router.py",
    }

    req = (ROOT / "requirements.txt").read_text(encoding="utf-8-sig", errors="replace").lower()
    for dep in ("pika", "amqp", "aiokafka"):
        assert dep not in req, f"в requirements.txt появился брокер: {dep}"

    offenders = []
    for path in (ROOT / "gex").rglob("*.py"):
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        lowered = text.lower()
        imports_broker = any(m in text for m in ("import pika", "import aio_pika", "import kombu", "import aiokafka"))
        mentions = any(m in lowered for m in ("rabbitmq", "amqp", "celery", "kafka"))
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        in_celery_layer = rel.startswith(celery_layer_prefix) or rel in celery_touchpoints
        if imports_broker and not in_celery_layer:
            offenders.append(f"{rel}: импортирует брокер")
        elif mentions and rel not in allowed_mentions and not in_celery_layer:
            offenders.append(f"{rel}: упоминание брокера вне allowlist")
    assert offenders == [], f"найдено: {offenders}"

    # Главное: Celery обязан работать поверх Redis, а не AMQP. Проверяем фактический
    # URL, а не комментарии — текстовая проверка здесь была бы бесполезной.
    try:
        from gex.workers import config as workers_config
    except Exception:  # noqa: BLE001 — слой опционален
        workers_config = None
    if workers_config is not None:
        scheme = workers_config.BROKER_URL.split("://", 1)[0]
        assert scheme == "redis", f"Celery-брокер должен быть Redis, получено: {scheme}://"
