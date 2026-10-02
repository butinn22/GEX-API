"""Системное: статистика, здоровье, метрики, логи, кэш и база.

Диагностика и обслуживание. Обработчики читают чужие подсистемы (пул БД, кэш,
логи), поэтому собраны отдельно от ресурсов.

Вынесено из ``admin_router.py`` (итерация 42). Обработчики перенесены дословно;
подроутер подключает фасад, поэтому пути не менялись — проверяет
``tests/test_admin_routes.py``.
"""
from __future__ import annotations

from ..models import (User, SubscriptionStatus, SUBSCRIPTION_VALUES)
from ..schemas import (AdminActivateIn, AdminStatsOut, AdminUserOut, AdminUserUpdateIn, AdminUsersPageOut, EmailConfigIn, FinAgentKeyIn, MessageOut, TelegramConfigIn)
from fastapi import (APIRouter, Depends, HTTPException, Query, Request, status)
from gex.adapters.persistence.database import (get_session)
from sqlalchemy.orm import (Session)
import asyncio
import os
import platform
import shlex
import subprocess
import threading
import time

from ._shared import (
    _APP_START_TIME,
    _require_admin,
    logger,
)


router = APIRouter()


def _queue_runtime(q) -> dict:  # type: ignore[no-untyped-def]
    """Публичное состояние очереди для админки.

    Очередь — это порт (``TaskPublisher`` над ``JobQueuePort``), а не прежний
    поток-объект: у него нет ни ``_running``, ни ``_consumer_thread``. Читаем то,
    что порт объявляет сам — ``describe()`` (длина, неподтверждённые, DLQ,
    опубликовано), плюс доступность самого порта.
    """
    try:
        state = q.describe() or {}
    except Exception as exc:  # noqa: BLE001 — диагностика не должна ронять страницу
        logger.warning("Очередь: describe() недоступен (%s)", exc)
        return {"consumer_running": False, "available": False}
    port = getattr(q, "port", None)
    return {
        "consumer_running": port is not None,
        "available": bool(getattr(port, "available", port is not None)),
        **{k: v for k, v in state.items() if k != "total"},
    }


@router.get("/stats", response_model=AdminStatsOut)
def get_admin_stats(
    user: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Статистика по пользователям.

    SQL живёт в репозиториях (``gex.application.auth.repositories``), сборка — в use-case
    (``AdminStatsService``): обработчик только отдаёт ответ. Раньше здесь было двадцать
    запросов подряд, и ни один из них нельзя было проверить без настоящей БД.
    """
    from gex.application.auth import AdminStatsService
    from gex.adapters.persistence.auth_repositories import PaymentRepository, UserRepository
    from ..payment_service import _get_usd_rub_rate

    service = AdminStatsService(
        UserRepository(db),
        PaymentRepository(db),
        usd_rub_rate=_get_usd_rub_rate,
    )
    return AdminStatsOut(**service.collect().as_dict())


@router.get("/system")
def get_system_health(
    admin: User = Depends(_require_admin),
):
    """Системная информация: Redis, БД, очереди, модули, версия."""
    from gex.adapters.persistence.database import check_db
    from gex.adapters.cache.redis_client import get_redis
    from gex.application.jobs import QUEUE_KINDS
    from gex.deps import get_task_queue
    from gex.auth.config import settings
    from gex.application.scheduler import get_scheduler
    from gex.deps import (
        get_auto_scanner_service,
        get_scan_service,
    )

    # ── Uptime / процесс ──────────────────────────────────────────────
    uptime_sec = int(time.monotonic() - _APP_START_TIME)

    process = {
        "pid": os.getpid(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": threading.active_count(),
    }
    try:
        import psutil  # type: ignore
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        process["memory_rss_mb"] = round(mem.rss / 1024 / 1024, 1)
        # cpu_percent(interval=None) при ПЕРВОМ вызове возвращает 0.0 —
        # вызываем дважды с небольшой паузой, чтобы получить честное значение.
        proc.cpu_percent(interval=None)
        time.sleep(0.15)
        process["cpu_percent"] = proc.cpu_percent(interval=None)
        process["open_fds"] = proc.num_fds() if hasattr(proc, "num_fds") else None
        vm = psutil.virtual_memory()
        process["host_memory_used_mb"] = round(vm.used / 1024 / 1024, 1)
        process["host_memory_total_mb"] = round(vm.total / 1024 / 1024, 1)
        process["host_memory_percent"] = vm.percent
    except Exception:  # noqa: BLE001
        pass

    # ── DB status ─────────────────────────────────────────────────────
    db_status = check_db()

    # ── Redis status ──────────────────────────────────────────────────
    redis_info = {
        "connected": False, "keys": 0, "memory_used": "N/A",
        "version": "N/A", "uptime_seconds": 0,
        "connected_clients": 0, "hits": 0, "misses": 0,
        "evicted_keys": 0, "expired_keys": 0, "maxmemory": "N/A",
    }
    rc = get_redis()
    if rc and rc.connected:
        redis_info["connected"] = True
        try:
            conn = rc._conn
            info = conn.info("memory")
            redis_info["memory_used"] = info.get("used_memory_human", "N/A")
            # Redis 3.0 не отдаёт maxmemory_human — берём из CONFIG GET
            mm = info.get("maxmemory_human")
            if not mm or mm == "N/A":
                try:
                    mm_cfg = conn.config_get("maxmemory") or {}
                    mm = mm_cfg.get("maxmemory", "N/A")
                except Exception:
                    mm = "N/A"
            redis_info["maxmemory"] = mm if mm != "0" else "без лимита"
            redis_info["keys"] = conn.dbsize()
            redis_info["evicted_keys"] = info.get("evicted_keys", 0)
            redis_info["expired_keys"] = info.get("expired_keys", 0)
            stats = conn.info("stats")
            redis_info["hits"] = stats.get("keyspace_hits", 0)
            redis_info["misses"] = stats.get("keyspace_misses", 0)
            srv = conn.info("server")
            redis_info["version"] = srv.get("redis_version", "N/A")
            redis_info["uptime_seconds"] = int(srv.get("uptime_in_seconds", 0))
            clients = conn.info("clients")
            redis_info["connected_clients"] = clients.get("connected_clients", 0)
        except Exception:  # noqa: BLE001
            pass

    # ── Queue status ──────────────────────────────────────────────────
    q = get_task_queue()
    # Виды очередей (ohlcv/chain/vol/gex), а не ключи Redis: транспорт — деталь порта.
    queue_info = {kind: q.queue_length(kind) for kind in sorted(QUEUE_KINDS)}
    queue_total = sum(queue_info.values())

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler = get_scheduler()
    scheduler_info = {
        "running": bool(scheduler and scheduler.is_running),
        "slots": {},
    }
    if scheduler is not None:
        # Состояние берём у планировщика, а не из его приватных полей: интервалы теперь
        # объявлены в плане прогрева, а «что выполнено/пропущено» знает воркер.
        # ``intervals`` может не быть словарём (это контракт планировщика, а не страницы):
        # вместо 500 на всю страницу — пустой блок слотов, остальные разделы отдаются.
        state = scheduler.describe()
        intervals = state.get("slots")
        if not isinstance(intervals, dict):
            logger.warning(
                "scheduler.describe()['slots'] не словарь (%s) — блок слотов пропущен",
                type(intervals).__name__,
            )
            intervals = {}
        fired = state.get("fired") or {}
        failed = state.get("failed") or {}
        for slot, interval in intervals.items():
            scheduler_info["slots"][slot] = {
                "interval_sec": interval,
                "fired": fired.get(slot, 0),
                "failed": failed.get(slot, 0),
            }
        scheduler_info["skipped_not_due"] = state.get("skipped_not_due", 0)
        scheduler_info["skipped_lease_held"] = state.get("skipped_lease_held", 0)
        scheduler_info["plan_problems"] = state.get("problems", [])

    # ── Scan / AutoScanner ────────────────────────────────────────────
    scan = None
    auto = None
    try:
        scan = get_scan_service()
        auto = get_auto_scanner_service()
    except Exception:  # noqa: BLE001
        pass

    scan_info = {
        "running": bool(scan and scan.is_running),
        "watchlist": len(scan.watchlist) if scan else 0,
        "interval_sec": scan.interval_seconds if scan else 0,
        "cached_tickers": len(scan.list_records()) if scan else 0,
        "last_report": None,
    }
    if scan and scan.last_report():
        rep = scan.last_report()
        scan_info["last_report"] = {
            "scanned_at": rep.scanned_at.isoformat() if rep.scanned_at else None,
            "total": rep.total,
            "ok": rep.ok,
            "failed": rep.failed,
        }

    auto_info = {}
    if auto:
        auto_info = auto.get_status()

    # ── Fetcher ───────────────────────────────────────────────────────
    from gex.application.background_fetcher import get_fetch_stats
    fetch_info = get_fetch_stats()

    # ── Visit tracker ─────────────────────────────────────────────────
    from gex.adapters.middleware.visits import get_visit_tracker
    tracker = get_visit_tracker()

    # ── HTTP-метрики (сводка) ────────────────────────────────────────
    from gex.adapters.middleware.http_metrics import get_http_stats
    http_stats = get_http_stats()

    # ── Коллектор системных метрик ───────────────────────────────────
    from gex.adapters.middleware.system_metrics import get_system_metric_collector
    _collector = get_system_metric_collector()

    return {
        "version": settings.VERSION,
        "role": settings.ROLE,
        "env": settings.APP_ENV,
        "uptime_seconds": uptime_sec,
        "process": process,
        "http": {
            "total": http_stats.get("total", 0),
            "errors": http_stats.get("errors", 0),
            "error_rate": http_stats.get("error_rate", 0.0),
        },
        "system_metric_collector": {
            "running": _collector.is_running,
            "last_snapshot_ts": (
                _collector.last_snapshot().isoformat()
                if _collector.last_snapshot()
                else None
            ),
        },
        "database": {
            "active_url": db_status["active_url"],
            "configured_url": db_status.get("configured_url", ""),
            "fallback_active": db_status["fallback_active"],
            "db_type": db_status.get("db_type", ""),
            "connected": db_status.get("connected", False),
        },
        "redis": redis_info,
        "queues": {
            "queues": queue_info,
            "total": queue_total,
            # Состояние очереди берём у порта (``TaskPublisher.describe``), а не из
            # приватных полей прежнего класса: после итер. 28 очередь — это
            # ``JobQueuePort``, у которого нет ни ``_running``, ни ``_consumer_thread``,
            # поэтому обращение к ним давало AttributeError и 500 на всей странице.
            **_queue_runtime(q),
        },
        "scheduler": scheduler_info,
        "scan_service": scan_info,
        "auto_scanner": auto_info,
        "fetcher": fetch_info,
        "visits_tracker": tracker.status(),
    }


@router.get("/metrics")
def get_metrics(
    days: int = Query(14, ge=1, le=90, description="Глубина агрегации, дней"),
    admin: User = Depends(_require_admin),
):
    """Метрики посещений: визиты, уникальные IP, топ путей, распределения."""
    from gex.adapters.middleware.visits import get_visit_stats
    return get_visit_stats(days=days)


@router.get("/metrics/history")
def get_system_metric_history(
    days: int = Query(7, ge=1, le=30, description="Глубина истории, дней"),
    admin: User = Depends(_require_admin),
):
    """История системных метрик из снапшотов (CPU/RAM/Redis/очереди/HTTP)."""
    from gex.adapters.middleware.system_metrics import get_metric_history
    return get_metric_history(days=days)


@router.get("/http-stats")
def get_http_metrics_stats(
    admin: User = Depends(_require_admin),
):
    """HTTP-метрики: запросы, ошибки, avg/p95 latency по маршрутам."""
    from gex.adapters.middleware.http_metrics import get_http_stats
    return get_http_stats()


@router.get("/logs")
def get_recent_logs(
    level: str = Query("INFO", description="Минимальный уровень: DEBUG/INFO/WARNING/ERROR/CRITICAL"),
    limit: int = Query(200, ge=1, le=1000, description="Максимум записей"),
    admin: User = Depends(_require_admin),
):
    """Последние логи приложения (кольцевой буфер, in-memory)."""
    from gex.adapters.middleware.logging_config import get_recent_logs as _get_recent_logs
    try:
        return {"logs": _get_recent_logs(level=level, limit=limit), "level": level.upper()}
    except Exception as exc:  # noqa: BLE001
        return {"logs": [], "level": level.upper(), "error": str(exc)}


@router.get("/db-stats")
def get_db_metrics_stats(
    admin: User = Depends(_require_admin),
):
    """Размеры таблиц и число строк (PostgreSQL: pg_stat_user_tables)."""
    from gex.adapters.middleware.system_metrics import get_db_stats
    return get_db_stats()


@router.post("/cache/clear")
def clear_cache(
    pattern: str = Query("gex:*", description="Паттерн ключей для удаления"),
    admin: User = Depends(_require_admin),
):
    """Очистить Redis-кэш по паттерну."""
    from gex.adapters.cache.redis_client import get_redis
    rc = get_redis()
    if not rc or not rc.connected:
        raise HTTPException(status_code=503, detail="Redis недоступен")
    try:
        keys = rc._conn.keys(pattern)
        if keys:
            rc._conn.delete(*keys)
        return {"deleted": len(keys), "pattern": pattern}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#: Сколько символов вывода команды показываем админу (полный лог ушёл бы в ответ целиком).
_OUTPUT_TAIL = 500


def _tail(text: str | None) -> str:
    """Хвост вывода команды — в ответе не нужны километры docker-лога."""
    raw = (text or "").strip()
    return raw[-_OUTPUT_TAIL:] if len(raw) > _OUTPUT_TAIL else raw


@router.post("/redis/restart")
async def restart_redis(
    admin: User = Depends(_require_admin),
):
    """Перезапустить Redis на сервере и переподключить приложение.

    Почему команда из конфигурации, а не из запроса
    ----------------------------------------------
    Перезапуск — действие на сервере, а не в браузере (откуда до сервиса не достать).
    Строку команды задаёт ``REDIS_RESTART_COMMAND``: клиент не передаёт её, значит
    подсунуть произвольную команду нельзя. Выполнение — ``shell=False``, с дедлайном.

    Вызов блокирующий (внешний процесс), поэтому уходит в поток: ``async def``-маршрут
    не должен морозить event loop (см. инцидент 2026-09-21 с /ohlcv).
    """
    from gex.auth.config import settings

    command = (settings.REDIS_RESTART_COMMAND or "").strip()
    if not command:
        raise HTTPException(
            status_code=501,
            detail="Перезапуск Redis не настроен: задайте REDIS_RESTART_COMMAND в .env.",
        )
    cwd = (settings.REDIS_RESTART_CWD or "").strip() or None
    timeout = float(settings.REDIS_RESTART_TIMEOUT_SECONDS or 60.0)

    def _run() -> "subprocess.CompletedProcess[str]":
        return subprocess.run(  # noqa: S603 — команда из конфигурации, без shell
            shlex.split(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    try:
        proc = await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        logger.warning("Admin Redis restart TIMEOUT (%s, %.0f с)", command, timeout)
        raise HTTPException(
            status_code=504,
            detail=f"Команда перезапуска не завершилась за {timeout:.0f} с: {command}",
        )
    except FileNotFoundError:
        logger.warning("Admin Redis restart: команда не найдена (%s)", command)
        raise HTTPException(
            status_code=502,
            detail=f"Команда не найдена: «{command}». Проверьте REDIS_RESTART_COMMAND и PATH.",
        )
    except Exception as exc:  # noqa: BLE001 — админка обязана ответить, а не упасть
        logger.exception("Admin Redis restart FAILED")
        raise HTTPException(status_code=500, detail=f"Перезапуск не удался: {exc}")

    stdout, stderr = _tail(proc.stdout), _tail(proc.stderr)
    if proc.returncode != 0:
        logger.warning(
            "Admin Redis restart: exit=%s stderr=%s", proc.returncode, stderr[:200]
        )
        raise HTTPException(
            status_code=502,
            detail=f"Команда завершилась с кодом {proc.returncode}. {stderr or stdout}".strip(),
        )

    # Команда успешна — переподключаемся сразу, не дожидаясь окна троттлинга:
    # иначе админка покажет «недоступен» на только что поднятом Redis.
    from gex.adapters.cache.redis_client import get_redis

    rc = get_redis()
    connected = bool(rc.reconnect()) if rc is not None else False

    logger.warning(
        "Admin Redis RESTART performed (command=%s, connected=%s)", command, connected
    )
    return {
        "status": "ok" if connected else "degraded",
        "message": (
            "Redis перезапущен, приложение подключилось к нему."
            if connected
            else "Команда перезапуска выполнена, но подключиться к Redis не удалось."
        ),
        "detail": (
            None if connected
            else "Проверьте, что сервис поднялся (docker ps), и повторите перезапуск."
        ),
        "command": command,
        "exit_code": proc.returncode,
        "redis_connected": connected,
        "stdout": stdout or None,
        "stderr": stderr or None,
    }


@router.post("/db/reset", response_model=MessageOut)
def reset_db(
    confirm: str = Query(
        ..., description="Подтверждение: передайте DROP (иным значением запрос отклоняется)"
    ),
    admin: User = Depends(_require_admin),
):
    """Принудительно сбросить базу данных в ноль.

    Дропает ВСЕ таблицы (пользователи, платежи, визиты, кэши SEC и т.д.),
    пересоздаёт схему и заново сидит Master Admin из .env.
    Это необратимо — сначала сделайте экспорт CSV/JSON.
    """
    if confirm.strip().upper() != "DROP":
        raise HTTPException(
            status_code=400,
            detail="Сброс отменён: требуется подтверждение confirm=DROP.",
        )

    from gex.adapters.persistence.database import SessionLocal, active_url, recreate_tables
    from gex.auth.router import seed_master_admin

    try:
        recreate_tables()
    except Exception as e:  # noqa: BLE001
        logger.exception("Admin DB reset FAILED")
        raise HTTPException(status_code=500, detail=f"Сброс не удался: {e}")

    # Пересоздаём Master Admin из .env (иначе потеряем доступ к админке)
    db = SessionLocal()
    try:
        seed_master_admin(db)
    except Exception as e:  # noqa: BLE001
        logger.exception("Admin DB reset: seed_master_admin failed")
        raise HTTPException(status_code=500, detail=f"Таблицы пересозданы, но Master Admin не создан: {e}")
    finally:
        db.close()

    logger.warning("Admin DB RESET performed (URL=%s)", active_url())
    return MessageOut(
        message="База данных сброшена в ноль: все таблицы пересозданы, Master Admin восстановлен из .env.",
        detail=f"Активная БД: {active_url()}. Восстановите данные из CSV/JSON, если это было нужно.",
    )
