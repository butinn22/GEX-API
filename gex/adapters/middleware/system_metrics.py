"""Системные метрики: снапшоты в БД, история и Telegram-алерты.

Что закрывает
-------------
* В разделе «Система» ранее показывались только «моментальные» значения
  (CPU/RAM/очереди сейчас). Упавший ночью сервис утром не виден.
* ``SystemMetricSnapshot`` — таблица с периодическими снимками (раз в минуту),
  хранение 7–30 дней. Позволяет строить графики истории.
* ``SystemMetricCollector`` — фоновый поток: раз в ``interval_seconds`` собирает
  текущие метрики (процесс/CPU, Redis, очереди, HTTP-статистика), пишет в БД,
  подчищает старые строки и отправляет Telegram-алерт при критических порогах.

Пороги алертов (константы ниже) и Telegram-канал берутся из settings
(``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID``); при пустых настройках
алерты молча отключаются.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, String, desc, func, text
from sqlalchemy.orm import Mapped, mapped_column

from gex.adapters.persistence.database import Base
from gex.adapters.persistence.sqlite_retry import run_with_retry

logger = logging.getLogger(__name__)

# Пороги алертов
ALERT_CPU_PERCENT = 90.0
ALERT_MEMORY_PERCENT = 90.0
ALERT_QUEUE_TOTAL = 200
ALERT_REDIS_MEMORY_MB = 400.0

# Срок хранения снапшотов и частота сбора
COLLECT_INTERVAL_SECONDS = 60
SNAPSHOT_RETENTION_DAYS = 14
COLLECTOR_STARTUP_DELAY = 3.0

# Не алертим чаще, чем раз в это время (смягчает спам)
ALERT_COOLDOWN_SECONDS = 1800


# ====================================================================== #
#  Модель
# ====================================================================== #
class SystemMetricSnapshot(Base):
    """Один снимок системных метрик (таблица ``system_metric_snapshots``)."""

    __tablename__ = "system_metric_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True,
    )

    # Процесс / хост
    cpu_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    memory_rss_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    host_memory_percent: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Redis
    redis_keys: Mapped[int | None] = mapped_column(Integer, nullable=True)
    redis_memory_mb: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Очереди
    queue_total: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # HTTP (из get_http_stats, берём сводку на момент сбора)
    http_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    http_errors: Mapped[int | None] = mapped_column(Integer, nullable=True)
    http_p95_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Статус сервисов (для отладки «кто упал»)
    services_ok: Mapped[str | None] = mapped_column(String(512), nullable=True)

    def __repr__(self) -> str:
        return f"<SystemMetricSnapshot {self.ts:%Y-%m-%d %H:%M} cpu={self.cpu_percent}>"


# ====================================================================== #
#  Коллектор (фоновый поток)
# ====================================================================== #
class SystemMetricCollector:
    """Периодически собирает системные метрики и пишет их в БД."""

    def __init__(
        self,
        interval_seconds: int = COLLECT_INTERVAL_SECONDS,
        retention_days: int = SNAPSHOT_RETENTION_DAYS,
    ):
        self._interval = int(interval_seconds)
        self._retention_days = int(retention_days)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False
        self._last_alert_ts = 0.0
        self._last_snapshot_ts: datetime | None = None

    # ── Lifecycle ────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="sys-metrics", daemon=True,
        )
        self._running = True
        self._thread.start()
        logger.info("SystemMetricCollector started (every %ds)", self._interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._running = False
        logger.info("SystemMetricCollector stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def last_snapshot(self) -> datetime | None:
        return self._last_snapshot_ts

    # ── Фоновый цикл ─────────────────────────────────────────────────
    def _loop(self) -> None:
        if self._stop_event.wait(COLLECTOR_STARTUP_DELAY):
            return
        while not self._stop_event.is_set():
            try:
                self.collect_once()
            except Exception:  # noqa: BLE001
                logger.exception("SystemMetricCollector collect_once failed")
            if self._stop_event.wait(self._interval):
                break

    # ── Один сбор ────────────────────────────────────────────────────
    def collect_once(self) -> None:
        """Собрать снимок и записать в БД (идемпотентно-безопасно)."""
        now = datetime.now(timezone.utc)
        payload = self._gather()

        try:
            # Повтор при «database is locked» оборачивает ВСЮ запись целиком
            # (сессия открывается внутри _write_snapshot): повтор одного commit()
            # после неудачного коммита бесполезен — сессия уже откатана.
            run_with_retry(
                lambda: self._write_snapshot(now, payload), what="снапшот метрик",
            )
            self._last_snapshot_ts = now
        except Exception:  # noqa: BLE001
            logger.warning("SystemMetricCollector: DB write failed", exc_info=True)

        self._maybe_alert(payload, now)

    def _write_snapshot(self, now: datetime, payload: dict[str, Any]) -> None:
        """Записать один снимок и подчистить старые (одна транзакция)."""
        from sqlalchemy import delete

        from gex.adapters.persistence.database import SessionLocal

        db = SessionLocal()
        try:
            snap = SystemMetricSnapshot(
                id=str(int(time.time() * 1000)),
                ts=now,
                cpu_percent=payload.get("cpu_percent"),
                memory_rss_mb=payload.get("memory_rss_mb"),
                host_memory_percent=payload.get("host_memory_percent"),
                redis_keys=payload.get("redis_keys"),
                redis_memory_mb=payload.get("redis_memory_mb"),
                queue_total=payload.get("queue_total"),
                http_total=payload.get("http_total"),
                http_errors=payload.get("http_errors"),
                http_p95_ms=payload.get("http_p95_ms"),
                services_ok=payload.get("services_ok"),
            )
            db.add(snap)
            # Подчистка старого
            cutoff = now - timedelta(days=self._retention_days)
            db.execute(
                delete(SystemMetricSnapshot).where(SystemMetricSnapshot.ts < cutoff)
            )
            db.commit()
        except Exception:  # noqa: BLE001
            db.rollback()
            raise
        finally:
            db.close()

    # ── Сбор значений ────────────────────────────────────────────────
    def _gather(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}

        # Процесс / хост (psutil — опционально)
        try:
            import psutil  # type: ignore
            proc = psutil.Process()
            mem = proc.memory_info()
            payload["memory_rss_mb"] = round(mem.rss / 1024 / 1024, 1)
            payload["cpu_percent"] = proc.cpu_percent(interval=0.1)
            vm = psutil.virtual_memory()
            payload["host_memory_percent"] = vm.percent
        except Exception:  # noqa: BLE001
            pass

        # Redis
        try:
            from gex.adapters.cache.redis_client import get_redis
            rc = get_redis()
            if rc and rc.connected:
                payload["redis_keys"] = rc._conn.dbsize()
                info = rc._conn.info("memory")
                used = info.get("used_memory", 0)
                payload["redis_memory_mb"] = round(used / 1024 / 1024, 1)
        except Exception:  # noqa: BLE001
            pass

        # Очереди
        try:
            from gex.deps import get_task_queue
            payload["queue_total"] = get_task_queue().queue_length()
        except Exception:  # noqa: BLE001
            pass

        # HTTP
        try:
            from gex.adapters.middleware.http_metrics import get_http_stats
            stats = get_http_stats()
            payload["http_total"] = stats.get("total", 0)
            payload["http_errors"] = stats.get("errors", 0)
            routes = stats.get("routes", [])
            p95s = [r.get("p95_ms") for r in routes if r.get("p95_ms") is not None]
            payload["http_p95_ms"] = max(p95s) if p95s else None
        except Exception:  # noqa: BLE001
            pass

        # Статусы сервисов (краткая сводка)
        payload["services_ok"] = self._services_summary()

        return payload

    def _services_summary(self) -> str | None:
        """Короткая строка: какие из фоновых сервисов сейчас не запущены."""
        bad: list[str] = []
        try:
            from gex.deps import get_scan_service
            scan = get_scan_service()
            if scan and not scan.is_running:
                bad.append("scan")
        except Exception:  # noqa: BLE001
            pass
        try:
            from gex.application.scheduler import get_scheduler
            sched = get_scheduler()
            if sched is not None and not sched.is_running:
                bad.append("scheduler")
        except Exception:  # noqa: BLE001
            pass
        if not bad:
            return None
        return ",".join(bad)

    # ── Алерты ────────────────────────────────────────────────────────
    def _maybe_alert(self, payload: dict[str, Any], now: datetime) -> None:
        problems: list[str] = []
        cpu = payload.get("cpu_percent")
        mem = payload.get("host_memory_percent")
        q = payload.get("queue_total")
        rm = payload.get("redis_memory_mb")

        if cpu is not None and cpu >= ALERT_CPU_PERCENT:
            problems.append(f"CPU {cpu:.0f}%")
        if mem is not None and mem >= ALERT_MEMORY_PERCENT:
            problems.append(f"RAM {mem:.0f}%")
        if q is not None and q >= ALERT_QUEUE_TOTAL:
            problems.append(f"Очереди={q}")
        if rm is not None and rm >= ALERT_REDIS_MEMORY_MB:
            problems.append(f"Redis {rm:.0f}MB")

        if not problems:
            return

        # Cooldown: не спамим
        now_mono = time.monotonic()
        if now_mono - self._last_alert_ts < ALERT_COOLDOWN_SECONDS:
            return
        self._last_alert_ts = now_mono

        try:
            from gex.auth.config import settings
            token = settings.TELEGRAM_BOT_TOKEN
            chat = settings.TELEGRAM_CHAT_ID
            if not token or not chat:
                return
            from gex.adapters.notifications.telegram_sender import send_telegram_message
            text = (
                "🚨 GEX Алерт (Система)\n"
                f"Время: {now:%Y-%m-%d %H:%M} UTC\n"
                + "\n".join(f"• {p}" for p in problems)
            )
            send_telegram_message(text, chat_id=chat, polish=False)
            logger.warning("System alert sent: %s", "; ".join(problems))
        except Exception:  # noqa: BLE001
            logger.warning("System alert send failed", exc_info=True)


_default_collector: SystemMetricCollector | None = None


def get_system_metric_collector() -> SystemMetricCollector:
    """Вернуть глобальный коллектор (создаёт при первом вызове)."""
    global _default_collector
    if _default_collector is None:
        _default_collector = SystemMetricCollector()
    return _default_collector


# ====================================================================== #
#  История для /auth/admin/metrics/history
# ====================================================================== #
def get_metric_history(days: int = 7) -> dict:
    """Вернуть временной ряд снапшотов за последние ``days`` дней.

    Returns
    -------
    dict
        days, points: [{ts, cpu_percent, memory_rss_mb, host_memory_percent,
                        redis_keys, redis_memory_mb, queue_total,
                        http_total, http_errors, http_p95_ms}]
    """
    from gex.adapters.persistence.database import SessionLocal

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    db = SessionLocal()
    try:
        rows = (
            db.query(SystemMetricSnapshot)
            .filter(SystemMetricSnapshot.ts >= start)
            .order_by(SystemMetricSnapshot.ts.asc())
            .all()
        )
        points = [
            {
                "ts": r.ts.isoformat(),
                "cpu_percent": r.cpu_percent,
                "memory_rss_mb": r.memory_rss_mb,
                "host_memory_percent": r.host_memory_percent,
                "redis_keys": r.redis_keys,
                "redis_memory_mb": r.redis_memory_mb,
                "queue_total": r.queue_total,
                "http_total": r.http_total,
                "http_errors": r.http_errors,
                "http_p95_ms": r.http_p95_ms,
            }
            for r in rows
        ]
        return {"days": days, "points": points}
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_metric_history failed: %s", exc)
        return {"days": days, "points": [], "error": str(exc)}
    finally:
        db.close()


# ====================================================================== #
#  Статистика БД (размеры таблиц, число строк)
# ====================================================================== #
def get_db_stats() -> dict:
    """Размеры таблиц и общее число строк (для раздела «Система»)."""
    from gex.adapters.persistence.database import SessionLocal, active_dialect

    db = SessionLocal()
    try:
        if active_dialect() == "postgresql":
            rows = db.execute(
                text(
                    "SELECT relname AS table_name, n_live_tup AS row_count "
                    "FROM pg_stat_user_tables ORDER BY n_live_tup DESC"
                )
            ).all()
            return {"tables": [{"table": r.table_name, "rows": r.row_count} for r in rows]}
        # SQLite
        tables = db.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ).all()
        result = []
        for (name,) in tables:
            try:
                cnt = db.execute(text(f'SELECT COUNT(*) FROM "{name}"')).scalar() or 0
                result.append({"table": name, "rows": int(cnt)})
            except Exception:  # noqa: BLE001
                result.append({"table": name, "rows": None})
        return {"tables": result}
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_db_stats failed: %s", exc)
        return {"tables": [], "error": str(exc)}
    finally:
        db.close()
