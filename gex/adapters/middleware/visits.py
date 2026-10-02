"""Аналитика посещений: модель, фоновый трекер и агрегаты для админ-панели.

Архитектура
-----------
* ``PageVisit`` — SQLAlchemy-модель (таблица ``page_visits``): IP, user_id
  (если авторизован), путь, User-Agent, время.
* ``VisitTracker`` — фоновый поток: запросы попадают в очередь (deque),
  поток сбрасывает их в БД пачками (каждые 5 секунд или 200 записей).
  Запись визита НЕ блокирует HTTP-запрос (O(1) enqueue).
* ``VisitTrackingMiddleware`` — Starlette-middleware: логирует только
  "человеческие" маршруты (без /js, /css, /img, /favicon, /health).
* ``get_visit_stats(days)`` — агрегаты для GET /auth/admin/metrics:
  визиты/уникальные IP по дням, распределение по часам, топ путей и агентов.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from gex.adapters.persistence.database import Base
from gex.adapters.persistence.sqlite_retry import run_with_retry

logger = logging.getLogger(__name__)

# ── Пути, которые не считаются визитами (статика/служебные) ─────────────
SKIP_PREFIXES = (
    "/js/", "/css/", "/img/", "/favicon", "/health", "/docs", "/redoc",
    "/openapi.json", "/visit",
)

FLUSH_INTERVAL_SECONDS = 5.0   # сброс очереди каждые 5 секунд
FLUSH_BATCH_SIZE = 200         # или по достижении 200 записей


# ====================================================================== #
#  Модель
# ====================================================================== #
class PageVisit(Base):
    """Один визит страницы (запись аналитики)."""

    __tablename__ = "page_visits"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    ip: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True,
    )
    path: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        return f"<PageVisit {self.path} from {self.ip} at {self.created_at:%Y-%m-%d %H:%M}>"


# ====================================================================== #
#  Трекер (фоновый поток + очередь)
# ====================================================================== #
class VisitTracker:
    """Асинхронный трекер посещений: очередь → пачковая запись в БД.

    Потокобезопасен: ``record()`` можно вызывать из любого потока
    (FastAPI middleware работает в event loop).
    """

    def __init__(
        self,
        flush_interval: float = FLUSH_INTERVAL_SECONDS,
        batch_size: int = FLUSH_BATCH_SIZE,
    ):
        self._queue: deque[tuple[str | None, str | None, str, str | None, datetime]] = deque()
        self._lock = threading.Lock()
        self._flush_interval = flush_interval
        self._batch_size = batch_size
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._total_queued = 0
        self._total_flushed = 0

    # ── Public API ────────────────────────────────────────────────────
    def record(
        self,
        ip: str | None,
        path: str,
        user_agent: str | None,
        user_id: str | None = None,
    ) -> None:
        """Поставить визит в очередь (не блокирует вызывающего)."""
        if not path or path.startswith(SKIP_PREFIXES):
            return
        try:
            with self._lock:
                self._queue.append((ip, user_id, path[:500], user_agent, datetime.now(timezone.utc)))
                self._total_queued += 1
                if len(self._queue) >= self._batch_size:
                    self._flush_locked()
        except Exception:  # noqa: BLE001 — трекер не должен ронять запросы
            logger.debug("VisitTracker.record failed", exc_info=True)

    def start(self) -> None:
        """Запустить фоновый поток сброса очереди в БД."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="visits-flush", daemon=True)
        self._thread.start()
        logger.info("VisitTracker started (flush every %.0fs)", self._flush_interval)

    def stop(self) -> None:
        """Остановить поток и сбросить остаток очереди."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        try:
            with self._lock:
                self._flush_locked()
        except Exception:  # noqa: BLE001
            logger.warning("VisitTracker final flush failed", exc_info=True)
        logger.info("VisitTracker stopped (flushed=%d)", self._total_flushed)

    # ── Internal ──────────────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    if self._queue:
                        self._flush_locked()
            except Exception:  # noqa: BLE001
                logger.warning("VisitTracker flush error", exc_info=True)
            self._stop_event.wait(self._flush_interval)

    def _flush_locked(self) -> None:
        """Сбросить очередь в БД (вызывается только под блокировкой)."""
        if not self._queue:
            return
        batch: list[PageVisit] = []
        for _ in range(min(len(self._queue), self._batch_size * 4)):
            ip, user_id, path, ua, ts = self._queue.popleft()
            batch.append(PageVisit(ip=ip, user_id=user_id, path=path, user_agent=ua, created_at=ts))
        def _write() -> int:
            from gex.adapters.persistence.database import SessionLocal
            db = SessionLocal()
            try:
                db.add_all(batch)
                db.commit()
                return len(batch)
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

        # Пачка визитов — не критичные данные, но терять их каждый раз, когда
        # БД на мгновение занята, не нужно: повтор при «database is locked»
        # закрывает этот случай (пауза короткая, очередь при этом не растёт —
        # сброс идёт под self._lock).
        try:
            self._total_flushed += run_with_retry(_write, what="пачка визитов")
        except Exception:  # noqa: BLE001 — БД недоступна: не теряем поток, дропаем пачку
            logger.warning("VisitTracker: DB flush failed, dropping %d visits", len(batch))

    # ── Состояние ─────────────────────────────────────────────────────
    def status(self) -> dict:
        with self._lock:
            return {
                "running": bool(self._thread and self._thread.is_alive()),
                "queued": len(self._queue),
                "total_queued": self._total_queued,
                "total_flushed": self._total_flushed,
            }


# ====================================================================== #
#  Middleware
# ====================================================================== #
def _extract_user_id(headers) -> str | None:
    """Достать user_id из Bearer-токена без обращения к БД (best-effort)."""
    try:
        auth = headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return None
        token = auth.split(" ", 1)[1].strip()
        if not token:
            return None
        from jose import jwt as jose_jwt
        from gex.auth.config import settings
        payload = jose_jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM],
        )
        if payload.get("type") != "access":
            return None
        sub = payload.get("sub")
        return str(sub) if sub else None
    except Exception:  # noqa: BLE001
        return None


class VisitTrackingMiddleware:
    """Записывает визиты страниц в фоновый трекер."""

    def __init__(self, app, tracker: VisitTracker | None = None):
        self.app = app
        self.tracker = tracker or _default_tracker

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        try:
            client = scope.get("client")
            ip = client[0] if client else None
            headers = dict(
                (k.decode("latin-1").lower(), v.decode("latin-1"))
                for k, v in scope.get("headers", [])
            )
            ua = headers.get("user-agent")
            user_id = _extract_user_id(headers)
            self.tracker.record(ip, path, ua, user_id)
        except Exception:  # noqa: BLE001
            pass

        await self.app(scope, receive, send)


# ====================================================================== #
#  Глобальный инстанс
# ====================================================================== #
_default_tracker = VisitTracker()


def get_visit_tracker() -> VisitTracker:
    return _default_tracker


# ====================================================================== #
#  Агрегаты (для /auth/admin/metrics)
# ====================================================================== #
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def get_visit_stats(days: int = 14) -> dict:
    """Агрегированная статистика посещений за последние ``days`` дней.

    Returns
    -------
    dict
        total, unique_ips, unique_users, today, series (по дням),
        hourly (последние 24 часа), top_paths, top_agents.
    """
    from sqlalchemy import and_

    from gex.adapters.persistence.database import SessionLocal

    now = _utcnow()
    start = now - timedelta(days=days)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    result: dict[str, Any] = {
        "days": days,
        "total": 0,
        "unique_ips": 0,
        "unique_users": 0,
        "today_visits": 0,
        "today_unique_ips": 0,
        "series": [],
        "hourly": [],
        "top_paths": [],
        "top_agents": [],
    }

    db = SessionLocal()
    try:
        # ── Общие счётчики ────────────────────────────────────────────
        result["total"] = (
            db.query(func.count(PageVisit.id))
            .filter(PageVisit.created_at >= start).scalar() or 0
        )
        result["unique_ips"] = (
            db.query(func.count(func.distinct(PageVisit.ip)))
            .filter(PageVisit.created_at >= start, PageVisit.ip.isnot(None)).scalar() or 0
        )
        result["unique_users"] = (
            db.query(func.count(func.distinct(PageVisit.user_id)))
            .filter(PageVisit.created_at >= start, PageVisit.user_id.isnot(None)).scalar() or 0
        )
        result["today_visits"] = (
            db.query(func.count(PageVisit.id))
            .filter(PageVisit.created_at >= day_start).scalar() or 0
        )
        result["today_unique_ips"] = (
            db.query(func.count(func.distinct(PageVisit.ip)))
            .filter(PageVisit.created_at >= day_start, PageVisit.ip.isnot(None)).scalar() or 0
        )

        # ── Серия по дням ─────────────────────────────────────────────
        from gex.adapters.persistence.database import active_dialect
        if active_dialect() == "postgresql":
            # date(timezone('UTC', ts)) — группировка строго по UTC
            day_expr = func.date(func.timezone("UTC", PageVisit.created_at))
        else:
            day_expr = func.date(PageVisit.created_at)

        day_rows = (
            db.query(
                day_expr.label("d"),
                func.count(PageVisit.id).label("visits"),
                func.count(func.distinct(PageVisit.ip)).label("ips"),
            )
            .filter(PageVisit.created_at >= start)
            .group_by("d")
            .all()
        )
        by_day = {}
        for d, visits, ips in day_rows:
            key = str(d)
            by_day[key] = {"visits": visits, "ips": ips}

        for i in range(days - 1, -1, -1):
            day = (now - timedelta(days=i)).date()
            key = day.isoformat()
            row = by_day.get(key, {})
            result["series"].append({
                "date": key,
                "visits": row.get("visits", 0),
                "unique_ips": row.get("ips", 0),
            })

        # ── Часовое распределение (последние 24 часа) ─────────────────
        from gex.adapters.persistence.database import active_dialect
        if active_dialect() == "postgresql":
            hour_expr = func.to_char(PageVisit.created_at, "YYYY-MM-DD HH24:00")
        else:
            hour_expr = func.strftime("%Y-%m-%d %H:00", PageVisit.created_at)

        h_start = now - timedelta(hours=24)
        hour_rows = (
            db.query(hour_expr.label("h"), func.count(PageVisit.id).label("visits"))
            .filter(PageVisit.created_at >= h_start)
            .group_by("h")
            .all()
        )
        by_hour = {h: v for h, v in hour_rows}
        hourly = []
        for i in range(23, -1, -1):
            ts = now - timedelta(hours=i)
            key = ts.strftime("%Y-%m-%d %H:00")
            hourly.append({
                "hour": ts.strftime("%H:00"),
                "visits": by_hour.get(key, 0),
            })
        result["hourly"] = hourly

        # ── Топ путей ─────────────────────────────────────────────────
        path_rows = (
            db.query(PageVisit.path, func.count(PageVisit.id).label("cnt"))
            .filter(PageVisit.created_at >= start)
            .group_by(PageVisit.path)
            .order_by(func.count(PageVisit.id).desc())
            .limit(12)
            .all()
        )
        result["top_paths"] = [{"path": p, "count": c} for p, c in path_rows]

        # ── Топ User-Agent ─────────────────────────────────────────────
        agent_rows = (
            db.query(PageVisit.user_agent, func.count(PageVisit.id).label("cnt"))
            .filter(
                PageVisit.created_at >= start,
                PageVisit.user_agent.isnot(None),
            )
            .group_by(PageVisit.user_agent)
            .order_by(func.count(PageVisit.id).desc())
            .limit(8)
            .all()
        )
        result["top_agents"] = [{"agent": a or "—", "count": c} for a, c in agent_rows]

    except Exception as exc:  # noqa: BLE001
        logger.warning("get_visit_stats failed: %s", exc)
        result["error"] = str(exc)
    finally:
        db.close()

    return result
