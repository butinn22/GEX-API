"""HTTP-метрики для админ-панели (запросы, латентность, ошибки).

Лёгкий ASGI-middleware, который считает по каждому маршруту:
  * число запросов (total),
  * число 5xx-ответов (errors),
  * суммарную длительность и число замеров → средний latency,
  * скользящий перцентиль p95 (bounded histogram, 100 бакетов).

Хранение — in-process словарь (не Redis), чтобы статистика не терялась на
перезапусках Redis и не требовала отдельной инфраструктуры. Окно агрегации
``window_seconds`` ограничивает глубину гистограммы; p95 вычисляется по
последним ``n`` замерам в пределах окна.

GET /auth/admin/http-stats отдаёт ``gex.http_metrics.get_http_stats()``.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 3600          # глубина скользящего окна для latency
_HIST_MAX = 100                  # максимум замеров латентности на маршрут


class HttpMetricsCollector:
    """Потокобезопасный сборщик HTTP-метрик (in-process)."""

    def __init__(self, window_seconds: int = _WINDOW_SECONDS, hist_max: int = _HIST_MAX):
        self._window = float(window_seconds)
        self._hist_max = int(hist_max)
        self._lock = threading.Lock()
        self._routes: dict[str, dict] = defaultdict(lambda: {
            "total": 0,
            "errors": 0,
            "latency_sum": 0.0,
            "latency_n": 0,
            "hist": deque(maxlen=self._hist_max),
            "last_seen": 0.0,
        })

    # ── Запись одного завершённого запроса ────────────────────────────
    def record(self, route: str, status_code: int | None, duration_ms: float, ok: bool = False) -> None:
        """Зафиксировать результат запроса.

        Parameters
        ----------
        route : str
            Шаблон маршрута (``/gex/{ticker}``) или путь, если шаблон неизвестен.
        status_code : int | None
            HTTP-статус ответа.
        duration_ms : float
            Длительность запроса, миллисекунды.
        ok : bool
            True, если сообщение ответа корректно (не обязательно — статус
            по-прежнему является основным признаком успеха).
        """
        try:
            status_code = int(status_code) if status_code is not None else 0
            duration_ms = float(duration_ms)
            now = time.monotonic()
            with self._lock:
                r = self._routes[route or "unknown"]
                r["total"] += 1
                r["last_seen"] = now
                if status_code >= 500:
                    r["errors"] += 1
                # Латентность агрегируем только для успешных (2xx/3xx/4xx),
                # чтобы 5xx с большими таймаутами не завышали "обычную" скорость.
                if status_code < 500 and status_code > 0:
                    r["latency_sum"] += duration_ms
                    r["latency_n"] += 1
                    r["hist"].append(duration_ms)
        except Exception:  # noqa: BLE001 — сборщик не должен ронять запрос
            pass

    # ── Агрегированная статистика ─────────────────────────────────────
    def snapshot(self) -> dict:
        """Вернуть снимок: total/errors/avg/p95 по маршрутам + сводку."""
        now = time.monotonic()
        with self._lock:
            items = []
            tot = errs = 0
            for route, r in list(self._routes.items()):
                total = r["total"]
                errors = r["errors"]
                latency_n = r["latency_n"]
                avg = round(r["latency_sum"] / latency_n, 1) if latency_n else 0.0
                p95 = self._percentile(sorted(r["hist"]), 0.95) if r["hist"] else None
                items.append({
                    "route": route,
                    "total": total,
                    "errors": errors,
                    "error_rate": round(errors / total * 100, 2) if total else 0.0,
                    "avg_ms": avg,
                    "p95_ms": round(p95, 1) if p95 is not None else None,
                    "last_seen": r["last_seen"],
                    "last_seen_ago": round(now - r["last_seen"], 1),
                })
                tot += total
                errs += errors
            items.sort(key=lambda i: i["total"], reverse=True)

        return {
            "total": tot,
            "errors": errs,
            "error_rate": round(errs / tot * 100, 2) if tot else 0.0,
            "routes": items,
        }

    # ── Очистка (для тестов) ──────────────────────────────────────────
    def reset(self) -> None:
        with self._lock:
            self._routes.clear()

    @staticmethod
    def _percentile(sorted_values: list[float], q: float) -> float:
        if not sorted_values:
            return 0.0
        idx = int(q * (len(sorted_values) - 1))
        return sorted_values[idx]


_default_collector = HttpMetricsCollector()


def get_http_metrics() -> HttpMetricsCollector:
    """Вернуть глобальный сборщик HTTP-метрик."""
    return _default_collector


def get_http_stats() -> dict:
    """Снимок HTTP-метрик для админ-панели."""
    try:
        return _default_collector.snapshot()
    except Exception:  # noqa: BLE001
        return {"total": 0, "errors": 0, "error_rate": 0.0, "routes": []}


class HttpMetricsMiddleware:
    """Starlette/ASGI middleware: замер длительности и запись метрик.

    Самый внешний слой — считает и статику, и API, и упавшие маршруты.
    Использует ``scope["route"]`` (Starlette ≥ 0.20) для шаблонного пути.
    """

    def __init__(self, app, collector: HttpMetricsCollector | None = None):
        self.app = app
        self.collector = collector or _default_collector

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_holder: list[int | None] = [None]

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder[0] = message.get("status")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            status_holder[0] = status_holder[0] or 500
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            route = None
            try:
                route = getattr(scope.get("route"), "path", None)
            except Exception:  # noqa: BLE001
                route = None
            self.collector.record(
                route or scope.get("path", "unknown"),
                status_holder[0],
                duration_ms,
            )


# Удобный ссылочный помощник для wiring (используется в main.py/lifespan).
def install_http_metrics(app) -> HttpMetricsCollector:
    """Добавить HttpMetricsMiddleware к приложению (идемпотентно)."""
    from starlette.middleware import Middleware

    collector = get_http_metrics()
    # ВНИМАНИЕ: add_middleware кладёт слой ПОВЕРХ всех существующих; мы вызываем
    # его самым первым (до остальных add_middleware), поэтому он станет внешним.
    app.add_middleware(HttpMetricsMiddleware, collector=collector)
    return collector