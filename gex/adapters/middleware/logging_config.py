"""Структурированное логирование: JSON + request_id (traceId).

Каждая запись — JSON::

    {"level":"WARNING","ts":"2026-08-04T09:00:00.123Z","logger":"gex.x",
     "message":"...","request_id":"ab12cd34ef56"}

``request_id`` берётся из contextvars и заполняется middleware
(см. main.py: RequestIdMiddleware). Дополнительные поля (userId и т.п.)
можно передавать через ``logger.info(..., extra={"userId": ...})``.

Помимо StreamHandler в корневой логгер добавляется ``RingBufferHandler``
(кольцевой буфер последних записей), из которого админ-панель читает логи
через ``get_recent_logs()``.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Поля из extra, которые попадают в JSON (безопасный белый список)
_EXTRA_FIELDS = ("userId", "traceId", "ticker", "duration_ms")


class JsonFormatter(logging.Formatter):
    """JSON-форматтер: level, ts, logger, message, request_id (+extra)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "level": record.levelname,
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key in _EXTRA_FIELDS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


_RING_MAXLEN = 2000
_ring_lock = threading.Lock()
_ring_buffer: deque[dict] = deque(maxlen=_RING_MAXLEN)


class RingBufferHandler(logging.Handler):
    """Handler, который хранит последние N записей в памяти (для админ-панели)."""

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level=level)
        self._json = JsonFormatter()
        self._plain = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "level": record.levelname,
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                "logger": record.name,
                "message": record.getMessage(),
                "request_id": request_id_var.get(),
            }
            if record.exc_info:
                entry["exc_info"] = self._json.formatException(record.exc_info)
            for key in _EXTRA_FIELDS:
                value = getattr(record, key, None)
                if value is not None:
                    entry[key] = value
            with _ring_lock:
                _ring_buffer.append(entry)
        except Exception:  # noqa: BLE001 — логгер не должен ронять приложение
            pass


def get_recent_logs(level: str = "DEBUG", limit: int = 200) -> list[dict]:
    """Вернуть последние записи из кольцевого буфера (от новых к старым).

    Parameters
    ----------
    level : str
        Минимальный уровень (DEBUG/INFO/WARNING/ERROR/CRITICAL).
    limit : int
        Максимум возвращаемых записей.
    """
    wanted = logging._nameToLevel.get(level.upper(), logging.DEBUG)
    with _ring_lock:
        items = [e for e in _ring_buffer if logging._nameToLevel.get(e["level"], 0) >= wanted]
    items.reverse()  # новые сверху
    return items[: max(1, min(limit, _RING_MAXLEN))]


def setup_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Настроить корневой логгер: один StreamHandler с JSON-форматом.

    Вызывается один раз при старте приложения (main.py).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler()
    if json_output:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
    root.addHandler(handler)
    # Кольцевой буфер последних записей (для админ-панели «Логи»).
    ring = RingBufferHandler(level=logging.DEBUG)
    root.addHandler(ring)
    return root


def reset_ring_buffer() -> None:
    """Очистить кольцевой буфер записей (для тестов)."""
    with _ring_lock:
        _ring_buffer.clear()
