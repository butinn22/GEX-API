"""Порт ограничения частоты обращений к провайдеру.

Требование, из-за которого порт существует: лимит обязан быть **распределённым** — иначе он
умножается на число uvicorn-воркеров (аудит 02/05/07/10: PC-05, SVC-08, B-03, F-04).
Реализация — ``gex/adapters/ratelimit/**`` (Redis + Lua), плюс локальный bounded-фоллбэк при
недоступном Redis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["Decision", "RateLimitPort"]


@dataclass(frozen=True)
class Decision:
    """Решение лимитера: разрешено ли обращение и когда повторить."""

    allowed: bool
    remaining: int = 0
    retry_after_ms: int = 0
    rule_name: str | None = None

    @property
    def retry_after_seconds(self) -> float:
        return max(0.0, self.retry_after_ms / 1000.0)


@runtime_checkable
class RateLimitPort(Protocol):
    """Per-provider бюджет с явным правилом (endpoint/scope) и стоимостью запроса."""

    def acquire(self, provider: str, *, endpoint: str = "*", cost: int = 1) -> Decision:
        """Попытаться занять «токен» без ожидания."""
        ...

    def wait(self, provider: str, *, endpoint: str = "*", cost: int = 1) -> Decision:
        """Дождаться разрешения (или вернуть отказ, если бюджет исчерпан надолго).

        Возвращает :class:`Decision`; отказ — сигнал вызывающему отдать устаревшее значение
        (serve-stale), а не 429 пользователю.
        """
        ...
