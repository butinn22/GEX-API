"""Audit log: append-only record of security-relevant actions.

In-memory by default (testable); wire to a DB-backed writer in production.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = ["AuditLog", "AuditEntry"]


@dataclass(frozen=True)
class AuditEntry:
    action: str
    actor: str
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class AuditLog:
    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(self, action: str, actor: str = "system", **detail) -> AuditEntry:
        entry = AuditEntry(action=action, actor=actor, detail=detail)
        self._entries.append(entry)
        return entry

    def entries(self, action: str | None = None) -> list[AuditEntry]:
        if action is None:
            return list(self._entries)
        return [e for e in self._entries if e.action == action]

    def __len__(self) -> int:
        return len(self._entries)
