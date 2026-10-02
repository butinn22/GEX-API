"""TTL cache (in-memory; Redis-backed variant is the production path)."""
from __future__ import annotations

import time
from typing import Any

__all__ = ["TtlCache"]


class TtlCache:
    """Async TTL cache with monotonic-clock expiry (single-process)."""

    def __init__(self, default_ttl: float = 60.0) -> None:
        self._default_ttl = default_ttl
        self._data: dict[str, tuple[float, Any]] = {}

    async def get(self, key: str) -> Any:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() >= expires_at:
            self._data.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        self._data[key] = (time.monotonic() + (ttl if ttl is not None else self._default_ttl), value)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        """Drop every entry."""
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
