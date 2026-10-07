"""Local signal client sessions: handshake, heartbeat, subscriptions, routing.

A *local client* is an external signal producer connected over ``/ws/client``
with no broker account integration. The registry tracks each session's state
(CONNECTED → ACTIVE after handshake), liveness (heartbeat timeout), and ticker
subscriptions (≤ 20 per client); the dispatcher fans ``signal_hub`` messages
out to exactly the sessions subscribed to each instrument.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ClientState",
    "ClientSession",
    "LocalClientError",
    "UnknownSessionError",
    "HandshakeRequiredError",
    "MaxTickersError",
    "LocalClientRegistry",
    "LocalClientDispatcher",
    "HEARTBEAT_INTERVAL",
    "HEARTBEAT_TIMEOUT",
    "MAX_TICKERS_PER_CLIENT",
]

HEARTBEAT_INTERVAL = 5.0
HEARTBEAT_TIMEOUT = 15.0
MAX_TICKERS_PER_CLIENT = 20


class ClientState(str, Enum):
    CONNECTED = "connected"  # socket open, handshake pending
    ACTIVE = "active"  # handshake completed


class LocalClientError(ValueError):
    """Base error for local client session operations."""


class UnknownSessionError(LocalClientError):
    pass


class HandshakeRequiredError(LocalClientError):
    pass


class MaxTickersError(LocalClientError):
    pass


@dataclass
class ClientSession:
    session_id: str
    client_id: str
    state: ClientState
    connected_at: float
    last_seen: float
    tickers: set[str] = field(default_factory=set)
    outbox: asyncio.Queue = field(default_factory=asyncio.Queue)


def _normalize_tickers(tickers: list[Any]) -> list[str]:
    seen: set[str] = set()
    for t in tickers:
        sym = str(t).strip().upper()
        if sym:
            seen.add(sym)
    return sorted(seen)


class LocalClientRegistry:
    """Session store with handshake state, heartbeat liveness, and routing."""

    def __init__(
        self,
        *,
        max_tickers: int = MAX_TICKERS_PER_CLIENT,
        heartbeat_timeout: float = HEARTBEAT_TIMEOUT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_tickers < 1:
            raise ValueError("max_tickers must be >= 1")
        if heartbeat_timeout <= 0:
            raise ValueError("heartbeat_timeout must be > 0")
        self.max_tickers = max_tickers
        self.heartbeat_timeout = heartbeat_timeout
        self._clock = clock
        self._sessions: dict[str, ClientSession] = {}

    # ── Lifecycle ────────────────────────────────────────────────────

    def connect(self, client_id: str = "") -> ClientSession:
        now = self._clock()
        session = ClientSession(
            session_id=uuid.uuid4().hex,
            client_id=client_id,
            state=ClientState.CONNECTED,
            connected_at=now,
            last_seen=now,
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> ClientSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise UnknownSessionError(f"unknown session: {session_id}")
        return session

    def handshake(self, session_id: str, client_id: str) -> ClientSession:
        session = self.get(session_id)
        session.client_id = client_id or session.client_id
        session.state = ClientState.ACTIVE
        session.last_seen = self._clock()
        return session

    def disconnect(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    # ── Heartbeat ────────────────────────────────────────────────────

    def touch(self, session_id: str) -> None:
        self.get(session_id).last_seen = self._clock()

    def is_stale(self, session_id: str) -> bool:
        return (self._clock() - self.get(session_id).last_seen) > self.heartbeat_timeout

    def stale_sessions(self) -> list[ClientSession]:
        now = self._clock()
        return [
            s for s in self._sessions.values()
            if (now - s.last_seen) > self.heartbeat_timeout
        ]

    # ── Subscriptions ────────────────────────────────────────────────

    def _require_active(self, session_id: str) -> ClientSession:
        session = self.get(session_id)
        if session.state is not ClientState.ACTIVE:
            raise HandshakeRequiredError("handshake required first")
        return session

    def subscribe(self, session_id: str, tickers: list[Any]) -> list[str]:
        session = self._require_active(session_id)
        merged = session.tickers | set(_normalize_tickers(tickers))
        if len(merged) > self.max_tickers:
            raise MaxTickersError(
                f"max {self.max_tickers} tickers per client "
                f"(have {len(session.tickers)}, requested {len(merged)})"
            )
        session.tickers = merged
        session.last_seen = self._clock()
        return sorted(session.tickers)

    def unsubscribe(self, session_id: str, tickers: list[Any]) -> list[str]:
        session = self._require_active(session_id)
        session.tickers -= set(_normalize_tickers(tickers))
        session.last_seen = self._clock()
        return sorted(session.tickers)

    # ── Routing / health ─────────────────────────────────────────────

    def sessions_for(self, symbol: str) -> list[ClientSession]:
        sym = symbol.strip().upper()
        return [
            s for s in self._sessions.values()
            if s.state is ClientState.ACTIVE and sym in s.tickers
        ]

    def snapshot(self) -> list[dict[str, Any]]:
        now = self._clock()
        return [
            {
                "session_id": s.session_id,
                "client_id": s.client_id,
                "state": s.state.value,
                "tickers": sorted(s.tickers),
                "last_seen_age_s": round(now - s.last_seen, 3),
                "stale": (now - s.last_seen) > self.heartbeat_timeout,
            }
            for s in self._sessions.values()
        ]


class LocalClientDispatcher:
    """Fan ``signal_hub`` messages out to subscribed client outboxes."""

    def __init__(self, registry: LocalClientRegistry, hub) -> None:
        self.registry = registry
        self.hub = hub
        self._task: asyncio.Task | None = None

    def dispatch(self, message: dict[str, Any]) -> None:
        symbol = message.get("symbol") or message.get("ticker")
        if not symbol:
            return  # heartbeats and non-signal traffic are not routable
        envelope = {"type": "signal", "ticker": str(symbol).upper(), "signal": message}
        for session in self.registry.sessions_for(str(symbol)):
            session.outbox.put_nowait(envelope)

    async def run(self) -> None:
        q = self.hub.subscribe()
        try:
            while True:
                self.dispatch(await q.get())
        finally:
            self.hub.unsubscribe(q)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
