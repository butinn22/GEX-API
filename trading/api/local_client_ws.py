"""Local signal client WebSocket API.

``/ws/client`` lets external signal producers connect with **no broker account
integration**: handshake → heartbeat → subscribe tickers → push raw signals and
receive native exchange order payloads (BingX v3 / TBank PostOrderRequest).

Protocol (JSON text frames) — see docs/SPEC-local-signal-sprint.md §1.
Connection health: any inbound frame resets liveness; a watchdog closes
sessions silent for longer than ``heartbeat_timeout`` with WS code 4001.
"""
from __future__ import annotations

import asyncio
import hmac
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from trading.application.local_client import (
    HEARTBEAT_INTERVAL,
    HEARTBEAT_TIMEOUT,
    MAX_TICKERS_PER_CLIENT,
    ClientSession,
    HandshakeRequiredError,
    LocalClientDispatcher,
    LocalClientError,
    LocalClientRegistry,
    MaxTickersError,
)
from trading.application.native_payloads import (
    SignalParseError,
    native_payloads_for,
    signal_from_raw,
)
from trading.application.signal_hub import signal_hub
from trading.config import settings
from trading.security import SecretError, decode_access_token

from .deps import require_auth
from .ws_limits import ws_limiter

__all__ = ["router", "registry", "dispatcher", "start_dispatcher", "stop_dispatcher"]

router = APIRouter()
registry = LocalClientRegistry()
dispatcher = LocalClientDispatcher(registry, signal_hub)


def _error(code: str, message: str) -> dict[str, Any]:
    return {"type": "error", "code": code, "message": message}


def _valid_client_token(token: str) -> bool:
    """A producer authenticates with the static ``TRADING_LOCAL_CLIENT_TOKEN``
    (preferred) or any valid JWT."""
    if not token:
        return False
    static = settings.local_client_token
    if static and hmac.compare_digest(token, static):
        return True
    try:
        decode_access_token(settings.secret_key, token)
        return True
    except SecretError:
        return False


_ERROR_CODES = {
    MaxTickersError: "MAX_TICKERS",
    HandshakeRequiredError: "NO_HANDSHAKE",
    SignalParseError: "BAD_SIGNAL",
}


def _handle_message(session: ClientSession, msg: dict[str, Any]) -> dict[str, Any]:
    mtype = msg.get("type")
    if mtype == "handshake":
        registry.handshake(session.session_id, str(msg.get("client_id") or session.client_id))
        return {
            "type": "handshake_ack",
            "session_id": session.session_id,
            "server_time": time.time(),
            "heartbeat_interval": HEARTBEAT_INTERVAL,
            "heartbeat_timeout": HEARTBEAT_TIMEOUT,
            "max_tickers": MAX_TICKERS_PER_CLIENT,
        }
    if mtype == "ping":
        registry.touch(session.session_id)
        return {"type": "pong", "ts": msg.get("ts", time.time())}
    if mtype == "subscribe":
        tickers = registry.subscribe(session.session_id, msg.get("tickers") or [])
        return {"type": "subscribe_ack", "tickers": tickers, "count": len(tickers)}
    if mtype == "unsubscribe":
        tickers = registry.unsubscribe(session.session_id, msg.get("tickers") or [])
        return {"type": "subscribe_ack", "tickers": tickers, "count": len(tickers)}
    if mtype == "signal":
        if session.state.value != "active":
            raise HandshakeRequiredError("handshake required first")
        sig = signal_from_raw(msg.get("payload"), source=session.client_id)
        payloads = native_payloads_for(sig, tbank_lots=max(1, int(sig.quantity.value)) if sig.quantity else None)
        registry.touch(session.session_id)
        signal_hub.publish({
            "type": "signal",
            "symbol": sig.symbol,
            "side": sig.side.value,
            "strength": sig.strength,
            "strategy": sig.strategy,
            "reason": sig.reason,
        })
        return {
            "type": "signal_ack",
            "signal_id": uuid.uuid4().hex,
            "symbol": sig.symbol,
            "native_payloads": payloads,
        }
    return _error("BAD_MESSAGE", f"unknown message type: {mtype!r}")


async def _pump_outbox(ws: WebSocket, session: ClientSession) -> None:
    while True:
        await ws.send_json(await session.outbox.get())


async def _watchdog(ws: WebSocket, session: ClientSession) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        try:
            if registry.is_stale(session.session_id):
                await ws.close(code=4001)
                return
        except LocalClientError:
            return  # session already gone


@router.websocket("/ws/client")
async def ws_client(ws: WebSocket) -> None:
    if not ws_limiter.acquire(ws):
        # Per-IP concurrent-connection cap (BaseHTTPMiddleware never sees WS).
        await ws.close(code=1013)  # try again later
        return
    try:
        await ws.accept()
        session = registry.connect()
        tasks: list[asyncio.Task] = []
        try:
            tasks = [
                asyncio.create_task(_pump_outbox(ws, session)),
                asyncio.create_task(_watchdog(ws, session)),
            ]
            while True:
                msg = await ws.receive_json()
                if not isinstance(msg, dict):
                    await ws.send_json(_error("BAD_MESSAGE", "frame must be a JSON object"))
                    continue
                # Authenticate the producer at the handshake. An invalid/missing
                # token never registers the session, so no signal frame can ever
                # be emitted (every signal path requires an active handshake).
                if msg.get("type") == "handshake" and not _valid_client_token(
                    str(msg.get("token") or "")
                ):
                    await ws.send_json(
                        _error("UNAUTHORIZED", "invalid or missing handshake token")
                    )
                    await ws.close(code=1008)
                    return
                try:
                    reply = _handle_message(session, msg)
                except LocalClientError as exc:
                    reply = _error(_ERROR_CODES.get(type(exc), "CLIENT_ERROR"), str(exc))
                except SignalParseError as exc:
                    reply = _error("BAD_SIGNAL", str(exc))
                await ws.send_json(reply)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            for task in tasks:
                task.cancel()
            registry.disconnect(session.session_id)
    finally:
        ws_limiter.release(ws)


@router.get("/api/v1/local-clients", dependencies=[Depends(require_auth)])
def local_clients() -> dict[str, Any]:
    """Connection health for every local signal client session."""
    return {"clients": registry.snapshot()}


async def start_dispatcher() -> None:
    dispatcher.start()


async def stop_dispatcher() -> None:
    await dispatcher.stop()
