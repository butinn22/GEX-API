"""WebSocket endpoints for live signals / orders / positions.

Each endpoint subscribes to its hub and streams messages to the client, with a
heartbeat when idle. A reader task detects client disconnect and cancels the
sender cleanly.

**Auth:** the browser WebSocket API cannot set an ``Authorization`` header, so
the JWT is carried in the ``Sec-WebSocket-Protocol`` subprotocol list
(``new WebSocket(url, ["gex.jwt", token])``) — with ``?token=`` as a fallback.
The token is validated **before** ``accept()``; failures close with code 1008.
"""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from trading.application.signal_hub import order_hub, position_hub, signal_hub

from .deps import WS_SUBPROTOCOL, validate_ws_jwt
from .ws_limits import ws_limiter

router = APIRouter()


async def _stream(ws: WebSocket, q: asyncio.Queue, heartbeat: float) -> None:
    while True:
        try:
            message = await asyncio.wait_for(q.get(), timeout=heartbeat)
            await ws.send_json(message)
        except asyncio.TimeoutError:
            await ws.send_json({"type": "heartbeat", "ts": time.time()})


async def _run_stream(ws: WebSocket, hub, hello: dict, heartbeat: float = 5.0) -> None:
    if not validate_ws_jwt(ws):
        # Reject before accept — the handshake fails (HTTP 403) and no stream
        # ever opens. 1008 = policy violation.
        await ws.close(code=1008)
        return
    if not ws_limiter.acquire(ws):
        # Per-IP concurrent-connection cap (the HTTP limiter never sees WS).
        await ws.close(code=1013)  # try again later
        return
    try:
        await ws.accept(subprotocol=WS_SUBPROTOCOL)
        q = hub.subscribe()
        try:
            await ws.send_json(hello)
            sender = asyncio.create_task(_stream(ws, q, heartbeat))
            reader = asyncio.create_task(ws.receive_text())
            done, pending = await asyncio.wait(
                {sender, reader}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                if task is sender:
                    await task  # surface any send error
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.unsubscribe(q)
    finally:
        ws_limiter.release(ws)


@router.websocket("/ws/signals")
async def ws_signals(ws: WebSocket) -> None:
    await _run_stream(ws, signal_hub, {"type": "hello", "stream": "signals"})


@router.websocket("/ws/orders")
async def ws_orders(ws: WebSocket) -> None:
    await _run_stream(ws, order_hub, {"type": "hello", "stream": "orders"})


@router.websocket("/ws/positions")
async def ws_positions(ws: WebSocket) -> None:
    await _run_stream(ws, position_hub, {"type": "hello", "stream": "positions"})
