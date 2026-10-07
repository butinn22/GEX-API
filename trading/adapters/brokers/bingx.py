"""BingX broker adapter (Swap perpetual futures, REST V3) with HMAC-SHA256 signing.

Signing (per BingX Open API V3):
  * signed params carry a millisecond ``timestamp``;
  * the signature is ``hex(HMAC-SHA256(api_secret, query_string))`` where
    ``query_string`` is the params sorted by key and URL-encoded (spaces as %20,
    i.e. ``quote``, not ``quote_plus``), excluding ``signature`` itself;
  * ``X-BX-APIKEY`` carries the API key.

The low-level ``BingxClient`` is HTTP/transport-agnostic (injectable ``httpx``
transport, so it is unit-testable offline). ``BingxBroker`` adapts it to the
domain ``BrokerAdapter`` and enforces rate limits via an optional ``RateLimiter``.
"""
from __future__ import annotations

import hashlib
import hmac
import time
import urllib.parse
from collections.abc import Mapping
from typing import Any

import httpx

from trading.adapters.ratelimit import RateLimiter
from trading.domain import (
    Account,
    BrokerError,
    Exchange,
    Order,
    OrderIntent,
    OrderStatus,
    Portfolio,
    Position,
    PositionSide,
)
from trading.ports import BrokerAdapter

__all__ = ["sign_hmac_sha256", "build_query", "BingxClient", "BingxBroker"]


def sign_hmac_sha256(secret: str, message: str) -> str:
    """BingX signature: lowercase hex of HMAC-SHA256(secret, message)."""
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def build_query(params: Mapping[str, Any]) -> str:
    """Sort params by key and URL-encode (spaces → %20)."""
    items = sorted((k, str(v)) for k, v in params.items() if v is not None)
    return urllib.parse.urlencode(items, quote_via=urllib.parse.quote)


class BingxClient:
    """Low-level async REST client for BingX Swap V3."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str = "https://open-api.bingx.com",
        transport: httpx.AsyncBaseTransport | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self._limiter = rate_limiter
        self._client = httpx.AsyncClient(base_url=self.base_url, transport=transport)

    async def close(self) -> None:
        await self._client.aclose()

    def _signed_params(self, params: dict[str, Any]) -> tuple[str, dict[str, str]]:
        params = {**params, "timestamp": int(time.time() * 1000)}
        query = build_query(params)
        signature = sign_hmac_sha256(self.api_secret, query)
        full = f"{query}&signature={signature}"
        headers = {"X-BX-APIKEY": self.api_key}
        return full, headers

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        signed: bool = True,
        rate_cost: float = 1.0,
        rate_capacity: float = 20.0,
        rate_refill: float = 20.0,
    ) -> dict[str, Any]:
        params = dict(params or {})
        url = path
        headers: dict[str, str] = {}
        if signed:
            query, headers = self._signed_params(params)
            url = f"{path}?{query}"
        elif params:
            url = f"{path}?{build_query(params)}"

        if self._limiter is not None:
            allowed, retry = self._limiter.acquire(
                f"bingx:{path}", rate_capacity, rate_refill, rate_cost
            )
            if not allowed:
                raise BrokerError(f"bingx rate limit hit, retry in {retry:.1f}s")

        resp = await self._client.request(method, url, headers=headers)
        if resp.status_code >= 400:
            raise BrokerError(f"bingx {method} {path} -> HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        # BingX wraps failures in {code, msg} with code != 0.
        if isinstance(data, dict) and data.get("code") not in (0, None, ""):
            raise BrokerError(f"bingx api error {data.get('code')}: {data.get('msg')}")
        return data

    # ── Endpoints ──────────────────────────────────────────────────────

    async def server_time(self) -> int:
        data = await self._request("GET", "/openApi/swap/v2/server/time", signed=False)
        return int(data["data"]["serverTime"])

    async def account_balance(self) -> dict[str, Any]:
        data = await self._request("GET", "/openApi/swap/v3/user/balance")
        return data.get("data", {})

    async def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params = {"symbol": symbol} if symbol else {}
        data = await self._request("GET", "/openApi/swap/v3/user/positions", params)
        return data.get("data", []) or []

    async def place_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        quantity: float,
        *,
        price: float | None = None,
        stop_price: float | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": type_.upper(),
            "quantity": quantity,
        }
        if price is not None:
            params["price"] = price
        if stop_price is not None:
            params["stopPrice"] = stop_price
        data = await self._request("POST", "/openApi/swap/v3/trade/order", params)
        return data.get("data", {}) or {}

    async def cancel_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        data = await self._request(
            "DELETE", "/openApi/swap/v3/trade/order",
            {"symbol": symbol, "orderId": order_id},
        )
        return data.get("data", {}) or {}

    async def get_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        data = await self._request(
            "GET", "/openApi/swap/v3/trade/order",
            {"symbol": symbol, "orderId": order_id},
        )
        return data.get("data", {}) or {}

    async def create_user_data_stream(self) -> str:
        """Request a listenKey for the private user-data WebSocket."""
        data = await self._request("POST", "/openApi/user/auth/userDataStream")
        key = data.get("listenKey") or (data.get("data") or {}).get("listenKey")
        if not key:
            raise BrokerError("bingx: no listenKey returned")
        return str(key)


class BingxBroker(BrokerAdapter):
    """``BrokerAdapter`` over :class:`BingxClient` (domain-type mapping)."""

    exchange = Exchange.BINGX

    def __init__(self, client: BingxClient) -> None:
        self.client = client

    async def get_accounts(self) -> list[Account]:
        bal = await self.client.account_balance()
        balance = bal.get("balance", {})
        return [
            Account(
                id=str(bal.get("uid") or "bingx-default"),
                currency="USDT",
                cash=float(balance.get("balance") or 0.0),
                buying_power=float(balance.get("availableMargin") or balance.get("balance") or 0.0),
                margin_used=float(balance.get("usedMargin") or 0.0),
            )
        ]

    async def get_positions(self) -> list[Position]:
        raw = await self.client.get_positions()
        out: list[Position] = []
        for p in raw:
            qty = abs(float(p.get("positionAmt") or 0.0))
            if qty <= 0:
                continue
            amt = float(p.get("positionAmt") or 0.0)
            side = PositionSide.LONG if amt > 0 else PositionSide.SHORT
            out.append(
                Position(
                    symbol=p.get("symbol", ""),
                    side=side,
                    quantity=qty,
                    average_entry_price=float(p.get("avgPrice") or 0.0),
                    realized_pnl=float(p.get("realizedProfit") or 0.0),
                )
            )
        return out

    async def get_portfolio(self) -> Portfolio:
        accounts = await self.get_accounts()
        positions = await self.get_positions()
        cash = accounts[0].cash if accounts else 0.0
        return Portfolio(cash=cash, positions=tuple(positions), currency="USDT")

    async def place_order(self, intent: OrderIntent) -> Order:
        type_ = {
            "market": "MARKET",
            "limit": "LIMIT",
            "stop_market": "STOP_MARKET",
        }.get(intent.order_type.value, "MARKET")
        resp = await self.client.place_order(
            intent.symbol,
            intent.side.value,
            type_,
            intent.quantity.value,
            price=intent.limit_price.value if intent.limit_price else None,
            stop_price=intent.stop_price.value if intent.stop_price else None,
        )
        return Order(
            id=str(resp.get("order", {}).get("orderId") or resp.get("orderId") or ""),
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity.value,
            order_type=intent.order_type,
            status=OrderStatus.OPEN,
            strategy=intent.strategy,
            reason=intent.reason,
        )

    async def cancel_order(self, order_id: str) -> Order:
        # Cancellation is symbol-scoped on BingX; keep a best-effort stub that
        # reflects the terminal state. Symbol must be supplied by the caller via
        # the order id convention "<symbol>:<orderId>" when known.
        raise BrokerError("cancel_order requires symbol; use client.cancel_order(symbol, order_id)")

    async def get_order_status(self, order_id: str) -> Order:
        raise BrokerError("get_order_status requires symbol; use client.get_order(symbol, order_id)")
