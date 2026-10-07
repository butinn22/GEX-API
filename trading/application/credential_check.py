"""Credential/connector validation used by the "Test connection" control.

Kept in the application layer so the API router only maps HTTP ⇄ domain and the
check itself is unit-testable with any ``BrokerAdapter`` (no network). The check
never returns secrets — only a short, safe status message.
"""
from __future__ import annotations

import asyncio
from typing import Any

from trading.domain import BrokerError
from trading.ports import BrokerAdapter

__all__ = ["build_broker", "check_broker"]

#: Default ceiling for a single credential probe.
CHECK_TIMEOUT_SECONDS = 8.0


def build_broker(
    exchange: str,
    api_key: str,
    api_secret: str,
    extra: dict[str, Any] | None = None,
) -> BrokerAdapter | None:
    """Build the live adapter for ``exchange`` (``None`` when unsupported)."""
    from trading.adapters.brokers.bingx import BingxBroker, BingxClient
    from trading.adapters.brokers.tbank import TbankBroker
    from trading.config import settings

    extra = extra or {}
    if exchange == "bingx":
        return BingxBroker(
            BingxClient(api_key, api_secret, base_url=settings.bingx_base_url)
        )
    if exchange == "tbank":
        return TbankBroker(
            api_key,
            str(extra.get("account_id") or ""),
            sandbox=settings.tbank_sandbox,
        )
    return None


async def _close(broker: BrokerAdapter) -> None:
    client = getattr(broker, "client", None)
    close = getattr(client, "close", None)
    if callable(close):
        try:
            await close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass


async def check_broker(
    broker: BrokerAdapter, *, timeout: float = CHECK_TIMEOUT_SECONDS
) -> tuple[bool, str]:
    """Probe ``broker`` with a cheap authenticated call.

    Returns ``(ok, safe_message)``. Never raises and never includes credentials.
    The adapter is closed afterwards so a probe cannot leak a connection.
    """
    if getattr(broker, "dry_run", False):
        # A dry-run adapter fabricates accounts; reporting "connected" would be
        # a false positive (TBANK with no token or a missing SDK).
        return False, "adapter is in dry-run (no live SDK/credentials) — cannot validate"
    try:
        accounts = await asyncio.wait_for(broker.get_accounts(), timeout=timeout)
    except TimeoutError:
        return False, f"connection timed out after {timeout:g}s"
    except BrokerError as exc:
        return False, f"broker rejected the credentials: {str(exc)[:160]}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"
    finally:
        await _close(broker)
    if not accounts:
        return True, "connected (no accounts returned)"
    account = accounts[0]
    currency = getattr(account, "currency", "") or ""
    cash = getattr(account, "cash", 0.0) or 0.0
    return True, f"connected — {currency} balance {cash:g}".strip()
