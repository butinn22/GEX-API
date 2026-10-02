"""API-key management service (encrypt-at-rest + CRUD + credential resolution)."""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from trading.adapters.persistence.key_repository import ApiKeyRepository
from trading.adapters.persistence.models import ApiKeyRow
from trading.security import decrypt, encrypt

__all__ = ["KeysService", "mask_secret"]

VALID_EXCHANGES = ("bingx", "tbank")


def mask_secret(secret: str, keep: int = 4) -> str:
    if len(secret) <= keep * 2:
        return "*" * len(secret)
    return f"{secret[:keep]}…{secret[-keep:]}"


class KeysService:
    def __init__(self, secret: str) -> None:
        self._secret = secret

    # ── CRUD ───────────────────────────────────────────────────────────

    async def add_key(
        self,
        session: AsyncSession,
        *,
        exchange: str,
        label: str,
        api_key: str,
        api_secret: str,
        extra: dict[str, Any] | None = None,
    ) -> ApiKeyRow:
        if exchange not in VALID_EXCHANGES:
            raise ValueError(f"unsupported exchange: {exchange}")
        if not api_key:
            raise ValueError("api_key is required")
        repo = ApiKeyRepository(session)
        return await repo.create(
            exchange,
            label,
            encrypt(self._secret, api_key),
            encrypt(self._secret, api_secret or ""),
            json.dumps(extra or {}),
        )

    async def list_keys(self, session: AsyncSession) -> list[ApiKeyRow]:
        return await ApiKeyRepository(session).list()

    async def delete_key(self, session: AsyncSession, key_id: int) -> bool:
        return await ApiKeyRepository(session).delete(key_id)

    # ── Credential resolution (for brokers) ────────────────────────────

    async def resolve_credentials(
        self, session: AsyncSession, exchange: str
    ) -> dict[str, Any] | None:
        """Return the first decrypted credential set for ``exchange`` (or None)."""
        rows = await ApiKeyRepository(session).list()
        for row in rows:
            if row.exchange == exchange:
                return {
                    "api_key": decrypt(self._secret, row.api_key_encrypted),
                    "api_secret": decrypt(self._secret, row.api_secret_encrypted),
                    "extra": json.loads(row.extra_json or "{}"),
                }
        return None
