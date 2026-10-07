"""API-key management service (encrypt-at-rest + CRUD + credential resolution)."""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from trading.adapters.persistence.key_repository import ApiKeyRepository
from trading.adapters.persistence.models import ApiKeyRow
from trading.application.account_router import (
    AccountRouter,
    AccountSettings,
    AccountView,
    select_accounts,
)
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
        # BingX authenticates every signed call with the secret → mandatory.
        if exchange == "bingx" and not (api_secret or "").strip():
            raise ValueError("api_secret is required for bingx credentials")
        # TBANK routes orders by account id (stored in ``extra``) → mandatory.
        # Its secret is optional (the token is the credential).
        if exchange == "tbank" and not str((extra or {}).get("account_id") or "").strip():
            raise ValueError("account_id is required for tbank credentials")
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

    # ── Multi-account routing settings (stored in extra_json) ───────

    def _row_credentials(self, row: ApiKeyRow) -> dict[str, Any]:
        return {
            "api_key": decrypt(self._secret, row.api_key_encrypted),
            "api_secret": decrypt(self._secret, row.api_secret_encrypted),
            "extra": json.loads(row.extra_json or "{}"),
        }

    async def update_settings(
        self, session: AsyncSession, key_id: int, patch: dict[str, Any]
    ) -> ApiKeyRow:
        """Merge routing/risk settings into a key's ``extra`` (validated)."""
        repo = ApiKeyRepository(session)
        row = await repo.get(key_id)
        if row is None:
            raise ValueError("key not found")
        extra = json.loads(row.extra_json or "{}")
        current = AccountSettings.from_extra(extra)
        updated = current.updated(patch)
        # Preserve non-routing extra fields (account_id, sandbox, …).
        extra.update(updated.as_dict())
        out = await repo.update_extra(key_id, json.dumps(extra))
        assert out is not None
        return out

    async def resolve_accounts_for_symbol(
        self, session: AsyncSession, symbol: str
    ) -> list[AccountView]:
        """All enabled accounts (credentials decrypted) that trade ``symbol``."""
        return select_accounts(await self.list_account_views(session), symbol)

    async def list_account_views(self, session: AsyncSession) -> list[AccountView]:
        """Every registered account as a routing view (credentials decrypted)."""
        rows = await ApiKeyRepository(session).list()
        return [
            AccountView(
                key_id=row.id,
                exchange=row.exchange,
                label=row.label,
                credentials=self._row_credentials(row),
                settings=AccountSettings.from_extra(json.loads(row.extra_json or "{}")),
            )
            for row in rows
        ]

    async def build_account_router(self, session: AsyncSession) -> AccountRouter:
        """An :class:`AccountRouter` over every registered account.

        Attach it to an ``ExecutionEngine`` to execute each intent on every
        account whose instrument scope covers it.
        """
        return AccountRouter(await self.list_account_views(session))
