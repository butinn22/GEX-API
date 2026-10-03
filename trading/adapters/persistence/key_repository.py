"""Repository for encrypted API-key rows."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ApiKeyRow

__all__ = ["ApiKeyRepository"]


class ApiKeyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        exchange: str,
        label: str,
        api_key_encrypted: str,
        api_secret_encrypted: str,
        extra_json: str = "{}",
    ) -> ApiKeyRow:
        row = ApiKeyRow(
            exchange=exchange,
            label=label,
            api_key_encrypted=api_key_encrypted,
            api_secret_encrypted=api_secret_encrypted,
            extra_json=extra_json,
        )
        self._session.add(row)
        await self._session.commit()
        await self._session.refresh(row)
        return row

    async def list(self) -> list[ApiKeyRow]:
        result = await self._session.execute(select(ApiKeyRow).order_by(ApiKeyRow.id))
        return list(result.scalars())

    async def get(self, key_id: int) -> ApiKeyRow | None:
        return await self._session.get(ApiKeyRow, key_id)

    async def delete(self, key_id: int) -> bool:
        row = await self.get(key_id)
        if row is None:
            return False
        await self._session.delete(row)
        await self._session.commit()
        return True

    async def update_extra(self, key_id: int, extra_json: str) -> ApiKeyRow | None:
        row = await self.get(key_id)
        if row is None:
            return None
        row.extra_json = extra_json
        await self._session.commit()
        await self._session.refresh(row)
        return row
