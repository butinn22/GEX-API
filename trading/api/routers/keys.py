"""API-key management endpoints (authenticated)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from trading.adapters.persistence.database import get_session
from trading.adapters.persistence.models import ApiKeyRow
from trading.application.keys_service import mask_secret
from trading.config import settings
from trading.security import decrypt

from ..deps import get_keys_service, require_auth
from ..schemas import ApiKeyCreate, ApiKeyOut

router = APIRouter(
    prefix="/keys",
    tags=["keys"],
    dependencies=[Depends(require_auth)],
)


def _to_out(row: ApiKeyRow) -> ApiKeyOut:
    api_key = decrypt(settings.secret_key, row.api_key_encrypted)
    return ApiKeyOut(
        id=row.id,
        exchange=row.exchange,
        label=row.label,
        api_key_masked=mask_secret(api_key),
        created_at=row.created_at,
    )


@router.post("", response_model=ApiKeyOut, status_code=status.HTTP_201_CREATED)
async def add_key(
    body: ApiKeyCreate,
    session: AsyncSession = Depends(get_session),
    svc=Depends(get_keys_service),
) -> ApiKeyOut:
    extra = {"account_id": body.account_id} if body.account_id else None
    try:
        row = await svc.add_key(
            session,
            exchange=body.exchange,
            label=body.label,
            api_key=body.api_key,
            api_secret=body.api_secret,
            extra=extra,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _to_out(row)


@router.get("", response_model=list[ApiKeyOut])
async def list_keys(
    session: AsyncSession = Depends(get_session),
    svc=Depends(get_keys_service),
) -> list[ApiKeyOut]:
    rows = await svc.list_keys(session)
    return [_to_out(r) for r in rows]


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_key(
    key_id: int,
    session: AsyncSession = Depends(get_session),
    svc=Depends(get_keys_service),
) -> None:
    if not await svc.delete_key(session, key_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "key not found")
