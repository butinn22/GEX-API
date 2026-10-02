"""Authentication: issue JWTs (OAuth2-style password login)."""
from __future__ import annotations

import hmac

from fastapi import APIRouter, HTTPException, status

from trading.config import settings
from trading.security import create_access_token

from .schemas import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse)
def login(body: LoginRequest) -> TokenResponse:
    ok_user = hmac.compare_digest(body.username, settings.admin_username)
    ok_pass = hmac.compare_digest(body.password, settings.admin_password)
    if not (ok_user and ok_pass):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    token = create_access_token(
        settings.secret_key,
        body.username,
        expires_minutes=settings.access_token_expire_minutes,
    )
    return TokenResponse(access_token=token)
