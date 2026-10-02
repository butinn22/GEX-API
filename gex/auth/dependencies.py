"""FastAPI dependencies (Barriers): JWT extraction, email verification, subscription check."""
from __future__ import annotations

import sys
from typing import Optional

# Очищаем пути uv-кэша из sys.path (конфликтует с python-jose)
sys.path = [p for p in sys.path if '\\cache\\' not in p and '/cache/' not in p]

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from .config import settings
from gex.adapters.persistence.database import get_session
from .models import User, subscription_is_active
from .service import (
    can_bypass_barriers,
    check_subscription,
    decode_token,
    get_user_by_id,
    is_master_admin,
)

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_session),
) -> User:
    """Извлечь текущего пользователя из JWT Access Token.

    Raises
    ------
    HTTPException(401)
        Если токен отсутствует, недействителен или пользователь не найден.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется авторизация",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_token(credentials.credentials)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Недействительный или просроченный токен",
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный тип токена",
        )

    user_id = payload.get("sub")
    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден",
        )

    # Заблокированный пользователь не может использовать API
    if user.is_blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Аккаунт заблокирован. Обратитесь к администратору.",
        )

    return user


# ================================================================= #
#  Barrier 1: Email Verification
# ================================================================= #
def require_verified_email(user: User = Depends(get_current_user)) -> User:
    """Barrier 1: пользователь должен подтвердить email.

    Master Admin (sadisting) обходит проверку.
    """
    if can_bypass_barriers(user):
        return user
    if not user.is_email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Подтвердите email перед использованием этого эндпоинта. "
                   "Используйте /auth/verify-email или /auth/resend-verification.",
        )
    return user


# ================================================================= #
#  Barrier 2: Subscription Level
# ================================================================= #
def require_subscription(level: str):
    """Barrier 2: минимальный требуемый уровень подписки.

    Использование::

        @app.get("/protected")
        def protected(user: User = Depends(require_subscription("BASIC"))):
            ...

    Master Admin (sadisting) обходит проверку.
    """
    if level not in ("INACTIVE", "BASIC", "EXTENDED", "ADMIN"):
        raise ValueError(f"Неизвестный уровень подписки: {level}")

    def _dependency(user: User = Depends(get_current_user)) -> User:
        if can_bypass_barriers(user):
            return user
        # Email-гейт исполняется здесь (а не только на /auth/login): любой
        # data-роут, закрытый подпиской, требует подтверждённый email.
        if not user.is_email_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Подтвердите email перед использованием этого раздела: "
                    "откройте ссылку из письма или активируйте аккаунт в Telegram "
                    "(POST /auth/send-verification — повторная отправка письма)."
                ),
            )
        if not check_subscription(user, level):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Требуется подписка уровня '{level}' или выше. "
                       f"Текущий статус: {user.subscription_status}.",
            )
        # Подписка должна быть активной (срок не истёк).
        if not subscription_is_active(user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Срок подписки истёк. Продлите подписку, чтобы продолжить использование.",
            )
        return user

    return _dependency


# ================================================================= #
#  Barrier 3: Master Admin
# ================================================================= #
def require_master_admin(user: User = Depends(get_current_user)) -> User:
    """Barrier: только Master Admin (внутренние/служебные эндпоинты)."""
    if not can_bypass_barriers(user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Доступно только администратору.",
        )
    return user


# ================================================================= #
#  Barrier Combo: Verified + Subscription
# ================================================================= #
def require_verified_and(level: str):
    """Barrier 1 + Barrier 2: email подтверждён + минимальная подписка."""
    def _dependency(user: User = Depends(get_current_user)) -> User:
        # Barrier 1
        if not can_bypass_barriers(user) and not user.is_email_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Подтвердите email перед использованием этого эндпоинта.",
            )
        # Barrier 2
        if not can_bypass_barriers(user) and not check_subscription(user, level):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Требуется подписка '{level}'. Текущая: {user.subscription_status}.",
            )
        return user

    return _dependency


def is_master_admin_user(user: User = Depends(get_current_user)) -> bool:
    """Проверка, является ли текущий пользователь Master Admin."""
    return is_master_admin(user)


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_session),
) -> Optional[User]:
    """Извлечь пользователя из JWT, если он авторизован. Иначе None.

    Не роняет 401 — возвращает None для неавторизованных запросов.
    Используется для опционального получения chat_id для Telegram-уведомлений.
    """
    if credentials is None:
        return None
    try:
        payload = decode_token(credentials.credentials)
    except JWTError:
        return None
    if payload.get("type") != "access":
        return None
    user_id = payload.get("sub")
    return get_user_by_id(db, user_id)
