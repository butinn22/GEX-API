"""Pydantic v2 схемы для auth API."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# Telegram-ник: @ + 5-32 символа [A-Za-z0-9_], по правилам Telegram.
TELEGRAM_USERNAME_RE = re.compile(r"^@[A-Za-z0-9_]{5,32}$")


def validate_telegram_username(value: str) -> str:
    """Проверить формат Telegram-ника: обязателен префикс '@'."""
    value = value.strip()
    if not TELEGRAM_USERNAME_RE.match(value):
        raise ValueError(
            "Ник Telegram должен начинаться с @ и содержать 5-32 символа "
            "(буквы, цифры, подчёркивание)"
        )
    return value


class RegisterIn(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    telegram_username: str = Field(
        ...,
        description="Ник Telegram вида @username (обязательно, с @).",
    )
    accept_terms: bool = Field(
        False,
        description="Согласие с условиями использования (EULA). Обязательно для регистрации.",
    )

    @field_validator("telegram_username")
    @classmethod
    def _check_telegram_username(cls, v: str) -> str:
        return validate_telegram_username(v)

    @field_validator("password")
    @classmethod
    def _check_password_strength(cls, v: str) -> str:
        """Минимум 8 символов + хотя бы одна буква и одна цифра."""
        if len(v) < 8:
            raise ValueError("Пароль должен содержать не менее 8 символов")
        if not re.search(r"[A-Za-zА-Яа-яЁё]", v):
            raise ValueError("Пароль должен содержать хотя бы одну букву")
        if not re.search(r"\d", v):
            raise ValueError("Пароль должен содержать хотя бы одну цифру")
        return v


class LoginIn(BaseModel):
    email: str
    password: str


class VerifyEmailIn(BaseModel):
    token: str = Field(..., description="Код/токен подтверждения email")


class OAuthCallbackIn(BaseModel):
    provider: str = Field(..., pattern="^(google|github)$")
    code: str


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 900  # 15 min


class RegisterOut(TokenOut):
    """Ответ на регистрацию: токены + ссылки активации (Telegram + email)."""
    verification_url: str
    telegram_activation_url: str
    telegram_delivered: bool = False
    message: str


class RefreshIn(BaseModel):
    refresh_token: str


class UserOut(BaseModel):
    """Публичный профиль пользователя."""
    id: str
    email: str
    is_email_verified: bool
    is_blocked: bool = False
    subscription_status: str
    oauth_provider: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    telegram_username: Optional[str] = None
    is_demo: bool = False
    is_master_admin: bool = False
    subscription_activated_at: Optional[datetime] = None
    subscription_expires_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserProfileOut(UserOut):
    """Профиль для /auth/me/admin: + master-флаг и статистика для админа."""
    is_master_admin: bool = False
    admin_stats: Optional[dict] = None


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class EmailChangeIn(BaseModel):
    """Смена email: новый адрес + текущий пароль (подтверждение)."""
    email: str = Field(..., min_length=3, max_length=255, description="Новый email-адрес")
    password: str = Field(..., min_length=1, max_length=128, description="Текущий пароль для подтверждения")

    @field_validator("email")
    @classmethod
    def _check_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_RE.match(v):
            raise ValueError("Некорректный email-адрес")
        return v


class EmailChangeOut(BaseModel):
    """Ответ на смену email: обновлённый профиль + новая пара токенов."""
    user: UserOut
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    message: str


class TelegramNicknameIn(BaseModel):
    """Обновление Telegram-ника (@username) в профиле."""
    telegram_username: str = Field(
        ...,
        description="Ник Telegram вида @username (обязательно, с @).",
    )

    @field_validator("telegram_username")
    @classmethod
    def _check_telegram_username(cls, v: str) -> str:
        return validate_telegram_username(v)


class TelegramNotifyIn(BaseModel):
    """Включение/выключение персональных Telegram-уведомлений."""
    notify: bool = False


class MessageOut(BaseModel):
    message: str
    detail: Optional[str] = None


class ErrorOut(BaseModel):
    detail: str


# =================================================================== #
#  Admin Schemas
# =================================================================== #
class AdminUserOut(BaseModel):
    """Пользователь для админ-панели (все поля)."""
    id: str
    email: str
    is_email_verified: bool
    is_blocked: bool = False
    subscription_status: str
    oauth_provider: Optional[str] = None
    telegram_username: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    subscription_activated_at: Optional[datetime] = None
    subscription_expires_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AdminUserUpdateIn(BaseModel):
    """Поля, которые админ может менять у пользователя."""
    subscription_status: Optional[str] = Field(
        None, pattern="^(INACTIVE|BASIC|EXTENDED|ADMIN)$"
    )
    is_email_verified: Optional[bool] = None
    is_blocked: Optional[bool] = None
    subscription_activated_at: Optional[str] = Field(
        None, description="ISO-8601 datetime или null для сброса"
    )
    subscription_expires_at: Optional[str] = Field(
        None, description="ISO-8601 datetime или null для сброса"
    )


class AdminUsersPageOut(BaseModel):
    """Страница пользователей для админ-панели."""
    users: list[AdminUserOut]
    total: int
    page: int
    page_size: int
    total_pages: int


class AdminStatsOut(BaseModel):
    """Статистика для админ-панели."""
    total_users: int
    active_subscriptions: int
    by_status: dict[str, int]
    by_provider: dict[str, int]

    # Расширенная аналитика
    new_users_24h: int = 0
    new_users_7d: int = 0
    expiring_7d: int = 0
    registrations_14d: list[dict] = []

    # Платёжная аналитика
    payments_total: int = 0
    payments_pending: int = 0
    payments_by_status: dict[str, int] = {}
    revenue_total_usd: float = 0.0
    revenue_30d_usd: float = 0.0
    usd_rub_rate: float = 0.0  # курс ЦБ на сегодня (для пересчёта выручки в ₽)


class AdminActivateIn(BaseModel):
    """Активация подписки админом."""
    plan: str = Field("BASIC", pattern="^(BASIC|EXTENDED)$")
    days: int = Field(30, ge=1, le=3650)


class EmailConfigIn(BaseModel):
    """Настройки корпоративной почты (SMTP)."""
    smtp_host: str = Field(..., min_length=1, max_length=255)
    smtp_port: int = Field(587, ge=1, le=65535)
    smtp_user: str = Field("", max_length=255)
    smtp_pass: str = Field("", max_length=255)
    from_email: str = Field("", max_length=255)


class TelegramConfigIn(BaseModel):
    """Настройки Telegram-бота."""
    bot_token: str = Field("", max_length=255)
    bot_username: str = Field("", max_length=128)
    chat_id: str = Field("", max_length=64)


class FinAgentKeyIn(BaseModel):
    """Входная модель PUT /auth/admin/finagent-key.

    ``api_key``: None (поле не прислано) — runtime-ключ не трогаем;
    "" — очистить runtime-ключ → фолбэк на OPENAI_COMPAT_API_KEY из .env.
    ``model`` / ``base_url``: None (поле не прислано) — runtime не трогаем;
    "" — очистить runtime → фолбэк на FINAGENT_MODEL / OPENAI_COMPAT_BASE_URL из .env.
    """
    api_key: Optional[str] = Field(
        None, max_length=512,
        description="API-ключ LLM-провайдера; '' = вернуться к .env",
    )
    model: Optional[str] = Field(
        None, max_length=128,
        description="Модель LLM (runtime поверх FINAGENT_MODEL); '' = вернуться к .env",
    )
    base_url: Optional[str] = Field(
        None, max_length=512,
        description="Base URL OpenAI-совместимого API (runtime поверх OPENAI_COMPAT_BASE_URL); '' = вернуться к .env",
    )
