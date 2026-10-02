"""Общие помощники админки: зависимости прав и выборка сущностей (ring: routers).

Вынесено из ``admin_router.py`` (итерация 42) **до** разбиения на подмодули: если бы
помощники остались в фасаде, каждый подмодуль импортировал бы из него, а фасад — подмодули,
то есть цикл. Здесь же лежит то, что нужно всем группам: проверка прав администратора и
«пользователь или 404».

``_user_or_404`` — одна формулировка на все девять обработчиков, которые раньше повторяли
выборку и проверку по отдельности (см. итерацию 34).
"""
from __future__ import annotations

from ..dependencies import (get_current_user)
from ..models import (User, SubscriptionStatus, SUBSCRIPTION_VALUES)
from ..service import (is_master_admin)
from datetime import (datetime, timedelta, timezone)
from fastapi import (APIRouter, Depends, HTTPException, Query, Request, status)
from sqlalchemy.orm import (Session)
from typing import (Optional)
import csv
import io
import logging
import time


logger = logging.getLogger(__name__)


_APP_START_TIME = time.monotonic()


def _user_or_404(db: Session, user_id: str) -> User:
    """Пользователь по id или 404.

    Одна формулировка на все девять обработчиков: раньше строка выборки и проверка «не найден»
    повторялись в каждом, и сообщение об ошибке успело разойтись, а сам SQL был не виден
    при чтении HTTP-кода. Здесь остаётся решение HTTP-слоя (какой код и что сказать),
    а выборка ушла в :class:`UserRepository`.
    """
    from gex.adapters.persistence.auth_repositories import UserRepository

    found = UserRepository(db).by_id(user_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return found


def _require_admin(user: User = Depends(get_current_user)) -> User:
    """Доступ в админку: Master Admin (по email из .env) ИЛИ пользователь со
    статусом подписки ADMIN (выдаётся Master'ом/CLI). ADMIN-статус даёт полный
    доступ к админ-панели, но НЕ позволяет менять Master'а и выдавать других
    администраторов (эти операции — только Master)."""
    if not is_master_admin(user) and user.subscription_status != SubscriptionStatus.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Доступ только для администратора.",
        )
    return user


def _is_master(admin: User) -> bool:
    """True если текущий админ — Master Admin (root, из .env)."""
    return is_master_admin(admin)


USER_EXPORT_FIELDS = [
    # AUTH-04: секреты НЕ экспортируются. Раньше CSV-выгрузка содержала bcrypt-хэши паролей,
    # живой `verification_token` (подтверждение чужого email) и `telegram_connect_token`
    # (привязка чужого Telegram) — утечка файла бэкапа = компрометация аккаунтов.
    # Импорт (`import_users`) по-прежнему ЧИТАЕТ эти колонки, если они есть в старом бэкапе,
    # но экспорт их больше не создаёт.
    "id", "email", "is_email_verified", "is_blocked",
    "telegram_chat_id", "telegram_username", "telegram_notify", "telegram_test_at",
    "subscription_status", "subscription_activated_at", "subscription_expires_at",
    "subscription_expiry_notified_for",
    "oauth_provider", "oauth_id", "created_at", "updated_at",
]


PAYMENT_EXPORT_FIELDS = [
    "id", "user_id", "user_email", "plan", "amount_rub", "amount_usd",
    "method", "crypto_currency", "crypto_amount", "status", "admin_note",
    "created_at", "confirmed_at", "expires_at",
]


def _user_export_row(u: User) -> dict:
    """Полная строка пользователя для экспорта/импорта (все поля)."""
    return {
        "id": u.id,
        "email": u.email,
        "is_email_verified": bool(u.is_email_verified),
        "is_blocked": bool(u.is_blocked),
        "telegram_chat_id": u.telegram_chat_id or "",
        "telegram_username": u.telegram_username or "",
        "telegram_notify": bool(u.telegram_notify),
        "telegram_test_at": u.telegram_test_at.isoformat() if u.telegram_test_at else "",
        "subscription_status": u.subscription_status,
        "subscription_activated_at": u.subscription_activated_at.isoformat()
            if u.subscription_activated_at else "",
        "subscription_expires_at": u.subscription_expires_at.isoformat()
            if u.subscription_expires_at else "",
        "subscription_expiry_notified_for": u.subscription_expiry_notified_for.isoformat()
            if u.subscription_expiry_notified_for else "",
        "oauth_provider": u.oauth_provider or "",
        "oauth_id": u.oauth_id or "",
        "created_at": u.created_at.isoformat() if u.created_at else "",
        "updated_at": u.updated_at.isoformat() if u.updated_at else "",
    }


def _payment_export_row(p) -> dict:
    """Полная строка платежа для экспорта/импорта (все поля)."""
    return {
        "id": p.id,
        "user_id": p.user_id,
        "user_email": p.user_email,
        "plan": p.plan,
        "amount_rub": p.amount_rub,
        "amount_usd": p.amount_usd,
        "method": p.method,
        "crypto_currency": p.crypto_currency or "",
        "crypto_amount": p.crypto_amount if p.crypto_amount is not None else "",
        "status": p.status,
        "admin_note": p.admin_note or "",
        "created_at": p.created_at.isoformat() if p.created_at else "",
        "confirmed_at": p.confirmed_at.isoformat() if p.confirmed_at else "",
        "expires_at": p.expires_at.isoformat() if p.expires_at else "",
    }


def _parse_iso_dt(value) -> Optional[datetime]:
    """ISO-строка / datetime → timezone-aware datetime (или None)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = datetime.fromisoformat(s)
            except ValueError:
                raise ValueError(f"Неверная дата: {s!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_bool(value, default: bool = False) -> bool:
    """true/false/1/0/да/нет → bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y", "да", "д"):
        return True
    if s in ("0", "false", "no", "n", "нет", "н", ""):
        return False
    return default


def _parse_float(value) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _detect_upload_format(body: bytes, hint: Optional[str]) -> str:
    """Определить формат: json / csv (по hint, затем по содержимому)."""
    if hint:
        h = hint.strip().lower()
        if h in ("json", "csv"):
            return h
    stripped = body.lstrip().lstrip(b"\xef\xbb\xbf")  # BOM
    if stripped[:1] in (b"[", b"{"):
        return "json"
    return "csv"


def _read_csv_records(body: bytes) -> list[dict]:
    """CSV (с BOM или без) → список dict по заголовкам."""
    text = body.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader if any((v or "").strip() for v in row.values())]


def _read_json_records(body: bytes) -> list[dict]:
    import json as _json
    text = body.decode("utf-8-sig", errors="replace")
    data = _json.loads(text)
    if isinstance(data, dict):
        # допускаем обёртку {users: [...]} / {payments: [...]}
        for key in ("users", "payments", "items", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("JSON должен быть списком объектов или объектом {users: [...]}")


async def _read_upload(request: Request, fmt_hint: Optional[str] = None):
    """Прочитать тело запроса и вернуть (records, format)."""
    body = await request.body()
    if not body or not body.strip():
        raise HTTPException(status_code=400, detail="Пустой файл.")
    fmt = _detect_upload_format(body, fmt_hint)
    try:
        if fmt == "json":
            records = _read_json_records(body)
        else:
            records = _read_csv_records(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Не удалось разобрать файл: {e}")
    if not records:
        raise HTTPException(status_code=400, detail="Файл не содержит записей.")
    return records, fmt


def _resend_email_verification(u: User, db: Session) -> str:
    """Пересоздать токен и отправить письмо подтверждения. Возвращает ссылку."""
    import uuid as _uuid
    from gex.adapters.notifications.email_service import send_verification_email
    from gex.auth.config import settings as _settings

    token = _uuid.uuid4().hex + _uuid.uuid4().hex
    u.verification_token = token
    db.commit()

    base_url = _settings.FRONTEND_URL.rstrip("/")
    verify_link = f"{base_url}/auth/verify-email?token={token}"
    sent = send_verification_email(u.email, token, base_url)
    logger.info("Admin re-sent email verification to %s (sent=%s)", u.email, sent)
    return verify_link


def _resend_telegram_activation(u: User, db: Session) -> str:
    """Пересоздать токен и отправить ссылку активации в Telegram. Возвращает ссылку."""
    import uuid as _uuid
    from ..router import _send_telegram_activation, _telegram_activation_link

    token = _uuid.uuid4().hex + _uuid.uuid4().hex
    u.verification_token = token
    db.commit()

    link = _telegram_activation_link(token)
    delivered = _send_telegram_activation(u, link)
    logger.info("Admin re-sent Telegram activation to %s (delivered=%s)", u.email, delivered)
    return link


def _payment_requisites_state(db: Session) -> dict:
    """Эффективные реквизиты + способы оплаты, видимые клиенту."""
    from ..payment_service import available_payment_methods, get_requisites_row, requisites_map

    row = get_requisites_row(db)
    req = requisites_map(row)
    state = dict(req)
    state["updated_at"] = row.updated_at
    state["methods"] = available_payment_methods(req)
    return state
