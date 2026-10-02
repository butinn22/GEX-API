"""Пользователи: список, карточка, модерация, подписка, импорт/выгрузка.

Обработчики одного ресурса. Вынесены вместе, потому что делят не только тему, но и
порядок регистрации: ``/users/export`` обязан идти до ``/users/{user_id}``.

Вынесено из ``admin_router.py`` (итерация 42). Обработчики перенесены дословно;
подроутер подключает фасад, поэтому пути не менялись — проверяет
``tests/test_admin_routes.py``.
"""
from __future__ import annotations

from ..models import (User, SubscriptionStatus, SUBSCRIPTION_VALUES)
from ..schemas import (AdminActivateIn, AdminStatsOut, AdminUserOut, AdminUserUpdateIn, AdminUsersPageOut, EmailConfigIn, FinAgentKeyIn, MessageOut, TelegramConfigIn)
from ..service import (is_master_admin)
from datetime import (datetime, timedelta, timezone)
from fastapi import (APIRouter, Depends, HTTPException, Query, Request, status)
from fastapi.responses import (StreamingResponse)
from gex.adapters.persistence.database import (get_session)
from sqlalchemy.orm import (Session)
from typing import (Optional)
import csv
import io

from ._shared import (
    USER_EXPORT_FIELDS,
    _is_master,
    _parse_bool,
    _parse_iso_dt,
    _read_upload,
    _require_admin,
    _resend_email_verification,
    _resend_telegram_activation,
    _user_export_row,
    _user_or_404,
    logger,
)


router = APIRouter()


@router.get("/users", response_model=AdminUsersPageOut)
def list_users(
    page: int = Query(1, ge=1, description="Номер страницы (1-based)"),
    page_size: int = Query(50, ge=1, le=500, description="Размер страницы"),
    search: Optional[str] = Query(None, description="Поиск по email"),
    status_filter: Optional[str] = Query(
        None, description="Фильтр по подписке (INACTIVE|BASIC|EXTENDED|ADMIN)"
    ),
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Список всех пользователей с пагинацией, поиском и фильтрацией.

    Запросы — в репозитории; здесь остаётся только валидация входа и форма ответа.
    """
    from gex.adapters.persistence.auth_repositories import UserRepository

    if status_filter:
        status_filter = status_filter.upper()
        if status_filter not in SUBSCRIPTION_VALUES:
            raise HTTPException(
                status_code=400,
                detail=f"Неизвестный статус: {status_filter}. Допустимые: {', '.join(SUBSCRIPTION_VALUES)}",
            )

    total_pages = 0  # пересчитывается после запроса
    offset = (page - 1) * page_size
    users, total = UserRepository(db).page(
        search=search, status=status_filter, offset=offset, limit=page_size
    )
    total_pages = max(1, (total + page_size - 1) // page_size)

    return AdminUsersPageOut(
        users=[AdminUserOut.model_validate(u) for u in users],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/users/export")
def export_users(
    format: str = Query("csv", description="Формат: csv или json"),
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Выгрузить всех пользователей в CSV или JSON (все поля, для полного восстановления)."""
    from gex.adapters.persistence.auth_repositories import UserRepository

    users = UserRepository(db).all_ordered()

    if format == "json":
        import json
        data = [_user_export_row(u) for u in users]
        content = json.dumps(data, indent=2, ensure_ascii=False)
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=gex_users.json"},
        )
    else:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(USER_EXPORT_FIELDS)
        for u in users:
            row = _user_export_row(u)
            writer.writerow([row[f] for f in USER_EXPORT_FIELDS])
        content = output.getvalue()
        output.close()
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8-sig")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=gex_users.csv"},
        )


@router.post("/users/import")
async def import_users(
    request: Request,
    format: Optional[str] = Query(None, description="csv|json (авто, если не задан)"),
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Импортировать пользователей из CSV/JSON (upsert по id или email).

    Формат — как в /users/export. Строки без email пропускаются с ошибкой.
    password_hash восстанавливается только если это bcrypt-хэш ($2...).
    """
    records, fmt = await _read_upload(request, format)

    imported = 0      # создано новых
    updated = 0       # обновлено существующих
    skipped = 0       # пропущено (ошибка в строке)
    errors: list[str] = []

    now = datetime.now(timezone.utc)

    for i, rec in enumerate(records, start=1):
        email = (rec.get("email") or "").strip().lower()
        if not email or "@" not in email:
            errors.append(f"строка {i}: отсутствует или некорректный email — пропущено")
            skipped += 1
            continue

        try:
            # ── находим существующего: по id, затем по email ──
            with db.begin_nested():
                from gex.adapters.persistence.auth_repositories import UserRepository

                user = None
                rid = (rec.get("id") or "").strip()
                repo = UserRepository(db)
                user = repo.by_id(rid) if rid else None
                if user is None:
                    user = repo.by_email(email)

                is_new = user is None
                if is_new:
                    user = User(id=rid, email=email) if rid else User(email=email)
                    db.add(user)

                # ── скалярные поля ──
                ph = (rec.get("password_hash") or "").strip()
                if ph.startswith("$2"):
                    user.password_hash = ph
                elif is_new:
                    user.password_hash = None

                user.is_email_verified = _parse_bool(rec.get("is_email_verified"))
                user.is_blocked = _parse_bool(rec.get("is_blocked"))
                user.telegram_chat_id = (rec.get("telegram_chat_id") or "").strip() or None
                user.telegram_username = (rec.get("telegram_username") or "").strip() or None
                user.telegram_connect_token = (rec.get("telegram_connect_token") or "").strip() or None
                user.telegram_notify = _parse_bool(rec.get("telegram_notify"))
                user.telegram_test_at = _parse_iso_dt(rec.get("telegram_test_at"))

                status = (rec.get("subscription_status") or SubscriptionStatus.INACTIVE).strip().upper()
                if status not in SUBSCRIPTION_VALUES:
                    raise ValueError(f"неизвестный subscription_status: {status}")
                user.subscription_status = status

                user.subscription_activated_at = _parse_iso_dt(rec.get("subscription_activated_at"))
                user.subscription_expires_at = _parse_iso_dt(rec.get("subscription_expires_at"))
                user.subscription_expiry_notified_for = _parse_iso_dt(rec.get("subscription_expiry_notified_for"))
                user.verification_token = (rec.get("verification_token") or "").strip() or None
                user.oauth_provider = (rec.get("oauth_provider") or "").strip() or None
                user.oauth_id = (rec.get("oauth_id") or "").strip() or None

                user.created_at = _parse_iso_dt(rec.get("created_at")) or now
                user.updated_at = _parse_iso_dt(rec.get("updated_at")) or now

                if is_new:
                    imported += 1
                else:
                    updated += 1
        except ValueError as e:
            errors.append(f"строка {i} ({email}): {e} — пропущено")
            skipped += 1
            continue
        except Exception as e:  # noqa: BLE001
            errors.append(f"строка {i} ({email}): {e} — пропущено")
            skipped += 1
            continue

    db.commit()
    logger.info(
        "Admin import users: %d imported, %d updated, %d skipped (%s)",
        imported, updated, skipped, fmt,
    )
    return {
        "imported": imported,
        "updated": updated,
        "skipped": skipped,
        "errors": errors[:50],
        "format": fmt,
    }


@router.get("/users/{user_id}", response_model=AdminUserOut)
def get_user(
    user_id: str,
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Получить одного пользователя по ID."""
    u = _user_or_404(db, user_id)
    return AdminUserOut.model_validate(u)


@router.patch("/users/{user_id}", response_model=AdminUserOut)
def update_user(
    user_id: str,
    body: AdminUserUpdateIn,
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Обновить пользователя: статус подписки, даты, верификацию."""
    u = _user_or_404(db, user_id)

    # Master Admin: нельзя менять себя или другого Master Admin
    if is_master_admin(u):
        raise HTTPException(
            status_code=403,
            detail="Нельзя изменить Master Admin через админ-панель.",
        )

    # Выдача/снятие прав администратора и изменение других админов — только Master
    granting_admin = body.subscription_status == SubscriptionStatus.ADMIN
    target_is_admin = u.subscription_status == SubscriptionStatus.ADMIN
    if not _is_master(admin) and (granting_admin or target_is_admin):
        raise HTTPException(
            status_code=403,
            detail="Только Master Admin может выдавать или изменять права администратора.",
        )

    changed = False

    if body.subscription_status is not None:
        if body.subscription_status not in SUBSCRIPTION_VALUES:
            raise HTTPException(
                status_code=400,
                detail=f"Неизвестный уровень подписки: {body.subscription_status}",
            )
        u.subscription_status = body.subscription_status
        changed = True

        # Авто-установка даты активации при первом назначении подписки
        if body.subscription_status != SubscriptionStatus.INACTIVE and not u.subscription_activated_at:
            u.subscription_activated_at = datetime.now(timezone.utc)
            logger.info(
                "Auto-set subscription_activated_at for user %s → %s",
                u.email, body.subscription_status,
            )

    if body.is_email_verified is not None:
        u.is_email_verified = body.is_email_verified
        changed = True

    if body.is_blocked is not None:
        u.is_blocked = body.is_blocked
        changed = True

    if body.subscription_activated_at is not None:
        if body.subscription_activated_at == "":
            u.subscription_activated_at = None
        else:
            try:
                u.subscription_activated_at = datetime.fromisoformat(
                    body.subscription_activated_at.replace("Z", "+00:00")
                )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Неверный формат даты для subscription_activated_at.",
                )
        changed = True

    if body.subscription_expires_at is not None:
        if body.subscription_expires_at == "":
            u.subscription_expires_at = None
        else:
            try:
                u.subscription_expires_at = datetime.fromisoformat(
                    body.subscription_expires_at.replace("Z", "+00:00")
                )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Неверный формат даты для subscription_expires_at.",
                )
        changed = True

    if changed:
        u.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(u)
        logger.info("Admin updated user %s: %s", u.email, body.model_dump(exclude_none=True))

    return AdminUserOut.model_validate(u)


@router.delete("/users/{user_id}", response_model=MessageOut)
def delete_user(
    user_id: str,
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Удалить пользователя (нельзя удалить самого Master Admin)."""
    u = _user_or_404(db, user_id)

    if is_master_admin(u):
        raise HTTPException(
            status_code=403,
            detail="Нельзя удалить Master Admin.",
        )

    email = u.email
    db.delete(u)
    db.commit()
    logger.info("Admin deleted user: %s (id=%s)", email, user_id)
    return MessageOut(message=f"Пользователь {email} удалён.")


@router.post("/users/{user_id}/moderate", response_model=AdminUserOut)
def moderate_user(
    user_id: str,
    action: str = Query(..., pattern="^(verify|unverify|activate|deactivate)$"),
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Быстрые модераторские действия."""
    u = _user_or_404(db, user_id)

    if is_master_admin(u):
        raise HTTPException(
            status_code=403,
            detail="Нельзя модерировать Master Admin.",
        )

    if action == "verify":
        u.is_email_verified = True
        msg = f"Email {u.email} подтверждён"
    elif action == "unverify":
        u.is_email_verified = False
        msg = f"Email {u.email} отозван"
    elif action == "activate":
        u.subscription_status = SubscriptionStatus.BASIC
        if not u.subscription_activated_at:
            u.subscription_activated_at = datetime.now(timezone.utc)
        msg = f"Подписка {u.email} активирована (BASIC)"
    elif action == "deactivate":
        u.subscription_status = SubscriptionStatus.INACTIVE
        msg = f"Подписка {u.email} деактивирована"

    u.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(u)
    logger.info("Admin moderated user: %s", msg)
    return AdminUserOut.model_validate(u)


@router.post("/users/{user_id}/resend-email", response_model=MessageOut)
def admin_resend_email(
    user_id: str,
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Переотправить письмо подтверждения email от имени админа."""
    u = _user_or_404(db, user_id)
    if is_master_admin(u):
        raise HTTPException(status_code=403, detail="Master Admin не требует подтверждения.")
    if u.is_email_verified:
        raise HTTPException(status_code=400, detail="Email уже подтверждён.")
    link = _resend_email_verification(u, db)
    return MessageOut(
        message=f"Письмо отправлено на {u.email}.",
        detail=link,
    )


@router.post("/users/{user_id}/resend-telegram", response_model=MessageOut)
def admin_resend_telegram(
    user_id: str,
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Переотправить ссылку активации в Telegram от имени админа."""
    u = _user_or_404(db, user_id)
    if is_master_admin(u):
        raise HTTPException(status_code=403, detail="Master Admin не требует подтверждения.")
    if u.is_email_verified:
        raise HTTPException(status_code=400, detail="Аккаунт уже активирован.")
    if not u.telegram_username:
        raise HTTPException(status_code=400, detail="У пользователя не указан Telegram-ник.")
    link = _resend_telegram_activation(u, db)
    return MessageOut(
        message=f"Ссылка активации отправлена в Telegram ({u.telegram_username}).",
        detail=link,
    )


@router.post("/users/{user_id}/activate", response_model=AdminUserOut)
def admin_activate_subscription(
    user_id: str,
    body: AdminActivateIn,
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Активировать/продлить подписку пользователя (выбор плана + срок)."""
    u = _user_or_404(db, user_id)
    if is_master_admin(u):
        raise HTTPException(
            status_code=403,
            detail="Нельзя менять подписку Master Admin.",
        )

    now = datetime.now(timezone.utc)
    current_expires = u.subscription_expires_at
    base = max(now, current_expires) if current_expires else now
    if isinstance(base, datetime) and base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)

    u.subscription_status = body.plan
    u.subscription_activated_at = u.subscription_activated_at or now
    u.subscription_expires_at = base + timedelta(days=body.days)
    u.updated_at = now
    db.commit()
    db.refresh(u)
    logger.info(
        "Admin activated subscription %s -> %s (days=%d, expires=%s)",
        u.email, body.plan, body.days, u.subscription_expires_at.isoformat(),
    )
    return AdminUserOut.model_validate(u)


@router.post("/users/{user_id}/block", response_model=AdminUserOut)
def admin_block_user(
    user_id: str,
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Заблокировать пользователя (теряет доступ к API/логину)."""
    u = _user_or_404(db, user_id)
    if is_master_admin(u):
        raise HTTPException(status_code=403, detail="Нельзя заблокировать Master Admin.")
    u.is_blocked = True
    u.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(u)
    logger.info("Admin blocked user: %s", u.email)
    return AdminUserOut.model_validate(u)


@router.post("/users/{user_id}/unblock", response_model=AdminUserOut)
def admin_unblock_user(
    user_id: str,
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Разблокировать пользователя."""
    u = _user_or_404(db, user_id)
    if is_master_admin(u):
        raise HTTPException(status_code=403, detail="Нельзя разблокировать Master Admin.")
    if not _is_master(admin) and u.subscription_status == SubscriptionStatus.ADMIN:
        raise HTTPException(status_code=403, detail="Только Master Admin управляет администраторами.")
    u.is_blocked = False
    u.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(u)
    logger.info("Admin unblocked user: %s", u.email)
    return AdminUserOut.model_validate(u)
