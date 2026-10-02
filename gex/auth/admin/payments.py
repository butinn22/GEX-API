"""Платежи: список, импорт/выгрузка, подтверждение и отклонение, реквизиты.

Модерация платежей — то, что админ делает руками; рядом реквизиты, которые она же
и настраивает.

Вынесено из ``admin_router.py`` (итерация 42). Обработчики перенесены дословно;
подроутер подключает фасад, поэтому пути не менялись — проверяет
``tests/test_admin_routes.py``.
"""
from __future__ import annotations

from ..models import (User, SubscriptionStatus, SUBSCRIPTION_VALUES)
from ..payment_schemas import (PaymentsPageOut, PaymentSettingsIn, PaymentSettingsOut)
from ..schemas import (AdminActivateIn, AdminStatsOut, AdminUserOut, AdminUserUpdateIn, AdminUsersPageOut, EmailConfigIn, FinAgentKeyIn, MessageOut, TelegramConfigIn)
from datetime import (datetime, timedelta, timezone)
from fastapi import (APIRouter, Depends, HTTPException, Query, Request, status)
from fastapi.responses import (StreamingResponse)
from gex.adapters.persistence.database import (get_session)
from sqlalchemy.orm import (Session)
from typing import (Optional)
import csv
import io

from ._shared import (
    PAYMENT_EXPORT_FIELDS,
    _parse_float,
    _parse_iso_dt,
    _payment_export_row,
    _payment_requisites_state,
    _read_upload,
    _require_admin,
    logger,
)


router = APIRouter()


@router.get("/payments/export")
def export_payments(
    format: str = Query("csv", description="Формат: csv или json"),
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Выгрузить все платежи в CSV или JSON (все поля, для полного восстановления)."""

    from gex.adapters.persistence.auth_repositories import PaymentRepository

    payments = PaymentRepository(db).all_ordered()

    if format == "json":
        import json
        data = [_payment_export_row(p) for p in payments]
        content = json.dumps(data, indent=2, ensure_ascii=False)
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=gex_payments.json"},
        )
    else:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(PAYMENT_EXPORT_FIELDS)
        for p in payments:
            row = _payment_export_row(p)
            writer.writerow([row[f] for f in PAYMENT_EXPORT_FIELDS])
        content = output.getvalue()
        output.close()
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8-sig")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=gex_payments.csv"},
        )


@router.post("/payments/import")
async def import_payments(
    request: Request,
    format: Optional[str] = Query(None, description="csv|json (авто, если не задан)"),
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Импортировать платежи из CSV/JSON (upsert по id).

    Формат — как в /payments/export. user_id должен ссылаться на существующего
    пользователя (импортируйте пользователей первыми). Если user_id пуст,
    ищется пользователь по user_email.
    """
    from ..payment_models import Payment, PaymentMethod, PaymentStatus

    records, fmt = await _read_upload(request, format)

    imported = 0
    updated = 0
    skipped = 0
    errors: list[str] = []

    valid_statuses = {
        PaymentStatus.PENDING, PaymentStatus.PAID_CLIENT,
        PaymentStatus.CONFIRMED, PaymentStatus.REJECTED, PaymentStatus.EXPIRED,
    }
    valid_methods = {PaymentMethod.SBP, PaymentMethod.CRYPTO, PaymentMethod.BANK}

    for i, rec in enumerate(records, start=1):
        try:
            with db.begin_nested():
                from gex.adapters.persistence.auth_repositories import PaymentRepository, UserRepository

                rid = (rec.get("id") or "").strip()
                payment = PaymentRepository(db).by_id(rid) if rid else None
                is_new = payment is None
                if is_new:
                    payment = Payment(id=rid) if rid else Payment()
                    db.add(payment)

                # ── user_id: явный, либо по user_email ──
                uid = (rec.get("user_id") or "").strip()
                if not uid:
                    uemail = (rec.get("user_email") or "").strip().lower()
                    existing = UserRepository(db).by_email(uemail) if uemail else None
                    if existing:
                        uid = existing.id
                if not uid:
                    raise ValueError("нет user_id и не найден пользователь по user_email")
                user = UserRepository(db).by_id(uid)
                if user is None:
                    raise ValueError(f"пользователь {uid} не найден (импортируйте пользователей первыми)")
                payment.user_id = uid
                payment.user_email = (rec.get("user_email") or user.email).strip()

                plan = (rec.get("plan") or "BASIC").strip().upper()
                if plan not in ("BASIC", "EXTENDED"):
                    raise ValueError(f"неизвестный plan: {plan}")
                payment.plan = plan

                amount_rub = _parse_float(rec.get("amount_rub"))
                amount_usd = _parse_float(rec.get("amount_usd"))
                if amount_rub is None or amount_usd is None:
                    raise ValueError("amount_rub и amount_usd обязательны")
                payment.amount_rub = amount_rub
                payment.amount_usd = amount_usd

                method = (rec.get("method") or "SBP").strip().upper()
                if method not in valid_methods:
                    raise ValueError(f"неизвестный method: {method}")
                payment.method = method

                payment.crypto_currency = (rec.get("crypto_currency") or "").strip() or None
                payment.crypto_amount = _parse_float(rec.get("crypto_amount"))

                status = (rec.get("status") or PaymentStatus.PENDING).strip().upper()
                if status not in valid_statuses:
                    raise ValueError(f"неизвестный status: {status}")
                payment.status = status

                payment.admin_note = (rec.get("admin_note") or "").strip() or None
                now = datetime.now(timezone.utc)
                payment.created_at = _parse_iso_dt(rec.get("created_at")) or now
                payment.confirmed_at = _parse_iso_dt(rec.get("confirmed_at"))
                payment.expires_at = _parse_iso_dt(rec.get("expires_at")) or (now + timedelta(hours=24))

                if is_new:
                    imported += 1
                else:
                    updated += 1
        except ValueError as e:
            errors.append(f"строка {i}: {e} — пропущено")
            skipped += 1
            continue
        except Exception as e:  # noqa: BLE001
            errors.append(f"строка {i}: {e} — пропущено")
            skipped += 1
            continue

    db.commit()
    logger.info(
        "Admin import payments: %d imported, %d updated, %d skipped (%s)",
        imported, updated, skipped, fmt,
    )
    return {
        "imported": imported,
        "updated": updated,
        "skipped": skipped,
        "errors": errors[:50],
        "format": fmt,
    }


@router.get("/payments", response_model=PaymentsPageOut)
def list_payments(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None),
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Список всех платежей."""
    from gex.adapters.persistence.auth_repositories import PaymentRepository
    from ..payment_service import PaymentService

    offset = (page - 1) * page_size
    payments, total = PaymentRepository(db).page(
        status=status_filter.upper() if status_filter else None,
        offset=offset,
        limit=page_size,
    )
    total_pages = max(1, (total + page_size - 1) // page_size)

    return PaymentsPageOut(
        payments=[PaymentService.to_out(p) for p in payments],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.post("/payments/{payment_id}/confirm", response_model=MessageOut)
def admin_confirm_payment(
    payment_id: str,
    note: Optional[str] = Query(None),
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Админ подтверждает платёж → активация подписки."""
    from ..payment_models import Payment
    from ..payment_service import PaymentService

    svc = PaymentService(db)
    p = svc.get_payment(payment_id)
    if not p:
        raise HTTPException(status_code=404, detail="Платёж не найден")
    try:
        svc.confirm_by_admin(p, admin, note)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return MessageOut(message=f"Платёж {p.user_email} подтверждён. Подписка {p.plan} активирована.")


@router.post("/payments/{payment_id}/reject", response_model=MessageOut)
def admin_reject_payment(
    payment_id: str,
    note: Optional[str] = Query(None),
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Админ отклоняет платёж."""
    from ..payment_models import Payment
    from ..payment_service import PaymentService

    svc = PaymentService(db)
    p = svc.get_payment(payment_id)
    if not p:
        raise HTTPException(status_code=404, detail="Платёж не найден")
    try:
        svc.reject_by_admin(p, admin, note)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return MessageOut(message=f"Платёж {p.user_email} отклонён.")


@router.get("/payment-requisites", response_model=PaymentSettingsOut)
def admin_get_payment_requisites(
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Текущие платёжные реквизиты (видны клиенту на странице оплаты)."""
    return _payment_requisites_state(db)


@router.put("/payment-requisites", response_model=PaymentSettingsOut)
def admin_set_payment_requisites(
    body: PaymentSettingsIn,
    admin: User = Depends(_require_admin),
    db: Session = Depends(get_session),
):
    """Обновить платёжные реквизиты (None = не менять, \"\" = очистить)."""
    from ..payment_service import get_requisites_row

    row = get_requisites_row(db)
    fields = (
        "sbp_phone", "sbp_bank", "sbp_name",
        "crypto_usdt_trc20", "crypto_usdt_bep20",
        "bank_name", "bank_bic", "bank_account", "bank_recipient",
    )
    changed = False
    for f in fields:
        val = getattr(body, f)
        if val is not None:
            setattr(row, f, val.strip())
            changed = True
    if changed:
        from datetime import datetime, timezone
        row.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(row)
        logger.info("payment-requisites обновлены админом %s", admin.email)
    return _payment_requisites_state(db)
