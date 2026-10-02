"""Payment router: создание и проверка платежей (СБП / крипта / банк)."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from gex.adapters.persistence.database import get_session
from .dependencies import get_current_user
from .models import User
from .payment_schemas import (
    PaymentConfirmIn,
    PaymentInitIn,
    PaymentOut,
    PlansListOut,
)
from .payment_service import (
    PaymentService,
    available_payment_methods,
    get_requisites_row,
    requisites_map,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payment", tags=["payment"])


# ================================================================= #
#  GET /payment/plans
# ================================================================= #
@router.get("/plans", response_model=PlansListOut)
def get_plans(db: Session = Depends(get_session)):
    """Список тарифных планов с ценами + доступные способы оплаты."""
    req = requisites_map(get_requisites_row(db))
    return PlansListOut(
        plans=PaymentService.get_plans(),
        methods=available_payment_methods(req),
    )


# ================================================================= #
#  POST /payment/init
# ================================================================= #
@router.post("/init", response_model=PaymentOut, status_code=status.HTTP_201_CREATED)
def init_payment(
    body: PaymentInitIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Создать платёж на оплату подписки."""
    # Акцепт условий платформы обязателен для создания платежа
    if not body.accept_terms:
        raise HTTPException(
            status_code=400,
            detail="Для оплаты необходимо принять условия платформы (платёж не подлежит обмену и возврату).",
        )

    svc = PaymentService(db)
    try:
        payment = svc.init_payment(
            user=user,
            plan=body.plan,
            method=body.method,
            crypto_currency=body.crypto_currency,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    req = requisites_map(get_requisites_row(db))
    return PaymentService.to_out(payment, req)


# ================================================================= #
#  GET /payment/my
# ================================================================= #
@router.get("/my", response_model=list[PaymentOut])
def get_my_payments(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """История платежей текущего пользователя."""
    svc = PaymentService(db)
    payments = svc.get_user_payments(user)
    req = requisites_map(get_requisites_row(db))
    return [PaymentService.to_out(p, req) for p in payments]


# ================================================================= #
#  GET /payment/{payment_id}
# ================================================================= #
@router.get("/{payment_id}", response_model=PaymentOut)
def get_payment(
    payment_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Получить статус конкретного платежа."""
    svc = PaymentService(db)
    payment = svc.get_payment(payment_id)
    if not payment:
        raise HTTPException(status_code=404, detail="Платёж не найден")
    if payment.user_id != user.id:
        raise HTTPException(status_code=403, detail="Чужой платёж")
    req = requisites_map(get_requisites_row(db))
    return PaymentService.to_out(payment, req)


# ================================================================= #
#  POST /payment/{payment_id}/confirm-client
# ================================================================= #
@router.post("/{payment_id}/confirm-client", response_model=PaymentOut)
def confirm_by_client(
    payment_id: str,
    body: PaymentConfirmIn = PaymentConfirmIn(),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Клиент нажал «Я оплатил» — платёж ждёт проверки админом."""
    svc = PaymentService(db)
    payment = svc.get_payment(payment_id)
    if not payment:
        raise HTTPException(status_code=404, detail="Платёж не найден")
    if payment.user_id != user.id:
        raise HTTPException(status_code=403, detail="Чужой платёж")
    try:
        payment = svc.confirm_by_client(payment, body.note)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    req = requisites_map(get_requisites_row(db))
    return PaymentService.to_out(payment, req)
