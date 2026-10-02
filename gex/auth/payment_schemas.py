"""Payment schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PlanOut(BaseModel):
    """Доступный тарифный план."""
    id: str
    label: str
    price_usd: float
    price_rub: float
    features: list[str]


class PlansListOut(BaseModel):
    """Список тарифов + доступные способы оплаты."""
    plans: list[PlanOut]
    methods: list[MethodOut] = []


class PaymentInitIn(BaseModel):
    """Запрос на создание платежа."""
    plan: str = Field(..., pattern="^(BASIC|EXTENDED)$")
    method: str = Field(..., pattern="^(SBP|CRYPTO|BANK)$")
    crypto_currency: Optional[str] = Field(
        None, pattern="^(USDT_TRC20|USDT_BEP20)$"
    )
    accept_terms: bool = Field(
        False,
        description="Акцепт условий платформы (вкл. безвозвратность платежа). Обязательно.",
    )


class MethodOut(BaseModel):
    """Способ оплаты, доступный клиенту на странице оплаты.

    Реквизиты не отдаются публично до создания платежа (кроме факта
    доступности метода и сетей крипты); полные реквизиты приходят
    в PaymentOut после /payment/init.
    """

    id: str
    # Для CRYPTO: сети, по которым настроен адрес (USDT_TRC20/USDT_BEP20)
    networks: list[str] = []


class PaymentSettingsOut(BaseModel):
    """Платёжные реквизиты сервиса (для админки).

    Все поля — строки; пустая строка = реквизит не настроен.
    """

    sbp_phone: str = ""
    sbp_bank: str = ""
    sbp_name: str = ""
    crypto_usdt_trc20: str = ""
    crypto_usdt_bep20: str = ""
    bank_name: str = ""
    bank_bic: str = ""
    bank_account: str = ""
    bank_recipient: str = ""
    updated_at: Optional[datetime] = None
    # Способы, которые сейчас видны клиенту на странице оплаты
    methods: list[MethodOut] = []


class PaymentSettingsIn(BaseModel):
    """PUT /auth/admin/payment-requisites: полная замена реквизитов.

    None-поле = не менять; пустая строка = очистить (метод скроется).
    """

    sbp_phone: Optional[str] = None
    sbp_bank: Optional[str] = None
    sbp_name: Optional[str] = None
    crypto_usdt_trc20: Optional[str] = None
    crypto_usdt_bep20: Optional[str] = None
    bank_name: Optional[str] = None
    bank_bic: Optional[str] = None
    bank_account: Optional[str] = None
    bank_recipient: Optional[str] = None


class PaymentOut(BaseModel):
    """Платёж."""
    id: str
    plan: str
    amount_rub: float
    amount_usd: float
    method: str
    crypto_currency: Optional[str] = None
    crypto_amount: Optional[float] = None
    status: str
    created_at: datetime
    expires_at: datetime
    confirmed_at: Optional[datetime] = None
    admin_note: Optional[str] = None

    # Payment details for client
    sbp_phone: Optional[str] = None
    sbp_bank: Optional[str] = None
    sbp_name: Optional[str] = None
    crypto_address: Optional[str] = None
    # Банковский перевод (метод BANK, реквизиты в ₽)
    bank_name: Optional[str] = None
    bank_bic: Optional[str] = None
    bank_account: Optional[str] = None
    bank_recipient: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class PaymentConfirmIn(BaseModel):
    """Клиент нажал «Я оплатил»."""
    note: Optional[str] = None


class AdminPaymentActionIn(BaseModel):
    """Админ подтверждает/отклоняет."""
    note: Optional[str] = None


class PaymentsPageOut(BaseModel):
    """Страница платежей."""
    payments: list[PaymentOut]
    total: int
    page: int
    page_size: int
    total_pages: int
