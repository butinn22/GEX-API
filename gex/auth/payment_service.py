"""Payment service: создание, проверка, подтверждение платежей."""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from .config import settings
from .models import SubscriptionStatus, User
from .payment_models import Payment, PaymentMethod, PaymentSettings, PaymentStatus
from .payment_schemas import PaymentOut

logger = logging.getLogger(__name__)

# Курс USDT к USD (упрощённо 1:1)
USDT_RATE = 1.0

# ── Платёжные реквизиты ─────────────────────────────────────────────
# Ключи БД-строки payment_settings → атрибуты settings (.env), которыми
# строка сидится при ПЕРВОМ обращении. Дальше источник истины — БД
# (правки из админки переживают пересоздание контейнеров).
PAYMENT_SETTINGS_FIELDS: dict[str, str] = {
    "sbp_phone": "SBP_PHONE",
    "sbp_bank": "SBP_BANK",
    "sbp_name": "SBP_NAME",
    "crypto_usdt_trc20": "CRYPTO_USDT_TRC20",
    "crypto_usdt_bep20": "CRYPTO_USDT_BEP20",
    "bank_name": "BANK_NAME",
    "bank_bic": "BANK_BIC",
    "bank_account": "BANK_ACCOUNT",
    "bank_recipient": "BANK_RECIPIENT",
}

# Поля без env-аналога (настраиваются только из админки)
PAYMENT_SETTINGS_DB_ONLY: tuple[str, ...] = ()


def _clean_requisite(value: Optional[str]) -> str:
    """Нормализация значения реквизита.

    Значения-плейсхолдеры (содержат '*', напр. дефолт SBP_PHONE
    "+791****1132") и 'none' считаются пустыми — иначе на страницу
    оплаты утекут фейковые реквизиты.
    """
    v = (value or "").strip()
    if not v or "*" in v or v.lower() in ("none", "n/a", "change-me"):
        return ""
    return v


def get_requisites_row(db: Session) -> PaymentSettings:
    """Строка реквизитов (id=1); при отсутствии — сид из settings/.env."""
    row = db.query(PaymentSettings).filter(PaymentSettings.id == 1).first()
    if row is None:
        row = PaymentSettings(id=1)
        for key, env_attr in PAYMENT_SETTINGS_FIELDS.items():
            setattr(row, key, _clean_requisite(getattr(settings, env_attr, "")))
        db.add(row)
        db.commit()
        db.refresh(row)
        logger.info("payment_settings: строка создана (сид из .env)")
    return row


def requisites_map(row: PaymentSettings) -> dict[str, str]:
    """Эффективные реквизиты: {ключ: значение} (плейсхолдеры → '')."""
    return {
        key: _clean_requisite(getattr(row, key))
        for key in list(PAYMENT_SETTINGS_FIELDS) + list(PAYMENT_SETTINGS_DB_ONLY)
    }


def available_payment_methods(req: dict[str, str]) -> list[dict]:
    """Способы оплаты, доступные клиенту (реквизиты настроены)."""
    methods: list[dict] = []
    if req.get("sbp_phone"):
        methods.append({"id": PaymentMethod.SBP, "networks": []})
    networks = []
    if req.get("crypto_usdt_trc20"):
        networks.append("USDT_TRC20")
    if req.get("crypto_usdt_bep20"):
        networks.append("USDT_BEP20")
    if networks:
        methods.append({"id": PaymentMethod.CRYPTO, "networks": networks})
    if req.get("bank_account") and req.get("bank_name"):
        methods.append({"id": PaymentMethod.BANK, "networks": []})
    return methods

def _addr_key(network: str) -> str:
    """Ключ реквизитов БД для сети USDT."""
    return {"USDT_TRC20": "crypto_usdt_trc20", "USDT_BEP20": "crypto_usdt_bep20"}.get(network, "")


# ── Курс USD/RUB по данным ЦБ РФ (на дату оплаты) ──────────────────────
# Источник: https://www.cbr-xml-daily.ru/daily_json.js (официальный курс ЦБ).
# Обновляется раз в сутки → кэшируем на 6 часов; при недоступности ЦБ
# используется fallback settings.USD_TO_RUB_RATE.
_CBR_CACHE: dict = {"ts": 0.0, "rate": None}
_CBR_TTL_SECONDS = 6 * 3600


def _get_usd_rub_rate() -> float:
    """Курс USD/RUB на дату оплаты по данным ЦБ РФ."""
    now = time.time()
    cached = _CBR_CACHE.get("rate")
    if cached is not None and (now - _CBR_CACHE.get("ts", 0.0)) < _CBR_TTL_SECONDS:
        return cached
    rate = settings.USD_TO_RUB_RATE
    try:
        import requests
        resp = requests.get(
            "https://www.cbr-xml-daily.ru/daily_json.js", timeout=8
        )
        data = resp.json()
        value = float((data.get("Valute") or {}).get("USD", {}).get("Value") or 0)
        if value > 0:
            rate = value
    except Exception as exc:  # noqa: BLE001 — ЦБ недоступен: fallback
        logger.warning("CBR rate unavailable, fallback %.2f: %s", settings.USD_TO_RUB_RATE, exc)
    _CBR_CACHE["ts"] = now
    _CBR_CACHE["rate"] = rate
    return rate


class PaymentService:
    """Сервис обработки платежей."""

    def __init__(self, db: Session):
        self.db = db

    # ── Планы ────────────────────────────────────────────────────
    @staticmethod
    def get_plans() -> list[dict]:
        usd_rub = _get_usd_rub_rate()
        return [
            {
                "id": "BASIC",
                "label": "Basic",
                "price_usd": settings.BASIC_PRICE_USD,
                "price_rub": round(settings.BASIC_PRICE_USD * usd_rub, 2),
                "features": [
                    "Персональный дашборд: любые акции, индексы и крипта, EMA-оверлеи и свои трендовые линии",
                    "Технический анализ по 4 таймфреймам (1h/2h/4h/1D): EMA, RSI, MACD, направление и сила тренда, вероятность разворота",
                    "Трендовые линии: уровни поддержки и сопротивления по экстремумам цены, консенсус по 4 таймфреймам",
                    "MACD-тренд: геометрия усреднённой линии, угол наклона, вероятность продолжения движения",
                    "Конус волатильности: вероятный квартальный диапазон цены — зоны 1σ/2σ, VWAP, Bollinger, Mean Reversion",
                    "Novel Candles: гибридные свечи (стандарт + Heikin-Ashi) с фильтрацией шума",
                    "Товарная динамика: равновзвешенный композит 9 товаров (GOLD, UKOIL, SILVER, …)",
                    "Рыночная ширина: McClellan-осциллятор и индекс коррекционного давления",
                    "Широта секторов: композит 12 секторов США (XLK–RSP)",
                ],
            },
            {
                "id": "EXTENDED",
                "label": "Extended",
                "price_usd": settings.EXTENDED_PRICE_USD,
                "price_rub": round(settings.EXTENDED_PRICE_USD * usd_rub, 2),
                "features": [
                    "Всё из тарифа Basic",
                    "GEX-анализ опционных рынков: Net GEX, Call/Put Wall, Zero Gamma, Power Zones, Price Band — 5 источников (US live, загруженные цепочки, MOEX, VIX/VVIX, крипта Bybit)",
                    "GEX-конус вероятностей: конус 1σ/2σ/3σ из OI-взвешенной волатильности, GEX-стены со ступеньками вероятностей",
                    "Сигнальный сканер: до 10 инструментов, фоновый опрос каждые 5 минут, уведомления в Telegram",
                    "Авто-сканер сигналов: анализ всего списка тикеров (~190: US, MOEX, крипта) на 4H и 1D с rate-limit",
                    "FinAgent AI: структурный анализ SMC + GEX + тренд и AI-заключение по направлению движения",
                ],
            },
        ]

    # ── Создание платежа ─────────────────────────────────────────
    def init_payment(
        self,
        user: User,
        plan: str,
        method: str,
        crypto_currency: Optional[str] = None,
        requisites: Optional[dict[str, str]] = None,
    ) -> Payment:
        if plan not in ("BASIC", "EXTENDED"):
            raise ValueError(f"Неизвестный план: {plan}")
        if method not in (PaymentMethod.SBP, PaymentMethod.CRYPTO, PaymentMethod.BANK):
            raise ValueError(f"Неизвестный метод: {method}")

        # Реквизиты: из БД (payment_settings) либо из .env (пока таблица
        # не тронута); без настроенных реквизитов метод недоступен
        req = requisites if requisites is not None else requisites_map(get_requisites_row(self.db))

        usd_rub = _get_usd_rub_rate()
        price_usd = (
            settings.BASIC_PRICE_USD if plan == "BASIC"
            else settings.EXTENDED_PRICE_USD
        )
        price_rub = round(price_usd * usd_rub, 2)

        crypto_amount = None
        if method == PaymentMethod.CRYPTO:
            networks = [
                n for n in ("USDT_TRC20", "USDT_BEP20") if req.get(_addr_key(n))
            ]
            if not networks:
                raise ValueError("Крипто-адреса не настроены — добавьте адрес в админке (Платежи → Реквизиты)")
            if not crypto_currency:
                crypto_currency = "USDT_TRC20" if "USDT_TRC20" in networks else "USDT_BEP20"
            if crypto_currency not in networks:
                raise ValueError(f"Адрес {crypto_currency} не настроен — доступны: {', '.join(networks)}")
            crypto_amount = round(price_usd * USDT_RATE, 2)

        if method == PaymentMethod.SBP and not req.get("sbp_phone"):
            raise ValueError("СБП не настроено — добавьте телефон в админке (Платежи → Реквизиты)")
        if method == PaymentMethod.BANK and not (
            req.get("bank_account") and req.get("bank_name")
        ):
            raise ValueError("Банковский счёт не настроен — добавьте реквизиты в админке (Платежи → Реквизиты)")

        payment = Payment(
            id=str(uuid.uuid4()),
            user_id=user.id,
            user_email=user.email,
            plan=plan,
            amount_rub=price_rub,
            amount_usd=price_usd,
            method=method,
            crypto_currency=crypto_currency if method == PaymentMethod.CRYPTO else None,
            crypto_amount=crypto_amount,
            status=PaymentStatus.PENDING,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )
        self.db.add(payment)
        self.db.commit()
        self.db.refresh(payment)
        logger.info(
            "Payment created: %s, plan=%s, method=%s, amount=%.2f USD",
            payment.id[:8], plan, method, price_usd,
        )
        return payment

    # ── Клиент отметил «оплатил» ─────────────────────────────────
    def confirm_by_client(self, payment: Payment, note: Optional[str] = None) -> Payment:
        if payment.status != PaymentStatus.PENDING:
            raise ValueError(f"Невозможно подтвердить платёж в статусе {payment.status}")
        payment.status = PaymentStatus.PAID_CLIENT
        if note:
            payment.admin_note = (payment.admin_note or "") + f"\n[Клиент]: {note}"
        self.db.commit()
        self.db.refresh(payment)
        logger.info("Payment %s marked as PAID_CLIENT by user", payment.id[:8])
        return payment

    # ── Админ подтвердил ─────────────────────────────────────────
    def confirm_by_admin(
        self, payment: Payment, admin: User, note: Optional[str] = None
    ) -> Payment:
        if payment.status not in (PaymentStatus.PAID_CLIENT, PaymentStatus.PENDING):
            raise ValueError(f"Невозможно подтвердить платёж в статусе {payment.status}")

        payment.status = PaymentStatus.CONFIRMED
        payment.confirmed_at = datetime.now(timezone.utc)
        if note:
            payment.admin_note = (payment.admin_note or "") + f"\n[Админ]: {note}"

        # Активировать подписку пользователю
        self._activate_subscription(payment)

        self.db.commit()
        logger.info(
            "Payment %s CONFIRMED by admin %s", payment.id[:8], admin.email,
        )
        return payment

    # ── Админ отклонил ───────────────────────────────────────────
    def reject_by_admin(
        self, payment: Payment, admin: User, note: Optional[str] = None
    ) -> Payment:
        if payment.status not in (PaymentStatus.PAID_CLIENT, PaymentStatus.PENDING):
            raise ValueError(f"Невозможно отклонить платёж в статусе {payment.status}")
        payment.status = PaymentStatus.REJECTED
        if note:
            payment.admin_note = (payment.admin_note or "") + f"\n[Админ]: {note}"
        self.db.commit()
        self.db.refresh(payment)
        logger.info("Payment %s REJECTED by admin %s", payment.id[:8], admin.email)
        return payment

    # ── Активация подписки ───────────────────────────────────────
    def _activate_subscription(self, payment: Payment) -> None:
        """Активировать/продлить подписку по подтверждённому платежу.

        Продление не теряет оставшиеся дни: новый срок отсчитывается
        от максимума (сейчас, текущий expires_at). Коммит выполняет
        вызывающий метод (один коммит на операцию).
        """
        user = self.db.query(User).filter(User.id == payment.user_id).first()
        if not user:
            logger.warning("User %s not found for payment %s", payment.user_id, payment.id)
            return

        now = datetime.now(timezone.utc)
        current_expires = user.subscription_expires_at
        # Если подписка ещё активна — продлеваем от её конца, иначе от сейчас
        base = max(now, current_expires) if current_expires else now
        if isinstance(base, datetime) and base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)

        user.subscription_status = payment.plan
        user.subscription_activated_at = user.subscription_activated_at or now
        user.subscription_expires_at = base + timedelta(days=30)
        logger.info(
            "Subscription activated: %s -> %s (expires %s)",
            user.email, payment.plan,
            user.subscription_expires_at.isoformat(),
        )

    # ── Получить платёж ──────────────────────────────────────────
    def get_payment(self, payment_id: str) -> Optional[Payment]:
        return self.db.query(Payment).filter(Payment.id == payment_id).first()

    # ── Платежи пользователя ─────────────────────────────────────
    def get_user_payments(self, user: User) -> list[Payment]:
        return (
            self.db.query(Payment)
            .filter(Payment.user_id == user.id)
            .order_by(Payment.created_at.desc())
            .all()
        )

    # ── Обогатить платёж платёжными реквизитами ──────────────────
    @staticmethod
    def to_out(payment: Payment, requisites: Optional[dict[str, str]] = None) -> PaymentOut:
        """Сериализация платежа в Pydantic-схему + реквизиты из настроек.

        Реквизиты берутся из строки payment_settings (БД, правится
        админкой); None — фолбэк на settings/.env (до первого чтения БД).
        """
        out = PaymentOut.model_validate(payment)
        if requisites is None:
            from .config import settings as _s
            requisites = {
                "sbp_phone": _s.SBP_PHONE or "",
                "sbp_bank": _s.SBP_BANK or "",
                "sbp_name": _s.SBP_NAME or "",
                "crypto_usdt_trc20": _s.CRYPTO_USDT_TRC20 or "",
                "crypto_usdt_bep20": _s.CRYPTO_USDT_BEP20 or "",
                "bank_name": _s.BANK_NAME or "",
                "bank_bic": _s.BANK_BIC or "",
                "bank_account": _s.BANK_ACCOUNT or "",
                "bank_recipient": _s.BANK_RECIPIENT or "",
            }
        if payment.method == PaymentMethod.SBP:
            out.sbp_phone = _clean_requisite(requisites.get("sbp_phone")) or None
            out.sbp_bank = _clean_requisite(requisites.get("sbp_bank")) or None
            out.sbp_name = _clean_requisite(requisites.get("sbp_name")) or None
        elif payment.method == PaymentMethod.CRYPTO:
            if payment.crypto_currency:
                out.crypto_address = (
                    _clean_requisite(requisites.get(_addr_key(payment.crypto_currency))) or None
                )
        elif payment.method == PaymentMethod.BANK:
            out.bank_name = _clean_requisite(requisites.get("bank_name")) or None
            out.bank_bic = _clean_requisite(requisites.get("bank_bic")) or None
            out.bank_account = _clean_requisite(requisites.get("bank_account")) or None
            out.bank_recipient = _clean_requisite(requisites.get("bank_recipient")) or None
        return out
