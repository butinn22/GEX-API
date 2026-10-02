"""Настройки уведомлений: SMTP, Telegram, ключ finagent и пробное письмо.

Конфигурация внешних каналов: чтение, запись и проверка «доходит ли». Рядом с
пользователями это держать смысла нет — каналы живут отдельно от аккаунтов.

Вынесено из ``admin_router.py`` (итерация 42). Обработчики перенесены дословно;
подроутер подключает фасад, поэтому пути не менялись — проверяет
``tests/test_admin_routes.py``.
"""
from __future__ import annotations

from ..models import (User, SubscriptionStatus, SUBSCRIPTION_VALUES)
from ..schemas import (AdminActivateIn, AdminStatsOut, AdminUserOut, AdminUserUpdateIn, AdminUsersPageOut, EmailConfigIn, FinAgentKeyIn, MessageOut, TelegramConfigIn)
from fastapi import (APIRouter, Depends, HTTPException, Query, Request, status)

from ._shared import (
    _require_admin,
)


router = APIRouter()


@router.get("/email-config")
def get_email_config(admin: User = Depends(_require_admin)):
    """Текущая конфигурация корпоративной почты (секреты замаскированы)."""
    from ..runtime_config import get_email_config, masked
    return masked(get_email_config())


@router.put("/email-config")
def set_email_config(
    body: EmailConfigIn,
    admin: User = Depends(_require_admin),
):
    """Сохранить конфигурацию корпоративной почты (SMTP)."""
    from ..runtime_config import set_email_config, masked
    result = set_email_config(
        smtp_host=body.smtp_host,
        smtp_port=body.smtp_port,
        smtp_user=body.smtp_user,
        smtp_pass=body.smtp_pass,
        from_email=body.from_email,
    )
    return masked(result)


@router.get("/telegram-config")
def get_telegram_config(admin: User = Depends(_require_admin)):
    """Текущая конфигурация Telegram-бота (токен замаскирован)."""
    from ..runtime_config import get_telegram_config, masked
    return masked(get_telegram_config())


@router.put("/telegram-config")
def set_telegram_config(
    body: TelegramConfigIn,
    admin: User = Depends(_require_admin),
):
    """Сохранить конфигурацию Telegram-бота."""
    from ..runtime_config import set_telegram_config, masked
    result = set_telegram_config(
        bot_token=body.bot_token,
        bot_username=body.bot_username,
        chat_id=body.chat_id,
    )
    return masked(result)


@router.get("/finagent-key")
def get_finagent_key_config(admin: User = Depends(_require_admin)):
    """Текущая конфигурация FinAgent: ключ (замаскирован) + модель + base URL + источники."""
    from ..runtime_config import finagent_state, finagent_key_source, masked
    out = masked(finagent_state())
    out["source"] = finagent_key_source()
    out["env_configured"] = finagent_key_source() in ("runtime", "env")
    return out


@router.put("/finagent-key")
def set_finagent_key_config(
    body: FinAgentKeyIn,
    admin: User = Depends(_require_admin),
):
    """Сохранить FinAgent-конфигурацию: ключ / модель / base URL.

    Пустая строка в любом поле = вернуться к .env для этого поля;
    отсутствующее поле (None) runtime не трогает.
    """
    from ..runtime_config import finagent_key_source, finagent_state, masked, set_finagent_config
    set_finagent_config(api_key=body.api_key, model=body.model, base_url=body.base_url)
    out = masked(finagent_state())
    out["source"] = finagent_key_source()
    out["env_configured"] = finagent_key_source() in ("runtime", "env")
    return out


@router.post("/test-email", response_model=MessageOut)
def admin_test_email(
    to: str = Query(..., min_length=3, max_length=255),
    admin: User = Depends(_require_admin),
):
    """Отправить тестовое письмо на указанный адрес (проверка SMTP)."""
    from gex.adapters.notifications.email_service import _send_email
    sent = _send_email(
        to_email=to,
        subject="GEX Analytics — тестовое письмо",
        html_body=(
            '<div style="font-family:Arial,sans-serif;padding:24px;color:#e8edf3;background:#0e1219">'
            '<h2 style="color:#fff">✅ SMTP настроен корректно</h2>'
            '<p>Это тестовое письмо с вашей корпоративной почты GEX Analytics.</p></div>'
        ),
        text_body="GEX Analytics — тестовое письмо. SMTP настроен корректно.",
    )
    if sent:
        return MessageOut(message=f"Тестовое письмо отправлено на {to}.")
    return MessageOut(
        message=f"Письмо НЕ отправлено (SMTP не настроен или ошибка). Проверьте настройки и логи.",
        detail="send returned False",
    )
