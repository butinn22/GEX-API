"""Порт уведомлений (Telegram и прочее).

Смысл порта: домен и application не должны знать ни про Bot API, ни про анти-спам, ни про лимиты
Telegram — они лишь просят «отправь это». Реализация — ``gex/adapters/providers/telegram.py``
(плюс pacing и дедупликация из ``telegram_sender``).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["NotifierPort"]


@runtime_checkable
class NotifierPort(Protocol):
    """Отправка сообщения; ошибка доставки не должна ломать вызывающий сценарий."""

    def send(self, text: str, *, chat_id: str | None = None, parse_mode: str | None = None) -> bool:
        """Отправить текст; ``False`` — доставка не удалась (без исключений)."""
        ...
