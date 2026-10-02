"""SubscriptionExpiryWatcher — уведомление в Telegram за 24ч до окончания подписки.

Фоновый поток (раз в час) находит пользователей, чья подписка BASIC/EXTENDED
истекает в ближайшие 24 часа, и отправляет им в Telegram сообщение
«Подписка истекает завтра». Повторная отправка для той же даты исключается
колонкой ``User.subscription_expiry_notified_for`` (хранит expires_at, о котором
уже уведомили). При продлении подписки expires_at меняется — уведомление
отправится заново для новой даты.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone

from gex.auth.models import SubscriptionStatus, User, _as_utc
from gex.adapters.persistence.database import SessionLocal
from gex.adapters.notifications.telegram_sender import send_telegram_message

logger = logging.getLogger(__name__)

# Интервал проверки (сек). 1 час — достаточно: окно уведомления 24 часа.
POLL_INTERVAL_SECONDS = 3600
# Отправляем, когда до окончания осталось не больше этого времени.
NOTIFY_BEFORE = timedelta(hours=24)


def _format_message(user: User) -> str:
    """HTML-сообщение «подписка истекает завтра»."""
    exp = _as_utc(user.subscription_expires_at)
    if exp is None:
        exp = datetime.now(timezone.utc) + timedelta(days=1)
    local = exp.astimezone()
    date_str = local.strftime("%d.%m.%Y %H:%M")
    plan = user.subscription_status or "BASIC"
    return (
        "<b>Подписка истекает завтра</b>\n\n"
        f"Ваша подписка <b>{plan}</b> действует до <b>{date_str}</b>.\n"
        "Продлите подписку, чтобы сканеры и уведомления о сигналах "
        "продолжали работать.\n\n"
        "Оплатить можно на сайте в разделе «Оплата»."
    )


class SubscriptionExpiryWatcher:
    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False

    # ── управление потоком ─────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="subscription-expiry-watcher"
        )
        self._thread.start()
        self._running = True
        logger.info("SubscriptionExpiryWatcher: поток запущен (интервал %ds)", POLL_INTERVAL_SECONDS)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._running = False
        logger.info("SubscriptionExpiryWatcher: остановлен")

    @property
    def is_running(self) -> bool:
        return self._running

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._check()
            except Exception as exc:
                logger.error("SubscriptionExpiryWatcher: ошибка проверки: %s", exc)
            self._stop_event.wait(POLL_INTERVAL_SECONDS)

    # ── проверка ───────────────────────────────────────────────────

    def _check(self) -> int:
        """Найти истекающие в ближайшие 24ч подписки и уведомить.

        Returns: число отправленных уведомлений.
        """
        now = datetime.now(timezone.utc)
        soon = now + NOTIFY_BEFORE

        db = SessionLocal()
        sent = 0
        try:
            users = (
                db.query(User)
                .filter(
                    User.subscription_status.in_([SubscriptionStatus.BASIC, SubscriptionStatus.EXTENDED]),
                    User.subscription_expires_at.isnot(None),
                    User.subscription_expires_at > now,
                    User.subscription_expires_at <= soon,
                    User.telegram_chat_id.isnot(None),
                )
                .all()
            )
            for user in users:
                try:
                    if self._notify(user, db):
                        sent += 1
                except Exception as exc:
                    logger.error("SubscriptionExpiryWatcher: ошибка отправки user=%s: %s", user.id, exc)
        finally:
            db.close()
        return sent

    def _notify(self, user: User, db) -> bool:
        """Отправить уведомление, если для этой даты ещё не отправляли."""
        exp = _as_utc(user.subscription_expires_at)
        notified_for = _as_utc(user.subscription_expiry_notified_for)
        if exp is not None and notified_for is not None and notified_for == exp:
            return False

        send_telegram_message(_format_message(user), parse_mode="HTML", chat_id=user.telegram_chat_id)
        user.subscription_expiry_notified_for = user.subscription_expires_at
        db.commit()
        logger.info(
            "SubscriptionExpiryWatcher: уведомление отправлено user=%s expires=%s",
            user.id, user.subscription_expires_at,
        )
        return True
