"""Telegram long-polling service — приём /start без webhook и публичного домена.

Бот сам забирает обновления через ``getUpdates`` (long-polling), обрабатывает их
тем же обработчиком, что и webhook (``handle_telegram_message``). Работает из
любого места с доступом к api.telegram.org — домен/HTTPS/nginx не нужны.

Правила:
  - Если у бота зарегистрирован webhook (getWebhookInfo.url не пуст) — polling
    отключается (Telegram запрещает одновременную работу webhook и getUpdates).
  - При ошибке 409 (конфликт с webhook) polling сам останавливается.
  - Повторные ошибки — экспоненциальный backoff.

Также содержит хелперы для работы с webhook:
  ``get_bot_token()``, ``get_webhook_info()``, ``set_telegram_webhook(url)``.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import requests

from gex.auth.config import settings

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org"
_POLL_TIMEOUT = 25  # long-polling, секунды
_REQUEST_TIMEOUT = _POLL_TIMEOUT + 15
# Backoff при ошибках: 1, 2, 5, 10, 30 сек
_ERROR_BACKOFF = (1, 2, 5, 10, 30)


# ---------------------------------------------------------------------------
# Хелперы (токен + webhook)
# ---------------------------------------------------------------------------

def get_bot_token() -> str:
    """Токен бота: runtime_config поверх settings/.env."""
    token = settings.TELEGRAM_BOT_TOKEN
    try:
        from gex.auth.runtime_config import get_telegram_config
        rt = get_telegram_config()
        if rt.get("bot_token"):
            token = rt["bot_token"]
    except Exception:  # noqa: BLE001
        pass
    return token or ""


def _api_call(method: str, token: str, payload: Optional[dict] = None) -> dict:
    resp = requests.post(
        f"{_API_BASE}/bot{token}/{method}",
        json=payload or {},
        timeout=_REQUEST_TIMEOUT,
    )
    try:
        return resp.json()
    except ValueError:
        return {"ok": False, "description": f"HTTP {resp.status_code}"}


def get_webhook_info(token: str) -> dict:
    """Вернуть getWebhookInfo (url, pending_update_count, last_error_message...)."""
    data = _api_call("getWebhookInfo", token)
    return data.get("result", {}) if data.get("ok") else {}


def set_telegram_webhook(
    url: str, token: Optional[str] = None, secret_token: Optional[str] = None
) -> dict:
    """Зарегистрировать webhook бота на ``url`` (например https://домен/telegram/webhook).

    ``secret_token`` — строка, которую Telegram будет присылать в заголовке
    ``X-Telegram-Bot-Api-Secret-Token`` каждого update; webhook-обработчик
    обязан сверить её (иначе подделка update). Если не передана — берём из
    настроек (TELEGRAM_WEBHOOK_SECRET). Без секрета webhook регистрировать
    не рекомендуется (см. telegram_router.webhook_router).

    Возвращает результат Telegram API ({"ok": bool, "result": {...}}).
    """
    token = token or get_bot_token()
    if not token:
        return {"ok": False, "description": "TELEGRAM_BOT_TOKEN не задан"}
    params: dict = {"url": url, "allowed_updates": ["message"]}
    secret = secret_token
    if secret is None:
        from gex.auth.config import settings as _st

        secret = (_st.TELEGRAM_WEBHOOK_SECRET or "").strip() or None
    if secret:
        params["secret_token"] = secret
    return _api_call("setWebhook", token, params)


# ---------------------------------------------------------------------------
# Long-polling service
# ---------------------------------------------------------------------------

class TelegramPollingService:
    """Фоновый поток: getUpdates → обработка сообщений тем же хендлером, что webhook."""

    def __init__(self, process_update: Optional[Callable[[dict], None]] = None):
        # process_update(update: dict) — вызывается для каждого обновления.
        # По умолчанию — handle_telegram_message из telegram_router со своей сессией БД.
        self._process_update = process_update or self._default_process
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._offset = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def _default_process(update: dict) -> None:
        """Обработка обновления по умолчанию: /start, /test и запись telegram_starts."""
        from gex.adapters.persistence.database import SessionLocal
        from gex.auth.telegram_router import handle_telegram_message

        msg = update.get("message") if isinstance(update, dict) else None
        if not msg:
            return
        db = SessionLocal()
        try:
            handle_telegram_message(msg, db)
        finally:
            db.close()

    def start(self) -> None:
        if self._running:
            return
        token = get_bot_token()
        if not token:
            logger.warning("Telegram polling отключён: TELEGRAM_BOT_TOKEN не задан")
            return

        # Если у бота уже зарегистрирован webhook — polling не запускаем.
        # Сеть до api.telegram.org может быть недоступна — не валим старт приложения.
        try:
            info = get_webhook_info(token)
            if info.get("url"):
                logger.info("Telegram webhook активен (%s) — long-polling отключён", info["url"])
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram: не удалось проверить webhook (%s) — запускаю long-polling", exc)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(token,), daemon=True, name="telegram-polling"
        )
        self._thread.start()
        self._running = True
        logger.info("Telegram long-polling запущен (getUpdates)")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._running = False

    def _loop(self, token: str) -> None:
        self._running = True
        backoff_idx = 0
        while not self._stop_event.is_set():
            try:
                data = _api_call(
                    "getUpdates",
                    token,
                    {
                        "offset": self._offset,
                        "timeout": _POLL_TIMEOUT,
                        "allowed_updates": ["message"],
                    },
                )
                if not data.get("ok"):
                    code = data.get("error_code")
                    desc = data.get("description", "?")
                    # 409 — конфликт с webhook: выключаемся навсегда
                    if code == 409:
                        logger.warning("Telegram polling: конфликт с webhook (409) — останавливаюсь: %s", desc)
                        self._running = False
                        return
                    logger.error("Telegram getUpdates error (%s): %s", code, desc)
                    backoff_idx = min(backoff_idx + 1, len(_ERROR_BACKOFF) - 1)
                    self._stop_event.wait(_ERROR_BACKOFF[backoff_idx])
                    continue

                backoff_idx = 0
                updates = data.get("result", [])
                for upd in updates:
                    if self._stop_event.is_set():
                        return
                    try:
                        self._process_update(upd)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Telegram update processing failed: %s", exc)
                    finally:
                        # Оффсет двигаем в любом случае, чтобы не зациклиться
                        self._offset = max(self._offset, int(upd.get("update_id", 0)) + 1)
            except Exception as exc:  # noqa: BLE001
                logger.error("Telegram polling error: %s", exc)
                backoff_idx = min(backoff_idx + 1, len(_ERROR_BACKOFF) - 1)
                self._stop_event.wait(_ERROR_BACKOFF[backoff_idx])

        self._running = False
