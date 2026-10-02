"""Per-user Telegram integration: connect personal Telegram, send targeted notifications.

Adds telegram_chat_id, telegram_username, telegram_connect_token, telegram_notify,
telegram_test_at to User model.

Endpoints (auth):
  GET  /auth/telegram/status      — статус подключения + тестовая проверка связи
  POST /auth/telegram/connect     — сгенерировать deep-link (t.me/<bot>?start=<token>)
  PUT  /auth/telegram/nickname    — сохранить ник вида @username
  PUT  /auth/telegram/notify      — вкл/выкл персональные уведомления
  POST /auth/telegram/test        — отправить тестовое сообщение в чат пользователя
  POST /auth/telegram/disconnect  — отвязать Telegram

Webhook (без auth):
  POST /telegram/webhook — обрабатывает /start <token>, /start activate_<token>, /test
"""
from __future__ import annotations

import logging
import secrets
import hmac
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from gex.adapters.persistence.database import get_session
from gex.auth.models import User
from gex.auth.dependencies import get_current_user
from gex.auth.config import settings
from gex.auth.schemas import TelegramNicknameIn, TelegramNotifyIn, validate_telegram_username
from gex.adapters.ratelimit.rate_limiter import IpRateLimiter
from gex.auth.service import (
    get_user_by_telegram_username,
    get_telegram_start_chat_id,
    record_telegram_start,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/telegram", tags=["telegram"])


# Публичная ручка /auth/telegram/check — оракул по @никнеймам: ограничиваем,
# чтобы нельзя было перебирать ники (enumeration) без последствий.
_tg_check_limiter = IpRateLimiter(rate=0.2, burst=10, scope="telegram")


def _check_public_rate(request: Request) -> None:
    if settings.TESTING:
        return
    ip = request.client.host if request.client else "unknown"
    if ip == "testclient":
        return
    if not _tg_check_limiter.allow(ip):
        raise HTTPException(
            status_code=429,
            detail="Слишком много проверок. Подождите и повторите.",
        )


import uuid as _uuid  # noqa: E402


def _bot_username() -> str:
    """Имя бота (@username) из runtime_config поверх settings."""
    bot = settings.TELEGRAM_BOT_USERNAME or "GexAnalyticsBot"
    try:
        from gex.auth.runtime_config import get_telegram_config
        rt = get_telegram_config()
        if rt.get("bot_username"):
            bot = rt["bot_username"]
    except Exception:  # noqa: BLE001
        pass
    return bot.lstrip("@")


BOT_USERNAME = _bot_username()


# Срок жизни connect-токена: 15 минут
CONNECT_TOKEN_TTL_MINUTES = 15

# Префикс для токена активации аккаунта (deep-link: /start activate_<verification_token>)
ACTIVATE_PREFIX = "activate_"


def _activation_link(token: str) -> str:
    """Deep-link активации: https://t.me/<bot>?start=activate_<token>"""
    return f"https://t.me/{_bot_username()}?start={ACTIVATE_PREFIX}{token}"


# ── Schemas ──────────────────────────────────────────────────────────


class TelegramStatusOut(BaseModel):
    connected: bool
    chat_id: Optional[str] = None
    username: Optional[str] = None
    notify: bool = False
    test_at: Optional[datetime] = None
    bot_configured: bool = False
    bot_username: str = ""


class TelegramConnectOut(BaseModel):
    connected: bool
    connect_url: str
    expires_in_minutes: int = CONNECT_TOKEN_TTL_MINUTES


class TelegramMessageOut(BaseModel):
    detail: str


class TelegramTestOut(BaseModel):
    ok: bool
    detail: str


class TelegramCheckIn(BaseModel):
    """Проверка ДО регистрации: ответил ли пользователь боту (ника @username)."""

    telegram_username: str = Field(
        ...,
        description="Ник Telegram вида @username (с @, 5-32 символа).",
    )

    @field_validator("telegram_username")
    @classmethod
    def _check_username(cls, v: str) -> str:
        return validate_telegram_username(v)


class TelegramCheckOut(BaseModel):
    """Результат проверки «пользователь написал боту» (telegram_starts)."""

    ok: bool = False
    username: str = ""
    bot_username: str = ""
    taken: bool = False
    message: str = ""


# ── Endpoints ────────────────────────────────────────────────────────


def _status(user: User) -> TelegramStatusOut:
    """Собрать статус подключения пользователя."""
    bot_token = settings.TELEGRAM_BOT_TOKEN
    try:
        from gex.auth.runtime_config import get_telegram_config
        rt = get_telegram_config()
        if rt.get("bot_token"):
            bot_token = rt["bot_token"]
    except Exception:  # noqa: BLE001
        pass
    return TelegramStatusOut(
        connected=bool(user.telegram_chat_id),
        chat_id=user.telegram_chat_id,
        username=user.telegram_username,
        notify=bool(user.telegram_notify),
        test_at=user.telegram_test_at,
        bot_configured=bool(bot_token),
        bot_username="@" + _bot_username(),
    )


@router.get("/status", response_model=TelegramStatusOut)
def telegram_status(user: User = Depends(get_current_user)) -> TelegramStatusOut:
    """Статус подключения Telegram (подвязан ли чат, включены ли уведомления)."""
    return _status(user)


@router.post("/check", response_model=TelegramCheckOut)
def telegram_check(
    body: TelegramCheckIn,
    _: None = Depends(_check_public_rate),
    db: Session = Depends(get_session),
) -> TelegramCheckOut:
    """Проверить (до регистрации), что владелец @username написал нашему боту.

    Смотрит таблицу ``telegram_starts``: она заполняется, когда бот получает
    от пользователя любое сообщение (/start или просто текст). Если записи
    нет — бот ещё не видел этого пользователя, регистрироваться нельзя
    (ссылку активации некому доставить).

    Публичная ручка: вызывается с формы регистрации ДО создания аккаунта.
    """
    nick = body.telegram_username  # валидирован: @ + 5-32 символа
    bot = "@" + _bot_username()

    # Ник уже привязан к существующему аккаунту — регистрация всё равно 409
    taken = get_user_by_telegram_username(db, nick)
    if taken is not None:
        return TelegramCheckOut(
            ok=False,
            username=nick,
            bot_username=bot,
            taken=True,
            message="Этот Telegram-ник уже привязан к другому аккаунту GEX Analytics.",
        )

    # Бот получал сообщение от этого пользователя (нажатие /start или любой текст)
    chat_id = get_telegram_start_chat_id(db, nick)
    if not chat_id:
        return TelegramCheckOut(
            ok=False,
            username=nick,
            bot_username=bot,
            message=(
                f"Бот ещё не получил от вас сообщение. Откройте {bot}, "
                "нажмите Start (или отправьте любое сообщение) и повторите проверку."
            ),
        )

    logger.info("Telegram pre-register check passed for %s (chat_id=%s)", nick, chat_id)
    return TelegramCheckOut(
        ok=True,
        username=nick,
        bot_username=bot,
        message="Telegram подтверждён — бот получил ваше сообщение. Ссылка активации придёт именно в этот чат.",
    )


@router.post("/connect", response_model=TelegramConnectOut)
def telegram_connect(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
) -> TelegramConnectOut:
    """Сгенерировать deep-link для подключения Telegram.

    URL ведёт на /start бота с одноразовым токеном. При переходе бот
    привязывает chat_id к аккаунту и проверяет ник из профиля.
    """
    # Уже подключён — ссылка не нужна
    if user.telegram_chat_id:
        return TelegramConnectOut(
            connected=True,
            connect_url=f"https://t.me/{_bot_username()}",
            expires_in_minutes=0,
        )

    # Генерируем connect-токен
    token = secrets.token_urlsafe(32)
    user.telegram_connect_token = token
    # Обновляем updated_at — он же используется как timestamp выдачи токена
    user.updated_at = datetime.now(timezone.utc)
    db.commit()

    connect_url = f"https://t.me/{_bot_username()}?start={token}"
    logger.info("Telegram connect link generated for user %s: %s", user.email, connect_url[:60] + "...")

    return TelegramConnectOut(
        connected=False,
        connect_url=connect_url,
        expires_in_minutes=CONNECT_TOKEN_TTL_MINUTES,
    )


@router.put("/nickname", response_model=TelegramStatusOut)
def telegram_update_nickname(
    body: TelegramNicknameIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
) -> TelegramStatusOut:
    """Сохранить Telegram-ник вида @username в профиле.

    Ник должен быть уникальным (не занят другим аккаунтом).
    """
    nickname = body.telegram_username
    other = get_user_by_telegram_username(db, nickname)
    if other and other.id != user.id:
        raise HTTPException(
            status_code=409,
            detail="Этот Telegram-ник уже привязан к другому аккаунту",
        )
    user.telegram_username = nickname
    db.commit()
    logger.info("Telegram nickname updated for user %s -> %s", user.email, nickname)
    return _status(user)


@router.put("/notify", response_model=TelegramStatusOut)
def telegram_set_notify(
    body: TelegramNotifyIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
) -> TelegramStatusOut:
    """Включить/выключить персональные уведомления в Telegram."""
    user.telegram_notify = bool(body.notify)
    db.commit()
    logger.info("Telegram notify for user %s -> %s", user.email, user.telegram_notify)
    return _status(user)


@router.post("/test", response_model=TelegramTestOut)
def telegram_test(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
) -> TelegramTestOut:
    """Отправить тестовое сообщение в Telegram пользователя — проверка связи.

    При успешной доставке фиксирует telegram_test_at (виден в статусе).
    """
    if not user.telegram_chat_id:
        raise HTTPException(status_code=400, detail="Telegram не подключён")

    from gex.adapters.notifications.telegram_sender import send_telegram_message

    text = (
        "✅ <b>Связь работает!</b>\n\n"
        "Это тестовое сообщение из GEX Analytics.\n"
        f"Аккаунт: <b>{user.email}</b>\n"
        f"Telegram: {user.telegram_username or '—'}\n\n"
        "Персональные уведомления будут приходить именно сюда."
    )
    res = send_telegram_message(text, parse_mode="HTML", chat_id=user.telegram_chat_id)

    if res.get("success"):
        user.telegram_test_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("Telegram test message delivered to user %s", user.email)
        return TelegramTestOut(ok=True, detail="Тестовое сообщение доставлено")

    errs = "; ".join(res.get("errors") or ["неизвестная ошибка"])
    logger.error("Telegram test message failed for user %s: %s", user.email, errs)
    return TelegramTestOut(ok=False, detail=f"Не удалось отправить: {errs}")


@router.post("/disconnect", response_model=TelegramMessageOut)
def telegram_disconnect(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
) -> TelegramMessageOut:
    """Отвязать Telegram от аккаунта."""
    user.telegram_chat_id = None
    user.telegram_username = None
    user.telegram_connect_token = None
    user.telegram_notify = False
    user.telegram_test_at = None
    db.commit()
    logger.info("Telegram disconnected for user %s", user.email)
    return TelegramMessageOut(detail="Telegram отвязан от аккаунта")


# ── Webhook handler (подключается как telegram_webhook_router) ──────


def handle_start_command(token: str, chat_id: str, username: str | None, db: Session) -> str:
    """Обработать /start <token> из Telegram-бота.

    Режимы:
      1. ``/start activate_<verification_token>`` — активация аккаунта
         (подтверждение владения Telegram при регистрации).
      2. ``/start <connect_token>`` — подключение уведомлений из профиля:
         привязывает чат к аккаунту, проверяет ник из профиля.

    Returns:
        Текст ответа бота пользователю в Telegram.
    """
    if not token:
        # Сервис узнал, что пользователь нажал /start — запоминаем chat_id.
        # Если по нику находится аккаунт — сразу привязываем чат и, при
        # незавершённой регистрации, отправляем подтверждение со ссылкой.
        if username:
            user = get_user_by_telegram_username(db, "@" + username.lstrip("@"))
            if user is not None:
                user.telegram_chat_id = chat_id
                user.telegram_username = "@" + username.lstrip("@")
                if not user.verification_token:
                    user.verification_token = _uuid.uuid4().hex + _uuid.uuid4().hex
                db.commit()

                if not user.is_email_verified:
                    link = _activation_link(user.verification_token)
                    logger.info("Telegram /start: pending activation for user=%s chat=%s", user.email, chat_id)
                    return (
                        f"👋 Мы получили ваш /start, {user.telegram_username}!\n\n"
                        f"Для email {user.email} идёт регистрация в GEX Analytics.\n"
                        f"Остался последний шаг — подтвердите, что Telegram принадлежит вам:\n\n"
                        f"{link}\n\n"
                        "После подтверждения вы сможете войти в терминал."
                    )

                logger.info("Telegram /start: chat bound to active user=%s chat=%s", user.email, chat_id)
                return (
                    f"✅ Ваш аккаунт {user.email} уже активирован!\n\n"
                    "Подключите персональные уведомления в профиле на сайте "
                    "— бот будет присылать выбранные вами данные прямо сюда."
                )

        return (
            f"👋 Добро пожаловать в GEX Analytics Bot!\n\n"
            f"Чтобы подключить уведомления, откройте профиль на сайте "
            f"и нажмите «Подключить Telegram» — бот привяжет ваш аккаунт."
        )

    # ── Режим 1: активация аккаунта ────────────────────────────────
    if token.startswith(ACTIVATE_PREFIX):
        verification_token = token[len(ACTIVATE_PREFIX):]
        user = db.query(User).filter(User.verification_token == verification_token).first()

        if user is None:
            logger.warning("Telegram activate: invalid token '%s...'", token[:20])
            return (
                "❌ Не удалось найти аккаунт по этой ссылке активации.\n\n"
                "Запросите новую ссылку в профиле или при регистрации "
                "на сайте GEX Analytics."
            )

        if user.is_email_verified:
            return (
                f"✅ Аккаунт {user.email} уже активирован. Добро пожаловать в GEX Analytics! 🎉"
            )

        # Привязываем Telegram-чат к аккаунту (подтверждение реальности)
        user.telegram_chat_id = chat_id
        if username:
            user.telegram_username = "@" + username.lstrip("@")
        user.is_email_verified = True
        user.verification_token = None  # одноразовый токен
        db.commit()

        logger.info(
            "Telegram activation: user=%s chat_id=%s username=%s",
            user.email, chat_id, username,
        )
        return (
            f"✅ Регистрация подтверждена! Аккаунт {user.email} активирован.\n\n"
            "Добро пожаловать в GEX Analytics — терминал микроструктурного "
            "анализа опционных рынков. Удачной торговли! 📈"
        )

    # ── Режим 2: подключение уведомлений (профиль) ──────────────────
    user = db.query(User).filter(User.telegram_connect_token == token).first()

    if user is None:
        logger.warning("Telegram connect: invalid or expired token '%s...'", token[:8])
        return "❌ Неверная или устаревшая ссылка. Пожалуйста, запросите новую в профиле."

    # Проверить срок действия токена (15 минут от updated_at)
    if user.updated_at:
        updated = user.updated_at
        if updated.tzinfo is None:  # SQLite не хранит tz — считаем UTC
            updated = updated.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - updated).total_seconds()
        if age > CONNECT_TOKEN_TTL_MINUTES * 60:
            user.telegram_connect_token = None
            db.commit()
            return "⏰ Ссылка устарела. Пожалуйста, запросите новую в профиле."

    # Проверка ника: если в профиле указан @nickname, то привязать можно
    # только чат с этим же ником (регистронезависимо).
    expected = (user.telegram_username or "").lstrip("@").lower()
    if expected and username:
        actual = username.lstrip("@").lower()
        if actual != expected:
            logger.warning(
                "Telegram connect: nickname mismatch for user=%s (expected @%s, got @%s)",
                user.email, expected, actual,
            )
            return (
                f"⚠️ Для аккаунта {user.email} указан Telegram-ник @{expected}, "
                f"но вы вошли в Telegram как @{actual}.\n\n"
                "Укажите в профиле именно ваш ник или войдите в Telegram под "
                "указанным ником и повторите попытку."
            )

    # Сохранить chat_id и username
    user.telegram_chat_id = chat_id
    if username:
        user.telegram_username = "@" + username.lstrip("@")
    user.telegram_connect_token = None  # одноразовый токен
    db.commit()

    logger.info(
        "Telegram connected: user=%s chat_id=%s username=%s",
        user.email, chat_id, username,
    )

    notify_hint = (
        "Уведомления включены — вы будете получать персональные данные, "
        "которые выбрали в личном кабинете."
        if user.telegram_notify
        else "Включите «Telegram-уведомления» в личном кабинете, чтобы получать "
             "персональные данные, которые вы выбрали."
    )
    return (
        f"✅ Telegram подключён к аккаунту {user.email}!\n\n"
        f"{notify_hint}"
    )


# ── Webhook router (отдельный, без auth) ─────────────────────────────


webhook_router = APIRouter(tags=["telegram-webhook"])


class TelegramUpdate(BaseModel):
    """Входящее обновление от Telegram Bot API."""

    update_id: int
    message: Optional[dict] = None


@webhook_router.post("/telegram/webhook")
def telegram_webhook(
    update: TelegramUpdate,
    request: Request,
    db: Session = Depends(get_session),
):
    """Webhook для Telegram Bot API (подлинность update проверяется секретом).

    Telegram присылает в каждом update заголовок ``X-Telegram-Bot-Api-Secret-Token``
    (тот secret_token, что был передан в setWebhook). Сверяем его в постоянном
    времени — иначе любой сможет подделать update: перехватить Telegram-канал
    уведомлений аккаунта, активировать чужой аккаунт и заставить бота писать
    в произвольные чаты. Принимает сообщения и обрабатывает /start /test.

    NOTE: секрет задаётся в .env (TELEGRAM_WEBHOOK_SECRET) и передаётся в
    setWebhook автоматически (main._maybe_register_telegram_webhook /
    scripts/set_telegram_webhook.py). Без заданного секрета webhook отвечает
    403 (в тестах settings.TESTING пропускает проверку).
    """
    if not settings.TESTING:
        expected = (settings.TELEGRAM_WEBHOOK_SECRET or "").strip()
        received = request.headers.get("x-telegram-bot-api-secret-token", "")
        if not expected:
            logger.error(
                "Telegram webhook: TELEGRAM_WEBHOOK_SECRET не задан — update отклонён. "
                "Задайте секрет в .env и перерегистрируйте webhook."
            )
            raise HTTPException(status_code=403, detail="Webhook secret is not configured")
        if not received or not hmac.compare_digest(received, expected):
            logger.warning("Telegram webhook: неверный secret token — update отклонён")
            raise HTTPException(status_code=403, detail="Invalid webhook secret")

    msg = update.message
    if msg:
        handle_telegram_message(msg, db)
    return {"ok": True}


def handle_telegram_message(msg: dict, db: Session) -> None:
    """Общая обработка входящего сообщения Telegram (webhook и long-polling).

    - Записывает факт /start (telegram_starts) — сервис узнаёт chat_id по нику.
    - Обрабатывает /start <token>, /start и /test, отвечает ботом.
    """
    text = (msg.get("text") or "").strip()
    chat = msg.get("chat", {})
    chat_id = str(chat.get("id", ""))
    # Только настоящий публичный username; first_name/last_name ником не являются
    username = chat.get("username") or None

    if not chat_id:
        return

    # Сервис узнаёт, что пользователь отправлял сообщения/нажимал /start боту:
    # запоминаем связку username → chat_id, чтобы уметь писать ему в Telegram.
    record_telegram_start(db, username, chat_id)

    # /test — проверка связи с сервером: отвечаем и фиксируем время
    if text == "/test":
        db_user = db.query(User).filter(User.telegram_chat_id == chat_id).first()
        if db_user:
            db_user.telegram_test_at = datetime.now(timezone.utc)
            db.commit()
            _send_telegram_reply(
                chat_id,
                f"✅ Связь работает, {db_user.telegram_username or ''}!\n"
                "Это ответ на ваш /test. Персональные уведомления будут приходить сюда.",
            )
        else:
            _send_telegram_reply(
                chat_id,
                "Ваш Telegram не привязан к аккаунту GEX Analytics.\n"
                "Подключите его в личном кабинете на сайте.",
            )
        return

    # /start — подключение или приветствие
    if text.startswith("/start"):
        # Извлечь токен: "/start ABC123" → "ABC123"
        parts = text.split(maxsplit=1)
        token = parts[1] if len(parts) > 1 else ""
        reply = handle_start_command(token, chat_id, username, db)

        # Отправить ответ через Bot API
        _send_telegram_reply(chat_id, reply)


# ── Helpers ───────────────────────────────────────────────────────────


def _send_telegram_reply(chat_id: str, text: str) -> None:
    """Отправить сообщение в Telegram через Bot API (синхронно)."""
    import requests

    bot_token = settings.TELEGRAM_BOT_TOKEN
    try:
        from gex.auth.runtime_config import get_telegram_config
        rt = get_telegram_config()
        if rt.get("bot_token"):
            bot_token = rt["bot_token"]
    except Exception:  # noqa: BLE001
        pass

    if not bot_token:
        logger.warning("Telegram-уведомления отключены: TELEGRAM_BOT_TOKEN не задан в .env")
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        if not resp.ok:
            logger.error("Telegram sendMessage failed: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logger.error("Telegram sendMessage error: %s", e)
