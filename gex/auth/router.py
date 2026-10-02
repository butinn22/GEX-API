"""FastAPI router для auth: регистрация, верификация email, логин, OAuth."""
from __future__ import annotations

import logging
import uuid as _uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from .config import settings
from .dependencies import get_current_user, get_optional_user
from gex.adapters.persistence.database import get_session
from .models import SubscriptionStatus, User
from .schemas import (
    EmailChangeIn,
    EmailChangeOut,
    LoginIn,
    MessageOut,
    OAuthCallbackIn,
    RefreshIn,
    RegisterIn,
    RegisterOut,
    TokenOut,
    UserOut,
    UserProfileOut,
)
from .service import (
    AuthService,
    create_access_token,
    create_refresh_token,
    is_demo_user,
    is_master_admin,
    hash_password,
    get_user_by_email,
    check_password,
)

#: Логгер обязан быть определён ДО ``try/except`` ниже: сообщения о недоступном SMTP и
#: Telegram пишутся именно в обработчиках, а определение стояло в конце модуля. При
#: отсутствии ``gex.email_service`` импорт падал с ``NameError: name 'logger'`` — то есть
#: «мягкая деградация» сама была отказом, причём на старте приложения.
logger = logging.getLogger(__name__)

# Email service (SMTP)
try:
    from gex.adapters.notifications.email_service import send_verification_email
    _email_available = True
except Exception as e:
    logger.warning("Email service not available: %s", e)
    _email_available = False
    def send_verification_email(email, token, base_url):
        logger.info("📧 Verification for %s: %s/auth/verify-email?token=%s", email, base_url, token)
        return False

# Telegram sender (бот для активации аккаунтов)
try:
    from gex.adapters.notifications.telegram_sender import send_telegram_message as _send_tg
    _telegram_available = True
except Exception as e:
    logger.warning("Telegram sender not available: %s", e)
    _telegram_available = False
    def _send_tg(text, *, parse_mode=None, chat_id=None):
        return {"success": False, "batches_sent": 0, "errors": ["sender unavailable"], "raw_responses": []}



def _telegram_activation_link(token: str) -> str:
    """Deep-link для активации: https://t.me/<bot>?start=activate_<token>."""
    bot = settings.TELEGRAM_BOT_USERNAME or "GexAnalyticsBot"
    try:
        from gex.auth.runtime_config import get_telegram_config
        rt = get_telegram_config()
        if rt.get("bot_username"):
            bot = rt["bot_username"]
    except Exception:  # noqa: BLE001
        pass
    return f"https://t.me/{bot}?start=activate_{token}"


def _send_telegram_activation(user: User, activation_link: str, db: Session | None = None) -> bool:
    """Отправить пользователю в Telegram ссылку активации (best-effort).

    Приоритет получателя:
      1. ``user.telegram_chat_id`` — чат уже привязан к аккаунту;
      2. ``telegram_starts`` — пользователь уже нажимал /start боту
         (сервис узнал chat_id по публичному нику); в этом случае chat_id
         сразу привязывается к аккаунту;
      3. ``@username`` — последний вариант: сработает только если чат
         публичный и бот уже «видел» пользователя.

    Ошибки не роняют регистрацию.
    """
    if not user.telegram_username or not _telegram_available:
        return False

    chat_id = user.telegram_chat_id
    if not chat_id and db is not None:
        from .service import get_telegram_start_chat_id
        chat_id = get_telegram_start_chat_id(db, user.telegram_username)
        if chat_id:
            user.telegram_chat_id = chat_id
            db.commit()
            logger.info("Telegram start chat bound to user %s (chat_id=%s)", user.email, chat_id)

    text = (
        f"👋 <b>GEX Analytics — активация аккаунта</b>\n\n"
        f"Для email <b>{_esc_html(user.email)}</b> создан аккаунт.\n"
        f"Нажмите кнопку, чтобы подтвердить, что Telegram принадлежит вам:\n\n"
        f"<a href=\"{_esc_html(activation_link)}\">✅ Активировать аккаунт</a>"
    )
    try:
        res = _send_tg(text, parse_mode="HTML", chat_id=chat_id or user.telegram_username)
        return bool(res and res.get("success"))
    except Exception as exc:
        logger.warning("Telegram activation send failed for %s: %s", user.email, exc)
        return False


def _esc_html(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")




from gex.adapters.ratelimit.rate_limiter import IpRateLimiter  # noqa: E402

# Брутфорс-защита: 5 попыток на IP, далее 1 раз в 4 сек
_auth_limiter = IpRateLimiter(rate=0.25, burst=5, scope="auth")


def _check_auth_rate_limit(request: Request) -> None:
    """429 при превышении числа попыток с одного IP (защита от brute-force)."""
    if settings.TESTING:
        return
    ip = request.client.host if request.client else "unknown"
    # TestClient (pytest) использует фиксированный host "testclient"
    if ip == "testclient":
        return
    if not _auth_limiter.allow(ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Слишком много попыток. Подождите и повторите.",
        )

router = APIRouter(prefix="/auth", tags=["auth"])


# ================================================================= #
#  POST /auth/register
# ================================================================= #
@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=RegisterOut)
def register(
    body: RegisterIn,
    request: Request = None,
    _: None = Depends(_check_auth_rate_limit),
    db: Session = Depends(get_session),
):
    """Регистрация: email + пароль + обязательный Telegram-ник.

    Создаёт пользователя, генерирует verification token, отправляет ссылку
    активации в Telegram (deep-link ``/start activate_<token>``) и письмо
    на email (резерв). Возвращает токены + ссылки активации.
    """
    # Обязательное согласие с условиями использования (EULA)
    if not body.accept_terms:
        raise HTTPException(
            status_code=400,
            detail="Необходимо принять условия использования платформы (EULA).",
        )

    svc = AuthService(db)
    try:
        user = svc.register(body.email, body.password, body.telegram_username)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Генерируем verification token
    token = _uuid.uuid4().hex + _uuid.uuid4().hex
    user.verification_token = token
    db.commit()

    # Ссылки для активации: Telegram (основной канал) + email (резерв)
    # Верификационные ссылки строим из фиксированного FRONTEND_URL, а НЕ из
    # заголовка Host (Host header poisoning → фишинговые ссылки, см. аудит 2026-09-04).
    base_url = settings.FRONTEND_URL.rstrip("/") or "http://localhost:8000"
    verify_link = f"{base_url}/auth/verify-email?token={token}"
    telegram_link = _telegram_activation_link(token)

    # Отправляем письмо через SMTP (или логируем если SMTP не настроен)
    email_sent = send_verification_email(user.email, token, base_url)
    if not email_sent and settings.APP_ENV != "production":
        logger.info("📧 Verification link (dev, console): %s", verify_link)

    # Отправляем ссылку активации в Telegram (best-effort)
    tg_delivered = _send_telegram_activation(user, telegram_link, db)
    if not tg_delivered:
        logger.info("📱 Telegram activation link: %s", telegram_link)

    # Возвращаем токены и ссылки активации
    access = create_access_token(user.id, user.email)
    refresh = create_refresh_token(user.id)

    return RegisterOut(
        access_token=access,
        refresh_token=refresh,
        verification_url=verify_link,
        telegram_activation_url=telegram_link,
        telegram_delivered=tg_delivered,
        message=(
            f"Ссылка активации отправлена в Telegram ({body.telegram_username}). "
            f"Также письмо отправлено на {user.email}"
        ),
    )


# ================================================================= #
#  GET /auth/verify-email
#  Два режима:
#   1) ?token=xxx  — из письма: подтверждает и редиректит на фронтенд
#   2) Bearer JWT  — из приложения: подтверждает текущего пользователя
# ================================================================= #
@router.get("/verify-email")
def verify_email_with_token(
    token: Optional[str] = Query(None, description="Токен из письма"),
    db: Session = Depends(get_session),
):
    """Подтвердить email по токену из письма/Telegram-активации.

    Самоподтверждение «по факту авторизации» намеренно УБРАНО (security-аудит
    2026-09-04): иначе любой зарегистрировавшийся на чужой email подтверждал бы
    его без доказательства владения. Верификация — только одноразовым токеном,
    отправленным на email или в Telegram владельца.
    """
    # Режим 1: ссылка из письма (?token=...) — редирект на фронтенд
    if token:
        u = db.query(User).filter(User.verification_token == token).first()
        if not u:
            raise HTTPException(status_code=404, detail="Недействительный или просроченный токен")
        u.is_email_verified = True
        u.verification_token = None  # сбрасываем токен
        db.commit()
        logger.info("Email verified via token: %s", u.email)
        frontend_url = settings.FRONTEND_URL.rstrip("/")
        return RedirectResponse(url=f"{frontend_url}/#/login?verified=1")

    # Режим 2 (запрещён): без токена email не подтверждаем. Направляем
    # пользователя в правильный флоу (ссылка в письме / Telegram / переотправка).
    raise HTTPException(
        status_code=403,
        detail=(
            "Подтверждение email возможно только по ссылке из письма или через "
            "Telegram-активацию. Запросите письмо повторно: POST /auth/send-verification."
        ),
    )


# ================================================================= #
#  PUT /auth/profile/email  (смена email с подтверждением паролем)
# ================================================================= #
@router.put("/profile/email", response_model=EmailChangeOut)
def change_email(
    body: EmailChangeIn,
    user: User = Depends(get_current_user),
    request: Request = None,
    db: Session = Depends(get_session),
):
    """Сменить email пользователя.

    - Проверяет текущий пароль (для OAuth-аккаунтов без пароля — пропускает).
    - Email должен быть свободен (уникальность, регистронезависимо).
    - После смены аккаунт переходит в статус is_email_verified=False:
      на новый адрес уходит письмо, в Telegram — ссылка активации.
    - Возвращает обновлённый профиль + новую пару токенов (email в JWT обновлён).
    """
    new_email = body.email.strip().lower()

    if new_email == settings.MASTER_EMAIL.strip().lower():
        raise HTTPException(status_code=409, detail="Этот email зарезервирован")

    if new_email == settings.DEMO_EMAIL.strip().lower():
        raise HTTPException(status_code=409, detail="Этот email зарезервирован")

    # Подтверждение текущим паролем (если он задан)
    if user.password_hash:
        if not check_password(body.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Неверный пароль",
            )

    # Уникальность нового email
    existing = get_user_by_email(db, new_email)
    if existing and existing.id != user.id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Пользователь с таким email уже существует",
        )

    # Меняем email и сбрасываем верификацию
    user.email = new_email
    user.is_email_verified = False
    token = _uuid.uuid4().hex + _uuid.uuid4().hex
    user.verification_token = token
    db.commit()

    # Верификационные ссылки строим из фиксированного FRONTEND_URL, а НЕ из
    # заголовка Host (Host header poisoning → фишинговые ссылки, см. аудит 2026-09-04).
    base_url = settings.FRONTEND_URL.rstrip("/") or "http://localhost:8000"
    verify_link = f"{base_url}/auth/verify-email?token={token}"
    telegram_link = _telegram_activation_link(token)

    email_sent = send_verification_email(user.email, token, base_url)
    if not email_sent and settings.APP_ENV != "production":
        logger.info("📧 Email change — verification link (dev, console): %s", verify_link)

    tg_delivered = _send_telegram_activation(user, telegram_link, db)
    if not tg_delivered:
        logger.info("📱 Email change — Telegram activation link: %s", telegram_link)

    # Новая пара токенов (access содержит актуальный email)
    access = create_access_token(user.id, user.email)
    refresh = create_refresh_token(user.id)

    logger.info("Email changed for user %s -> %s", user.id, user.email)
    return EmailChangeOut(
        user=UserOut.model_validate(user),
        access_token=access,
        refresh_token=refresh,
        message=(
            "Email изменён. На новый адрес отправлено письмо с подтверждением"
            " — аккаунт снова требует активации."
        ),
    )


# ================================================================= #
#  POST /auth/send-verification (переотправка письма)
# ================================================================= #
@router.post("/send-verification", response_model=MessageOut)
def send_verification(
    user: User = Depends(get_current_user),
    request=None,
    db: Session = Depends(get_session),
):
    """Переотправить письмо с подтверждением email."""
    if user.is_email_verified:
        raise HTTPException(status_code=400, detail="Email уже подтверждён")

    token = _uuid.uuid4().hex + _uuid.uuid4().hex
    user.verification_token = token
    db.commit()

    # Верификационные ссылки строим из фиксированного FRONTEND_URL, а НЕ из
    # заголовка Host (Host header poisoning → фишинговые ссылки, см. аудит 2026-09-04).
    base_url = settings.FRONTEND_URL.rstrip("/") or "http://localhost:8000"
    verify_link = f"{base_url}/auth/verify-email?token={token}"

    # Отправляем письмо
    email_sent = send_verification_email(user.email, token, base_url)
    if not email_sent and settings.APP_ENV != "production":
        logger.info("📧 Re-send verification link (dev, console): %s", verify_link)

    return MessageOut(message=f"Письмо отправлено на {user.email}." + (" Ссылка: " + verify_link if not email_sent else ""))


# ================================================================= #
#  POST /auth/send-telegram-activation (переотправка ссылки в Telegram)
# ================================================================= #
@router.post("/send-telegram-activation", response_model=MessageOut)
def send_telegram_activation(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Переотправить ссылку активации в Telegram (если бот ещё не смог)."""
    if user.is_email_verified:
        raise HTTPException(status_code=400, detail="Аккаунт уже активирован")
    if not user.telegram_username:
        raise HTTPException(status_code=400, detail="Telegram-ник не указан")

    token = _uuid.uuid4().hex + _uuid.uuid4().hex
    user.verification_token = token
    db.commit()

    telegram_link = _telegram_activation_link(token)
    tg_delivered = _send_telegram_activation(user, telegram_link, db)
    logger.info("📱 Re-send Telegram activation: %s (delivered=%s)", telegram_link, tg_delivered)

    detail = (
        f"Ссылка отправлена в Telegram ({user.telegram_username}). "
        f"Если бот не написал — откройте ссылку вручную: {telegram_link}"
    )
    return MessageOut(message="Ссылка активации Telegram", detail=detail)


# ================================================================= #
#  POST /auth/login  (проверяет is_email_verified)
# ================================================================= #
@router.post("/login", response_model=TokenOut)
def login(
    body: LoginIn,
    request: Request = None,
    _: None = Depends(_check_auth_rate_limit),
    db: Session = Depends(get_session),
):
    """Вход. Проверяет email+password и is_email_verified."""
    svc = AuthService(db)
    try:
        if body.email.strip().lower() == settings.MASTER_EMAIL.strip().lower():
            user = svc.master_admin_login(body.password)
        else:
            user = svc.login(body.email, password=body.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    # Проверка: email должен быть подтверждён (кроме Master Admin)
    if not is_master_admin(user) and not user.is_email_verified:
        raise HTTPException(
            status_code=403,
            detail="Email не подтверждён. Проверьте почту или запросите новое письмо.",
        )

    # Заблокированный аккаунт не может войти
    if user.is_blocked:
        raise HTTPException(
            status_code=403,
            detail="Аккаунт заблокирован. Обратитесь к администратору.",
        )

    access = create_access_token(user.id, user.email)
    refresh = create_refresh_token(user.id)

    logger.info("Login: %s (id=%s)", user.email, user.id)
    return TokenOut(
        access_token=access,
        refresh_token=refresh,
    )


# ================================================================= #
#  Demo-аккаунт (seed + вход без пароля)
# ================================================================= #
def seed_demo_user(db: Session) -> User:
    """Создать/обновить demo-аккаунт («Просмотр демо» на /register).

    Подписка EXTENDED с далёким сроком (проходит барьеры разрешённых
    разделов), email verified, Telegram не привязан. Пароль — случайный:
    войти можно только через POST /auth/demo, middleware ограничивает
    такого пользователя allowlist-ом (просмотр тикера ES).
    """
    from datetime import datetime, timedelta, timezone

    email = settings.DEMO_EMAIL.strip().lower()
    if not email:
        raise ValueError("DEMO_EMAIL не задан")
    existing = get_user_by_email(db, email)
    if existing:
        existing.is_email_verified = True
        existing.is_blocked = False
        existing.subscription_status = SubscriptionStatus.EXTENDED
        existing.subscription_activated_at = existing.subscription_activated_at or datetime.now(timezone.utc)
        existing.subscription_expires_at = datetime.now(timezone.utc) + timedelta(days=36500)
        db.commit()
        logger.info("Demo user updated: %s", email)
        return existing
    user = User(
        id=str(_uuid.uuid4()),
        email=email,
        password_hash=hash_password(_uuid.uuid4().hex + _uuid.uuid4().hex),
        is_email_verified=True,
        is_blocked=False,
        subscription_status=SubscriptionStatus.EXTENDED,
        subscription_activated_at=datetime.now(timezone.utc),
        subscription_expires_at=datetime.now(timezone.utc) + timedelta(days=36500),
    )
    db.add(user)
    db.commit()
    logger.info("Demo user seeded: %s", email)
    return user


@router.post("/demo", response_model=TokenOut)
def demo_login(
    _: None = Depends(_check_auth_rate_limit),
    db: Session = Depends(get_session),
):
    """Вход в demo-режим (публичный, без пароля/регистрации).

    Возвращает пару токенов служебного demo-аккаунта. Доступ к данным
    ограничен middleware-ом DemoScopeMiddleware: только чтение тикера ES
    в разделах тех. анализа / GEX / GEX Cone; остальное — 403.
    """
    user = seed_demo_user(db)
    if user.is_blocked:
        raise HTTPException(status_code=403, detail="Демо-доступ временно недоступен")
    access = create_access_token(user.id, user.email)
    refresh = create_refresh_token(user.id, user.email)
    logger.info("Demo login (id=%s)", user.id)
    return TokenOut(access_token=access, refresh_token=refresh)


# ================================================================= #
#  POST /auth/refresh
# ================================================================= #
@router.post("/refresh", response_model=TokenOut)
def refresh(body: RefreshIn, db: Session = Depends(get_session)):
    """Обновить access token через refresh token."""
    svc = AuthService(db)
    try:
        access, new_refresh = svc.refresh_tokens(body.refresh_token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    return TokenOut(
        access_token=access,
        refresh_token=new_refresh,
    )


# ================================================================= #
#  GET /auth/me  (с проверкой is_email_verified)
# ================================================================= #
@router.get("/me", response_model=UserOut)
def get_me(user: User = Depends(get_current_user)):
    """Профиль."""
    out = UserOut.model_validate(user)
    out.is_demo = is_demo_user(user)
    out.is_master_admin = is_master_admin(user)
    return out


# ================================================================= #
#  GET /auth/me/admin
# ================================================================= #
@router.get("/me/admin", response_model=UserProfileOut)
def get_me_admin(user: User = Depends(get_current_user), db: Session = Depends(get_session)):
    """Расширенный профиль (Master Admin видит статистику)."""
    profile = UserProfileOut.model_validate(user)
    profile.is_master_admin = is_master_admin(user)
    if is_master_admin(user):
        total = db.query(User).count()
        active = db.query(User).filter(User.subscription_status != "INACTIVE").count()
        profile.admin_stats = {"total_users": total, "active_subscriptions": active}
    return profile


# ================================================================= #
#  POST /auth/oauth/callback (stub)
# ================================================================= #
@router.post("/oauth/callback")
def oauth_callback(body: OAuthCallbackIn, db: Session = Depends(get_session)):
    """OAuth stub — обмен code на токен."""
    svc = AuthService(db)
    email = f"oauth_{body.provider}_{body.code[:8]}@example.com"
    try:
        user = svc.oauth_login_or_register(body.provider, email, body.code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    access = create_access_token(user.id, user.email)
    refresh = create_refresh_token(user.id)
    return {"access_token": access, "refresh_token": refresh, "token_type": "bearer"}


# ================================================================= #
#  GET /auth/logout
# ================================================================= #
@router.get("/logout", response_model=MessageOut)
def logout():
    """Выход."""
    return MessageOut(message="Выход выполнен. Удалите токены на клиенте.")


# ================================================================= #
#  Seed Master Admin
# ================================================================= #
def seed_master_admin(db: Session) -> None:
    """Создать Master Admin (реквизиты из настроек)."""
    if not settings.MASTER_PASSWORD:
        logger.error("MASTER_PASSWORD не задан — Master Admin не создан. Задайте в .env.")
        return
    email = settings.MASTER_EMAIL.strip().lower()
    existing = get_user_by_email(db, email)
    if existing:
        existing.subscription_status = SubscriptionStatus.ADMIN
        existing.is_email_verified = True
        existing.password_hash = hash_password(settings.MASTER_PASSWORD)
        db.commit()
        logger.info("Master Admin updated: %s", email)
        return
    user = User(
        id=str(_uuid.uuid4()),
        email=email,
        password_hash=hash_password(settings.MASTER_PASSWORD),
        is_email_verified=True,
        subscription_status=SubscriptionStatus.ADMIN,
    )
    db.add(user)
    db.commit()
    logger.info("Master Admin seeded: %s", email)
