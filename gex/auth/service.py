"""Auth business logic: register, login, verify, OAuth, JWT, Master Admin."""
from __future__ import annotations

import hmac
import re
import sys
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

# Очищаем пути uv-кэша из sys.path (конфликтует с python-jose)
sys.path = [p for p in sys.path if '\\\\cache\\\\' not in p and '/cache/' not in p]

import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from .config import settings
from .models import User, SubscriptionStatus, SUBSCRIPTION_ORDER
from .schemas import validate_telegram_username


_PASSWORD_RE_LETTER = re.compile(r"[A-Za-zА-Яа-яЁё]")
_PASSWORD_RE_DIGIT = re.compile(r"\d")


def validate_password_policy(password: str) -> None:
    """Проверка требований к паролю: >=8 символов, буква и цифра."""
    if len(password) < 8:
        raise ValueError("Пароль должен содержать не менее 8 символов")
    if not _PASSWORD_RE_LETTER.search(password):
        raise ValueError("Пароль должен содержать хотя бы одну букву")
    if not _PASSWORD_RE_DIGIT.search(password):
        raise ValueError("Пароль должен содержать хотя бы одну цифру")


# ================================================================= #
#  Хэширование пароля
# ================================================================= #
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


# ================================================================= #
#  JWT
# ================================================================= #
def is_demo_user(user: User) -> bool:
    """Demo-аккаунт (кнопка «Просмотр демо») — по email (регистронезависимо)."""
    email = (getattr(user, "email", "") or "").strip().lower()
    return bool(email) and email == settings.DEMO_EMAIL.strip().lower()


def _demo_claim(email: str) -> dict:
    """В JWT demo-пользователя кладём claim demo=1: middleware ограничивает
    его ВСЕХ data-ручках только allowlist-ом (просмотр тикера ES)."""
    return {"demo": 1} if (email or "").strip().lower() == settings.DEMO_EMAIL.strip().lower() else {}


def create_access_token(user_id: str, email: str) -> str:
    import uuid
    payload = {
        "sub": user_id,
        "email": email,
        "type": "access",
        "jti": str(uuid.uuid4()),
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + settings.ACCESS_TOKEN_EXPIRE,
        **_demo_claim(email),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user_id: str, email: str = "") -> str:
    import uuid
    payload = {
        "sub": user_id,
        "type": "refresh",
        "jti": str(uuid.uuid4()),
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + settings.REFRESH_TOKEN_EXPIRE,
        **_demo_claim(email),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Декодировать JWT. Возвращает payload или кидает JWTError."""
    payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    return payload


# ================================================================= #
#  User CRUD
# ================================================================= #
def get_user_by_email(db: Session, email: str) -> Optional[User]:
    email = email.strip().lower()
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_telegram_username(db: Session, telegram_username: str) -> Optional[User]:
    """Найти пользователя по Telegram-нику (регистронезависимо)."""
    nick = telegram_username.strip().lower()
    return db.query(User).filter(User.telegram_username.ilike(nick)).first()


def get_telegram_start_chat_id(db: Session, telegram_username: str) -> Optional[str]:
    """Вернуть chat_id пользователя, если он уже нажимал /start боту.

    Ищет по публичному username (без @, в нижнем регистре) в таблице
    telegram_starts. Позволяет писать пользователю в Telegram даже до
    активации аккаунта на сайте.
    """
    from .models import TelegramStart
    key = telegram_username.strip().lstrip("@").lower()
    if not key:
        return None
    row = db.query(TelegramStart).filter(TelegramStart.username == key).first()
    return row.chat_id if row else None


def record_telegram_start(db: Session, username: str, chat_id: str) -> None:
    """Записать/обновить факт нажатия /start боту (username → chat_id)."""
    from .models import TelegramStart
    key = (username or "").strip().lstrip("@").lower()
    if not key or not chat_id:
        return
    row = db.query(TelegramStart).filter(TelegramStart.username == key).first()
    if row:
        row.chat_id = chat_id
    else:
        db.add(TelegramStart(username=key, chat_id=chat_id))
    db.commit()


def is_master_admin(user: User) -> bool:
    """Master Admin проверка — по email (регистронезависимо)."""
    return user.email.strip().lower() == settings.MASTER_EMAIL.strip().lower()


# ================================================================= #
#  Auth Service
# ================================================================= #
class AuthService:
    """Единый сервис аутентификации."""

    def __init__(self, db: Session):
        self.db = db

    # ---------------------- Email Registration ---------------------- #
    def register(self, email: str, password: str, telegram_username: str) -> User:
        """Регистрация: email + пароль + обязательный Telegram-ник.

        Возвращает User с is_email_verified=False (активация через Telegram).
        """
        email = email.strip().lower()
        telegram_username = validate_telegram_username(telegram_username)

        # Требования к паролю (дублируем схему — защита прямых вызовов)
        validate_password_policy(password)

        existing = get_user_by_email(self.db, email)
        if existing:
            raise ValueError("Пользователь с таким email уже существует")

        # Telegram-ник не должен быть занят другим аккаунтом
        tg_user = get_user_by_telegram_username(self.db, telegram_username)
        if tg_user:
            raise ValueError("Этот Telegram-ник уже привязан к другому аккаунту")

        # Master Admin нельзя создать через регистрацию
        if email == settings.MASTER_EMAIL.strip().lower():
            raise ValueError("Этот email зарезервирован")

        # Demo-аккаунт тоже служебный — только через POST /auth/demo
        if email == settings.DEMO_EMAIL.strip().lower():
            raise ValueError("Этот email зарезервирован")

        user = User(
            id=str(uuid.uuid4()),
            email=email,
            password_hash=hash_password(password),
            is_email_verified=False,
            subscription_status=SubscriptionStatus.INACTIVE,
            telegram_username=telegram_username,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    # ---------------------- Email Login ---------------------- #
    def login(self, email: str, password: str) -> User:
        """Логин по email+password. Возвращает User или кидает ValueError."""
        email = email.strip().lower()
        user = get_user_by_email(self.db, email)
        if not user or not user.password_hash:
            raise ValueError("Неверный email или пароль")
        if not check_password(password, user.password_hash):
            raise ValueError("Неверный email или пароль")
        return user

    # ---------------------- Master Admin (backdoor) ---------------------- #
    def master_admin_login(self, password: str) -> User:
        """Специальный вход для Master Admin с email=sadisting."""
        email = settings.MASTER_EMAIL.strip().lower()
        # AUTH-08: сравнение в постоянном времени. Оператор `!=` прекращает сравнение на первом
        # несовпавшем байте и по времени ответа выдаёт длину общего префикса (утечка секрета).
        # Пустой/незаданный MASTER_PASSWORD теперь отклоняет вход целиком (fail-closed), иначе
        # POST с пустым паролем проходил как master-вход.
        if not settings.MASTER_PASSWORD or not hmac.compare_digest(
            str(password), str(settings.MASTER_PASSWORD)
        ):
            raise ValueError("Неверный email или пароль")

        user = get_user_by_email(self.db, email)
        if not user:
            raise ValueError("Master Admin не найден. Выполните seed.")

        # Гарантируем ADMIN + verified
        if user.subscription_status != SubscriptionStatus.ADMIN:
            user.subscription_status = SubscriptionStatus.ADMIN
        if not user.is_email_verified:
            user.is_email_verified = True
        self.db.commit()
        self.db.refresh(user)
        return user

    # ---------------------- Email Verification ---------------------- #
    def verify_email(self, user: User) -> User:
        """Подтвердить email (упрощённо: без кода, просто эндпоинт)."""
        user.is_email_verified = True
        self.db.commit()
        self.db.refresh(user)
        return user

    # ---------------------- OAuth ---------------------- #
    def oauth_login_or_register(self, provider: str, email: str, oauth_id: str) -> User:
        """OAuth вход: найти или создать пользователя. Email считается verified."""
        email = email.strip().lower()
        user = get_user_by_email(self.db, email)
        if user:
            # Обновить OAuth-данные
            user.oauth_provider = provider
            user.oauth_id = oauth_id
            user.is_email_verified = True
        else:
            # Создать нового пользователя
            user = User(
                id=str(uuid.uuid4()),
                email=email,
                is_email_verified=True,
                subscription_status=SubscriptionStatus.INACTIVE,
                oauth_provider=provider,
                oauth_id=oauth_id,
            )
            self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    # ---------------------- Token Refresh ---------------------- #
    def refresh_tokens(self, refresh_token: str) -> tuple[str, str]:
        """Проверить refresh_token и выдать новую пару токенов."""
        try:
            payload = decode_token(refresh_token)
        except JWTError:
            raise ValueError("Недействительный или просроченный refresh token")

        if payload.get("type") != "refresh":
            raise ValueError("Неверный тип токена")

        user_id = payload.get("sub")
        user = get_user_by_id(self.db, user_id)
        if not user:
            raise ValueError("Пользователь не найден")
        if user.is_blocked:
            raise ValueError("Аккаунт заблокирован")
        # Email-гейт: обновлять сессию может только подтверждённый пользователь
        # (иначе регистрация на чужой email давала бы бессрочный доступ через refresh).
        if not user.is_email_verified:
            raise ValueError("Email не подтверждён. Подтвердите email, чтобы продолжить.")

        access = create_access_token(user.id, user.email)
        refresh = create_refresh_token(user.id, user.email)
        return access, refresh

    # ---------------------- User Info ---------------------- #
    def get_user_profile(self, user: User) -> dict:
        return {
            "id": user.id,
            "email": user.email,
            "is_email_verified": user.is_email_verified,
            "subscription_status": user.subscription_status,
            "oauth_provider": user.oauth_provider,
            "is_master_admin": is_master_admin(user),
            "created_at": user.created_at.isoformat() if user.created_at else None,
        }


# ================================================================= #
#  Barrier helpers (используются в dependencies.py)
# ================================================================= #
def check_subscription(user: User, required: str) -> bool:
    """True если подписка пользователя >= требуемого уровня."""
    user_level = SUBSCRIPTION_ORDER.get(user.subscription_status, -1)
    required_level = SUBSCRIPTION_ORDER.get(required, 999)
    return user_level >= required_level


def can_bypass_barriers(user: User) -> bool:
    """Master Admin обходит все барьеры."""
    return is_master_admin(user)
