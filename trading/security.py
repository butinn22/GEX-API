"""Security primitives: at-rest encryption (Fernet) and JWT tokens.

API keys are encrypted at rest with Fernet, never stored in plaintext. JWTs
authenticate API calls. The two use **separate keying material**:
``settings.secret_key`` signs JWTs, while ``settings.encryption_secret``
(``TRADING_BROKER_KEY_SECRET``, falling back to ``secret_key``) derives the
Fernet key — so a leaked signing key does not also expose stored broker
credentials. A real deployment should use a KMS (see ARCHITECTURE.md).
"""
from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timedelta, timezone

from cryptography.fernet import Fernet, InvalidToken
from jose import JWTError, jwt

__all__ = [
    "encrypt",
    "decrypt",
    "create_access_token",
    "decode_access_token",
    "SecretError",
]


class SecretError(Exception):
    """Raised when a secret cannot be decrypted/decoded (wrong key or tampered)."""


def _fernet(secret: str) -> Fernet:
    # Fernet needs a 32-byte urlsafe-b64 key; derive it deterministically from the secret.
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def encrypt(secret: str, plaintext: str) -> str:
    return _fernet(secret).encrypt(plaintext.encode()).decode()


def decrypt(secret: str, token: str) -> str:
    try:
        return _fernet(secret).decrypt(token.encode()).decode()
    except InvalidToken as exc:  # pragma: no cover - trivial
        raise SecretError("failed to decrypt secret (wrong key or tampered)") from exc


def create_access_token(secret: str, subject: str, *, expires_minutes: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_access_token(secret: str, token: str) -> str:
    """Return the subject of a valid token, else raise SecretError."""
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload["sub"]
    except (JWTError, KeyError) as exc:  # pragma: no cover - trivial
        raise SecretError("invalid or expired token") from exc
