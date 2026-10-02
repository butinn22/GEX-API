"""Runtime-настройки почты (SMTP) и Telegram-бота.

Хранятся в JSON-файле ``runtime_config.json`` в корне проекта (gitignored),
чтобы админ мог менять корпоративную почту/бота из админ-панели без
перезапуска сервера. Fallback — переменные из settings/.env.

Никаких внешних зависимостей: стандартный json + threading.Lock.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "runtime_config.json")

_lock = threading.Lock()

# Ключи, которые разрешено хранить/читать (email SMTP + telegram).
_EMAIL_KEYS = ("smtp_host", "smtp_port", "smtp_user", "smtp_pass", "from_email")
_TELEGRAM_KEYS = ("telegram_bot_token", "telegram_bot_username", "telegram_chat_id")
#: Ключ FinAgent (LLM API) в runtime_config.json — поверх OPENAI_COMPAT_API_KEY из .env.
_FINAGENT_KEY = "finagent_api_key"
#: Модель LLM — поверх FINAGENT_MODEL из .env.
_FINAGENT_MODEL = "finagent_model"
#: Base URL OpenAI-совместимого API — поверх OPENAI_COMPAT_BASE_URL из .env.
_FINAGENT_BASE_URL = "finagent_base_url"


def _load() -> dict[str, Any]:
    if not os.path.isfile(_CONFIG_PATH):
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("runtime_config: load failed: %s", exc)
        return {}


def _save(data: dict[str, Any]) -> None:
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.error("runtime_config: save failed: %s", exc)
        raise


def get_email_config() -> dict[str, Any]:
    """Текущая почтовая конфигурация (runtime поверх settings/.env)."""
    from .config import settings
    with _lock:
        data = _load()
    return {
        "smtp_host": data.get("smtp_host") or settings.SMTP_HOST,
        "smtp_port": int(data.get("smtp_port") or settings.SMTP_PORT or 0),
        "smtp_user": data.get("smtp_user") or settings.SMTP_USER,
        "smtp_pass": data.get("smtp_pass") or settings.SMTP_PASS,
        "from_email": data.get("from_email") or settings.FROM_EMAIL,
    }


def set_email_config(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str = "",
    smtp_pass: str = "",
    from_email: str = "",
) -> dict[str, Any]:
    """Сохранить почтовую конфигурацию (корпоративная почта)."""
    with _lock:
        data = _load()
        data["smtp_host"] = (smtp_host or "").strip()
        data["smtp_port"] = int(smtp_port or 0)
        data["smtp_user"] = (smtp_user or "").strip()
        data["smtp_pass"] = (smtp_pass or "")
        data["from_email"] = (from_email or "").strip()
        _save(data)
    logger.info("runtime_config: SMTP updated (host=%s, from=%s)", data["smtp_host"], data["from_email"])
    return get_email_config()


def get_telegram_config() -> dict[str, Any]:
    """Текущая конфигурация Telegram-бота (runtime поверх settings/.env)."""
    from .config import settings
    with _lock:
        data = _load()
    return {
        "bot_token": data.get("telegram_bot_token") or settings.TELEGRAM_BOT_TOKEN,
        "bot_username": data.get("telegram_bot_username") or settings.TELEGRAM_BOT_USERNAME,
        "chat_id": data.get("telegram_chat_id") or settings.TELEGRAM_CHAT_ID,
    }


def set_telegram_config(
    bot_token: str = "",
    bot_username: str = "",
    chat_id: str = "",
) -> dict[str, Any]:
    """Сохранить конфигурацию Telegram-бота."""
    with _lock:
        data = _load()
        data["telegram_bot_token"] = (bot_token or "").strip()
        data["telegram_bot_username"] = (bot_username or "").strip().lstrip("@")
        data["telegram_chat_id"] = (chat_id or "").strip()
        _save(data)
    logger.info("runtime_config: Telegram updated (username=@%s)", data["telegram_bot_username"])
    return get_telegram_config()


def _env_finagent_key() -> str:
    """Ключ из окружения как его видит FinAgent.

    finagent/config.py делает ``load_dotenv()`` при импорте и кэширует
    значение в константе ``OPENAI_COMPAT_API_KEY`` (роутер импортирует
    конфиг лениво, поэтому os.getenv в рантайме может быть пуст до
    первого LLM-вызова). Берём константу, инициируя её загрузку.
    """
    env_key = os.getenv("OPENAI_COMPAT_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        from finagent.config import OPENAI_COMPAT_API_KEY as _c
        return (_c or "").strip()
    except Exception:  # noqa: BLE001 - finagent не импортируется (тесты/изоляция)
        return ""


def get_finagent_key() -> str:
    """Действующий API-ключ FinAgent: runtime (JSON) поверх OPENAI_COMPAT_API_KEY (env)."""
    with _lock:
        data = _load()
    runtime = (data.get(_FINAGENT_KEY) or "").strip()
    if runtime:
        return runtime
    return _env_finagent_key()


def finagent_key_source() -> str:
    """Источник действующего ключа FinAgent: "runtime" | "env" | "none"."""
    with _lock:
        data = _load()
    if (data.get(_FINAGENT_KEY) or "").strip():
        return "runtime"
    if _env_finagent_key():
        return "env"
    return "none"


def set_finagent_key(api_key: str) -> dict[str, Any]:
    """Сохранить API-ключ FinAgent (пустая строка = вернуться к env)."""
    with _lock:
        data = _load()
        data[_FINAGENT_KEY] = (api_key or "").strip()
        _save(data)
    logger.info(
        "runtime_config: FinAgent key updated (runtime_set=%s)",
        bool(data[_FINAGENT_KEY]),
    )
    return {_FINAGENT_KEY: data[_FINAGENT_KEY]}


# ================================================================= #
#  FinAgent: модель + base URL (runtime поверх .env)
# ================================================================= #
def _env_finagent_model() -> str:
    """Модель из окружения: FINAGENT_MODEL → finagent.config (default deepseek-chat)."""
    m = os.getenv("FINAGENT_MODEL", "").strip()
    if m:
        return m
    try:
        from finagent.config import FINAGENT_MODEL as _c
        return (_c or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _env_finagent_base_url() -> str:
    """Base URL из окружения: OPENAI_COMPAT_BASE_URL → finagent.config."""
    u = os.getenv("OPENAI_COMPAT_BASE_URL", "").strip()
    if u:
        return u
    try:
        from finagent.config import OPENAI_COMPAT_BASE_URL as _c
        return (_c or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def get_finagent_model() -> str:
    """Действующая модель FinAgent: runtime (JSON) поверх FINAGENT_MODEL (.env)."""
    with _lock:
        data = _load()
    runtime = (data.get(_FINAGENT_MODEL) or "").strip()
    if runtime:
        return runtime
    return _env_finagent_model()


def get_finagent_base_url() -> str:
    """Действующий base URL FinAgent: runtime (JSON) поверх OPENAI_COMPAT_BASE_URL (.env)."""
    with _lock:
        data = _load()
    runtime = (data.get(_FINAGENT_BASE_URL) or "").strip()
    if runtime:
        return runtime
    return _env_finagent_base_url()


def finagent_model_source() -> str:
    """Источник модели: \"runtime\" | \"env\" | \"none\"."""
    with _lock:
        data = _load()
    if (data.get(_FINAGENT_MODEL) or "").strip():
        return "runtime"
    return "env" if _env_finagent_model() else "none"


def finagent_base_url_source() -> str:
    """Источник base URL: \"runtime\" | \"env\" | \"none\"."""
    with _lock:
        data = _load()
    if (data.get(_FINAGENT_BASE_URL) or "").strip():
        return "runtime"
    return "env" if _env_finagent_base_url() else "none"


def set_finagent_config(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> dict[str, Any]:
    """Сохранить FinAgent-конфигурацию (ключ + модель + base URL).

    Семантика значений:
      None  — runtime-значение не трогаем (поле не пришло от клиента);
      ""    — очистить runtime → фолбэк на .env;
      иначе — записать в runtime_config.json (поверх .env).
    """
    with _lock:
        data = _load()
        if api_key is not None:
            data[_FINAGENT_KEY] = (api_key or "").strip()
        if model is not None:
            data[_FINAGENT_MODEL] = (model or "").strip()
        if base_url is not None:
            data[_FINAGENT_BASE_URL] = (base_url or "").strip()
        _save(data)
    logger.info(
        "runtime_config: FinAgent config updated (key_set=%s, model_set=%s, url_set=%s)",
        bool(data.get(_FINAGENT_KEY)),
        bool(data.get(_FINAGENT_MODEL)),
        bool(data.get(_FINAGENT_BASE_URL)),
    )
    return finagent_state()


def finagent_state() -> dict[str, Any]:
    """Полное состояние FinAgent-конфигурации: эффективные значения + источники."""
    return {
        "finagent_api_key": get_finagent_key(),
        "model": get_finagent_model(),
        "base_url": get_finagent_base_url(),
        "key_source": finagent_key_source(),
        "model_source": finagent_model_source(),
        "base_url_source": finagent_base_url_source(),
    }


def masked(d: dict[str, Any]) -> dict[str, Any]:
    """Копия конфига с замаскированными секретами (для вывода в UI)."""
    out = dict(d)
    for k in ("smtp_pass", "bot_token", "finagent_api_key"):
        if out.get(k):
            out[k] = "••••" + str(out[k])[-4:]
    return out
