"""Admin router: управление пользователями (только Master Admin)."""

# --------------------------------------------------------------------------- #
#  Разбор файла (итерация 42)
# --------------------------------------------------------------------------- #
# 1 509 строк и 35 маршрутов разложены на `gex/auth/admin/*` по владельцу ресурса:
#   * users.py    — список, карточка, модерация, подписка, импорт/выгрузка;
#   * payments.py — платежи и реквизиты;
#   * notify.py   — конфигурация SMTP / Telegram / ключа finagent и пробное письмо;
#   * system.py   — статистика, метрики, логи, кэш, база;
#   * _shared.py  — общие помощники (права администратора, «пользователь или 404»).
#
# Помощники лежат в отдельном модуле, а не здесь: подмодули импортируют их из _shared,
# и если бы они остались в фасаде, получился бы цикл «фасад ↔ подмодули».
#
# Пути НЕ менялись: подроутеры не имеют префиксов, префикс по-прежнему объявлен здесь
# (`/auth/admin`). Это проверяет `tests/test_admin_routes.py` — инвентарь маршрутов
# по AST, работающий без fastapi (в окружении аудита fastapi нет, поэтому проверка
# статическая; она же видит и состояние до разбора, и после).
#
# Отдельно закреплено требование к порядку: `/users/export` обязан регистрироваться
# до `/users/{user_id}`, иначе FastAPI примет слово export за идентификатор.

from __future__ import annotations

from fastapi import APIRouter

from .admin import users
from .admin import payments
from .admin import notify
from .admin import system
from .admin._shared import _require_admin, _user_or_404

from .admin.users import (
    list_users,
    export_users,
    import_users,
    get_user,
    update_user,
    delete_user,
    moderate_user,
    admin_resend_email,
    admin_resend_telegram,
    admin_activate_subscription,
    admin_block_user,
    admin_unblock_user,
)
from .admin.payments import (
    export_payments,
    import_payments,
    list_payments,
    admin_confirm_payment,
    admin_reject_payment,
    admin_get_payment_requisites,
    admin_set_payment_requisites,
)
from .admin.notify import (
    get_email_config,
    set_email_config,
    get_telegram_config,
    set_telegram_config,
    get_finagent_key_config,
    set_finagent_key_config,
    admin_test_email,
)
from .admin.system import (
    get_admin_stats,
    get_system_health,
    get_metrics,
    get_system_metric_history,
    get_http_metrics_stats,
    get_recent_logs,
    get_db_metrics_stats,
    clear_cache,
    restart_redis,
    reset_db,
)


__all__ = [
    "_require_admin",
    "_user_or_404",
    "admin_activate_subscription",
    "admin_block_user",
    "admin_confirm_payment",
    "admin_get_payment_requisites",
    "admin_reject_payment",
    "admin_resend_email",
    "admin_resend_telegram",
    "admin_set_payment_requisites",
    "admin_test_email",
    "admin_unblock_user",
    "clear_cache",
    "delete_user",
    "export_payments",
    "export_users",
    "get_admin_stats",
    "get_db_metrics_stats",
    "get_email_config",
    "get_finagent_key_config",
    "get_http_metrics_stats",
    "get_metrics",
    "get_recent_logs",
    "get_system_health",
    "get_system_metric_history",
    "get_telegram_config",
    "get_user",
    "import_payments",
    "import_users",
    "list_payments",
    "list_users",
    "moderate_user",
    "reset_db",
    "restart_redis",
    "router",
    "set_email_config",
    "set_finagent_key_config",
    "set_telegram_config",
    "update_user",
]


router = APIRouter(prefix="/auth/admin", tags=["admin"])

# Порядок включения не влияет на разбор путей: конфликтов «литерал против параметра»
# между группами нет (динамические сегменты есть только внутри users и payments),
# а внутри группы исходный порядок сохранён — это и проверяет инвентарь маршрутов.
router.include_router(users.router)
router.include_router(payments.router)
router.include_router(notify.router)
router.include_router(system.router)
