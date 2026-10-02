"""Аренда на Redis: «это делает ровно один процесс» (ring: adapters).

Зачем отдельный модуль
----------------------
Одну и ту же задачу «выполнить один раз на кластер» решают три места: слоты прогрева
(итер. 32), право владения сканером (итер. 33) и лидер пересчёта страницы (итер. 26).
У каждого была (или была бы) своя копия `SET NX PX` — с разными ошибками: где-то без NX,
где-то с освобождением чужой аренды по простому `DEL`.

Два свойства, которые здесь принципиальны
----------------------------------------
1. **Захват атомарен** (``SET key token NX PX``). Проверка ``GET`` + ``SET`` не атомарна:
   между ними вклинится другой воркер, и оба решат, что они лидеры — это и был дефект
   прогрева (слот публиковался каждой репликой).
2. **Освобождается только своя аренда.** ``DEL`` по ключу снял бы аренду, которую уже
   перехватил другой процесс (своя истекла, он взял новую) — и тогда слот выполнится дважды.
   Поэтому ``release`` сравнивает токен.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Минимальная аренда: меньше секунды — смысла нет (лидер не успеет ничего сделать).
MIN_TTL_S = 1


class RedisLease:
    """Именованные аренды поверх Redis (``SET NX PX``)."""

    def __init__(self, redis: Any, *, prefix: str = "gex:lease:") -> None:
        self._redis = redis
        self._prefix = prefix
        self.held = 0
        self.denied = 0

    def key(self, name: str) -> str:
        return f"{self._prefix}{name}"

    def acquire(self, name: str, ttl_s: int) -> Optional[str]:
        """Взять аренду. Возвращает токен или ``None``, если она занята.

        Токен нужен не для красоты: освобождать можно только своё (см. модульный docstring).
        """
        if self._redis is None:
            return None
        token = uuid.uuid4().hex
        ttl_ms = max(int(ttl_s), MIN_TTL_S) * 1000
        try:
            ok = self._redis.set(self.key(name), token, nx=True, px=ttl_ms)
        except Exception as exc:  # noqa: BLE001 — Redis недоступен
            logger.warning("Аренда %s не взята (Redis недоступен): %s", name, exc)
            return None
        if not ok:
            self.denied += 1
            return None
        self.held += 1
        return token

    def release(self, name: str, token: Optional[str]) -> bool:
        """Освободить **свою** аренду (по токену). ``False`` — аренда уже не наша."""
        if self._redis is None or token is None:
            return False
        try:
            current = self._redis.get(self.key(name))
            if current is None:
                return False
            if isinstance(current, bytes):
                current = current.decode("utf-8", errors="replace")
            if current != token:
                # Аренда успела истечь и её перехватил другой процесс: снимать нельзя.
                logger.debug("Аренда %s принадлежит другому процессу — не освобождаю", name)
                return False
            return bool(self._redis.delete(self.key(name)))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Аренда %s не освобождена: %s", name, exc)
            return False

    def renew(self, name: str, token: Optional[str], ttl_s: int) -> bool:
        """Продлить **свою** аренду. ``False`` — аренда уже не наша (истекла и перехвачена).

        Нужна для длинных операций: полный прогон сканера идёт минутами и легко переживёт
        исходный TTL. Без продления владелец потерял бы аренду посреди работы, и её забрал
        бы другой процесс — то есть задача пошла бы параллельно сама с собой.
        """
        if self._redis is None or token is None:
            return False
        try:
            current = self._redis.get(self.key(name))
            if current is None:
                return False
            if isinstance(current, bytes):
                current = current.decode("utf-8", errors="replace")
            if current != token:
                return False
            return bool(self._redis.set(self.key(name), token, px=max(int(ttl_s), MIN_TTL_S) * 1000, nx=False))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Аренда %s не продлена: %s", name, exc)
            return False

    def is_held(self, name: str) -> bool:
        """Занята ли аренда (для админки и диагностики)."""
        if self._redis is None:
            return False
        try:
            return self._redis.get(self.key(name)) is not None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Состояние аренды %s недоступно: %s", name, exc)
            return False

    def describe(self) -> dict:
        return {"held": self.held, "denied": self.denied, "available": self._redis is not None}


__all__ = ["MIN_TTL_S", "RedisLease"]
