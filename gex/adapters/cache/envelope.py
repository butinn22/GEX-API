"""Конверт кэша: версия схемы, метка времени, etag и источник значения (ring: adapters).

Зачем конверт, а не «значение + ts»
-----------------------------------
В `gex/result_cache.py` результат хранился как ``{"v": …, "ts": …}``. Этого хватает,
чтобы посчитать возраст, но не хватает для трёх вещей, которые нужны дальше (итер. 31, 36):

1. **Версия схемы.** Изменение формы значения неотличимо от «старых данных»: читатель
   молча отдаёт payload, собранный прежним кодом. Версия в конверте делает это явным.
2. **etag.** Ревалидация (``If-None-Match`` → 304) должна отвечать «данные те же» без
   передачи тела. Сравнивать значения целиком дорого, сравнивать хэш — дёшево.
3. **Источник.** Провайдер, отдавший значение, — то же требование, что и в ключах
   (итер. 25), но здесь он нужен для диагностики и для UI («данные от yfinance, 12 минут»).

Что модуль **не** делает
------------------------
Он не владеет сериализацией байтов: слой Redis сериализует через ``serialize_value``
(pickle+zlib), а тесты подставляют свой сериализатор. Поэтому модуль не тянет ни pandas,
ни redis и проверяется без зависимостей.

Имя модуля: план называл его ``redis_json.py``, но фактический формат байтов — pickle+zlib,
а не JSON, поэтому модуль назван по сути (конверт), а не по формату.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Protocol

logger = logging.getLogger(__name__)

#: Версия формы конверта. Меняется при несовместимом изменении полей.
ENVELOPE_VERSION = 2

#: Версия «до конверта»: ``{"v": …, "ts": …}`` из result_cache. Читается, но не пишется.
LEGACY_VERSION = 0

#: Длина etag в hex-символах (64 бита достаточно для сравнения «изменилось ли»).
ETAG_LEN = 16

#: Префикс служебного поля: значения прикладных payload'ов могут содержать ``v``/``ts``,
#: поэтому форма конверта помечена отдельным ключом, а не угадывается по составу.
_MARKER = "__env__"

#: Сериализатор и десериализатор. Возвращаемое значение — ``object``, а не ``Any``:
#: десериализатор отдаёт «что-то неизвестное», и это ровно то, что проверяет
#: :meth:`Envelope.from_payload` через ``isinstance``.
Serializer = Callable[[object], object]
Deserializer = Callable[[object], object]


class RedisLike(Protocol):
    """Минимум, который нужен конверту (RedisClient и тестовый фейк совместимы)."""

    def get(self, key: str) -> object: ...
    def set(self, key: str, value: object, ex: Optional[int] = None) -> object: ...
    def delete(self, key: str) -> object: ...


def etag_for(raw: bytes) -> str:
    """etag по **сериализованному** значению: сравнивает байты, а не объекты.

    Берём именно байты, а не ``repr`` значения: repr двух разных типов может совпасть,
    а для ревалидации важно точное «payload тот же / не тот».
    """
    return hashlib.sha256(raw).hexdigest()[:ETAG_LEN]


@dataclass(frozen=True)
class Envelope:
    """Значение кэша с метаданными.

    * :attr:`ttl` — срок **свежести**, не срок жизни ключа в Redis (ключ живёт ``ttl*2``,
      чтобы устаревшее значение можно было отдать, пока считается новое);
    * :attr:`version` — версия формы; ``LEGACY_VERSION`` означает «прочитано из старой
      схемы», такие записи считаются устаревшими и перезаписываются в новом виде.
    """

    value: Any
    stored_at: float
    ttl: int
    version: int = ENVELOPE_VERSION
    etag: str = ""
    source: str = ""
    extra: dict = field(default_factory=dict)

    # -- возраст и свежесть ------------------------------------------------ #
    def age(self, now: Optional[float] = None) -> float:
        """Секунды с момента записи (может быть отрицательным при сдвиге часов)."""
        return (time.time() if now is None else now) - self.stored_at

    def is_fresh(self, now: Optional[float] = None) -> bool:
        """Свежее ли значение.

        ``ttl == 0`` означает «свежим не считается» — так помечаются записи прежней схемы
        (``LEGACY_VERSION``, срок неизвестен) и принудительно устаревшие (:meth:`to_stale`).
        Без этого правила запись с нулевым ttl была бы «свежей» ровно в тот же момент
        времени, и поведение зависело бы от разрешения часов.
        """
        if self.ttl <= 0 or self.is_legacy():
            return False
        return self.age(now) <= self.ttl

    def is_usable_stale(self, max_age: int, now: Optional[float] = None) -> bool:
        """Годится для отдачи «как есть» с фоновым пересчётом (SWR)."""
        return self.ttl < self.age(now) <= max_age

    def is_legacy(self) -> bool:
        return self.version < ENVELOPE_VERSION

    # -- сериализация ------------------------------------------------------ #
    def as_payload(self) -> dict:
        payload = {
            _MARKER: self.version,
            "v": self.value,
            "ts": self.stored_at,
            "ttl": self.ttl,
            "etag": self.etag,
        }
        if self.source:
            payload["src"] = self.source
        if self.extra:
            payload["extra"] = self.extra
        return payload

    @classmethod
    def from_payload(cls, payload: object) -> Optional["Envelope"]:
        """Разобрать payload; ``None`` — если это не конверт и не legacy-запись.

        Терпимость к legacy обязательна: в работающем Redis лежат записи прежней формы,
        и после выката читатель обязан их понимать (иначе «холодный кэш» на ровном месте).
        """
        if not isinstance(payload, dict):
            return None
        ts = payload.get("ts")
        if not isinstance(ts, (int, float)) or isinstance(ts, bool):
            return None

        version = payload.get(_MARKER)
        if version is None:
            # legacy {"v", "ts"} → считаем устаревшим (version=0, ttl=0)
            if "v" not in payload:
                return None
            return cls(value=payload["v"], stored_at=float(ts), ttl=0, version=LEGACY_VERSION)
        if not isinstance(version, int) or version > ENVELOPE_VERSION:
            # Конверт новее, чем умеет этот код: не гадаем о форме значения.
            logger.warning("Конверт версии %s не поддерживается — пропускаю", version)
            return None
        if "v" not in payload:
            return None
        ttl = payload.get("ttl")
        return cls(
            value=payload["v"],
            stored_at=float(ts),
            ttl=int(ttl) if isinstance(ttl, (int, float)) and not isinstance(ttl, bool) else 0,
            version=version,
            etag=str(payload.get("etag") or ""),
            source=str(payload.get("src") or ""),
            extra=payload.get("extra") if isinstance(payload.get("extra"), dict) else {},
        )

    def to_stale(self) -> "Envelope":
        """Копия, гарантированно считающаяся устаревшей (для принудительного пересчёта)."""
        return replace(self, ttl=0)


class RedisEnvelopeCache:
    """Чтение/запись конвертов в Redis (или в любой ``RedisLike``).

    Сериализатор инжектируется: продовый путь берёт ``serialize_value``/``deserialize_value``
    из ``gex.redis_client`` (pickle+zlib), тесты — что угодно дешёвое. Благодаря этому
    модуль проверяется без pandas и redis.
    """

    def __init__(
        self,
        redis: RedisLike,
        *,
        serialize: Optional[Serializer] = None,
        deserialize: Optional[Deserializer] = None,
        clock: Callable[[], float] = time.time,
        key_prefix: str = "",
    ) -> None:
        self._redis = redis
        self._clock = clock
        self._prefix = key_prefix
        if serialize is None or deserialize is None:
            from .serialize import default_deserializer, default_serializer

            serialize = serialize or default_serializer
            deserialize = deserialize or default_deserializer
        self._serialize = serialize
        self._deserialize = deserialize

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}" if self._prefix else key

    def read(self, key: str) -> Optional[Envelope]:
        try:
            raw = self._redis.get(self._k(key))
        except Exception as exc:  # noqa: BLE001 — Redis может быть недоступен
            logger.debug("Конверт не прочитан %s: %s", key, exc)
            return None
        if raw is None:
            return None
        try:
            payload = self._deserialize(raw)
        except Exception as exc:  # noqa: BLE001 — битые байты не должны ронять запрос
            logger.warning("Конверт %s не десериализуется: %s", key, exc)
            return None
        envelope = Envelope.from_payload(payload)
        if envelope is None:
            logger.warning("Конверт %s неизвестной формы — игнорирую", key)
        return envelope

    def write(self, key: str, value: object, ttl: int, *, source: str = "", **extra: object) -> Optional[Envelope]:
        """Записать значение. Ключ живёт ``ttl*2`` — чтобы устаревшее можно было отдать."""
        try:
            raw = self._serialize(value)
        except Exception as exc:  # noqa: BLE001 — несериализуемое значение не кэшируем
            logger.warning("Значение %s не сериализуется: %s", key, exc)
            return None
        envelope = Envelope(
            value=value,
            stored_at=self._clock(),
            ttl=int(ttl),
            etag=etag_for(raw if isinstance(raw, bytes) else str(raw).encode()),
            source=source,
            extra=dict(extra),
        )
        try:
            self._redis.set(self._k(key), self._serialize(envelope.as_payload()), ex=max(int(ttl), 1) * 2)
        except Exception as exc:  # noqa: BLE001 — запись кэша не критична
            logger.warning("Конверт %s не записан: %s", key, exc)
            return None
        return envelope

    def invalidate(self, key: str) -> bool:
        try:
            result = self._redis.delete(self._k(key))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Конверт %s не удалён: %s", key, exc)
            return False
        return bool(result) if result is not None else False


__all__ = [
    "ENVELOPE_VERSION",
    "Deserializer",
    "Serializer",
    "LEGACY_VERSION",
    "ETAG_LEN",
    "Envelope",
    "RedisLike",
    "RedisEnvelopeCache",
    "etag_for",
]
