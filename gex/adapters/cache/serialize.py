"""Сериализация значений кэша без обязательной зависимости от pandas/redis (ring: adapters).

Продовый путь — ``serialize_value`` / ``deserialize_value`` из :mod:`gex.redis_client`:
это pickle+zlib с терпимостью к записям старых форматов. Но модуль ``redis_client`` тянет
pandas, поэтому конверт и single-flight не должны импортировать его напрямую — иначе их
нельзя проверить без внешних пакетов, а именно это требование к набору тестов.

Пара сериализаторов **разрешается один раз и целиком**: смешивать продовый сериализатор
(с zlib) с запасным десериализатором (без zlib) нельзя — чтение таких байтов упадёт.
"""

from __future__ import annotations

import pickle
from typing import Callable, Optional, Tuple

#: Пара «сериализовать / разобрать». Тип значения — ``object``: сериализатор принимает
#: что угодно, десериализатор возвращает неизвестное (проверяется ``isinstance`` у вызывающего).
Pair = Tuple[Callable[[object], object], Callable[[object], object]]

_pair: Optional[Pair] = None


def resolve_serializers() -> Pair:
    """(serialize, deserialize) — из ``gex.redis_client``, если он доступен, иначе pickle."""
    global _pair
    if _pair is None:
        try:
            from gex.adapters.cache.redis_client import deserialize_value, serialize_value

            _pair = (serialize_value, deserialize_value)
        except ImportError:
            _pair = (pickle.dumps, pickle.loads)
    return _pair


def default_serializer(value: object) -> object:
    return resolve_serializers()[0](value)


def default_deserializer(raw: object) -> object:
    return resolve_serializers()[1](raw)


def reset_cache() -> None:
    """Сбросить запомненную пару (для тестов)."""
    global _pair
    _pair = None


__all__ = ["Pair", "default_deserializer", "default_serializer", "reset_cache", "resolve_serializers"]
