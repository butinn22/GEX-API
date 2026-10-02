"""Конус вероятностей: модель, экспирации, вероятности и путь (ring: domain).

Разложено из ``gex/gexcone.py`` (1 114 строк, итерация 40). Точка входа прежняя —
``gex.gexcone.build_gex_cone``; наружные импорты не менялись, числа закреплены эталоном
``tests/test_cone_golden.py``: три конфигурации (в том числе ``wall_decay=0`` и
``oi_quantile=1.0`` — границы), лестница вероятностей по каждому уровню, путь из 18 точек.

* ``model``         — константы, структуры данных, общие примитивы;
* ``expiries``      — отбор цепочки и построение экспираций с уровнями;
* ``probabilities`` — логнормальная база, поправка на стены, лестница вероятностей;
* ``path``          — путь конуса и глобальные уровни.
"""

from .model import ConeExpiry, ConeExpiryLevel, GexConeData, GlobalLevel

__all__ = ["ConeExpiry", "ConeExpiryLevel", "GexConeData", "GlobalLevel"]
