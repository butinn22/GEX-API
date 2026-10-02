"""GEX (Gamma Exposure) analytic engine.

Модульный пакет для микроструктурного анализа опционных рынков.
Все публичные классы/функции импортируются напрямую из подмодулей:
  ``from gex.application.service import GEXService``
  ``from gex.domain.data_loader import GEXDataLoader``
  и т.д.

Реэкспорты в этом ``__init__`` удалены для предотвращения циклических
зависимостей и ускорения импорта.
"""
