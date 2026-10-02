"""Job queue adapters (ring: adapters) — реализация JobQueuePort: Redis Streams (по умолчанию), RabbitMQ (при срабатывании триггеров).

Гарантия: at-least-once; handler идемпотентен по job.idempotency_key.
"""
