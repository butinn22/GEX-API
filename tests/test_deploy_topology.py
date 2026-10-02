"""Deploy-контракт: у каждой очереди Celery есть потребитель в prod-compose.

Дефект, ради которого заведён набор (2026-09-25): очередь ``gex_market`` (снапшоты
страниц ``/breadth``, ``/sector``, ``/breadth-imoex``) не имела воркера в
``docker-compose.prod.yml`` — Beat и запросы исправно ставили задачи пересчёта,
а страницы навсегда оставались в состоянии «данные готовятся» (503). Ни один тест
этого не видел: очередь объявлена в коде, воркеры — в compose, а связи между ними
не проверял никто.

Правило: каждый Queue из ``gex.workers.celery_app.QUEUES`` обязан встречаться в
команде хотя бы одного сервиса prod-compose (профиль split не учитывается — базовый
профиль должен работать без флагов).
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _prod_compose() -> dict:
    text = (ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8-sig")
    return yaml.safe_load(text)


def test_every_celery_queue_has_a_worker_in_prod_compose():
    from gex.workers.celery_app import QUEUES

    compose = _prod_compose()
    commands = " ".join(
        str(svc.get("command", ""))
        for svc in compose.get("services", {}).values()
    )
    for queue in QUEUES:
        assert queue.name in commands, (
            f"очередь '{queue.name}' не потребляется ни одним воркером "
            "docker-compose.prod.yml: задачи будут копиться без исполнения"
        )


def test_beat_schedule_targets_only_known_queues():
    """Beat не должен класть задачи в очередь, которой нет в топологии."""
    from gex.workers.celery_app import QUEUES, _beat_schedule

    names = {q.name for q in QUEUES}
    for entry_name, entry in _beat_schedule().items():
        queue = (entry.get("options") or {}).get("queue")
        assert queue in names, (
            f"beat-запись '{entry_name}' шлёт в неизвестную очередь '{queue}'"
        )


def test_prod_compose_is_valid_yaml_with_expected_services():
    compose = _prod_compose()
    services = compose.get("services", {})
    assert "celery-beat" in services
    assert "celery-market" in services
    # Один beat на кластер: реплик быть не должно (синглтон обеспечивается
    # RedisLease внутри задач, но только ОДИН планировщик ставит работу).
    assert not services["celery-beat"].get("deploy", {}).get("replicas")


def test_systemd_units_cover_every_queue():
    """То же правило для bare-metal деплоя (deploy/systemd/).

    Шаблон gex-celery-scan@.service покрывает произвольную очередь (%i), но
    контракт — в инструкциях: если очереди нет в комментариях enable-листа
    или в ExecStart явного юнита, оператор её не поднимет — и задачи будут
    копиться без исполнения (реальный дефект 2026-09-25: gex_market).
    """
    from gex.workers.celery_app import QUEUES

    systemd_dir = ROOT / "deploy" / "systemd"
    units_text = "\n".join(
        p.read_text(encoding="utf-8-sig", errors="replace")
        for p in sorted(systemd_dir.glob("*.service"))
    )
    for queue in QUEUES:
        assert queue.name in units_text, (
            f"очередь '{queue.name}' не упомянута ни в одном юните deploy/systemd: "
            "оператор не сможет поднять воркера для неё"
        )
