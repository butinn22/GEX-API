"""Status endpoint for offloaded tasks (the polling half of ``202 Accepted``).

``POST`` endpoints that may exceed ``CELERY_WAIT_TIMEOUT`` answer ``202`` with a
``task_id``; the client polls this route until ``ready`` is true. States are
Celery's: ``PENDING`` -> ``STARTED`` -> ``SUCCESS`` / ``FAILURE`` / ``RETRY``.

``PENDING`` is ambiguous by nature: it means "not finished", which covers both
"queued" and "unknown id" (e.g. the result already expired). Clients must treat a
long-lived ``PENDING`` as a timeout, not as a definitive failure.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from gex.auth.dependencies import require_subscription
from gex.auth.models import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tasks"])

#: Same access tier as the endpoints that create these tasks.
require_extended = require_subscription("EXTENDED")


@router.get("/tasks/{task_id}")
def get_task(task_id: str, user: User = Depends(require_extended)) -> JSONResponse:
    """Current state and (when finished) result of a background task.

    Celery is imported lazily: it is an optional dependency and must never break
    application startup (the same rule the web/worker role split follows).
    """
    from celery.result import AsyncResult

    from gex.workers.celery_app import app

    async_result = AsyncResult(task_id, app=app)

    payload: dict[str, object] = {
        "task_id": task_id,
        "state": async_result.state,
        "ready": async_result.ready(),
        "successful": async_result.successful(),
        "failed": async_result.failed(),
    }

    headers = {}
    if async_result.ready():
        if async_result.successful():
            payload["result"] = async_result.get(propagate=False)
        else:
            payload["error"] = str(async_result.result)
            logger.warning("Задача %s завершилась ошибкой: %s", task_id, async_result.result)
    else:
        headers["Retry-After"] = "2"

    return JSONResponse(content=payload, headers=headers)
