"""Background fetcher routes: /fetcher/prewarm, /fetcher/status."""
from fastapi import APIRouter

from gex.application.scheduler import trigger_prewarm
from gex.application.jobs import QUEUE_KINDS
from gex.deps import get_task_queue

from fastapi import Depends
from gex.auth.dependencies import require_master_admin
router = APIRouter(dependencies=[Depends(require_master_admin)], prefix="/fetcher", tags=["fetcher"])


@router.post("/prewarm")
def post_prewarm() -> dict:
    return trigger_prewarm()


@router.get("/status")
def get_fetcher_status() -> dict:
    q = get_task_queue()
    return {
        "queue_length": q.queue_length(),
        "queues": sorted(QUEUE_KINDS),
    }
