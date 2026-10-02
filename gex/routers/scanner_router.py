"""Scanner: /scans, /scans/{ticker}, POST /scans/run."""
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from gex.deps import provide_scan_service
from gex.schemas import ScanRecordOut, ScanReportOut
from gex.adapters.notifications.telegram_sender import notify_scan_report, notify_scan_record

from fastapi import Depends
from gex.auth.dependencies import require_master_admin
router = APIRouter(dependencies=[Depends(require_master_admin)], tags=["scanner"])

def _record_to_schema(r) -> ScanRecordOut:
    return ScanRecordOut(ticker=r.ticker, scanned_at=r.scanned_at, status=r.status,
        ta_ok=r.ta_ok if hasattr(r, "ta_ok") else (r.status == "ok"),
        gex_ok=r.gex_ok if hasattr(r, "gex_ok") else (r.status == "ok"),
        error=r.error if hasattr(r, "error") else None)

@router.get("/scans", response_model=ScanReportOut)
def get_scans(notify: bool = Query(False), background_tasks: BackgroundTasks = None,
    svc=Depends(provide_scan_service)) -> ScanReportOut:
    report = svc.last_report(); records = svc.list_records()
    if report is None:
        rec_out = [_record_to_schema(r) for r in records]
        ok = sum(1 for r in rec_out if r.status == "ok")
        failed = sum(1 for r in rec_out if r.status == "error")
        out = ScanReportOut(scanned_at=datetime.now(timezone.utc),
            interval_hours=svc.interval_seconds / 3600.0, running=svc.is_running,
            total=len(svc.watchlist), ok=ok, failed=failed, records=rec_out)
    else:
        out = ScanReportOut(scanned_at=report.scanned_at, interval_hours=report.interval_hours,
            running=svc.is_running, total=report.total, ok=report.ok, failed=report.failed,
            records=[_record_to_schema(r) for r in report.records])
    if notify: background_tasks.add_task(notify_scan_report, out)
    return out

@router.get("/scans/{ticker}", response_model=ScanRecordOut)
def get_scan_ticker(ticker: str, notify: bool = Query(False),
    background_tasks: BackgroundTasks = None,
    svc=Depends(provide_scan_service)) -> ScanRecordOut:
    record = svc.get_latest(ticker)
    if record is None: raise HTTPException(404, f"Записей сканера для '{ticker.upper()}' не найдено.")
    out = _record_to_schema(record)
    if notify: background_tasks.add_task(notify_scan_record, out)
    return out

@router.post("/scans/run")
def run_scans_now(notify: bool = Query(False), background_tasks: BackgroundTasks = None,
    svc=Depends(provide_scan_service)):
    report = svc.scan_all(auto_notify=False)
    rec_out = [_record_to_schema(r) for r in report.records]
    out = ScanReportOut(scanned_at=report.scanned_at, interval_hours=report.interval_hours,
        running=svc.is_running, total=report.total, ok=report.ok, failed=report.failed, records=rec_out)
    if notify: background_tasks.add_task(notify_scan_report, out)
    return out
