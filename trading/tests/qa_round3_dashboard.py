"""QA Round-3 — INDEPENDENT dashboard XSS + HTML-error verification (B8/B9).

Injects hostile symbol/strategy values into a stored key's ``config_json`` and
proves the anonymous ``/API_KEY/{key}`` page renders them escaped (no stored
XSS), and that invalid/revoked keys render an HTML error page (not raw JSON).

Run explicitly::

    venv/Scripts/python.exe -m pytest trading/tests/qa_round3_dashboard.py -q
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest_asyncio

from trading.adapters.persistence import database as db
from trading.adapters.persistence.models import SignalKeyRow
from trading.main import app

SCRIPT = '"><script>alert(1)</script>'
IMG = "<img src=x onerror=alert(1)>"
AMP = "a&b<c>d"


@pytest_asyncio.fixture
async def client():
    await db.init_db()
    async with db._session_factory() as s:
        row = SignalKeyRow(
            key="sk_qa_xss",
            exchange="tbank",
            label="qa",
            active=True,
            config_json=json.dumps(
                {
                    "strategy": SCRIPT,
                    "tickers": [{"symbol": IMG}, {"symbol": AMP}],
                }
            ),
        )
        s.add(row)
        await s.commit()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    async with db._session_factory() as s:
        from sqlalchemy import delete

        await s.execute(delete(SignalKeyRow).where(SignalKeyRow.key == "sk_qa_xss"))
        await s.commit()


async def test_no_stored_xss_in_symbol_and_strategy(client):
    r = await client.get("/API_KEY/sk_qa_xss")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    text = r.text
    # raw injection payloads must NOT survive
    assert SCRIPT not in text, "raw <script> payload was echoed unescaped"
    assert IMG not in text, "raw <img onerror> payload was echoed unescaped"
    assert "<script>alert(1)</script>" not in text
    # escaped forms must be present
    assert "&lt;img src=x onerror=alert(1)&gt;" in text, "symbol was not HTML-escaped"
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text, "strategy was not HTML-escaped"
    assert "a&amp;b&lt;c&gt;d" in text, "ampersand/angle brackets not escaped"


async def test_refresh_interval_is_60s(client):
    r = await client.get("/API_KEY/sk_qa_xss")
    assert "setInterval(() => refresh(true), 60000)" in r.text
    assert "every 15 s" not in r.text


async def test_unknown_key_is_html_404(client):
    r = await client.get("/API_KEY/sk_missing_xyz")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("text/html"), "raw JSON leaked to browser"
    assert r.text.lstrip().startswith("<!DOCTYPE html>")
    assert "unknown API key" in r.text
    assert '{"detail"' not in r.text


async def test_revoked_key_is_html_410(client):
    async with db._session_factory() as s:
        row = SignalKeyRow(
            key="sk_qa_revoked",
            exchange="tbank",
            label="qa",
            active=True,
            revoked_at=datetime.now(UTC),
            config_json="{}",
        )
        s.add(row)
        await s.commit()
    try:
        r = await client.get("/API_KEY/sk_qa_revoked")
        assert r.status_code == 410
        assert r.headers["content-type"].startswith("text/html")
        assert r.text.lstrip().startswith("<!DOCTYPE html>")
        assert "revoked" in r.text
        assert '{"detail"' not in r.text
    finally:
        async with db._session_factory() as s:
            from sqlalchemy import delete

            await s.execute(
                delete(SignalKeyRow).where(SignalKeyRow.key == "sk_qa_revoked")
            )
            await s.commit()
