"""One-click dev launcher for the GEX Trading API.

Starts uvicorn, waits until ``/health`` answers, then opens the browser. Used by
``run.bat`` and the desktop shortcut, but it is also a perfectly good CLI:

    .venv/Scripts/python.exe scripts/serve.py                 # port 8000, opens browser
    .venv/Scripts/python.exe scripts/serve.py --reload        # auto-reload on edits
    .venv/Scripts/python.exe scripts/serve.py --port 9000     # pick a starting port
    .venv/Scripts/python.exe scripts/serve.py --no-browser    # headless / CI

Why a Python launcher instead of a bare ``uvicorn`` call:

* The app resolves ``trading.db`` **relative to the cwd**, so the working
  directory must be the repo root no matter where the shortcut is launched from.
* If port 8000 is already taken (a stale server, another project) the launcher
  walks forward to the next free port instead of dying with a confusing error.
* Opening the browser *before* the server is ready shows a connection error, so we
  poll ``/health`` first and only then hand the URL to the OS.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

#: Repo root = the parent of ``scripts/``. Everything below is anchored to it.
ROOT = Path(__file__).resolve().parents[1]

#: How many consecutive ports to try before giving up.
_PORT_SCAN = 25

BANNER = r"""
  ____ _____ __  __   ___  ____  ___
 / ___| ____|\ \/ /  / _ \|  _ \_ _|
| |  _|  _|   \  /  | |_| | |_) || |
| |_| | |___  /  \  |  _  |  __/ | |
 \____|_____|/_/\_\ |_| |_|_|   |___|

  GEX Trading API
  UI        {ui}
  Swagger   {docs}
  Health    {health}

  Press Ctrl+C to stop.
"""


def _port_is_free(host: str, port: int) -> bool:
    """True if nothing is listening on ``port``.

    Deliberately a plain ``bind()`` with **no** ``SO_REUSEADDR``: on Windows that
    option lets you bind a port another process is already listening on, which
    would make this probe always report "free" and hand us a port that uvicorn
    then fails to take. On Linux/macOS a plain bind is the truthful test too
    (TIME_WAIT sockets are not listening and don't block a fresh server).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def free_port(host: str, preferred: int, scan: int = _PORT_SCAN) -> int:
    """Return ``preferred`` if free, else the next free port within ``scan``."""
    for port in range(preferred, preferred + max(scan, 1)):
        if _port_is_free(host, port):
            return port
    raise SystemExit(
        f"[serve] no free port in {preferred}..{preferred + scan}. "
        "Stop something, or pass --port with a different starting point."
    )


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.0) as resp:  # noqa: S310 - localhost only
            return 200 <= resp.status < 400
    except Exception:
        return False


def _open_browser_when_ready(page: str, health: str, timeout: float = 90.0) -> None:
    """Poll ``health`` then open ``page`` — in a daemon thread, so it never blocks."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _healthy(health):
            try:
                webbrowser.open(page)
            except Exception:
                pass  # a headless box just doesn't get a browser tab
            return
        time.sleep(0.3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serve.py",
        description="Run the GEX Trading API and open the browser.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="starting port (default: 8000)")
    parser.add_argument("--scan", type=int, default=_PORT_SCAN, help="ports to try if busy")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="uvicorn log level (default: info)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # The shortcut may start us from anywhere; the app expects the repo root
    # (relative SQLite path, `trading/` imports, `static/` files).
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.environ.setdefault("PYTHONUTF8", "1")

    port = free_port(args.host, args.port, args.scan)
    if port != args.port:
        print(f"[serve] port {args.port} is busy -> using {port}", flush=True)

    base = f"http://{args.host}:{port}"
    health = f"{base}/health"

    if not args.no_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(f"{base}/", health),
            name="open-browser",
            daemon=True,
        ).start()

    print(BANNER.format(ui=f"{base}/", docs=f"{base}/docs", health=health), flush=True)

    import uvicorn  # imported late so `--help` stays fast and dependency-free

    try:
        uvicorn.run(
            "trading.main:app",
            host=args.host,
            port=port,
            reload=args.reload,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\n[serve] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
