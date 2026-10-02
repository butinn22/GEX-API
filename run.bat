@echo off
REM ===================================================================
REM  GEX Trading API - one-click launcher
REM  Double-click this file, or use the desktop shortcut.
REM  Pass through flags such as:  run.bat --reload   /   run.bat --port 9000
REM ===================================================================
setlocal
cd /d "%~dp0"

title GEX Trading API

set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo.
  echo   [!] Python virtual environment not found:
  echo       %PY%
  echo.
  echo   Create it once from this folder, then re-run:
  echo       uv venv .venv
  echo       uv pip install --python .venv -e ".[dev]"
  echo.
  pause
  exit /b 1
)

"%PY%" "%~dp0scripts\serve.py" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo   Server exited with code %RC%. See the messages above.
  echo.
  pause
)

endlocal & exit /b %RC%
