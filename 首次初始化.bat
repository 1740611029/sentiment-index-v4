@echo off
rem ============================================
rem  情绪指标 v4 - first-time setup
rem  English/ASCII only + CRLF, safe for cmd.exe
rem  1) create .venv (if missing)  2) install deps
rem  3) rebuild panels. Any step fails -> stop.
rem ============================================
cd /d "%~dp0"

rem ---- Python discovery: prefer project .venv, else system python ----
set SYS=
if exist ".venv\Scripts\python.exe" (
  goto :have_venv
)
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found. Install Python 3.x and tick
  echo         "Add python.exe to PATH" (https://www.python.org/downloads/).
  pause
  exit /b 1
)
for /f "delims=" %%i in ('where python') do (
  if not defined SYS set SYS=%%i
)

echo [1/3] Creating virtual environment (.venv) ...
"%SYS%" -m venv .venv
if errorlevel 1 (
  echo [ERROR] Failed to create .venv.
  pause
  exit /b 1
)

:have_venv
set PY=.venv\Scripts\python.exe

echo [2/3] Installing dependencies (first run ~few minutes) ...
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Dependency install failed. Check your network.
  pause
  exit /b 1
)

echo [3/3] Rebuilding stock indicators and all panels (first run ~few minutes) ...
"%PY%" run.py build
if errorlevel 1 (
  echo [ERROR] Panel rebuild failed.
  pause
  exit /b 1
)

echo.
echo Done! Now double-click [启动网站.bat] to open the site, or
echo [更新数据.bat] to refresh data.
pause
