@echo off
rem ============================================
rem  情绪指标 v4 - daily update
rem  refresh = fetch new data (the ONLY network step) + rebuild all panels
rem  Enc: ASCII-only + CRLF, safe for cmd.exe
rem ============================================
cd /d "%~dp0"

set SYS=
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.x with "Add to PATH" ticked.
    echo         Then run [首次初始化.bat] once, or just this bat after install.
    pause
    exit /b 1
  )
  for /f "delims=" %%i in ('where python') do (
    if not defined SYS set "SYS=%%i"
  )
  set "PY=%SYS%"
)

echo [1/2] Fetching new data and rebuilding all panels (about 5-10 min) ...
"%PY%" run.py refresh
if errorlevel 1 (
  echo [ERROR] refresh failed. See output above.
  pause
  exit /b 1
)

echo.
echo [2/2] Hit-rate stats:
"%PY%" run.py stats
echo.
echo Tip: if the web server is running, just reload the page -
echo      it picks up new data automatically via meta.built_at.
pause
