@echo off
rem ============================================
rem  情绪指标 v4 - daily update
rem  refresh = fetch new data (the ONLY network step) + rebuild all panels
rem  Encoding: ASCII only + CRLF, safe for cmd.exe
rem ============================================
cd /d D:\情绪指标4

set PY=C:\Users\wr\.workbuddy-ai\binaries\python\envs\default\Scripts\python.exe

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
