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
echo       Do NOT close this window while it is running.
"%PY%" run.py refresh
rem  Why two tests: "if errorlevel 1" only catches exit codes >= 1.
rem  A hard crash (0xC0000005) or Ctrl-C (0xC000013A) shows up as a NEGATIVE
rem  errorlevel, so the old single test let a dead refresh pass as success --
rem  then step [2/2] died on missing panels. Second test catches that.
if errorlevel 1 goto :ref_fail
if not errorlevel 0 goto :ref_fail
goto :ref_ok

:ref_fail
echo.
echo [ERROR] refresh failed or was interrupted (exit code %ERRORLEVEL%).
echo         - If you pressed Ctrl-C or closed the window, just run this bat again.
echo         - If this PC never ran [first_setup.bat], run that once first.
pause
exit /b 1

:ref_ok

rem  data\panels* is derived data and is NOT shipped with the repo, so a
rem  previously interrupted run can leave it empty. Rebuild before stats.
if not exist "data\panels\SH.parquet" (
  echo [!] Panel cache missing - rebuilding it now ...
  "%PY%" run.py build
  if errorlevel 1 goto :build_fail
  if not errorlevel 0 goto :build_fail
)

echo.
echo [2/2] Hit-rate stats:
"%PY%" run.py stats
echo.
echo Tip: if the web server is running, just reload the page -
echo      it picks up new data automatically via meta.built_at.
pause
exit /b 0

:build_fail
echo.
echo [ERROR] Panel rebuild failed. See output above.
pause
exit /b 1
