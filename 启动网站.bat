@echo off
rem ============================================
rem  情绪指标 v4 - launch website (port 8779)
rem  Enc: ASCII-only + CRLF. cd to script dir,
rem  auto-detect python (.venv -> system).
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

"%PY%" run.py serve
pause
