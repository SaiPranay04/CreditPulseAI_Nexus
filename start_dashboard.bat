@echo off
title CreditPulse AI — Dashboard

echo.
echo  ==========================================
echo   CreditPulse AI  ^|  IDBI Innovate 2026
echo  ==========================================
echo.

REM ── Start FastAPI backend in a new window
echo [1/2] Starting FastAPI backend on http://localhost:8000 ...
start "CreditPulse API" cmd /k "cd /d %~dp0 && venv\Scripts\python -m uvicorn app.api.main:app --reload --port 8000 --host 0.0.0.0"

REM ── Wait 3 seconds for API to initialize
timeout /t 3 /nobreak >nul

REM ── Start Next.js dev server in a new window
echo [2/2] Starting Next.js frontend on http://localhost:3000 ...
start "CreditPulse Web" cmd /k "cd /d %~dp0web && npm run dev"

REM ── Wait then open browser
timeout /t 5 /nobreak >nul
echo.
echo  Dashboard ready:
echo   Frontend  →  http://localhost:3000
echo   API Docs  →  http://localhost:8000/docs
echo.
start http://localhost:3000

echo  Both servers are running in separate windows.
echo  Close those windows to stop them.
echo.
pause
