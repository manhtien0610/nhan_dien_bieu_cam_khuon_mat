@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Chua co .venv! Chay: python -m venv .venv
    pause
    exit /b 1
)
.venv\Scripts\python.exe train.py %*
if errorlevel 1 pause
