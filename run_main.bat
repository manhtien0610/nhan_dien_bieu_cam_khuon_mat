@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Chua co .venv! Chay: python -m venv .venv
    echo Sau do: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
.venv\Scripts\python.exe main.py
if errorlevel 1 pause
