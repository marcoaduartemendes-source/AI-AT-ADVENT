@echo off
REM One-shot setup for Windows.
REM Run from the project root: setup.bat

echo.
echo === Crypto Investment Bot - Local Setup ===
echo.

REM 1. Find Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [X] Python not found. Install it from https://www.python.org/downloads/
    echo     During install, check "Add Python to PATH".
    exit /b 1
)
echo [OK] Python detected
python --version

REM 2. Create virtualenv
if not exist "venv" (
    python -m venv venv
    echo [OK] Created virtualenv .\venv
) else (
    echo [OK] Virtualenv .\venv already exists
)
call venv\Scripts\activate.bat
echo [OK] Activated virtualenv

REM 3. Install requirements
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
echo [OK] Installed dependencies from requirements.txt

REM 4. Create .env from template if missing
if not exist ".env" (
    copy /Y .env.example .env >nul
    echo [OK] Created .env from .env.example
) else (
    echo [!] .env already exists - not overwriting
)

REM 5. Make data dir
if not exist "data" mkdir data
echo [OK] Data directory ready

echo.
echo ======================================================
echo   NEXT STEPS
echo ======================================================
echo.
echo   1. Open .env in Notepad and paste your Coinbase keys
echo      on lines 24 and 25:
echo         notepad .env
echo.
echo   2. Validate setup:
echo         venv\Scripts\activate.bat
echo         python scripts\check_setup.py
echo.
echo   3. Run one test cycle (paper trading):
echo         python src\main_trading.py --once
echo.
echo   4. Run continuously:
echo         python src\main_trading.py
echo.
echo   5. Go live (only when ready):
echo         Edit .env: set DRY_RUN=false
echo         python src\main_trading.py
echo.
echo ======================================================
echo.
