@echo off
:: WC 2026 Predictor — Windows Setup Script
:: Run this once from the project root in PowerShell or CMD

echo ============================================
echo  WC 2026 AI Predictor — Environment Setup
echo ============================================

:: 1. Check Python version
python --version 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.11+ from python.org
    pause
    exit /b 1
)

:: 2. Create virtual environment
echo.
echo [1/6] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

:: 3. Upgrade pip
echo [2/6] Upgrading pip...
python -m pip install --upgrade pip

:: 4. Install requirements
echo [3/6] Installing requirements (this takes 3-5 min)...
pip install -r requirements.txt

:: 5. Download spaCy English model (for NLP/NER in psych module)
echo [4/6] Downloading spaCy NLP model...
python -m spacy download en_core_web_sm

:: 6. Create .env file if it doesn't exist
echo [5/6] Setting up .env file...
if not exist .env (
    echo # WC 2026 Predictor — Environment Variables > .env
    echo FOOTBALL_DATA_API_KEY= >> .env
    echo # Get your free key at: https://www.football-data.org/client/register >> .env
    echo.
    echo .env created. Add your football-data.org API key when ready.
) else (
    echo .env already exists, skipping.
)

:: 7. Create __init__ files
echo [6/6] Creating package structure...
type nul > src\__init__.py
type nul > src\pipeline\__init__.py
type nul > src\models\__init__.py
type nul > src\psych\__init__.py
type nul > src\tactics\__init__.py
type nul > src\utils\__init__.py
type nul > tests\__init__.py

echo.
echo ============================================
echo  Setup complete!
echo ============================================
echo.
echo Next steps:
echo   1. Run: python -m src.pipeline.build_pipeline
echo      (Downloads data + computes ELO ratings)
echo.
echo   2. Open the project in VS Code:
echo      code .
echo.
echo   3. Select interpreter: .venv\Scripts\python.exe
echo      (Ctrl+Shift+P → Python: Select Interpreter)
echo.
pause
