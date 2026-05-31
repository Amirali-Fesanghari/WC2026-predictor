@echo off
:: WC 2026 Predictor — Windows Setup Script (v2)
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
echo [1/7] Creating virtual environment (.venv)...
python -m venv .venv
call .venv\Scripts\activate.bat

:: 3. Upgrade pip
echo [2/7] Upgrading pip...
python -m pip install --upgrade pip

:: 4. Install requirements
echo [3/7] Installing requirements (this may take 5-10 min)...
pip install -r requirements.txt

:: 5. Download spaCy English model (for NLP/NER in psych module)
echo [4/7] Downloading spaCy NLP model (en_core_web_sm)...
python -m spacy download en_core_web_sm

:: 6. Create .env file if it doesn't exist
echo [5/7] Setting up .env file...
if not exist .env (
    echo # WC 2026 Predictor — Environment Variables > .env
    echo FOOTBALL_DATA_API_KEY= >> .env
    echo # Get your free key at: https://www.football-data.org/client/register >> .env
    echo.
    echo .env created. Add your football-data.org API key when ready.
) else (
    echo .env already exists, skipping.
)

:: 7. Create directory structure (including new dirs)
echo [6/7] Creating directory structure...

:: Core data dirs
if not exist data\cache mkdir data\cache
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist logs mkdir logs
if not exist notebooks mkdir notebooks

:: New cache sub-dirs
if not exist data\cache\news mkdir data\cache\news
if not exist data\cache\injuries mkdir data\cache\injuries

:: src sub-dirs
if not exist src\simulation mkdir src\simulation
if not exist src\models\saved mkdir src\models\saved

:: Dashboard dir
if not exist dashboard mkdir dashboard

:: 8. Create __init__.py files
echo [7/7] Creating package __init__.py files...
type nul > src\__init__.py
type nul > src\pipeline\__init__.py
type nul > src\models\__init__.py
type nul > src\psych\__init__.py
type nul > src\tactics\__init__.py
type nul > src\utils\__init__.py
type nul > src\simulation\__init__.py
type nul > dashboard\__init__.py
type nul > tests\__init__.py

echo.
echo ============================================
echo  Setup complete!
echo ============================================
echo.
echo Packages installed (key ones):
echo   pandas / numpy / scipy    - data science core
echo   scikit-learn / xgboost    - ML models
echo   torch                     - neural network model
echo   vaderSentiment / spacy    - NLP / sentiment analysis
echo   feedparser                - RSS news feeds
echo   streamlit / plotly        - interactive dashboard
echo   rich                      - colourful CLI formatting
echo   loguru / joblib / tqdm    - utilities
echo.
echo Directories created:
echo   data\cache\news        - cached news articles
echo   data\cache\injuries    - cached injury reports
echo   src\simulation\        - simulation engine module
echo   src\models\saved\      - saved trained models
echo   dashboard\             - Streamlit dashboard app
echo.
echo Next steps:
echo   1. Add your football-data.org API key to .env
echo.
echo   2. Build the data pipeline:
echo      python -m src.pipeline.build_pipeline
echo      (or: python cli.py update)
echo.
echo   3. Run a prediction:
echo      python cli.py predict --home "France" --away "Brazil" --stage "quarter-final"
echo.
echo   4. Launch the dashboard:
echo      streamlit run dashboard\app.py
echo.
echo   5. Open in VS Code:
echo      code .
echo      (Ctrl+Shift+P -> Python: Select Interpreter -> .venv\Scripts\python.exe)
echo.
pause
