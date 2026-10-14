@echo off
REM =============================================================================
REM setup.bat — Create .venv, install deps, download NLTK data (Windows)
REM Requires Python 3.9–3.12 (3.13/3.14 cannot install this stack from wheels).
REM
REM Usage:
REM   setup.bat
REM Then:
REM   run.bat              (default experiment + dashboard)
REM   run.bat --smoke      (quick install check)
REM   run.bat --full       (large experiment)
REM =============================================================================
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

if not exist "%CD%\requirements.txt" (
  echo ERROR: requirements.txt not found in %CD%
  echo Run this script from the project root.
  exit /b 1
)

REM ---- Pick a supported Python (3.9–3.12) ------------------------------------
set "PY="

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  for %%V in (3.12 3.11 3.10 3.9) do (
    if "!PY!"=="" (
      py -%%V -c "import sys" >nul 2>&1
      if !ERRORLEVEL!==0 set "PY=py -%%V"
    )
  )
)

if "!PY!"=="" (
  where python >nul 2>&1
  if !ERRORLEVEL!==0 (
    python -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1
    if !ERRORLEVEL!==0 set "PY=python"
  )
)

if not defined PY (
  echo.
  echo Python 3.9-3.12 is required.
  echo Python 3.13/3.14 cannot install NumPy/Numba/UMAP from wheels.
  echo.
  echo Install 3.12 from https://www.python.org/downloads/
  echo Tick "Add python.exe to PATH" during setup, then re-run setup.bat
  echo.
  exit /b 1
)

echo.
echo ==========================================================
echo   Hybrid LDA-LLM Topic Modeling - Setup
echo ==========================================================
echo.
echo Using Python:
%PY% -c "import sys; print('  ', sys.executable); print('  ', '.'.join(map(str, sys.version_info[:3])))"
if errorlevel 1 exit /b 1

REM ---- Recreate .venv if it is the wrong Python version ----------------------
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1
  if errorlevel 1 (
    echo Existing .venv uses unsupported Python. Recreating...
    rmdir /s /q .venv
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment.
    exit /b 1
  )
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo ERROR: venv python not found at %VENV_PY%
  exit /b 1
)

REM ---- Cache / data dirs (writable without admin) ----------------------------
set "NLTK_DATA=%CD%\data\nltk_data"
set "NUMBA_CACHE_DIR=%CD%\data\cache\numba"
set "MPLCONFIGDIR=%CD%\data\cache\matplotlib"
set "HF_HOME=%CD%\data\cache\huggingface"
set "TRANSFORMERS_CACHE=%CD%\data\cache\huggingface\transformers"
set "TOKENIZERS_PARALLELISM=false"
set "PYTHONWARNINGS=ignore"

if not exist "%NLTK_DATA%" mkdir "%NLTK_DATA%"
if not exist "%NUMBA_CACHE_DIR%" mkdir "%NUMBA_CACHE_DIR%"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%CD%\results" mkdir "%CD%\results"
if not exist "%CD%\results\figures" mkdir "%CD%\results\figures"

REM ---- Install packages ------------------------------------------------------
echo.
echo Upgrading pip / setuptools / wheel ...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1

echo.
echo Installing packages from requirements.txt ...
"%VENV_PY%" -m pip install --prefer-binary -r "%CD%\requirements.txt"
if errorlevel 1 (
  echo.
  echo pip install failed. Common fixes:
  echo   - Use Python 3.9-3.12 only
  echo   - Check network / proxy settings
  echo   - Delete .venv and re-run setup.bat
  exit /b 1
)

REM ---- NLTK corpora ----------------------------------------------------------
echo.
echo Downloading NLTK data ...
"%VENV_PY%" -c "import os,nltk; from pathlib import Path; d=Path('data/nltk_data'); d.mkdir(parents=True, exist_ok=True); os.environ['NLTK_DATA']=str(d.resolve()); pkgs=('punkt','punkt_tab','stopwords','wordnet','omw-1.4'); [print('  nltk:',p) or nltk.download(p, download_dir=str(d), quiet=True) for p in pkgs]"
if errorlevel 1 exit /b 1

REM ---- Import smoke check ----------------------------------------------------
echo.
echo Checking imports ...
"%VENV_PY%" -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('.').resolve())); from src.quiet import configure_warnings; configure_warnings(); import numpy,pandas,sklearn,scipy,gensim,nltk,torch,umap,matplotlib,streamlit,plotly; from sentence_transformers import SentenceTransformer; print('imports ok'); print('  python', '.'.join(map(str, sys.version_info[:3]))); print('  numpy', numpy.__version__); print('  gensim', gensim.__version__); print('  torch', torch.__version__); print('  streamlit', streamlit.__version__)"
if errorlevel 1 (
  echo Import check failed. Delete .venv and re-run setup.bat
  exit /b 1
)

echo.
echo ==========================================================
echo   Setup finished successfully.
echo ==========================================================
echo.
echo Next steps:
echo   run.bat              - default experiment + Streamlit dashboard
echo   run.bat --smoke      - tiny install check ^(150 docs^)
echo   run.bat --full       - large experiment ^(8k docs^)
echo   run.bat --dashboard  - dashboard only ^(skip pipeline^)
echo.
echo Or activate manually:
echo   .venv\Scripts\activate
echo   python run_pipeline.py --smoke
echo.
endlocal
exit /b 0
