@echo off
REM =============================================================================
REM setup.bat — Create .venv, install deps, download NLTK data (Windows)
REM Supports Python 3.9–3.15. Uses binary wheels only for compiled packages
REM (avoids Visual Studio / meson source builds).
REM
REM Usage:
REM   setup.bat
REM Then:
REM   run.bat --smoke
REM =============================================================================
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

if not exist "%CD%\requirements.txt" (
  echo ERROR: requirements.txt not found in %CD%
  echo Run this script from the project root.
  exit /b 1
)

REM ---- Pick a supported Python (3.9–3.15) ------------------------------------
set "PY="

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  for %%V in (3.15 3.14 3.13 3.12 3.11 3.10 3.9) do (
    if "!PY!"=="" (
      py -%%V -c "import sys" >nul 2>&1
      if !ERRORLEVEL!==0 set "PY=py -%%V"
    )
  )
)

if "!PY!"=="" (
  where python >nul 2>&1
  if !ERRORLEVEL!==0 (
    python -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 15) else 1)" >nul 2>&1
    if !ERRORLEVEL!==0 set "PY=python"
  )
)

if "!PY!"=="" (
  echo.
  echo Python 3.9-3.15 is required.
  echo Install from https://www.python.org/downloads/
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
  ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 15) else 1)" >nul 2>&1
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

REM ---- Cache / data dirs -----------------------------------------------------
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

REM Packages that must never be compiled from source on Windows
set "ONLY_BIN=numpy,pandas,scipy,scikit-learn,numba,llvmlite,pyarrow,pillow,matplotlib,torch,PyYAML"

echo.
echo Upgrading pip / setuptools / wheel ...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1

for /f "delims=" %%M in ('"%VENV_PY%" -c "import sys; print(sys.version_info.minor)"') do set "PY_MINOR=%%M"
for /f "delims=" %%M in ('"%VENV_PY%" -c "import sys; print(sys.version_info[0])"') do set "PY_MAJOR=%%M"

REM PyTorch cp315 wheels are on download.pytorch.org, not PyPI.
if !PY_MINOR! GEQ 15 (
  echo.
  echo Python 3.!PY_MINOR!: installing torch from download.pytorch.org ...
  "%VENV_PY%" -m pip install --only-binary=:all: "torch>=2.2.0,<3" --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 (
    echo.
    echo torch has no binary wheel for this Python on Windows.
    echo Install Python 3.12 from https://www.python.org/downloads/
    echo then:  rmdir /s /q .venv ^& setup.bat
    exit /b 1
  )
)

REM On 3.15, bump floors so pip resolves packages that publish cp315 wheels.
set "EXTRA_PINS="
if !PY_MINOR! GEQ 15 (
  set "EXTRA_PINS=numpy>=2.5.0 pandas>=3.0.6 scipy>=1.18.0 scikit-learn>=1.9.0 pyarrow>=26.0.0 pillow>=12.0.0 numba>=0.68.0 llvmlite>=0.50.0 matplotlib>=3.10.0"
  echo.
  echo Python 3.!PY_MINOR!: requiring packages with cp315 wheels ...
  "%VENV_PY%" -m pip install --only-binary=%ONLY_BIN% !EXTRA_PINS!
  if errorlevel 1 goto :wheel_fail
)

echo.
echo Installing packages from requirements.txt ^(binary wheels only for compiled deps^) ...
"%VENV_PY%" -m pip install --prefer-binary --only-binary=%ONLY_BIN% -r "%CD%\requirements.txt"
if errorlevel 1 goto :wheel_fail

echo.
echo Downloading NLTK data ...
"%VENV_PY%" -c "import os,nltk; from pathlib import Path; d=Path('data/nltk_data'); d.mkdir(parents=True, exist_ok=True); os.environ['NLTK_DATA']=str(d.resolve()); pkgs=('punkt','punkt_tab','stopwords','wordnet','omw-1.4'); [print('  nltk:',p) or nltk.download(p, download_dir=str(d), quiet=True) for p in pkgs]"
if errorlevel 1 exit /b 1

echo.
echo Checking imports ...
"%VENV_PY%" -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('.').resolve())); from src.quiet import configure_warnings; configure_warnings(); import numpy,pandas,sklearn,scipy,gensim,nltk,torch,umap,matplotlib,streamlit,plotly; from sentence_transformers import SentenceTransformer; print('imports ok'); print('  python', '.'.join(map(str, sys.version_info[:3]))); print('  numpy', numpy.__version__); print('  pandas', pandas.__version__); print('  torch', torch.__version__); print('  streamlit', streamlit.__version__)"
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
endlocal
exit /b 0

:wheel_fail
echo.
echo ==========================================================
echo   pip could not find a BINARY wheel ^(no compiler needed^).
echo ==========================================================
echo.
echo This usually means a package has no wheel for your Python yet.
echo Do NOT install Visual Studio to fix this — use a Python with wheels.
echo.
echo Fix ^(recommended^):
echo   1. Install Python 3.12 from https://www.python.org/downloads/
echo   2. In this folder run:
echo        rmdir /s /q .venv
echo        setup.bat
echo.
echo Or stay on 3.15 only if every package publishes cp315 wheels.
echo.
endlocal
exit /b 1
