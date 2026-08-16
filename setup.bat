@echo off
REM Create a local virtualenv and install dependencies (Windows).
REM After this finishes:
REM   .venv\Scripts\activate
REM   python run_pipeline.py --smoke
setlocal EnableExtensions

cd /d "%~dp0"

set "PY="
where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3.12 -c "import sys" >nul 2>&1
  if %ERRORLEVEL%==0 set "PY=py -3.12"
)
if not defined PY (
  where py >nul 2>&1
  if %ERRORLEVEL%==0 (
    py -3.11 -c "import sys" >nul 2>&1
    if %ERRORLEVEL%==0 set "PY=py -3.11"
  )
)
if not defined PY (
  where py >nul 2>&1
  if %ERRORLEVEL%==0 (
    py -3.10 -c "import sys" >nul 2>&1
    if %ERRORLEVEL%==0 set "PY=py -3.10"
  )
)
if not defined PY (
  where py >nul 2>&1
  if %ERRORLEVEL%==0 (
    py -3.9 -c "import sys" >nul 2>&1
    if %ERRORLEVEL%==0 set "PY=py -3.9"
  )
)
if not defined PY (
  where python >nul 2>&1
  if %ERRORLEVEL%==0 (
    python -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1
    if %ERRORLEVEL%==0 set "PY=python"
  )
)

if not defined PY (
  echo Python 3.9-3.12 is required. Python 3.13/3.14 cannot install NumPy/Numba/UMAP from wheels.
  echo Install 3.12 from https://www.python.org/downloads/ and re-run setup.bat
  echo Tick "Add python.exe to PATH" during setup.
  exit /b 1
)

echo Using Python:
%PY% -c "import sys; print(sys.executable); print('.'.join(map(str, sys.version_info[:3])))"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1
  if errorlevel 1 (
    echo Existing .venv is an unsupported Python. Recreating.
    rmdir /s /q .venv
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  %PY% -m venv .venv
  if errorlevel 1 exit /b 1
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "NLTK_DATA=%CD%\data\nltk_data"
set "NUMBA_CACHE_DIR=%CD%\data\cache\numba"
set "MPLCONFIGDIR=%CD%\data\cache\matplotlib"
if not exist "%NLTK_DATA%" mkdir "%NLTK_DATA%"
if not exist "%NUMBA_CACHE_DIR%" mkdir "%NUMBA_CACHE_DIR%"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"

echo Installing packages from requirements.txt ...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b 1
"%VENV_PY%" -m pip install --prefer-binary -r "%CD%\requirements.txt"
if errorlevel 1 exit /b 1

echo Downloading NLTK data ...
"%VENV_PY%" -c "import os,nltk; from pathlib import Path; d=Path('data/nltk_data'); d.mkdir(parents=True, exist_ok=True); os.environ['NLTK_DATA']=str(d.resolve()); [nltk.download(p, download_dir=str(d), quiet=True) for p in ('punkt','punkt_tab','stopwords','wordnet','omw-1.4')]"
if errorlevel 1 exit /b 1

echo Checking imports ...
"%VENV_PY%" -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('.').resolve())); from src.quiet import configure_warnings; configure_warnings(); import numpy,pandas,sklearn,scipy,gensim,nltk,torch,umap,matplotlib; from sentence_transformers import SentenceTransformer; print('imports ok')"
if errorlevel 1 exit /b 1

echo.
echo Setup finished.
echo Activate the environment, then verify with a small run:
echo.
echo   .venv\Scripts\activate
echo   python run_pipeline.py --smoke
echo.
echo Full experiment:
echo   python run_pipeline.py
echo.
echo Optional:
echo   python run_pipeline.py --full
echo   python run_dashboard.py
echo.
endlocal
exit /b 0
