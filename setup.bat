@echo off
REM Create a local virtualenv and install dependencies (Windows).
REM After this finishes:
REM   .venv\Scripts\activate
REM   python run_pipeline.py
setlocal EnableExtensions

cd /d "%~dp0"

set "PY="
where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
  if %ERRORLEVEL%==0 set "PY=py -3"
)
if not defined PY (
  where python >nul 2>&1
  if %ERRORLEVEL%==0 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
    if %ERRORLEVEL%==0 set "PY=python"
  )
)

if not defined PY (
  echo Python 3.9 or newer is required.
  echo Install Python from https://www.python.org/downloads/ and re-run setup.bat
  echo If Python is already installed, tick "Add python.exe to PATH" during setup.
  exit /b 1
)

echo Using Python:
%PY% -c "import sys; print(sys.executable); print('.'.join(map(str, sys.version_info[:3])))"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  %PY% -m venv .venv
  if errorlevel 1 exit /b 1
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"

echo Installing packages from requirements.txt ...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%VENV_PY%" -m pip install -r "%CD%\requirements.txt"
if errorlevel 1 exit /b 1

echo Downloading NLTK data ...
"%VENV_PY%" -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','stopwords','wordnet','omw-1.4')]"
if errorlevel 1 exit /b 1

echo.
echo Setup finished.
echo Activate the environment, then run the pipeline with the same command on every OS:
echo.
echo   .venv\Scripts\activate
echo   python run_pipeline.py
echo.
echo Optional:
echo   python run_pipeline.py --full
echo   python run_dashboard.py
echo.
endlocal
exit /b 0
