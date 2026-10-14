@echo off
REM =============================================================================
REM run.bat — End-to-end: pipeline + Streamlit dashboard (Windows)
REM
REM Prerequisites: run setup.bat once first.
REM
REM Usage:
REM   run.bat                 Default experiment (both corpora) + dashboard
REM   run.bat --smoke         Tiny verify run, then dashboard
REM   run.bat --full          Large experiment, then dashboard
REM   run.bat --dashboard     Launch dashboard only (skip pipeline)
REM   run.bat --pipeline      Pipeline only (skip dashboard)
REM   run.bat --help          Show this help
REM
REM Any other flags are forwarded to run_pipeline.py, e.g.:
REM   run.bat --corpora arxiv --n-docs 2000 --k 8,12,16
REM =============================================================================
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

echo.
echo ==========================================================
echo   Hybrid LDA-LLM Topic Modeling - End-to-End Execution
echo ==========================================================
echo.

REM ---- Help ------------------------------------------------------------------
if /I "%~1"=="--help" goto :help
if /I "%~1"=="-h" goto :help
if /I "%~1"=="/?" goto :help

REM ---- Ensure venv exists ----------------------------------------------------
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo [!] No .venv found. Running setup.bat first...
  echo.
  call "%CD%\setup.bat"
  if errorlevel 1 (
    echo Setup failed. Fix errors above, then re-run run.bat
    exit /b 1
  )
  echo.
)

if not exist "%VENV_PY%" (
  echo ERROR: Still no .venv\Scripts\python.exe after setup.
  exit /b 1
)

"%VENV_PY%" -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 (
  echo ERROR: .venv Python is not 3.9-3.12. Delete .venv and run setup.bat
  exit /b 1
)

REM ---- Environment (match setup.bat) ----------------------------------------
set "NLTK_DATA=%CD%\data\nltk_data"
set "NUMBA_CACHE_DIR=%CD%\data\cache\numba"
set "MPLCONFIGDIR=%CD%\data\cache\matplotlib"
set "HF_HOME=%CD%\data\cache\huggingface"
set "TRANSFORMERS_CACHE=%CD%\data\cache\huggingface\transformers"
set "TOKENIZERS_PARALLELISM=false"
set "PYTHONWARNINGS=ignore"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "STREAMLIT_LOGGER_LEVEL=error"

if not exist "%NLTK_DATA%" mkdir "%NLTK_DATA%"
if not exist "%NUMBA_CACHE_DIR%" mkdir "%NUMBA_CACHE_DIR%"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%CD%\results" mkdir "%CD%\results"
if not exist "%CD%\results\figures" mkdir "%CD%\results\figures"

REM Put project root on PYTHONPATH for `import src`
set "PYTHONPATH=%CD%;%PYTHONPATH%"

REM ---- Parse run.bat-only flags; forward the rest to the pipeline ------------
set "RUN_PIPELINE=1"
set "RUN_DASHBOARD=1"
set "PIPE_ARGS="

:parse
if "%~1"=="" goto :parsed
if /I "%~1"=="--dashboard" (
  set "RUN_PIPELINE=0"
  set "RUN_DASHBOARD=1"
  shift
  goto :parse
)
if /I "%~1"=="--pipeline" (
  set "RUN_PIPELINE=1"
  set "RUN_DASHBOARD=0"
  shift
  goto :parse
)
REM Collect remaining args for run_pipeline.py
if defined PIPE_ARGS (
  set "PIPE_ARGS=!PIPE_ARGS! %~1"
) else (
  set "PIPE_ARGS=%~1"
)
shift
goto :parse

:parsed

echo [*] Using: %VENV_PY%
"%VENV_PY%" -c "import sys; print('[*] Python', '.'.join(map(str, sys.version_info[:3])))"
echo.

REM ---- Pipeline --------------------------------------------------------------
if "%RUN_PIPELINE%"=="1" (
  echo [*] Running analysis pipeline ^(run_pipeline.py^)...
  if defined PIPE_ARGS (
    echo [*] Args: !PIPE_ARGS!
    "%VENV_PY%" "%CD%\run_pipeline.py" !PIPE_ARGS!
  ) else (
    "%VENV_PY%" "%CD%\run_pipeline.py"
  )
  if errorlevel 1 (
    echo.
    echo [!] Pipeline failed. Dashboard will not start.
    exit /b 1
  )
  echo.
  echo [*] Pipeline completed successfully.
  echo [*] Results: %CD%\results\
  echo.
) else (
  echo [*] Skipping pipeline ^(--dashboard^).
  echo.
)

REM ---- Dashboard -------------------------------------------------------------
if "%RUN_DASHBOARD%"=="1" (
  if not exist "%CD%\run_dashboard.py" (
    echo ERROR: run_dashboard.py not found.
    exit /b 1
  )
  if not exist "%CD%\app\streamlit_app.py" (
    echo ERROR: app\streamlit_app.py not found.
    exit /b 1
  )
  echo [*] Starting Streamlit dashboard...
  echo [*] Close the browser tab / press Ctrl+C here to stop.
  echo.
  "%VENV_PY%" "%CD%\run_dashboard.py"
  set "DASH_EXIT=!ERRORLEVEL!"
  if !DASH_EXIT! neq 0 (
    echo.
    echo [!] Dashboard exited with code !DASH_EXIT!
    exit /b !DASH_EXIT!
  )
) else (
  echo [*] Skipping dashboard ^(--pipeline^).
)

echo.
echo Done.
endlocal
exit /b 0

:help
echo.
echo run.bat — Hybrid LDA-LLM Topic Modeling
echo.
echo   run.bat                 Default experiment + dashboard
echo   run.bat --smoke         Tiny install check, then dashboard
echo   run.bat --full          Large experiment, then dashboard
echo   run.bat --dashboard     Dashboard only
echo   run.bat --pipeline      Pipeline only
echo   run.bat --help          This help
echo.
echo Pipeline flags ^(forwarded to run_pipeline.py^):
echo   --smoke
echo   --full
echo   --n-docs N
echo   --k 8,12,16
echo   --corpora 20newsgroups,arxiv
echo   --embedding-model NAME
echo   --log-level INFO
echo.
echo Examples:
echo   run.bat --smoke
echo   run.bat --corpora arxiv --n-docs 2000 --k 8,12,16
echo   run.bat --pipeline --full
echo   run.bat --dashboard
echo.
endlocal
exit /b 0
