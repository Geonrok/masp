@echo off
setlocal
pushd "%~dp0\.."
set "REPO_ROOT=%CD%"
set "VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"

echo ========================================
echo Multi-Asset Strategy Platform - Install
echo ========================================
echo.
echo REPO_ROOT: %REPO_ROOT%
echo VENV_PYTHON: %VENV_PYTHON%
echo.

if not exist "%REPO_ROOT%\.venv" (
  echo [1/4] Creating virtual environment...
  python -m venv ".venv"
  if errorlevel 1 (
    set "RC=%ERRORLEVEL%"
    echo ERROR: failed to create venv (exit %RC%)
    popd
    endlocal & exit /b %RC%
  )
)

if not exist "%VENV_PYTHON%" (
  echo ERROR: venv python not found at "%VENV_PYTHON%"
  popd
  endlocal & exit /b 3
)

echo [2/4] Upgrading pip...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
  set "RC=%ERRORLEVEL%"
  echo ERROR: pip upgrade failed (exit %RC%)
  popd
  endlocal & exit /b %RC%
)

echo [3/4] Installing requirements...
"%VENV_PYTHON%" -m pip install -r "requirements.txt"
if errorlevel 1 (
  set "RC=%ERRORLEVEL%"
  echo ERROR: requirements install failed (exit %RC%)
  popd
  endlocal & exit /b %RC%
)

echo [4/4] Installing package (editable)...
"%VENV_PYTHON%" -m pip install -e .
if errorlevel 1 (
  set "RC=%ERRORLEVEL%"
  echo ERROR: editable install failed (exit %RC%)
  popd
  endlocal & exit /b %RC%
)

echo.
echo INSTALL OK
popd
endlocal & exit /b 0
