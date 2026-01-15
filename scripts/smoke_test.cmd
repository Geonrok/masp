@echo off
setlocal
pushd "%~dp0\.."
set "REPO_ROOT=%CD%"
set "VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"

echo ========================================
echo Multi-Asset Strategy Platform - Smoke Test
echo ========================================
echo.
echo REPO_ROOT: %REPO_ROOT%
echo.

if not exist "%VENV_PYTHON%" (
  echo ERROR: venv python not found at "%VENV_PYTHON%"
  echo Run: scripts\install.cmd
  popd
  endlocal & exit /b 3
)

echo [FORMAT_AUDIT] Running...
"%VENV_PYTHON%" "scripts\format_audit.py"
if errorlevel 1 (
  set "RC=%ERRORLEVEL%"
  echo ERROR: FORMAT_AUDIT failed (exit %RC%)
  popd
  endlocal & exit /b %RC%
)

"%VENV_PYTHON%" "scripts\smoke_test.py"
set "RC=%ERRORLEVEL%"
popd
endlocal & exit /b %RC%
