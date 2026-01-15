@echo off
setlocal
pushd "%~dp0\.."
set "REPO_ROOT=%CD%"
set "VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"

rem Phase 0: PORT 기본값
set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"

rem Critical Fix: 숫자만 허용 (for /f delims 방식)
for /f "delims=0123456789" %%A in ("%PORT%") do (
    set "RC=2"
    echo ERROR: PORT must be digits only, got "%PORT%"
    goto :END
)

rem 범위 체크: 1..65535
set /a PORTNUM=%PORT% 2>nul
if %PORTNUM% LSS 1 (
    set "RC=2"
    echo ERROR: PORT out of range [1-65535], got "%PORT%"
    goto :END
)
if %PORTNUM% GTR 65535 (
    set "RC=2"
    echo ERROR: PORT out of range [1-65535], got "%PORT%"
    goto :END
)

if not exist "%VENV_PYTHON%" (
    set "RC=3"
    echo ERROR: venv not found. Run: scripts\install.cmd
    goto :END
)

echo [start_api] Starting on port %PORT%...
echo Starting MASP API Server (Single Worker)...
echo ============================================
echo WARNING: Multi-worker mode disabled for state safety
echo ============================================
"%VENV_PYTHON%" -m uvicorn services.api.main:app ^
  --host 0.0.0.0 ^
  --port %PORT% ^
  --workers 1 ^
  --reload
set "RC=%ERRORLEVEL%"

:END
popd
endlocal & exit /b %RC%
