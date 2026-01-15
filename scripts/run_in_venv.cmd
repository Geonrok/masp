@echo off
setlocal
pushd "%~dp0\.."
set "REPO_ROOT=%CD%"
set "VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
  echo ERROR: venv python not found at "%VENV_PYTHON%"
  echo Run: scripts\install.cmd
  popd
  endlocal & exit /b 3
)

if "%~1"=="" goto :USAGE

if /i "%~1"=="python" goto :RUN_PY
if /i "%~1"=="pip" goto :RUN_PIP
if /i "%~1"=="uvicorn" goto :RUN_UVICORN

echo ERROR: unsupported command "%~1"
goto :USAGE

:RUN_PY
"%VENV_PYTHON%" %2 %3 %4 %5 %6 %7 %8 %9
set "RC=%ERRORLEVEL%"
popd
endlocal & exit /b %RC%

:RUN_PIP
"%VENV_PYTHON%" -m pip %2 %3 %4 %5 %6 %7 %8 %9
set "RC=%ERRORLEVEL%"
popd
endlocal & exit /b %RC%

:RUN_UVICORN
"%VENV_PYTHON%" -m uvicorn %2 %3 %4 %5 %6 %7 %8 %9
set "RC=%ERRORLEVEL%"
popd
endlocal & exit /b %RC%

:USAGE
echo Usage:
echo   scripts\run_in_venv.cmd python [args...]
echo   scripts\run_in_venv.cmd pip [args...]
echo   scripts\run_in_venv.cmd uvicorn [args...]
echo.
echo NOTE: Maximum 8 arguments supported after command name
popd
endlocal & exit /b 2