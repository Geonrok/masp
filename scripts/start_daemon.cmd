@echo off
setlocal
pushd "%~dp0\.."
set "REPO_ROOT=%CD%"
set "VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"

echo ============================================
echo  MASP Daemon Mode Launcher
echo ============================================

if not exist "%VENV_PYTHON%" (
    set "RC=3"
    echo ERROR: venv not found. Run: scripts\install.cmd
    goto :END
)

rem Default values
set "STRATEGY=kama_tsmom_gate"
set "EXCHANGE=paper"

rem Parse arguments
:PARSE_ARGS
if "%~1"=="" goto :RUN
if /i "%~1"=="--strategy" (
    set "STRATEGY=%~2"
    shift
    shift
    goto :PARSE_ARGS
)
if /i "%~1"=="--exchange" (
    set "EXCHANGE=%~2"
    shift
    shift
    goto :PARSE_ARGS
)
if /i "%~1"=="--help" goto :HELP
shift
goto :PARSE_ARGS

:RUN
echo.
echo Strategy: %STRATEGY%
echo Exchange: %EXCHANGE%
echo.
echo Starting daemon... (Press Ctrl+C to stop)
echo.

"%VENV_PYTHON%" "%REPO_ROOT%\scripts\run_daemon.py" --strategy %STRATEGY% --exchange %EXCHANGE% %*
set "RC=%ERRORLEVEL%"
goto :END

:HELP
echo.
echo Usage: start_daemon.cmd [options]
echo.
echo Options:
echo   --strategy NAME    Strategy ID (default: kama_tsmom_gate)
echo   --exchange NAME    Exchange (default: paper)
echo   --help             Show this help
echo.
echo Examples:
echo   start_daemon.cmd
echo   start_daemon.cmd --strategy atlas_futures_p04 --exchange binance_futures
echo   start_daemon.cmd --exchange upbit
echo.
set "RC=0"

:END
popd
endlocal & exit /b %RC%
