@echo off
setlocal
pushd "%~dp0\.."
set "REPO_ROOT=%CD%"

echo ========================================
echo Phase 1 - CI Local Gate (with Security)
echo ========================================
echo.
echo This script runs install + smoke_test + security regression tests.
echo REPO_ROOT: %REPO_ROOT%
echo.

echo [STEP 1/4] Running install.cmd...
call "scripts\install.cmd"
if errorlevel 1 (
  set "RC=%ERRORLEVEL%"
  echo.
  echo [FATAL] install.cmd FAILED (exit %RC%)
  popd
  endlocal & exit /b %RC%
)

echo.
echo [STEP 2/4] Running smoke_test.cmd...
call "scripts\smoke_test.cmd"
if errorlevel 1 (
  set "RC=%ERRORLEVEL%"
  echo.
  echo [FATAL] smoke_test.cmd FAILED (exit %RC%)
  popd
  endlocal & exit /b %RC%
)

echo.
echo [STEP 3/4] Security Regression Tests...

rem Test 1: PORT 범위 초과 차단
call "scripts\start_api.cmd" 99999 >nul 2>&1
set "RC=%ERRORLEVEL%"
if %RC% NEQ 2 (
  echo   [FAIL] PORT range check failed (expected EXIT=2, got %RC%)
  popd
  endlocal & exit /b 1
)
echo   [OK] PORT range check

rem Test 2: Config 마스킹 확인
scripts\run_in_venv.cmd python -c "from libs.core.config import Config, AssetClass; c = Config(asset_class=AssetClass.CRYPTO_SPOT); assert '<MASKED>' in str(c), 'API key masking failed'" >nul 2>&1
if errorlevel 1 (
  echo   [FAIL] Config masking check failed
  popd
  endlocal & exit /b 1
)
echo   [OK] Config masking

rem Test 3: model_dump 제외 확인
scripts\run_in_venv.cmd python -c "from libs.core.config import Config, AssetClass; c = Config(asset_class=AssetClass.CRYPTO_SPOT); assert 'upbit_access_key' not in c.model_dump(), 'API key in dump'" >nul 2>&1
if errorlevel 1 (
  echo   [FAIL] model_dump exclusion check failed
  popd
  endlocal & exit /b 1
)
echo   [OK] model_dump exclusion

echo.
echo [STEP 4/4] Running service smoke test...
scripts\run_in_venv.cmd python -m apps.crypto_spot_service --once >nul 2>&1
if errorlevel 1 (
  echo   [FAIL] Service smoke test failed
  popd
  endlocal & exit /b 1
)
echo   [OK] Service execution

echo.
echo ========================================
echo CI LOCAL GATE: PASSED (with Security Tests)
echo ========================================
popd
endlocal & exit /b 0
