# Multi-Asset Strategy Platform

**Phase 0: Mock only (no real trading).**

## Quick Start (Windows CMD)

```cmd
cd /d "E:\투자\Multi-Asset Strategy Platform"
scripts\install.cmd
scripts\smoke_test.cmd
scripts\start_api.cmd
```

Dashboard: http://127.0.0.1:8000/

## 중지 방법

- **API 서버**: Ctrl+C로 종료 (exit code 1이어도 정상)

## 흔한 문제

### PowerShell 사용 시 오류

PowerShell에서 dir/findstr 같은 CMD 명령이 오동작할 수 있습니다. **외부 Windows CMD를 사용하세요.**

### venv 격리 확인

`smoke_test.cmd` 실행 시 Python 경로가 `...\.venv\Scripts\python.exe`로 표시되어야 합니다. 시스템 Python 경로가 보이면 `install.cmd`를 다시 실행하세요.

### 데이터베이스

초기 실행 시 `storage/local.db` 파일이 자동 생성됩니다. 삭제 시 재생성됩니다.

## 수동 실행

서비스를 수동으로 실행할 때는 venv python을 사용하세요:

```cmd
REM 방법 1: venv python 직접 사용
.venv\Scripts\python.exe -m apps.crypto_spot_service --once

REM 방법 2: helper 스크립트 사용
scripts\run_in_venv.cmd python -m apps.crypto_spot_service --once
```

## Phase 0 Freeze

Phase 1 개발 시에도 아래 명령으로 회귀 확인:

```cmd
scripts\ci_local.cmd
```

이 게이트가 실패하면 Phase 0 baseline이 깨진 것입니다.

## 시크릿 정책

**API 키는 `.env` / 환경변수만 사용**

- `.env.example`을 복사해 `.env` 생성 (이미 `.gitignore`에 포함됨)
- Phase 0에서는 키 불필요 (`adapter_mode=mock` 기본값)
- Phase 1에서 real adapter 구현 시에만 사용
- **리포에 절대 하드코딩 금지**
