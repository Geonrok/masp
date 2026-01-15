---
description: MASP Phase 3 전략 파이프라인 연동 작업 지시서 (Task 1.2~5)
---

# MASP 전략 파이프라인 연동 작업 지시서

> **작성일**: 2026-01-14
> **대상 AI**: GPT-5.2-Codex
> **선행 조건**: `get_ohlcv()` 메서드가 `real_upbit_spot.py`에 추가됨 (검증 완료)

---

## 📋 작업 개요

KAMA-TSMOM-Gate 전략을 MASP 시스템에 완전히 통합하고 검증합니다.

---

## Task 1.2: 전략 목록 확인

### 목적
`list_available_strategies()`가 `kama_tsmom_gate` 전략을 반환하는지 확인

### 실행 명령
```powershell
// turbo
scripts\run_in_venv.cmd python -c "
from libs.strategies.loader import list_available_strategies
strategies = list_available_strategies()
print(f'Total strategies: {len(strategies)}')
for s in strategies:
    print(f'  - {s.get(\"id\")}: {s.get(\"name\")} [{s.get(\"status\")}]')
found = any('kama_tsmom_gate' == s.get('id') for s in strategies)
print(f'\\nkama_tsmom_gate found: {found}')
exit(0 if found else 1)
"
```

### 성공 기준
- `kama_tsmom_gate found: True` 출력
- Exit code: 0

---

## Task 2.1: Paper E2E 테스트

### 목적
Paper Trading 모드에서 전략이 정상 작동하는지 확인

### 실행 명령
```powershell
// turbo
$env:MASP_ENABLE_LIVE_TRADING = "0"
scripts\run_in_venv.cmd python -c "
from services.strategy_runner import StrategyRunner

runner = StrategyRunner(
    strategy_name='kama_tsmom_gate',
    exchange='paper',
    symbols=['BTC/KRW'],
    position_size_krw=10000
)

print('[1] StrategyRunner created')
print(f'    Strategy: {runner.strategy.__class__.__name__}')
print(f'    Exchange: paper')

result = runner.run_once()
print(f'[2] run_once result: {result}')

for symbol, details in result.items():
    action = details.get('action', 'UNKNOWN')
    reason = details.get('reason', 'No reason')
    print(f'    {symbol}: {action} - {reason}')

print('[3] Paper E2E: PASS')
"
```

### 성공 기준
- StrategyRunner 생성 성공
- `run_once()` 호출 성공
- 각 심볼에 대해 `action` (BUY/SELL/HOLD/BLOCKED) 반환
- 예외 없이 완료

---

## Task 2.2: Scheduler + Runner 통합 테스트

### 목적
`DailyScheduler.run_once()`가 `StrategyRunner.run_once()`를 올바르게 호출하는지 확인

### 실행 명령
```powershell
// turbo
$env:MASP_ENABLE_LIVE_TRADING = "0"
scripts\run_in_venv.cmd python -c "
from services.strategy_runner import StrategyRunner
from services.scheduler import DailyScheduler

# 1. StrategyRunner 생성
runner = StrategyRunner(
    strategy_name='kama_tsmom_gate',
    exchange='paper',
    symbols=['BTC/KRW'],
    position_size_krw=10000
)
print('[1] StrategyRunner created')

# 2. DailyScheduler 생성
scheduler = DailyScheduler(runner=runner)
print('[2] DailyScheduler created')
print(f'    Trigger: {scheduler.trigger}')
print(f'    Jitter: {scheduler.jitter}s')

# 3. run_once 실행
print('[3] Executing scheduler.run_once()...')
success = scheduler.run_once()
print(f'    Result: {\"SUCCESS\" if success else \"FAILED\"}')

# 4. 정리
scheduler.stop()
print('[4] Scheduler stopped')

print()
print('=== Task 2.2 PASS ===' if success else '=== Task 2.2 FAIL ===')
exit(0 if success else 1)
"
```

### 성공 기준
- `scheduler.run_once()` 반환값: `True`
- 예외 없이 완료
- Exit code: 0

---

## Task 3: 실거래 단발 테스트 (주의: 실제 API 호출)

### 목적
실거래 모드에서 전략이 정상 작동하는지 확인 (Kill-Switch 및 Gate 조건으로 안전 보장)

### 사전 조건 확인
```powershell
# 필수 환경변수 확인
echo "UPBIT_ACCESS_KEY: $env:UPBIT_ACCESS_KEY"
echo "UPBIT_SECRET_KEY: $env:UPBIT_SECRET_KEY"
```

### 실행 명령 (⚠️ 실거래 - 수동 승인 필요)
```powershell
$env:MASP_ENABLE_LIVE_TRADING = "1"
$env:MASP_ACK_REAL_MONEY = "1"
$env:MASP_ACK_STRATEGY = "kama_tsmom_gate"

scripts\run_in_venv.cmd python -c "
import os
from services.strategy_runner import StrategyRunner

# Multi-factor ACK 확인
ack1 = os.getenv('MASP_ENABLE_LIVE_TRADING') == '1'
ack2 = os.getenv('MASP_ACK_REAL_MONEY') == '1'
ack3 = os.getenv('MASP_ACK_STRATEGY') == 'kama_tsmom_gate'

print('[1] ACK Check:')
print(f'    MASP_ENABLE_LIVE_TRADING: {ack1}')
print(f'    MASP_ACK_REAL_MONEY: {ack2}')
print(f'    MASP_ACK_STRATEGY: {ack3}')

if not all([ack1, ack2, ack3]):
    print('[!] ACK failed. Aborting.')
    exit(1)

# StrategyRunner 생성 (실거래)
runner = StrategyRunner(
    strategy_name='kama_tsmom_gate',
    exchange='upbit',
    symbols=['BTC/KRW'],
    position_size_krw=10000
)
print('[2] StrategyRunner created (LIVE)')

# 실행
result = runner.run_once()
print(f'[3] Result: {result}')

for symbol, details in result.items():
    action = details.get('action', 'UNKNOWN')
    reason = details.get('reason', 'No reason')
    order_id = details.get('order_id', 'N/A')
    print(f'    {symbol}: {action} - {reason} (order_id: {order_id})')

print('[4] Live E2E Complete')
"
```

### 성공 기준
- ACK 3개 모두 통과
- 전략 실행 후 결과 반환 (BUY/SELL/HOLD/BLOCKED 중 하나)
- Gate CLOSED 또는 HOLD일 경우 주문 없음 (정상)
- 오류 없이 완료

---

## Task 4: CronTrigger 검증

### 목적
스케줄러의 CronTrigger가 올바른 시간(09:00 KST)에 실행되도록 설정되었는지 확인

### 실행 명령
```powershell
// turbo
scripts\run_in_venv.cmd python -c "
from datetime import datetime
from zoneinfo import ZoneInfo
from apscheduler.triggers.cron import CronTrigger

trigger = CronTrigger(hour=9, minute=0, timezone=ZoneInfo('Asia/Seoul'))
now = datetime.now(ZoneInfo('Asia/Seoul'))
next_run = trigger.get_next_fire_time(None, now)

print(f'Current time (KST): {now.strftime(\"%Y-%m-%d %H:%M:%S\")}')
print(f'Next scheduled run: {next_run.strftime(\"%Y-%m-%d %H:%M:%S\")}')
print(f'Hour: {next_run.hour}, Minute: {next_run.minute}')

is_valid = next_run.hour == 9 and next_run.minute == 0
print(f'\\nCronTrigger valid: {is_valid}')
exit(0 if is_valid else 1)
"
```

### 성공 기준
- `next_run.hour == 9` 및 `next_run.minute == 0`
- Exit code: 0

---

## Task 5: pytest 회귀 테스트

### 목적
기존 테스트가 모두 통과하는지 확인

### 실행 명령
```powershell
// turbo
scripts\run_in_venv.cmd python -m pytest tests/ -v --tb=short -x 2>&1 | Select-Object -First 100
```

### 성공 기준
- 모든 테스트 PASSED
- 실패 시 즉시 수정 필요

---

## 📊 작업 완료 체크리스트

```
[ ] Task 1.2: 전략 목록 확인 - kama_tsmom_gate 발견
[ ] Task 2.1: Paper E2E - StrategyRunner.run_once() 성공
[ ] Task 2.2: Scheduler 통합 - DailyScheduler.run_once() 성공
[ ] Task 3: 실거래 단발 - ACK 통과, 결과 반환 (선택적)
[ ] Task 4: CronTrigger - 09:00 KST 설정 확인
[ ] Task 5: pytest 회귀 - 전체 PASS
```

---

## ⚠️ 문제 발생 시 대응

### get_ohlcv AttributeError
```
원인: adapter에 get_ohlcv 메서드 없음
해결: 이미 2026-01-14에 real_upbit_spot.py에 추가됨. 확인 필요.
```

### run_once() returns False
```
원인: 이미 실행 중인 asyncio 이벤트 루프에서 호출
해결: 동기식 컨텍스트에서 호출해야 함
```

### 429 Rate Limit
```
원인: Upbit API 호출 과다
해결: Circuit Breaker가 자동으로 60초 대기
```

---

## 🔚 작업 완료 후 보고 형식

```
=== MASP Phase 3 Pipeline Integration Report ===
Date: 2026-01-14
Executor: GPT-5.2-Codex

Task 1.2: [PASS/FAIL] - 전략 목록 확인
Task 2.1: [PASS/FAIL] - Paper E2E
Task 2.2: [PASS/FAIL] - Scheduler 통합
Task 3:   [PASS/FAIL/SKIP] - 실거래 단발
Task 4:   [PASS/FAIL] - CronTrigger 검증
Task 5:   [PASS/FAIL] - pytest 회귀

Overall: [X/6 PASSED]
Notes: [이상 사항 기록]
```
