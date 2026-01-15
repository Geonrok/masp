---
description: 빗썸 현물 자동매매 통합 작업 지시서
---

# 빗썸(Bithumb) 현물 자동매매 통합 작업

> **작성일**: 2026-01-14
> **목표**: 빗썸 현물 거래를 MASP 봇에서 자동으로 실행 가능하도록 완전 통합
> **선행 완료**: Upbit 현물 거래 (Phase 3 완료)

---

## 📋 현재 상태

### ✅ 완료된 항목
1. `BithumbSpotMarketData` - 시세 조회 어댑터 (get_quote, get_ohlcv)
2. `BithumbExecutionAdapter` - 실행 어댑터 (place_order, get_balance)
3. `AdapterFactory` - bithumb/bithumb_spot 지원
4. 환경변수 - `BITHUMB_API_KEY`, `BITHUMB_SECRET_KEY`

### ❌ 미완료 항목
1. `StrategyRunner`에서 bithumb 거래소 지원
2. 실거래 안전장치 (Multi-Factor ACK)
3. E2E 테스트
4. pybithumb 의존성 확인

---

## 🎯 Task 목록

### Task 1: pybithumb 설치 확인

```powershell
// turbo
.venv\Scripts\python.exe -c "import pybithumb; print('pybithumb installed:', pybithumb.__version__ if hasattr(pybithumb, '__version__') else 'OK')"
```

실패 시:
```powershell
.venv\Scripts\python.exe -m pip install pybithumb
```

### Task 2: StrategyRunner에 bithumb 지원 추가

**파일**: `services/strategy_runner.py`

**수정 위치**: `__init__` 메서드 (라인 91-106 근처)

```python
# 현재 코드 (upbit만 지원)
execution_exchange = exchange
adapter_mode = "paper"
if exchange in {"upbit", "upbit_spot"}:
    execution_exchange = "upbit_spot"
    adapter_mode = "live"

# 수정 코드 (bithumb 추가)
execution_exchange = exchange
adapter_mode = "paper"
if exchange in {"upbit", "upbit_spot"}:
    execution_exchange = "upbit_spot"
    adapter_mode = "live"
elif exchange in {"bithumb", "bithumb_spot"}:
    execution_exchange = "bithumb"
    adapter_mode = "live"
```

**수정 위치**: 시세 어댑터 (라인 105 근처)

```python
# 현재 코드
md_exchange = "upbit_spot" if exchange in ["paper", "upbit", "upbit_spot"] else "bithumb_spot"

# 수정 코드
if exchange in ["paper", "upbit", "upbit_spot"]:
    md_exchange = "upbit_spot"
elif exchange in ["bithumb", "bithumb_spot"]:
    md_exchange = "bithumb_spot"
else:
    md_exchange = "upbit_spot"  # 기본값
```

### Task 3: AdapterFactory에 bithumb 실거래 안전장치 추가

**파일**: `libs/adapters/factory.py`

**수정 위치**: `create_execution` 메서드, bithumb 섹션 (라인 169-176)

```python
# 현재 코드
if exchange_name == "bithumb":
    from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
    if config is None:
        raise ValueError("Config required for Bithumb execution adapter")
    adapter = BithumbExecutionAdapter(config, **kwargs)
    if trade_logger:
        adapter.set_trade_logger(trade_logger)
    return adapter

# 수정 코드 (안전장치 추가)
if exchange_name in {"bithumb", "bithumb_spot"}:
    if adapter_mode in {"live", "execution"}:
        if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
            raise RuntimeError(
                "[Factory] Bithumb live trading disabled. "
                "Set MASP_ENABLE_LIVE_TRADING=1 or use adapter_mode='paper'"
            )
        from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
        from libs.core.config import Config as ConfigClass
        if config is None:
            config = ConfigClass()
        adapter = BithumbExecutionAdapter(config, **kwargs)
        if trade_logger:
            adapter.set_trade_logger(trade_logger)
        return adapter
    
    # Paper mode for bithumb
    from libs.adapters.paper_execution import PaperExecutionAdapter
    market_data = AdapterFactory.create_market_data("bithumb_spot")
    return PaperExecutionAdapter(
        market_data_adapter=market_data,
        initial_balance=kwargs.pop("initial_balance", 1_000_000),
        config=config,
        trade_logger=trade_logger,
        **kwargs,
    )
```

### Task 4: BithumbExecutionAdapter 인터페이스 정규화

**파일**: `libs/adapters/real_bithumb_execution.py`

**추가**: `order_id` 속성을 위한 호환성 (StrategyRunner가 `order.order_id`를 사용)

BithumbOrderResult가 이미 `order_id` 필드를 가지고 있으므로 추가 작업 불필요.
단, StrategyRunner가 사용하는 `OrderResult` 인터페이스와 호환되는지 확인 필요.

**확인사항**:
```python
# StrategyRunner에서 사용하는 패턴
order = self.execution.place_order(symbol, "BUY", amount, order_type="MARKET")
order_id = order.order_id or order.symbol
```

BithumbOrderResult는 `order_id` 속성이 있으므로 호환됨 ✅

### Task 5: Paper Trading 검증 (bithumb)

```powershell
// turbo
$env:MASP_ENABLE_LIVE_TRADING = "0"
.venv\Scripts\python.exe -c "
from services.strategy_runner import StrategyRunner
from libs.strategies.loader import get_strategy

runner = StrategyRunner(
    strategy_name='kama_tsmom_gate',
    exchange='bithumb',  # Paper mode (live trading disabled)
    symbols=['BTC/KRW'],
    position_size_krw=10000
)

print('[1] StrategyRunner created')
print(f'    Exchange: bithumb')
print(f'    Strategy: {runner.strategy.__class__.__name__}')

result = runner.run_once()
print(f'[2] Result: {result}')
print('[3] Paper Trading Test: PASS')
"
```

### Task 6: 실거래 검증 (bithumb) - 선택적

⚠️ **주의**: 실제 주문이 발생합니다. 사용자 승인 필요.

```powershell
# 환경변수 확인
Write-Host "BITHUMB_API_KEY: $(if ($env:BITHUMB_API_KEY) { 'SET' } else { 'NOT SET' })"
Write-Host "BITHUMB_SECRET_KEY: $(if ($env:BITHUMB_SECRET_KEY) { 'SET' } else { 'NOT SET' })"
```

사용자 승인 후 실행:
```powershell
$env:MASP_ENABLE_LIVE_TRADING = "1"
$env:MASP_ACK_REAL_MONEY = "1"
$env:MASP_ACK_STRATEGY = "kama_tsmom_gate"

.venv\Scripts\python.exe -c "
import os
from services.strategy_runner import StrategyRunner

# ACK Check
ack1 = os.getenv('MASP_ENABLE_LIVE_TRADING') == '1'
ack2 = os.getenv('MASP_ACK_REAL_MONEY') == '1'
ack3 = os.getenv('MASP_ACK_STRATEGY') == 'kama_tsmom_gate'

print('[1] ACK:', 'ALL PASS' if all([ack1, ack2, ack3]) else 'FAILED')

if not all([ack1, ack2, ack3]):
    exit(1)

runner = StrategyRunner(
    strategy_name='kama_tsmom_gate',
    exchange='bithumb',  # LIVE MODE
    symbols=['BTC/KRW'],
    position_size_krw=10000
)

print('[2] StrategyRunner created (BITHUMB LIVE)')
result = runner.run_once()
print('[3] Result:', result)
"
```

---

## 📊 검증 체크리스트

```
[ ] Task 1: pybithumb 설치 확인
[ ] Task 2: StrategyRunner bithumb 지원 추가
[ ] Task 3: AdapterFactory 안전장치 추가
[ ] Task 4: 인터페이스 호환성 확인
[ ] Task 5: Paper Trading 검증
[ ] Task 6: 실거래 검증 (선택적 - 사용자 승인 필요)
```

---

## ⚠️ 문제 발생 시 대응

### pybithumb ImportError
```
원인: pybithumb 미설치
해결: pip install pybithumb
```

### BITHUMB_API_KEY not set
```
원인: 환경변수 미설정
해결: .env 파일에 BITHUMB_API_KEY, BITHUMB_SECRET_KEY 추가
```

### Config required for Bithumb
```
원인: Config 객체 누락
해결: AdapterFactory에서 Config 자동 생성 로직 추가
```

---

## 🔚 완료 후 보고 형식

```
=== Bithumb Integration Report ===
Date: 2026-01-14
Executor: GPT-5.2-Codex

Task 1: [PASS/FAIL] - pybithumb 설치
Task 2: [PASS/FAIL] - StrategyRunner 수정
Task 3: [PASS/FAIL] - AdapterFactory 수정
Task 4: [PASS/FAIL] - 인터페이스 확인
Task 5: [PASS/FAIL] - Paper Trading 검증
Task 6: [PASS/FAIL/SKIP] - 실거래 검증

Overall: [X/6 PASSED]
Notes: [이상 사항 기록]
```
