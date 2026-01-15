# Phase 2C Readiness Checklist

**Protocol**: MASP-v1.0  
**Date**: 2026-01-10  
**Status**: Phase 2 APPROVED → Phase 2C PENDING  
**Lead AI**: Claude (Anthropic)

---

## Executive Summary

Phase 2C는 **Live Trading (실제 주문 실행)** 단계입니다.  
Phase 2A/2B에서 구축한 인프라를 기반으로, 실제 거래소 API 인증 및 주문 실행을 구현합니다.

**⚠️ 주의**: Phase 2C는 **실제 자금 손실 위험**이 있습니다. 모든 안전장치를 충분히 검증 후 진입하세요.

---

## Phase 2C 진입 조건 (Must-Have)

### ✅ Phase 2 완료 조건 (모두 충족)

| # | 조건 | 상태 | 증빙 |
|---|------|------|------|
| 1 | Phase 2A Real Market Data | ✅ DONE | `PHASE2_COMPLETE.md` |
| 2 | Phase 2B Paper Trading | ✅ DONE | `PHASE2_COMPLETE.md` |
| 3 | Cross-Model Review APPROVED | ✅ DONE | GPT/Gemini/Perplexity 3/3 |
| 4 | 필수보강 3건 반영 | ✅ DONE | Patch #1-3 |
| 5 | ci_local.cmd EXIT=0 | ✅ DONE | 회귀 없음 |

### 🔴 Phase 2C 진입 전 필수 항목 (9개)

| # | 항목 | 상태 | 조치 |
|---|------|------|------|
| 1 | **Paper Trading 1개월** | ⏳ PENDING | 최소 200 trades, MDD < 15% |
| 2 | **API 키 발급 (Upbit)** | ⏳ PENDING | Access Key + Secret Key |
| 3 | **API 키 발급 (Binance)** | ⏳ PENDING | API Key + Secret (선택) |
| 4 | **.env 파일 설정** | ⏳ PENDING | API 키 입력 + 검증 |
| 5 | **Kill-Switch 파일 생성** | ⏳ PENDING | 경로 확정 + 리허설 3회 |
| 6 | **손실 허용 범위 설정** | ⏳ PENDING | 최대 손실 금액 결정 |
| 7 | **최소 자금 입금** | ⏳ PENDING | Upbit 1M KRW 이상 권장 |
| 8 | **Order Validator 최종 검증** | ⏳ PENDING | 한도 설정 확인 |
| 9 | **연속 오류 3회 대응 계획** | ⏳ PENDING | Kill-Switch 자동 활성화 |

---

## ✅ Phase 2C-0: Strategy Health Monitor (완료)

**Date**: 2026-01-11  
**Status**: ✅ **COMPLETE**

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 1 | strategy_health.py | ✅ DONE | 377줄, 4 classes |
| 2 | paper_execution 통합 | ✅ DONE | get_health_status() |
| 3 | pytest | ✅ DONE | 7/7 PASS |
| 4 | 수동 검증 | ✅ DONE | 7/7 PASS |
| 5 | MDD 계산 버그 수정 | ✅ DONE | equity curve 기반 |

### 업계 표준 임계값

| 트리거 | 임계값 | 상태 | 근거 |
|--------|--------|------|------|
| Sharpe Floor | < 0.5 (30일) | WARNING | 암호화폐 특성 완화 |
| Sharpe Critical | < 0.0 | CRITICAL | 음수 Sharpe |
| MDD Warning | > 10% | WARNING | 조기 경고 |
| MDD Critical | > 15% | CRITICAL | 프롭 트레이딩 기준 |
| Consecutive Loss | 5회 | WARNING | 50% 승률 기준 |
| Consecutive Loss | 8회 | CRITICAL | 1% 확률 |
| Daily Loss | > 3% | CRITICAL (당일 HALT) | 암호화폐 변동성 |

### Health Status (4가지)

- ✅ **HEALTHY**: 정상 운영
- ⚠️ **WARNING**: 파라미터 검토 권장
- 🔴 **CRITICAL**: 거래 중단 필요
- ⛔ **HALTED**: Kill-Switch 활성

### 사용 방법

```python
from libs.adapters.paper_execution import PaperExecutionAdapter
from libs.adapters.factory import AdapterFactory

# PaperExecution 초기화 (Health Monitor 자동 포함)
md = AdapterFactory.create_market_data("upbit_spot")
pe = PaperExecutionAdapter(md, initial_balance=10_000_000)

# 거래 실행
order = pe.place_order("BTC/KRW", "BUY", 0.001)

# 건강 상태 확인
health = pe.get_health_status()
print(f"Status: {health['status']}")
print(f"MDD: {health['mdd_pct']:.2f}%")
print(f"Sharpe (30d): {health['sharpe_30d']}")
print(f"Recommendation: {health['recommendation']}")
```

---

## Pre-Development Tasks

### Task #1: Paper Trading 1개월 시뮬레이션

**목적**: 실전 환경 검증 (최소 200 trades 이상)

**체크리스트**:
- [ ] PaperExecutionAdapter로 1개월 운영 (2026-01-10 ~ 2026-02-10)
- [ ] 최소 거래 수: 200 trades 이상
- [ ] Max Drawdown < 15%
- [ ] Sharpe Ratio > 1.0 (최소 30 샘플 이상)
- [ ] Kill-Switch 리허설 3회 이상
- [ ] 연속 오류 3회 발생 시 자동 중지 확인

**성과 목표**:
- Win Rate: > 50%
- Profit Factor: > 1.5
- Max Daily Loss: < 5%
- Avg Trade PnL: > 0

**산출물**:
- `paper_trading_report_202601.md` (1개월 성과 보고서)
- `paper_trading_trades.csv` (거래 내역)
- `paper_trading_equity_curve.png` (자산 곡선)

---

### Task #2: API 키 발급 및 보안 설정

#### Upbit API 키 발급

**절차**:
1. Upbit 웹사이트 로그인
2. **[고객센터] → [Open API 안내]** 이동
3. **[Open API 사용하기]** 클릭
4. **권한 설정**:
   - ✅ 자산 조회 (필수)
   - ✅ 주문 조회 (필수)
   - ✅ 주문하기 (Phase 2C)
   - ❌ 출금하기 (보안상 비활성화)
5. Access Key + Secret Key 발급
6. **IP 화이트리스트 설정** (권장)

**보안 주의사항**:
- Secret Key는 **즉시 .env 파일에 저장** (재조회 불가)
- .env 파일은 **절대 git commit 금지** (.gitignore 확인)
- API 키는 **주기적으로 재발급** (3개월 권장)

#### Binance API 키 발급 (선택)

**절차**:
1. Binance 웹사이트 로그인
2. **[API Management]** 이동
3. **[Create API]** 클릭
4. **권한 설정**:
   - ✅ Enable Reading
   - ✅ Enable Spot & Margin Trading
   - ❌ Enable Withdrawals (보안상 비활성화)
5. API Key + Secret 발급
6. **IP Restriction 설정** (필수)

---

### Task #3: .env 파일 설정

**파일 위치**: `프로젝트 루트/.env`

**템플릿**:
```bash
# Upbit API Keys (Phase 2C)
UPBIT_ACCESS_KEY=your_access_key_here
UPBIT_SECRET_KEY=your_secret_key_here

# Binance API Keys (Optional)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Kill-Switch Configuration
KILL_SWITCH_FILE=E:\투자\Multi-Asset Strategy Platform\kill_switch.txt

# Trading Limits (Phase 2C)
MAX_ORDER_VALUE_KRW=10000000
MAX_POSITION_PCT=0.10
MAX_DAILY_LOSS_KRW=5000000
```

**검증**:
```cmd
rem API 키 로드 확인
scripts\run_in_venv.cmd python -c "from libs.core.config import Config; c = Config(); print(f'Upbit Key: {c.upbit_access_key}')"
rem Expected: <SecretStr('**********')>
```

---

### Task #4: Kill-Switch 최종 검증

**Kill-Switch 파일 경로 확정**:
```
E:\투자\Multi-Asset Strategy Platform\kill_switch.txt
```

**리허설 3회**:

**1회차**:
```cmd
rem 1. Kill-Switch 파일 생성
echo EMERGENCY > kill_switch.txt

rem 2. 활성화 확인
scripts\run_in_venv.cmd python -c "from libs.core.config import Config; c = Config(); print(f'Active: {c.is_kill_switch_active()}')"
rem Expected: Active: True

rem 3. 주문 차단 확인 (Paper)
scripts\run_in_venv.cmd python -c "from libs.adapters.paper_execution import PaperExecutionAdapter; from libs.adapters.factory import AdapterFactory; md = AdapterFactory.create_market_data('upbit_spot'); pe = PaperExecutionAdapter(md); pe.place_order('BTC/KRW', 'BUY', 0.01)"
rem Expected: (정상 실행 - Config 주입 필요)

rem 4. 파일 삭제
del kill_switch.txt

rem 5. 비활성화 확인
scripts\run_in_venv.cmd python -c "from libs.core.config import Config; c = Config(); print(f'Active: {c.is_kill_switch_active()}')"
rem Expected: Active: False
```

**2회차**: 위 절차 반복  
**3회차**: 위 절차 반복

---

### Task #5: 손실 허용 범위 설정

**최대 손실 한도 결정**:

| 구분 | 권장 | 최소 | 최대 |
|------|------|------|------|
| 초기 자금 | 10M KRW | 1M KRW | 100M KRW |
| 일일 손실 한도 | 5% (500K) | 3% (300K) | 10% (1M) |
| 주문당 최대 금액 | 10M KRW | 5M KRW | 50M KRW |
| 포지션 비율 | 10% | 5% | 20% |
| Max Drawdown 허용 | 15% | 10% | 30% |

**OrderValidator 설정**:
```python
# libs/core/order_validator.py

MAX_POSITION_PCT = 0.10  # 총 자산의 10%
MAX_ORDER_VALUE_KRW = 10_000_000  # 1천만 원
MIN_ORDER_VALUE_KRW = 5_000  # 5천 원
```

---

## Phase 2C Development Tasks

### Item #1: Upbit 실주문 API 인증 (8h)

**파일**: `libs/adapters/real_upbit_spot.py`

**구현 사항**:
1. JWT 토큰 생성 (`uuid4`, `hashlib`, `jwt`)
2. `UpbitSpotExecution.place_order()` 구현
3. POST `/v1/orders` API 호출
4. Order ID 반환

**AC**:
- [ ] 실제 주문 생성 (시장가)
- [ ] 실제 주문 생성 (지정가)
- [ ] 주문 체결 확인
- [ ] Kill-Switch 활성 시 차단

---

### Item #2: Order Validator와 PaperExecution 통합 (4h)

**파일**: `libs/adapters/paper_execution.py`

**구현 사항**:
```python
def __init__(self, market_data_adapter, initial_balance, config):
    self.config = config
    self.validator = OrderValidator(config)
    # ...

def place_order(self, symbol, side, quantity, order_type, price):
    # [1] Kill-Switch 체크
    if self.config.is_kill_switch_active():
        raise RuntimeError("Kill-Switch is active")
    
    # [2] OrderValidator 검증
    result = self.validator.validate(
        symbol, side, quantity, price or last_price,
        self.balance, self.get_total_equity()
    )
    if not result.valid:
        raise ValueError(f"Order validation failed: {result.reason}")
    
    # [3] 주문 생성
    # ...
```

**AC**:
- [ ] Config 주입
- [ ] OrderValidator 호출
- [ ] 검증 실패 시 OrderResult(success=False) 반환

---

### Item #3: 거래 로그 저장 (6h)

**파일**: `libs/core/trade_logger.py` (신규)

**구현 사항**:
```python
import csv
from datetime import datetime

class TradeLogger:
    """거래 내역을 CSV로 저장"""
    
    def __init__(self, log_file: str = "trades.csv"):
        self.log_file = log_file
    
    def log_order(self, order: PaperOrder):
        """주문 내역 저장"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                order.order_id,
                order.symbol,
                order.side,
                order.quantity,
                order.filled_price,
                order.status
            ])
```

**AC**:
- [ ] CSV 파일 생성
- [ ] 주문마다 1줄 기록
- [ ] 타임스탬프, Order ID, 심볼, 가격 포함

---

### Item #4: 모니터링 대시보드 (20h - Optional)

**기술 스택**: Streamlit / Dash / FastAPI + React

**화면 구성**:
1. **실시간 Equity 차트**
2. **포지션 현황**
3. **최근 거래 내역 (10건)**
4. **Kill-Switch 상태**
5. **경고 알림**

**Phase 2C에서는 선택 사항**, Phase 3에서 고도화

---

## Post-Development Verification

### Acceptance Criteria (Phase 2C)

| # | 항목 | 기준 | 검증 방법 |
|---|------|------|----------|
| 1 | Upbit 실주문 | 시장가 주문 성공 | API 호출 → Order ID 반환 |
| 2 | 지정가 주문 | 지정가 주문 성공 | API 호출 → Order ID 반환 |
| 3 | 체결 확인 | 주문 상태 조회 | GET `/v1/order` |
| 4 | Kill-Switch | 활성 시 차단 | RuntimeError 발생 |
| 5 | Order Validator | 검증 실패 시 차단 | ValueError 발생 |
| 6 | 거래 로그 | CSV 저장 | trades.csv 파일 확인 |
| 7 | Balance 확인 | 잔고 조회 | GET `/v1/accounts` |
| 8 | ci_local.cmd | EXIT=0 | 회귀 방지 |

---

## Security Checklist (Phase 2C)

| # | 항목 | 상태 | 비고 |
|---|------|------|------|
| 1 | .env 파일 .gitignore | ⏳ | git status 확인 |
| 2 | API 키 SecretStr | ✅ | Phase 1 완료 |
| 3 | Kill-Switch 3회 리허설 | ⏳ | Task #4 |
| 4 | Order Validator 한도 | ⏳ | Task #5 |
| 5 | IP 화이트리스트 (Upbit) | ⏳ | API 키 발급 시 |
| 6 | IP Restriction (Binance) | ⏳ | API 키 발급 시 |
| 7 | 출금 권한 비활성화 | ⏳ | API 키 발급 시 |
| 8 | 연속 오류 3회 대응 | ⏳ | Phase 2C Item #X |

---

## Troubleshooting Guide

### Issue #1: API 인증 실패

**증상**: 401 Unauthorized  
**해결**:
1. `.env` 파일 API 키 확인
2. JWT 토큰 생성 로직 확인
3. Upbit API 문서 확인: https://docs.upbit.com/docs/authorization-request

### Issue #2: Kill-Switch 미작동

**증상**: kill_switch.txt 생성했지만 주문 실행됨  
**해결**:
1. Config에 `kill_switch_file` 경로 올바른지 확인
2. `is_kill_switch_active()` 호출 여부 확인
3. PaperExecution `place_order()`에 Config 주입 확인

### Issue #3: Order Validator 검증 실패

**증상**: "Order exceeds 10% of equity"  
**해결**:
1. `total_equity` 계산 확인
2. `MAX_POSITION_PCT` 값 조정 (0.10 → 0.20)
3. 주문 금액 축소

---

## Phase 2C Timeline

| 주차 | 작업 | 예상 시간 | 산출물 |
|------|------|----------|--------|
| Week 1 | Paper Trading (계속) | 40h | 200+ trades |
| Week 2 | API 키 발급 + .env 설정 | 4h | .env 파일 |
| Week 3 | Upbit 실주문 구현 | 8h | real_upbit_spot.py |
| Week 4 | Order Validator 통합 | 4h | paper_execution.py |
| Week 5 | 거래 로그 + 검증 | 6h | trade_logger.py |
| Week 6 | 최종 검증 + 리허설 | 8h | PHASE2C_COMPLETE.md |

**총 예상 시간**: 70h (버퍼 포함: 80h)

---

## Phase 2C → Phase 3 브릿지

### Phase 2C 완료 조건

| # | 조건 | 기준 |
|---|------|------|
| 1 | 실주문 100건 이상 | Upbit 실제 체결 |
| 2 | Win Rate > 50% | 실제 거래 성과 |
| 3 | Max Drawdown < 15% | 실제 자금 운용 |
| 4 | Kill-Switch 정상 동작 | 3회 이상 실제 차단 |
| 5 | 연속 오류 0건 | 안정성 검증 |

### Phase 3 진입 시점

Phase 2C 완료 후 **최소 1개월 실전 운영** 성공 시 Phase 3 (고도화) 진입

---

**Phase 2C Status**: ⏳ **READY TO START**  
**Entry Date**: TBD (Paper Trading 1개월 후)

---

_Generated: 2026-01-10 22:38 KST_  
_Protocol: MASP-v1.0_  
_Lead AI: Claude (Anthropic)_
