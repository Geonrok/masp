# MASP Bithumb Live 테스트 결과 검수 요청

## 📋 검수 요청 개요

```yaml
프로젝트: MASP (Multi-Asset Strategy Platform)
작업: Bithumb 현물 자동매매 Live 단발 테스트
날짜: 2026-01-14
이전_판정: CONDITIONALLY APPROVED (ChatGPT)
현재_단계: Live 단발 테스트 완료 → 재검수 요청
```

---

## 🔴 Live 테스트 중 발견된 문제 및 즉시 수정

### 문제 발견
```
[BithumbExecution] REJECTED: BTC/KRW BUY 6000 - Order value 836,022,000,000 exceeds limit 1,000,000
```

### 원인 분석
- **StrategyRunner**가 `position_size_krw` (6000 KRW)를 `quantity` (코인 수량)로 직접 전달
- BithumbExecutionAdapter가 6000을 **6000 BTC**로 해석
- 6000 BTC × 139,337,000 KRW = **836조 원** 주문 시도

### 즉시 수정
1. **StrategyRunner** 수정:
   - BUY: `amount_krw=self.position_size_krw` 파라미터 사용
   - SELL: `units=balance` 파라미터 사용

2. **PaperExecutionAdapter** 수정:
   - `units=`, `amount_krw=` 파라미터 지원 추가
   - `amount_krw` → `units` 내부 변환 로직 추가

---

## ✅ 수정 후 Live 테스트 결과

```json
{
  "timestamp": "2026-01-14T21:13:47.878976",
  "symbol": "BTC/KRW",
  "position_size_krw": 6000,
  "status": "COMPLETED",
  "execution_type": "BithumbExecutionAdapter",
  "result": {
    "BTC/KRW": {
      "action": "BUY",
      "order_id": "BTC/KRW"
    }
  },
  "pass_criteria": {
    "contract_violation": 0,
    "kill_switch_ready": true,
    "log_consistency": true,
    "safe_exit": true
  }
}
```

---

## 📁 검수 대상 파일

### 핵심 수정 파일 (필수 검토)

1. **`libs/adapters/real_bithumb_execution.py`**
   - 시그니처 분리: `units=`, `amount_krw=` keyword-only
   - 상호배타 체크: BUY XOR, SELL units only
   - `amount_krw` → `units` 변환 (fee_buffer 0.3%)

2. **`services/strategy_runner.py`**
   - BUY: `amount_krw=self.position_size_krw`
   - SELL: `units=balance`

3. **`libs/adapters/paper_execution.py`**
   - `units=`, `amount_krw=` 파라미터 지원
   - `amount_krw` → `units` 변환

### 관련 파일

4. **`libs/adapters/real_bithumb_spot.py`**
   - OHLCV 정렬: `df.sort_index()`

5. **`tests/test_bithumb_order_contract.py`**
   - 계약 테스트 10개

6. **`docs/BITHUMB_DEPLOYMENT_APPROVAL.md`**
   - 배포 승인 문서

---

## ❓ 검수 요청 사항

### 1. 코드 검토
- [ ] `amount_krw` → `units` 변환 로직이 올바른가?
- [ ] StrategyRunner의 BUY/SELL 호출이 계약을 준수하는가?
- [ ] PaperExecutionAdapter와 BithumbExecutionAdapter의 인터페이스가 일관적인가?

### 2. 안전성 검토
- [ ] Live 거래에서 "KRW를 unit으로 전달하는 사고"가 완전히 방지되었는가?
- [ ] Kill-Switch가 주문 전에 체크되는가?
- [ ] 최소/최대 주문 금액이 올바르게 적용되는가?

### 3. 테스트 검토
- [ ] 기존 10개 테스트가 모두 통과하는가?
- [ ] 추가 테스트가 필요한 경로가 있는가?

### 4. 최종 판정
- [ ] Live 배포 가능 여부
- [ ] 추가 조건 또는 권고사항

---

## 📊 테스트 결과

```
pytest tests/test_bithumb_order_contract.py: 10/10 PASS
pytest tests/: 143 passed, 5 skipped
Paper Trading: ✅ PASS
Live Trading: ✅ COMPLETED (수정 후)
```

---

## 📝 참고: 수정 전후 비교

### StrategyRunner (수정 전)
```python
# BUY
order = self.execution.place_order(
    symbol, "BUY",
    self.position_size_krw,  # ❌ 6000을 quantity(코인)로 전달
    order_type="MARKET",
)
```

### StrategyRunner (수정 후)
```python
# BUY
order = self.execution.place_order(
    symbol, "BUY",
    order_type="MARKET",
    amount_krw=self.position_size_krw,  # ✅ amount_krw=로 명시
)

# SELL
order = self.execution.place_order(
    symbol, "SELL",
    order_type="MARKET",
    units=balance,  # ✅ units=로 명시
)
```

---

## 🎯 검수 결과 양식

검수 후 아래 양식으로 응답해주세요:

```yaml
검수자: [AI 이름]
판정: [PASS / CONDITIONAL PASS / FAIL]
조건: [있다면]

코드_검토:
  amount_krw_변환: [OK / 문제점]
  StrategyRunner_호출: [OK / 문제점]
  인터페이스_일관성: [OK / 문제점]

안전성_검토:
  KRW_unit_혼동_방지: [OK / 문제점]
  Kill_Switch_위치: [OK / 문제점]
  주문_상한하한: [OK / 문제점]

추가_권고:
  - [있다면]
```
