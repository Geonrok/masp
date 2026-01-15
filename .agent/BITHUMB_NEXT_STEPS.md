# GPT-5.2-Codex 작업 지시서: Bithumb 통합 최종 단계

## 📋 현재 상태 요약

```yaml
프로젝트: MASP (Multi-Asset Strategy Platform)
작업: Bithumb 현물 자동매매 통합
판정: CONDITIONALLY APPROVED (ChatGPT)
날짜: 2026-01-14
```

### ✅ 완료된 항목

| # | 항목 | 상태 | 검증 |
|---|------|------|------|
| 1 | pybithumb API 확인 | ✅ | `buy_market_order(ticker, unit)` - unit = 코인 수량 |
| 2 | 시그니처 분리 | ✅ | `units=`, `amount_krw=` keyword-only |
| 3 | 상호배타 체크 | ✅ | BUY: XOR, SELL: units only |
| 4 | amount_krw 변환 | ✅ | fee_buffer 0.3% |
| 5 | OHLCV 정렬 | ✅ | `df.sort_index()` |
| 6 | 최소 주문 5,000 KRW | ✅ | `MIN_ORDER_KRW = 5000` |
| 7 | Kill-Switch Fail-Fast | ✅ | 주문 전 체크 |
| 8 | 계약 테스트 10개 | ✅ | pytest 통과 |
| 9 | 전체 회귀 143 passed | ✅ | 5 skipped |
| 10 | 배포 승인 문서 | ✅ | `docs/BITHUMB_DEPLOYMENT_APPROVAL.md` |
| 11 | Live 테스트 스크립트 | ✅ | `scripts/bithumb_live_test.ps1` |

---

## 🎯 다음 작업 목표

### Task 1: Live 단발 테스트 (사용자 승인 시)
```yaml
조건: 사용자가 API 키 제공 및 승인 시에만 진행
환경변수:
  - BITHUMB_API_KEY: (사용자 제공)
  - BITHUMB_SECRET_KEY: (사용자 제공)
  - MASP_ENABLE_LIVE_TRADING: "1"
  - MASP_ACK_BITHUMB_LIVE: "1"
스크립트: scripts\bithumb_live_test.ps1
통과기준:
  - 주문 계약 위반 0건
  - Kill-Switch 작동 가능
  - 로그 모순 없음
  - 안전 실패
```

### Task 2: Upbit 어댑터 동일 인터페이스 적용 (권장)
```yaml
목표: Bithumb과 동일한 시그니처 분리 적용
파일: libs/adapters/real_upbit_spot.py
변경:
  - place_order(*, units=None, amount_krw=None) 추가
  - 상호배타 체크 추가
  - 하위 호환성 (quantity → units 매핑)
테스트: tests/test_upbit_order_contract.py 생성
```

### Task 3: 거래소별 주문 계약 문서 완성 (ChatGPT 권장)
```yaml
파일: docs/BITHUMB_DEPLOYMENT_APPROVAL.md
추가내용:
  - Upbit 계약 정보
  - 변환/버퍼 정책
  - 운영자 가이드
```

---

## 🚫 절대 금지

```yaml
금지사항:
  1. 사용자 승인 없이 Live 거래 실행 금지
  2. API 키 값 로그/코드에 노출 금지
  3. kill_switch.flag 존재 시 모든 거래 금지
  4. 테스트 실패 상태에서 진행 금지
  5. amount_krw를 unit 자리에 직접 전달 금지
```

---

## 📝 작업 진행 시 체크리스트

### Live 테스트 전 필수 확인
- [ ] API 키 환경변수 설정 완료
- [ ] 3중 ACK 설정 완료
- [ ] Kill-Switch 비활성 확인
- [ ] Paper Trading 테스트 통과
- [ ] 사용자 최종 승인

### 코드 변경 시 필수 확인
- [ ] pytest 전체 통과 (143+ passed)
- [ ] Syntax 검사 통과
- [ ] Paper Trading 테스트 통과
- [ ] 기존 기능 회귀 없음

---

## 🔧 즉시 실행 가능 명령어

### Paper Trading 테스트
```powershell
$env:MASP_ENABLE_LIVE_TRADING = "0"
scripts\run_in_venv.cmd python -c "
from services.strategy_runner import StrategyRunner
runner = StrategyRunner('kama_tsmom_gate', 'bithumb', ['BTC/KRW'], 10000)
print(f'Execution: {runner.execution.__class__.__name__}')
result = runner.run_once()
print(f'Result: {result}')
"
```

### pytest 전체 회귀
```powershell
scripts\run_in_venv.cmd python -m pytest tests/ --tb=line -q
```

### Bithumb 계약 테스트
```powershell
scripts\run_in_venv.cmd python -m pytest tests/test_bithumb_order_contract.py -v
```

---

## 📊 현재 파일 구조

```
libs/adapters/
├── real_bithumb_execution.py  # ✅ 시그니처 분리 완료
├── real_bithumb_spot.py       # ✅ OHLCV 정렬 완료
├── real_upbit_spot.py         # 🔶 시그니처 분리 권장
└── factory.py                 # ✅ Bithumb 지원 완료

tests/
├── test_bithumb_order_contract.py  # ✅ 10개 테스트
└── (기타 테스트 파일들)

docs/
└── BITHUMB_DEPLOYMENT_APPROVAL.md  # ✅ 배포 승인 문서

scripts/
└── bithumb_live_test.ps1      # ✅ Live 테스트 스크립트
```

---

## ✅ 작업 완료 기준

```yaml
Live_테스트:
  - 주문 계약 위반 0건
  - Kill-Switch 정상 작동
  - 로그 모순 없음
  - 결과 JSON 저장됨

Upbit_동일화 (선택):
  - 시그니처 분리 적용
  - 테스트 10개 통과
  - pytest 전체 회귀 통과

문서화:
  - 거래소별 계약 문서 완성
  - 운영자 가이드 포함
```

---

**작업 시작 전 사용자에게 확인할 사항:**
1. Live 테스트 진행 여부
2. API 키 제공 가능 여부
3. Upbit 동일화 작업 진행 여부
