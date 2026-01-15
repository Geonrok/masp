# GPT-5.2-Codex 작업 지시서: Bithumb 배포 후 다음 단계

## 📋 현재 상태 요약

```yaml
프로젝트: MASP (Multi-Asset Strategy Platform)
완료_작업: Bithumb 현물 자동매매 통합
판정: ✅ APPROVED (4/4 AI 승인)
날짜: 2026-01-14 22:15 KST
테스트: 143 passed, 5 skipped
```

---

## ✅ 완료된 Bithumb 통합 항목

| # | 항목 | 상태 |
|---|------|------|
| 1 | pybithumb API 확인 (unit = 코인 수량) | ✅ |
| 2 | 시그니처 분리 (units=, amount_krw=) | ✅ |
| 3 | Live deprecated quantity 거부 | ✅ |
| 4 | Paper get_balance(asset) 수정 | ✅ |
| 5 | Paper 계약 강제 (BUY XOR / SELL units-only) | ✅ |
| 6 | OHLCV 정렬 (df.sort_index()) | ✅ |
| 7 | 최소 주문 5,000 KRW | ✅ |
| 8 | 수수료 버퍼 0.3% | ✅ |
| 9 | Kill-Switch Fail-Fast | ✅ |
| 10 | 테스트 143 passed | ✅ |
| 11 | 배포 승인 문서 | ✅ |

---

## 🎯 다음 작업 옵션 (우선순위순)

### Option 1: ChatGPT 권장 보강 (선택)
```yaml
목표: 운영 품질 향상
항목:
  A. Live order_id 실제 주문ID 저장
     - 현재: 심볼로 fallback ("BTC/KRW")
     - 권장: 거래소 응답의 원문/주문ID 저장
  
  B. Live ACK 게이트 테스트 고정
     - 주문 직전에 ACK 강제되는지 테스트 추가

블로커: ❌ 아님 (권장)
```

### Option 2: Upbit 어댑터 동일화 (권장)
```yaml
목표: Bithumb과 동일한 시그니처/계약 적용
파일: libs/adapters/real_upbit_spot.py
변경:
  - place_order(*, units=None, amount_krw=None) 추가
  - 상호배타 체크 추가
  - Live deprecated quantity 거부
테스트: tests/test_upbit_order_contract.py 생성
```

### Option 3: 카나리아 배포 모니터링 (Gemini 권장)
```yaml
목표: 실제 운영 검증
전략:
  1단계: 10만원 Pilot (24시간)
  2단계: 매수/매도 사이클 관찰
  3단계: Scale-up
모니터링: 에러 로그, 주문 실패, Kill-Switch 작동
```

### Option 4: 다른 프로젝트 태스크
```yaml
참조: /masp_phase3_pipeline_tasks 워크플로우
```

---

## 🚫 절대 금지

```yaml
금지사항:
  1. 승인된 Bithumb 코드 임의 변경 금지
  2. API 키 로그/코드 노출 금지
  3. kill_switch.flag 존재 시 모든 거래 금지
  4. 테스트 실패 상태에서 배포 금지
```

---

## 📝 작업 진행 시 체크리스트

### 코드 변경 전
- [ ] 현재 pytest 상태 확인 (143 passed)
- [ ] git status 확인
- [ ] 변경 대상 파일 백업/커밋

### 코드 변경 후
- [ ] 문법 검사 (py_compile)
- [ ] pytest 전체 회귀 통과
- [ ] Paper Trading 테스트 통과

---

## 🔧 즉시 실행 가능 명령어

### pytest 전체 회귀
```powershell
scripts\run_in_venv.cmd python -m pytest tests/ --tb=line -q
```

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

### Live 배포 (승인됨)
```powershell
$env:MASP_ENABLE_LIVE_TRADING = "1"
$env:MASP_ACK_BITHUMB_LIVE = "1"
scripts\run_in_venv.cmd python -m services.strategy_runner `
    --exchange bithumb --strategy kama_tsmom_gate `
    --symbols BTC/KRW --position-size-krw 100000
```

---

## 📁 참조 파일

| 파일 | 설명 |
|------|------|
| `docs/BITHUMB_DEPLOYMENT_APPROVAL.md` | 배포 승인 문서 (APPROVED) |
| `.agent/BITHUMB_NEXT_STEPS.md` | 이전 작업 지시서 |
| `.agent/BITHUMB_LIVE_TEST_REVIEW_REQUEST.md` | 검수 요청서 |
| `.agent/workflows/bithumb_integration.md` | 통합 워크플로우 |

---

## ✅ 작업 시작 확인

**사용자에게 확인할 사항:**
1. Option 1~4 중 어느 작업을 진행할지 선택
2. 추가 지시사항 확인

**기본 권장:**
- Option 2 (Upbit 동일화) - Bithumb과 동일한 안전성 확보
