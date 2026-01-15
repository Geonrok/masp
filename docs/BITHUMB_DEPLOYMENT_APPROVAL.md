# MASP Bithumb Integration - 배포 승인 문서

## 📋 배포 판정

| 항목 | 내용 |
|------|------|
| **판정** | ✅ **FINAL PASS (최종 배포 승인)** |
| **날짜** | 2026-01-15 12:38 KST |
| **검수자** | ChatGPT (PASS), Gemini (PASS), DeepSeek (PASS), Perplexity (PASS) |

### 승인 근거
- ✅ 블로커 3건 완료 (ChatGPT)
- ✅ 잔여 이슈 2건 완료 (ChatGPT)
- ✅ 기술적 보완 완료 (Gemini)
- ✅ pytest 150 passed, 5 skipped
- ✅ Live 테스트 COMPLETED
- ✅ **4/4 AI 만장일치 PASS**

---

## ✅ 충족된 게이트

### 필수 게이트 (ChatGPT)

| # | 게이트 | 상태 | 구현 위치 |
|---|--------|------|-----------|
| 1 | Live ACK 2단계 | ✅ | `MASP_ENABLE_LIVE_TRADING`, `MASP_ACK_BITHUMB_LIVE` |
| 2 | Kill-Switch Fail-Fast | ✅ | `real_bithumb_execution.py:170-173` |
| 3 | 최소/최대 주문 상한 | ✅ | `MIN_ORDER_KRW=5000`, `max_order_value_krw` |

### 코드 계약 (ChatGPT)

| # | 항목 | 상태 | 구현 |
|---|------|------|------|
| 1 | 시그니처 분리 | ✅ | `units=`, `amount_krw=` keyword-only |
| 2 | 상호배타 체크 | ✅ | BUY: XOR, SELL: units only |
| 3 | amount_krw 변환 | ✅ | fee_buffer 0.3% 포함 |

> ⚠️ **중요**: 운영자가 실수로 SELL에 `amount_krw`를 넣어도 **런타임에서 즉시 거부**됩니다.
> BUY에서 `units`와 `amount_krw`를 동시에 넣어도 **런타임에서 즉시 거부**됩니다.

### 데이터 무결성 (Gemini)

| # | 항목 | 상태 | 구현 |
|---|------|------|------|
| 1 | OHLCV 정렬 | ✅ | `df.sort_index()` |
| 2 | 최소 주문 5,000 KRW | ✅ | `MIN_ORDER_KRW = 5000` |

---

## 📊 테스트 결과

| 테스트 | 결과 |
|--------|------|
| Bithumb 계약 테스트 | 10/10 PASS |
| pytest 전체 | 143 passed, 5 skipped |
| Paper Trading | ✅ PASS |

---

## 🔒 pybithumb API 계약

```python
# 확인됨 (2026-01-14)
def buy_market_order(self, order_currency, unit, payment_currency="KRW"):
    """
    :param unit: 주문수량 (코인 수량)
    """
```

**결론**: `unit` = 코인 수량 (KRW 금액 아님)

---

## ⚠️ 주의사항

### Live 거래 전 필수 확인

1. **환경변수 설정**
   - `BITHUMB_API_KEY` 설정
   - `BITHUMB_SECRET_KEY` 설정
   - `MASP_ENABLE_LIVE_TRADING=1`
   - `MASP_ACK_BITHUMB_LIVE=1` (또는 `MASP_STRATEGY_PIPELINE_ACK`)

2. **Kill-Switch 확인**
   - `storage/kill_switch.flag` 없음 확인
   - `STOP_TRADING` 환경변수 없음 확인

3. **최소 주문**
   - 5,000 KRW 이상만 실행됨

---

## 📁 수정된 파일 목록

| 파일 | 변경 내용 |
|------|-----------|
| `libs/adapters/real_bithumb_execution.py` | 시그니처 분리, 계약 강제 |
| `libs/adapters/real_bithumb_spot.py` | OHLCV 정렬 |
| `tests/test_bithumb_order_contract.py` | 10개 테스트 |
| `services/strategy_runner.py` | Bithumb 지원 추가 |
| `libs/adapters/factory.py` | Bithumb execution 팩토리 |

---

## 🚀 배포 절차

### Paper Trading (권장 선행)
```powershell
$env:MASP_ENABLE_LIVE_TRADING = "0"
scripts\run_in_venv.cmd python -m services.strategy_runner --exchange bithumb --strategy kama_tsmom_gate --symbols BTC/KRW
```

### Live Trading
```powershell
$env:MASP_ENABLE_LIVE_TRADING = "1"
$env:MASP_ACK_BITHUMB_LIVE = "1"
scripts\run_in_venv.cmd python -m services.strategy_runner --exchange bithumb --strategy kama_tsmom_gate --symbols BTC/KRW --position-size-krw 10000
```

---

## 📝 거래소별 주문 계약 문서 (ChatGPT 권장)

| 거래소 | BUY 단위 | SELL 단위 | amount_krw 지원 | 변환 버퍼 |
|--------|----------|-----------|-----------------|-----------|
| **Bithumb** | 코인 수량 (unit) | 코인 수량 (unit) | ✅ (내부 변환) | 0.3% |
| **Upbit** | (추가 예정) | (추가 예정) | - | - |

---

## ✅ 최종 승인

```
판정: FINAL PASS (최종 배포 승인)
조건: 모든 블로커 및 잔여 이슈 해결 완료
서명: AI 검수 팀 (ChatGPT, Gemini, DeepSeek, Perplexity)
날짜: 2026-01-15 12:38 KST

수정 완료 항목:
- 블로커 #1: quantity→units 전달
- 블로커 #2: 빈 테스트→실제 테스트
- 추가 권고: status=UNKNOWN
- 잔여 #1: ACK 테스트 assert 활성화
- 잔여 #2: UNKNOWN 로깅 방지
```
