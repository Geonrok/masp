# Bithumb Live Dry Run 검수 요청

## 📋 검수 요청 개요

```yaml
프로젝트: MASP (Multi-Asset Strategy Platform)
작업: Bithumb Live Dry Run (10,000 KRW 소액 테스트)
날짜: 2026-01-15 13:41 KST
이전_판정: FINAL PASS (4/4 AI 만장일치)
현재_단계: Live Dry Run 실행 → API 키 오류 발생
```

---

## 📊 Dry Run 실행 결과

### Step 1: Paper Trading ✅ 성공
```
Execution: PaperExecutionAdapter
Result: {'BTC/KRW': {'action': 'BUY', 'order_id': 'a6c3a756'}}
```
- Paper 모드에서 정상 작동 확인

### Step 2: Live Dry Run ❌ 실패
```
Execution: BithumbExecutionAdapter
[BithumbExecution] REJECTED: BTC/KRW BUY 7.07e-05 - API error: Invalid Apikey
Result: {'BTC/KRW': {'action': 'BUY', 'order_id': 'BTC/KRW'}}
```
- Live 모드에서 API 키 오류 발생

---

## 🔍 진단 결과

### 확인된 항목 ✅

| 항목 | 결과 |
|------|------|
| 공인 IP | 1.233.173.27 (Bithumb 허용 목록에 등록됨) |
| .env 로드 경로 | `E:\투자\Multi-Asset Strategy Platform\.env` |
| BITHUMB_API_KEY 로드 | ✅ (길이: 46) |
| BITHUMB_SECRET_KEY 로드 | ✅ (길이: 84) |
| 공백/따옴표 | ❌ 없음 |
| Placeholder 값 | ❌ 아님 |
| pybithumb 버전 | 1.0.21 (최신) |
| Kill-Switch | 비활성 |

### 문제점 ❌

| 항목 | 상태 |
|------|------|
| Bithumb API 응답 | `Invalid Apikey` |
| order_id | `BTC/KRW` (실제 ID 아님) |
| status | `REJECTED` |

---

## 📋 코드 동작 분석

### BithumbExecutionAdapter.place_order() 흐름

```python
1. Kill-Switch 체크 → ✅ 비활성
2. 현재가 조회 → ✅ 성공 (약 141,400,000 KRW)
3. units 계산 → ✅ 7.07e-05 BTC
4. pybithumb.buy_market_order() 호출 → ❌ Invalid Apikey
5. _parse_result() → REJECTED 반환
```

### pybithumb 내부 동작

```python
# pybithumb 1.0.21
def buy_market_order(self, order_currency, unit, payment_currency="KRW"):
    # HMAC-SHA512 서명 생성
    # POST /trade/market_buy 호출
    # → "Invalid Apikey" 응답
```

---

## ❓ 의심되는 원인

1. **API 키 타입**
   - Bithumb KR Open API가 아닌 다른 서비스용 키?
   - Trading 권한이 아닌 Read-only 키?

2. **활성화 상태**
   - 발급 직후 아직 활성화/승인 대기?
   - 계정 2FA/보안 설정 미완료?

3. **IP 제한**
   - 허용됨 상태라도 적용 지연?
   - 실제 요청 IP와 허용 IP 불일치?

4. **pybithumb 호환성**
   - Bithumb API 버전 변경?
   - 서명 규격 변경?

---

## ❓ 검수 요청 사항

1. **코드 레벨**: `real_bithumb_execution.py`의 API 호출 로직에 문제가 있는가?
2. **pybithumb 레벨**: 라이브러리가 올바르게 서명을 생성하는가?
3. **환경 레벨**: .env 로드, 환경변수 전달에 문제가 없는가?
4. **인프라 레벨**: Bithumb API 키 타입, 권한, IP 설정에 문제가 있는가?

---

## 📁 참조 파일

| 파일 | 설명 |
|------|------|
| `libs/adapters/real_bithumb_execution.py` | Bithumb 실행 어댑터 |
| `libs/core/config.py` | 설정 로드 (API 키 포함) |
| `.env` | 환경변수 파일 (API 키 저장) |
| `services/strategy_runner.py` | 전략 실행기 |

---

## 🎯 검수 결과 양식

```yaml
검수자: [AI 이름]
판정: [코드 문제 / 환경 문제 / Bithumb 설정 문제]

진단:
  코드_문제: [있음 / 없음: 설명]
  pybithumb_호환성: [정상 / 문제: 설명]
  환경변수_로드: [정상 / 문제: 설명]
  추천_조치: [내용]
```
