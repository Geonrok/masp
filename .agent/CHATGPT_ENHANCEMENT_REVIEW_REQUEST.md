# ChatGPT 권장 보강 검수 요청

## 📋 검수 요청 개요

```yaml
프로젝트: MASP (Multi-Asset Strategy Platform)
작업: ChatGPT 권장 보강 (Option 1)
날짜: 2026-01-14 22:55 KST
이전_판정: APPROVED (4/4 AI 승인)
현재_단계: 권장 보강 완료 → 재검수 요청
```

---

## ✅ 완료된 보강 항목

### 보강 A: Live order_id 실제 주문ID 저장

**문제점 (ChatGPT 지적)**
- order_id가 심볼로 fallback ("BTC/KRW")되어 사후 추적/취소 불가능

**해결책**
- pybithumb 튜플 응답 파싱: `("bid", "BTC", "order_12345", "KRW")`
- order_id 유효성 검사 추가
- 심볼로 fallback 시 경고 로그 + 타임스탬프 기반 ID 생성

**수정 코드 위치**
- `libs/adapters/real_bithumb_execution.py` → `_parse_result()` 메서드

### 보강 B: Live ACK 게이트 테스트 고정

**문제점 (ChatGPT 지적)**
- Live ACK 게이트가 "주문 직전"에 강제되는지 테스트로 고정되지 않음

**해결책**
- 7개 테스트 추가 (`tests/test_live_ack_gate.py`)
  1. Paper 모드에서 Paper 어댑터 사용 확인
  2. Live 모드 ACK 테스트 (향후 구현용)
  3. Kill-Switch가 주문 전 차단 확인
  4. StrategyRunner 환경변수 체크 확인
  5. order_id가 심볼로 fallback되지 않는지 확인
  6. pybithumb 튜플 응답에서 order_id 추출 확인
  7. None 응답 시 적절히 처리 확인

---

## 📁 검수 대상 파일

### 핵심 수정 파일

1. **`libs/adapters/real_bithumb_execution.py`**
   - `_parse_result()` 메서드 개선
   - pybithumb 튜플 파싱
   - order_id 유효성 검사

2. **`tests/test_live_ack_gate.py`** (신규)
   - 7개 테스트
   - Live ACK 게이트 검증
   - Order ID 추적 검증

---

## 📊 테스트 결과

```
pytest tests/: 150 passed, 5 skipped ✅ (+7 신규)
pytest tests/test_live_ack_gate.py: 7/7 passed ✅
```

---

## 📝 핵심 코드 변경

### _parse_result() 개선 (보강 A)

```python
def _parse_result(self, result, symbol, side, quantity, order_type, price) -> BithumbOrderResult:
    """
    API 응답 파싱
    
    [ChatGPT 권장 보강 A] 실제 order_id 추출
    
    pybithumb 반환값 형식:
    - 성공: ("bid"/"ask", ticker, order_id, payment_currency)
    - 실패: None 또는 에러 dict
    """
    if result is None:
        return self._rejected_order(symbol, side, quantity, "Order failed: None response")
    
    order_id = None
    raw_response = str(result)
    
    if isinstance(result, tuple) and len(result) >= 3:
        # 정상 응답: ("bid", "BTC", "order_12345", "KRW")
        order_id = result[2]  # 세 번째 요소가 order_id
        logger.info(f"[BithumbExecution] Order ID extracted: {order_id}")
    elif isinstance(result, dict):
        # API 에러 응답
        order_id = result.get("order_id") or result.get("orderId")
        if not order_id:
            error_msg = result.get("message") or str(result)
            return self._rejected_order(symbol, side, quantity, f"API error: {error_msg}")
    elif isinstance(result, str):
        order_id = result
    else:
        order_id = str(result)
        logger.warning(f"[BithumbExecution] Unknown result format: {type(result)}")
    
    # order_id 유효성 검사 (ChatGPT 권고: 심볼로 fallback 방지)
    if not order_id or order_id == symbol or order_id == "None":
        logger.warning(f"[BithumbExecution] Invalid order_id: {order_id}, raw: {raw_response}")
        order_id = f"UNKNOWN_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
    # ... (나머지 코드)
```

### 테스트 예시 (보강 B)

```python
def test_order_id_not_fallback_to_symbol(self, mock_config):
    """order_id가 심볼로 fallback되지 않는지 확인"""
    with patch('libs.adapters.real_bithumb_execution.pybithumb') as mock_pybithumb:
        mock_bithumb_instance = MagicMock()
        mock_pybithumb.Bithumb.return_value = mock_bithumb_instance
        
        # pybithumb 정상 응답: ("bid", "BTC", "order_12345", "KRW")
        mock_bithumb_instance.buy_market_order.return_value = ("bid", "BTC", "order_12345", "KRW")
        
        from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
        adapter = BithumbExecutionAdapter(mock_config)
        adapter.get_current_price = MagicMock(return_value=50_000_000)
        
        result = adapter.place_order("BTC/KRW", "BUY", units=0.001)
        
        # order_id가 실제 주문 ID여야 함
        assert result.order_id == "order_12345"
        assert result.order_id != "BTC/KRW"  # 심볼로 fallback 안됨
```

---

## ❓ 검수 요청 사항

### 1. 보강 A 검토
- [ ] pybithumb 튜플 파싱 로직이 올바른가?
- [ ] order_id 유효성 검사가 충분한가?
- [ ] 심볼로 fallback 방지가 완전한가?

### 2. 보강 B 검토
- [ ] 7개 테스트가 필요한 시나리오를 커버하는가?
- [ ] Kill-Switch 테스트가 "주문 직전"을 검증하는가?
- [ ] 추가 테스트가 필요한 경로가 있는가?

### 3. 최종 확인
- [ ] 기존 기능에 영향 없는가? (150 passed)
- [ ] ChatGPT 권장 보강 완료로 판정 가능한가?

---

## 🎯 검수 결과 양식

```yaml
검수자: [AI 이름]
판정: [PASS / CONDITIONAL PASS / FAIL]

보강A_검토:
  튜플_파싱: [OK / 문제점]
  order_id_유효성: [OK / 문제점]
  fallback_방지: [OK / 문제점]

보강B_검토:
  테스트_커버리지: [OK / 문제점]
  Kill_Switch_테스트: [OK / 문제점]
  추가_테스트_필요: [없음 / 있음: 내용]

추가_권고:
  - [있다면]
```
