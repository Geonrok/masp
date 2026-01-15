# Bithumb API 2.0 구현 완료 검수 요청

## 📋 검수 요청 개요

```yaml
프로젝트: MASP (Multi-Asset Strategy Platform)
작업: Bithumb API 2.0 (JWT) 네이티브 어댑터 구현
날짜: 2026-01-15 15:02 KST
상태: 구현 완료 - 테스트 통과
```

---

## ✅ 구현 완료 현황

### 테스트 결과
```
pytest tests/ : 156 passed, 5 skipped ✅
```

### 변경된 파일

| # | 파일 | 작업 | 상태 |
|---|------|------|------|
| 1 | `libs/adapters/bithumb_api_v2.py` | **신규** - API 2.0 (JWT) 클라이언트 | ✅ |
| 2 | `libs/adapters/real_bithumb_execution.py` | **수정** - pybithumb → BithumbAPIV2 교체 | ✅ |
| 3 | `requirements.txt` | **수정** - PyJWT>=2.8.0 | ✅ |
| 4 | `tests/test_bithumb_api_v2.py` | **신규** - API 2.0 클라이언트 테스트 | ✅ |
| 5 | `tests/test_live_ack_gate.py` | **수정** - BithumbAPIV2 패치로 변경 | ✅ |

---

## 🔧 핵심 구현 내용

### 1. BithumbAPIV2 클라이언트 (`bithumb_api_v2.py`)

```python
class BithumbAPIV2:
    """Bithumb Open API 2.0 (JWT) client"""
    
    # JWT 생성
    def _generate_jwt(self, params) -> str:
        payload = {
            "access_key": self.api_key,
            "nonce": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),
        }
        if params:
            query = self._encode_query(params)
            payload["query_hash"] = self._make_query_hash(query)
            payload["query_hash_alg"] = "SHA512"
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    # 쿼리 인코딩 (배열: key[]=v1&key[]=v2)
    @staticmethod
    def _encode_query(params) -> str:
        items = []
        for key in sorted(params.keys()):  # 정렬
            value = params[key]
            if isinstance(value, (list, tuple)):
                for item in value:
                    items.append((f"{key}[]", item))
            else:
                items.append((key, value))
        return urllib.parse.urlencode(items)
    
    # API 메서드
    def get_accounts(self) -> list[dict]           # 잔고 조회
    def get_ticker(self, markets: list[str])       # 현재가 조회
    def post_order(...) -> dict                    # 주문하기
    def get_order(uuid_value: str) -> dict         # 주문 조회
    def cancel_order(uuid_value: str) -> dict      # 주문 취소
```

### 2. real_bithumb_execution.py 변경 사항

```python
# Before (pybithumb)
import pybithumb
self.bithumb = pybithumb.Bithumb(api_key, secret_key)
ticker = self._convert_symbol(symbol)  # "BTC"
result = self.bithumb.buy_market_order(ticker, units)

# After (BithumbAPIV2)
from libs.adapters.bithumb_api_v2 import BithumbAPIV2
self.bithumb = BithumbAPIV2(api_key, secret_key)
market = self._convert_symbol(symbol)  # "KRW-BTC"
result = self.bithumb.post_order(
    market=market,
    side="bid",
    ord_type="price",
    price=str(int(amount_krw))
)
```

### 3. 심볼 변환

```python
# Before
def _convert_symbol(self, symbol: str) -> str:
    return symbol.split("/")[0]  # "BTC/KRW" → "BTC"

# After
def _convert_symbol(self, symbol: str) -> str:
    base, quote = symbol.split("/")
    return f"{quote}-{base}"  # "BTC/KRW" → "KRW-BTC"
```

### 4. 주문 결과 파싱

```python
# Before (pybithumb tuple)
# result: ("bid", "BTC", "order_12345", "KRW")

# After (API v2 dict)
# result: {"uuid": "order_12345", ...}
order_id = result.get("uuid") or result.get("order_id")
```

---

## 🧪 테스트 검증

### API 2.0 클라이언트 테스트 (5/5 PASS)
- `test_encode_query_with_list` - 배열 key[] 인코딩 ✅
- `test_query_hash_sha512_length` - SHA512 해시 길이 ✅
- `test_jwt_includes_query_hash` - JWT payload 검증 ✅
- `test_request_raises_on_api_error` - API 에러 처리 ✅
- `test_request_raises_on_http_error` - HTTP 에러 처리 ✅

### Live ACK 게이트 테스트 (6/6 PASS)
- `test_live_adapter_requires_ack_env_vars` ✅
- `test_live_mode_requires_ack` ✅
- `test_kill_switch_blocks_before_order` ✅
- `test_strategy_runner_checks_env_before_execution` ✅
- `test_order_id_not_fallback_to_symbol` ✅
- `test_order_id_handles_none_response` ✅

---

## ❓ 검수 요청 사항

1. **JWT 생성 로직**: access_key, nonce, timestamp, query_hash 구조가 올바른가?
2. **query_hash 생성**: sorted + urlencode + SHA512 규칙이 맞는가?
3. **엔드포인트**: `/v1/accounts`, `/v1/orders`, `/v1/ticker` 경로가 맞는가?
4. **주문 파라미터**: market, side, ord_type, volume, price 구조가 맞는가?
5. **기존 기능 호환**: 150개+ 테스트 통과 유지 (156 passed) ✅

---

## 📁 검수 대상 파일

| # | 파일 | 설명 |
|---|------|------|
| 1 | `libs/adapters/bithumb_api_v2.py` | **핵심** - API 2.0 클라이언트 |
| 2 | `libs/adapters/real_bithumb_execution.py` | **핵심** - 실행 어댑터 |
| 3 | `tests/test_bithumb_api_v2.py` | API 2.0 테스트 |
| 4 | `tests/test_live_ack_gate.py` | Live ACK 게이트 테스트 |

---

## 🎯 검수 결과 양식

```yaml
검수자: [AI 이름]
판정: [PASS / FAIL / 조건부 PASS]

항목별_검증:
  JWT_생성: [OK / 문제점]
  query_hash: [OK / 문제점]
  엔드포인트: [OK / 문제점]
  주문_파라미터: [OK / 문제점]
  테스트_호환: [OK / 문제점]

추천_조치: [없음 / 내용]
```

---

## 🚀 다음 단계

1. **AI 검수 완료 후**: Live Dry Run 실행
2. **Dry Run 성공 시**: 소액 자동매매 테스트
3. **안정화 후**: 정규 운영 전환
