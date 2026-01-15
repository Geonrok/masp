# MASP: Bithumb Open API 2.0(JWT) 네이티브 어댑터 구현 지시서 (pybithumb 대체)

## 📋 배경

- 기존 pybithumb 기반 Live Dry Run에서 "Invalid Apikey"가 발생.
- 빗썸 Open API 2.0은 인증 방식이 Sign → JWT로 변경됨.
- **중요**: 기존 API KEY는 v1.2.0만 지원할 수 있으므로, Private API 2.0 사용 시 **"신규 API KEY"**가 필요하다.
  - 구현/테스트 전에 운영 키가 "신규 API KEY(Private 2.0)"인지 확인할 것.
  - 사용자가 API 2.0 키를 이미 보유 확인됨 ✅

---

## 🚫 절대 제약

- daemon 금지.
- Live 실행은 별도 ACK 없이 절대 수행 금지 (기존 Live ACK 게이트 유지/강화).
- 주문 계약 유지:
  - BUY: `amount_krw` XOR `units` (기존 계약 유지)
  - SELL: `units` only (`amount_krw` 거부)
  - Live에서는 deprecated `quantity` 파라미터를 즉시 ValueError로 거부 (기 구현 유지)

---

## 🎯 목표

- pybithumb를 제거(또는 최소화)하고, Open API 2.0(JWT) 기반 네이티브 호출로 교체한다.
- 기존 ExecutionAdapter 인터페이스 (StrategyRunner/TradeLogger/ACK 게이트/계약 테스트)와 호환되게 유지한다.
- 기존 pytest 150개 테스트를 깨지 않고 통과시킨다.

---

## 📍 엔드포인트 (API 2.0 문서 기준)

### Base URL
```
https://api.bithumb.com
```

### Public API (인증 불필요)
```
GET /v1/market/all           # 마켓 코드 조회
GET /v1/ticker               # 현재가 정보 (markets 파라미터)
GET /v1/candles/minutes/1    # 분봉 캔들 (선택)
```

### Private API (JWT 인증 필요)
```
GET  /v1/accounts            # 전체 계좌 조회 (잔고)
POST /v1/orders              # 주문하기
GET  /v1/order               # 개별 주문 조회
DELETE /v1/order             # 주문 취소 접수
```

> **⚠️ 주의**: pybithumb의 `/info/balance`, `/trade/market_buy`는 v1.2.0 경로임.
> API 2.0에서는 반드시 `/v1/*` 경로 사용.

---

## 📁 구현 파일/변경 범위

| # | 파일 | 작업 |
|---|------|------|
| 1 | `libs/adapters/bithumb_api_v2.py` | **신규** - API 2.0 클라이언트 |
| 2 | `libs/adapters/real_bithumb_execution.py` | **수정** - pybithumb → BithumbAPIV2 교체 |
| 3 | `requirements.txt` | **수정** - PyJWT>=2.8.0 추가 |
| 4 | `tests/test_bithumb_api_v2.py` | **신규** - JWT/API 테스트 |

---

## 🔐 핵심 구현 요구사항

### A. query_hash 생성 규칙 (문서 준수)

```python
def _encode_query(params: dict) -> str:
    """
    URL 인코딩 쿼리 문자열 생성
    - 키 기준 정렬 (sorted)
    - list/tuple 값은 key[]=v1&key[]=v2 형태 ★중요★
    - None 값 제외
    """
    import urllib.parse
    
    items = []
    for k, v in sorted(params.items()):
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            for item in v:
                items.append((f"{k}[]", item))
        else:
            items.append((k, v))
    
    return urllib.parse.urlencode(items)


def _make_query_hash(query: str) -> str:
    """SHA512 해시 생성"""
    import hashlib
    return hashlib.sha512(query.encode()).hexdigest()
```

### B. JWT 생성 규칙

```python
import jwt
import uuid
import time

def _generate_jwt(self, params: dict = None) -> str:
    """JWT 토큰 생성"""
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
```

### C. HTTP 요청 공통 레이어

```python
import requests

class BithumbAPIV2:
    BASE_URL = "https://api.bithumb.com"
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.session = requests.Session()
    
    def _request(self, method: str, endpoint: str, params: dict = None) -> dict:
        """API 요청 (에러 핸들링 포함)"""
        url = f"{self.BASE_URL}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self._generate_jwt(params)}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        try:
            if method == "GET":
                resp = self.session.get(url, headers=headers, params=params, timeout=10)
            elif method == "POST":
                resp = self.session.post(url, headers=headers, json=params, timeout=10)
            elif method == "DELETE":
                resp = self.session.delete(url, headers=headers, params=params, timeout=10)
            
            resp.raise_for_status()
            result = resp.json()
            
            # API 2.0 에러 체크 (error 필드)
            if "error" in result:
                raise ValueError(f"Bithumb API Error: {result['error']}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Network Error: {e}")
```

### D. BithumbAPIV2 표면 API

```python
class BithumbAPIV2:
    """Bithumb API 2.0 (JWT) 클라이언트"""
    
    # === Public API ===
    
    def get_ticker(self, markets: list[str]) -> list[dict]:
        """현재가 조회"""
        params = {"markets": ",".join(markets)}
        return requests.get(f"{self.BASE_URL}/v1/ticker", params=params).json()
    
    # === Private API ===
    
    def get_accounts(self) -> list[dict]:
        """전체 계좌 조회 (잔고)"""
        return self._request("GET", "/v1/accounts")
    
    def post_order(self, market: str, side: str, ord_type: str, 
                   volume: str = None, price: str = None) -> dict:
        """주문하기
        
        Args:
            market: 마켓 코드 (예: "KRW-BTC")
            side: "bid" (매수) 또는 "ask" (매도)
            ord_type: "market" (시장가) 또는 "limit" (지정가)
            volume: 주문 수량 (시장가 매도, 지정가)
            price: 주문 가격 (시장가 매수: KRW 금액, 지정가: 1주 가격)
        """
        params = {
            "market": market,
            "side": side,
            "ord_type": ord_type,
        }
        if volume:
            params["volume"] = volume
        if price:
            params["price"] = price
        
        return self._request("POST", "/v1/orders", params)
    
    def get_order(self, uuid: str) -> dict:
        """개별 주문 조회"""
        return self._request("GET", "/v1/order", {"uuid": uuid})
    
    def cancel_order(self, uuid: str) -> dict:
        """주문 취소"""
        return self._request("DELETE", "/v1/order", {"uuid": uuid})
```

### E. 심볼 변환 SSOT

```python
def _convert_symbol(symbol: str) -> str:
    """
    MASP 심볼 → Bithumb 마켓 코드 변환
    "BTC/KRW" → "KRW-BTC"
    """
    base, quote = symbol.split("/")
    return f"{quote}-{base}"

def _reverse_symbol(market: str) -> str:
    """
    Bithumb 마켓 코드 → MASP 심볼 변환
    "KRW-BTC" → "BTC/KRW"
    """
    quote, base = market.split("-")
    return f"{base}/{quote}"
```

### F. real_bithumb_execution.py 통합 포인트

```python
# pybithumb 의존 제거

# 현재가 조회
def get_current_price(self, symbol: str) -> float:
    market = self._convert_symbol(symbol)
    result = self.bithumb.get_ticker([market])
    return float(result[0]["trade_price"])

# 잔고 조회
def get_balance(self, currency: str) -> float:
    accounts = self.bithumb.get_accounts()
    for acc in accounts:
        if acc["currency"] == currency:
            return float(acc["balance"])
    return 0.0

# 시장가 매수 (KRW 금액 기준)
def _buy_market(self, symbol: str, amount_krw: float) -> dict:
    market = self._convert_symbol(symbol)
    return self.bithumb.post_order(
        market=market,
        side="bid",
        ord_type="price",  # 시장가 매수 (KRW 금액)
        price=str(int(amount_krw))
    )

# 시장가 매도 (코인 수량 기준)
def _sell_market(self, symbol: str, units: float) -> dict:
    market = self._convert_symbol(symbol)
    return self.bithumb.post_order(
        market=market,
        side="ask",
        ord_type="market",  # 시장가 매도
        volume=f"{units:.8f}"
    )
```

---

## 🧪 테스트 요구사항

### 1. 순수 유닛테스트 (외부 호출 없음)

```python
def test_encode_query_with_list():
    """list 값이 key[]로 직렬화되는지"""
    params = {"markets": ["KRW-BTC", "KRW-ETH"]}
    query = _encode_query(params)
    assert "markets[]=KRW-BTC" in query
    assert "markets[]=KRW-ETH" in query

def test_query_hash_sha512():
    """query_hash가 sha512로 생성되는지"""
    query = "market=KRW-BTC"
    hash_val = _make_query_hash(query)
    assert len(hash_val) == 128  # SHA512 = 128자

def test_jwt_with_query_hash():
    """params 유무에 따라 query_hash 포함/미포함"""
    # ...
```

### 2. Mock API 테스트

```python
def test_api_error_handling():
    """401/429/5xx 케이스"""
    # requests mock으로 에러 응답 주입
```

### 3. 회귀 테스트

```bash
pytest tests/ : 150 passed, 5 skipped 유지
```

---

## 📦 의존성

### 추가
```
PyJWT>=2.8.0
```

### 제거 가능
```
pybithumb  # API 2.0 전환 후 제거 권장
```

---

## ⚠️ 주의사항

1. **API 2.0 문서 참조**: https://apidocs.bithumb.com
2. **마켓 코드 형식**: `KRW-BTC` (Bithumb 2.0) vs `BTC/KRW` (MASP)
3. **에러 핸들링**: `error.name`, `error.message` 파싱
4. **Rate Limit**: 초당 요청 제한 확인
5. **테스트**: 실거래 전 Mock 테스트 필수
6. **v2.1.5(BETA)**: `/v2/orders`, `order_type` 등 차이 있을 수 있음

---

## 📊 작업 순서

| # | 작업 | 예상 시간 |
|---|------|----------|
| 1 | PyJWT 의존성 추가 | 5분 |
| 2 | `bithumb_api_v2.py` 생성 | 45분 |
| 3 | `real_bithumb_execution.py` 수정 | 30분 |
| 4 | 단위 테스트 작성 | 30분 |
| 5 | 회귀 테스트 확인 | 15분 |
| 6 | Dry Run 테스트 | 15분 |
| **합계** | | **~2시간 20분** |

---

## ✅ 성공 기준

| 항목 | 기준 |
|------|------|
| JWT 생성 | 올바른 형식 (HS256, SHA512 query_hash) |
| 잔고 조회 | `/v1/accounts` 정상 응답 |
| 시장가 매수 | `uuid` (order_id) 반환 |
| 기존 테스트 | 150 passed 유지 |
| Dry Run | 10,000 KRW 매수 성공 |

---

## 🚫 Live ACK 게이트

- Live Dry Run은 **반드시 ACK 설정 후** 실행
- `MASP_ENABLE_LIVE_TRADING=1`
- `MASP_ACK_BITHUMB_LIVE=1`
- ACK 없이는 절대 실행 금지
