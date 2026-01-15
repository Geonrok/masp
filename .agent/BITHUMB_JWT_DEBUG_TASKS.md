# Bithumb API 2.0 JWT 인증 401 오류 디버깅

## 📋 작업 개요

```yaml
프로젝트: MASP (Multi-Asset Strategy Platform)
작업: Bithumb API 2.0 JWT 인증 401 오류 해결
날짜: 2026-01-15 15:43 KST
문제: GET /v1/accounts 호출 시 401 Unauthorized
목표: JWT 인증 로직 수정하여 Private API 호출 성공
```

---

## 📊 현재 상황

### 테스트 결과
```
✅ Public API (현재가): 성공
   GET /v1/ticker → SUCCESS
   BTC price: 141,366,000 KRW

❌ Private API (잔고): 실패
   GET /v1/accounts → 401 Unauthorized
```

### 문제점
- Public API는 정상 작동 (JWT 없이 호출)
- Private API에서 JWT 인증 실패 (401)

---

## 🔍 의심 원인

### 1. API 키 타입 문제
- 사용자가 API 2.0 키를 보유 (확인됨)
- 하지만 키가 Private API 권한이 없을 수 있음

### 2. JWT 생성 로직 문제
- Bithumb API 2.0 문서와 불일치 가능성
- query_hash 계산 방식, nonce, timestamp 형식 등

### 3. 요청 형식 문제
- GET /v1/accounts에 파라미터가 필요할 수 있음
- Content-Type, Accept 헤더 등

---

## 🎯 디버깅 작업

### Task 1: JWT 토큰 디버깅
```python
# libs/adapters/bithumb_api_v2.py에 디버깅 로그 추가
def _generate_jwt(self, params: Optional[Dict] = None) -> str:
    payload = {
        "access_key": self.api_key,
        "nonce": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
    }
    
    if params:
        query = self._encode_query(params)
        payload["query_hash"] = self._make_query_hash(query)
        payload["query_hash_alg"] = "SHA512"
    
    # DEBUG: payload 출력 (API 키 마스킹)
    debug_payload = {**payload, "access_key": payload["access_key"][:8] + "..."}
    logger.debug(f"[BithumbAPIV2] JWT Payload: {debug_payload}")
    
    token = jwt.encode(payload, self.secret_key, algorithm="HS256")
    return token
```

### Task 2: Bithumb API 문서 확인
Bithumb API 2.0 공식 문서에서 다음 확인:
1. JWT payload 필수 필드
2. query_hash 생성 규칙 (params 없을 때도 필요?)
3. Authorization 헤더 형식

### Task 3: 대안 시도
```python
# params 없어도 query_hash를 빈 문자열로 생성
def _generate_jwt(self, params: Optional[Dict] = None) -> str:
    payload = {
        "access_key": self.api_key,
        "nonce": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
    }
    
    # [시도 1] params가 없어도 query_hash 포함
    query = self._encode_query(params) if params else ""
    payload["query_hash"] = self._make_query_hash(query)
    payload["query_hash_alg"] = "SHA512"
    
    token = jwt.encode(payload, self.secret_key, algorithm="HS256")
    return token
```

### Task 4: 실제 요청 디버깅
```python
# _request 메서드에 디버깅 추가
def _request(self, method: str, endpoint: str, params: Optional[Dict] = None):
    url = f"{self.BASE_URL}{endpoint}"
    jwt_token = self._generate_jwt(params)
    
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    
    # DEBUG: 요청 정보 출력
    logger.debug(f"[BithumbAPIV2] Request: {method} {url}")
    logger.debug(f"[BithumbAPIV2] Headers: Authorization: Bearer {jwt_token[:50]}...")
    logger.debug(f"[BithumbAPIV2] Params: {params}")
    
    # ... 나머지 로직
```

---

## 📁 수정 대상 파일

| # | 파일 | 작업 |
|---|------|------|
| 1 | `libs/adapters/bithumb_api_v2.py` | JWT 디버깅 로그 추가, 로직 수정 |
| 2 | `tools/test_bithumb_api_v2.py` | 디버깅 테스트 스크립트 보강 |

---

## 🧪 테스트 방법

```bash
# 디버그 로깅 활성화하여 테스트
PYTHONPATH=. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)

from libs.adapters.bithumb_api_v2 import BithumbAPIV2
from dotenv import load_dotenv
import os

load_dotenv()
client = BithumbAPIV2(os.getenv('BITHUMB_API_KEY'), os.getenv('BITHUMB_SECRET_KEY'))

try:
    accounts = client.get_accounts()
    print('SUCCESS:', accounts)
except Exception as e:
    print('FAILED:', e)
"
```

---

## ✅ 성공 기준

| 항목 | 기준 |
|------|------|
| GET /v1/accounts | 200 OK 또는 정상 응답 |
| 잔고 조회 | KRW/코인 잔고 반환 |
| 기존 테스트 | 156 passed 유지 |

---

## 🔗 참조 자료

- Bithumb API 2.0 문서: https://apidocs.bithumb.com
- JWT 생성 예시 (Python): 문서에서 확인
- 에러 코드 목록: 401 = Unauthorized

---

## ⚠️ 주의사항

1. **API 키 노출 금지**: 로그에 키 전체 출력 금지
2. **실제 주문 금지**: 디버깅 중 주문 API 호출 금지
3. **테스트 유지**: 기존 156개 테스트 통과 유지
