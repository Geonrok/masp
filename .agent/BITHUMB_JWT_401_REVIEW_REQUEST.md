# Bithumb API 2.0 JWT 401 오류 검수 요청

## 📋 검수 요청 개요

```yaml
프로젝트: MASP (Multi-Asset Strategy Platform)
작업: Bithumb API 2.0 JWT 인증 401 오류 디버깅
날짜: 2026-01-15 15:52 KST
상태: 디버깅 완료 - query_hash 무관하게 401 발생
```

---

## 📊 테스트 결과

### 환경
- API 키: 로드됨 (cd68235a...)
- Secret 키: 로드됨

### Test 1: query_hash 없이 (기본)
```
JWT Payload: {
  'access_key': 'cd68235a...',
  'nonce': '97905452-d4a5-42e8-983e-a8b94b00c77c',
  'timestamp': 1768460242263
}
결과: 401 Unauthorized ❌
```

### Test 2: query_hash 포함 (강제)
```
JWT Payload: {
  'access_key': 'cd68235a...',
  'nonce': '1d8ea1b1-d0b6-4b90-86f7-51f70f5d96eb',
  'timestamp': 1768460320273,
  'query_hash': 'cf83e1357eef...',  # SHA512 of empty string
  'query_hash_alg': 'SHA512'
}
결과: 401 Unauthorized ❌
```

### Public API (인증 불필요)
```
GET /v1/ticker → 200 OK ✅
BTC price: 141,414,000 KRW
```

---

## 🔍 분석

### 핵심 발견
**query_hash 유무와 관계없이 401 발생** → JWT 로직이 아닌 문제

### 의심 원인
1. **API 키 권한**: Bithumb 웹에서 "읽기(Read)" 권한 미체크
2. **IP 제한**: 현재 IP가 허용 목록에 없음
3. **API 키 활성화 안됨**: 발급 후 이메일/SMS 인증 미완료
4. **API 키 타입**: v1.2.0 키가 아닌 다른 타입

---

## ❓ 검수 요청 사항

1. JWT 인증 로직이 Bithumb API 2.0 문서와 일치하는가?
2. 401 Unauthorized의 실제 원인은 무엇인가?
3. 코드 레벨에서 추가 수정이 필요한가?
4. API 키 설정 확인이 필요한가?

---

## 📁 검수 대상 파일

| # | 파일 | 설명 |
|---|------|------|
| 1 | `libs/adapters/bithumb_api_v2.py` | JWT 생성 로직 + 디버그 로깅 |
| 2 | `tools/test_bithumb_api_v2.py` | 테스트 스크립트 |
| 3 | `tests/test_bithumb_api_v2.py` | 단위 테스트 (6개 PASS) |

---

## 🧪 테스트 명령어

```powershell
# query_hash 없이 테스트
$env:BITHUMB_JWT_DEBUG = "1"
$env:BITHUMB_JWT_INCLUDE_EMPTY_QUERY_HASH = "0"
python tools/test_bithumb_api_v2.py

# query_hash 포함 테스트
$env:BITHUMB_JWT_DEBUG = "1"
$env:BITHUMB_JWT_INCLUDE_EMPTY_QUERY_HASH = "1"
python tools/test_bithumb_api_v2.py

# 단위 테스트
pytest tests/test_bithumb_api_v2.py -v
```

---

## ✅ 변경 사항 요약

### bithumb_api_v2.py
1. `import logging, os` 추가
2. `logger = logging.getLogger(__name__)` 추가
3. `_generate_jwt()`: BITHUMB_JWT_INCLUDE_EMPTY_QUERY_HASH 환경변수 지원
4. 디버그 로깅 추가 (API 키 마스킹)
5. `_request()`: Accept 헤더 추가, 디버그 로깅 추가

### test_bithumb_api_v2.py
1. `test_jwt_includes_empty_query_hash_when_forced()` 추가

### tools/test_bithumb_api_v2.py
1. 디버그 플래그 지원 (BITHUMB_JWT_DEBUG)
2. 강제 query_hash 플래그 지원 (BITHUMB_JWT_INCLUDE_EMPTY_QUERY_HASH)

---

## 🎯 테스트 결과
```
pytest tests/test_bithumb_api_v2.py: 6 passed ✅
pytest tests/: 157 passed, 5 skipped ✅ (+1)
```
