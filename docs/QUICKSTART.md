# MASP Quick Start Guide

Multi-Asset Strategy Platform (MASP) 빠른 시작 가이드입니다.

## 1. 요구 사항

- Python 3.11 이상
- pip (Python 패키지 관리자)
- Git

## 2. 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/masp.git
cd masp

# 가상환경 생성 (선택사항이지만 권장)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 3. 환경 설정

```bash
# 환경 설정 파일 복사
cp .env.example .env

# .env 파일을 편집하여 API 키 설정
```

### 필수 설정 항목

| 변수 | 설명 |
|------|------|
| `MASP_ADMIN_TOKEN` | API 인증 토큰 |
| `MASP_DASHBOARD_PASSWORD` | 대시보드 로그인 비밀번호 |
| `UPBIT_API_KEY` | Upbit API 키 |
| `UPBIT_API_SECRET` | Upbit API 시크릿 |

## 4. 실행

### Config API 서버
```bash
python -m services.config_api.main
# http://localhost:8000 에서 실행
```

### 대시보드
```bash
streamlit run services/dashboard/app.py
# http://localhost:8501 에서 실행
```

### 전체 스택 (Docker)
```bash
docker-compose up -d
```

## 5. 테스트

```bash
# 전체 테스트 실행
python -m pytest

# 특정 모듈 테스트
python -m pytest tests/adapters/ -v
```

## 6. 주요 기능

### 대시보드 탭

1. **Overview**: 시스템 상태, 포트폴리오 요약
2. **Trading**: 주문 패널, 포지션, 거래 내역, 멀티 거래소
3. **Analytics**: 전략 성과, 백테스트, 리스크 지표
4. **Monitoring**: 로그, 알림, 스케줄러
5. **Settings**: 전략 설정, API 키, 텔레그램, 알림

### API 엔드포인트

- `GET /health`: 헬스 체크
- `GET /config`: 설정 조회
- `POST /config`: 설정 업데이트
- `GET /signals`: 시그널 조회
- `POST /execute`: 주문 실행

## 7. 문제 해결

### 일반적인 문제

1. **ModuleNotFoundError**: `pip install -r requirements.txt` 재실행
2. **API 인증 실패**: `.env` 파일의 토큰 확인
3. **거래소 연결 실패**: API 키 및 네트워크 확인

### 로그 확인
```bash
# 로그 디렉토리
ls logs/

# 실시간 로그 확인
tail -f logs/masp.log
```

## 8. 추가 리소스

- [API 문서](./API.md)
- [전략 개발 가이드](./STRATEGY.md)
- [배포 가이드](./DEPLOYMENT.md)
