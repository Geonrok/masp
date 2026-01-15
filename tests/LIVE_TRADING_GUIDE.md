# Phase 2C-3: 소액 실거래 테스트 가이드

## ⚠️ 경고

**이 테스트는 실제 자금을 사용합니다!**
- 약 5,000 KRW의 실제 자금이 사용됩니다
- 수수료 약 5 KRW (0.05% × 2)
- 슬리피지로 인한 소액 손실 가능

## 사전 준비

### 1. API 키 확인
`.env` 파일에 실제 Upbit API 키가 설정되어 있어야 합니다:
```
UPBIT_ACCESS_KEY=your_real_access_key
UPBIT_SECRET_KEY=your_real_secret_key
```

### 2. 잔고 확인
- 최소 5,000 KRW 이상의 KRW 잔고 필요
- 현재 잔고 확인:
```cmd
scripts\run_in_venv.cmd python tests\test_upbit_execution.py
```

### 3. Kill-Switch 비활성화
`kill_switch.txt` 파일이 없어야 합니다:
```cmd
# 파일 존재 확인
dir kill_switch.txt

# 있다면 삭제 (실거래 허용 시에만)
del kill_switch.txt
```

## 실거래 테스트 실행

### 실행 명령
```cmd
scripts\run_in_venv.cmd python tests\test_live_trading_minimal.py
```

### 실행 단계 (12단계)

1. **모듈 로드**: Config, TradeLogger, AdapterFactory
2. **Kill-Switch 체크**: 활성화 시 중단
3. **Upbit 어댑터 생성**: API 키 검증
4. **사전 잔고 확인**: KRW, BTC 잔고 확인
5. **BTC 현재가**: 매수 수량 계산
6. **BTC 매수**: 5,000 KRW 시장가 매수
7. **체결 확인**: 3초 대기 후 잔고 확인
8. **BTC 매도**: 전량 시장가 매도
9. **최종 잔고**: KRW, BTC 잔고 확인
10. **PnL 계산**: 손익 및 손익률 계산
11. **거래 로그**: TradeLogger에 기록된 거래 확인
12. **Daily Report**: Markdown 리포트 생성

## 예상 결과

### 정상 케이스
```
투자금: 5,000 KRW
PnL: -5 KRW (-0.10%)
예상 수수료: ~5 KRW
```
- 수수료만큼 손실 (0.05% × 2 = 0.1%)
- 슬리피지로 인한 소액 추가 손실 가능

### 생성되는 파일
```
logs/live_trades/
├── trades/
│   └── 2026-01/
│       └── trades_2026-01-11.csv  (거래 기록)
└── reports/
    └── daily_2026-01-11.md  (일일 리포트)
```

## 주의사항

### ⚠️ 반드시 확인
- [ ] API 키가 실제 키로 설정되어 있는가?
- [ ] KRW 잔고가 5,000 이상인가?
- [ ] Kill-Switch가 비활성화되어 있는가?
- [ ] 테스트 실행 전 "yes" 입력 확인

### ⚠️ 실행 중 문제 발생 시
1. **매수 후 매도 실패**: BTC가 계정에 남아있을 수 있음
   - Upbit 앱/웹에서 수동 매도
   - 또는 스크립트 재실행 (매수 스킵하고 매도만 실행)

2. **Kill-Switch 활성화**: 
   - `kill_switch.txt` 파일 삭제 후 재실행

3. **API 키 오류**:
   - `.env` 파일 확인
   - API 키 권한 확인 (조회, 주문)

## 다음 단계

실거래 테스트 성공 후:
1. **Paper Trading**: 200+ 거래 시뮬레이션
2. **Health Monitor 검증**: MDD, Sharpe 추적
3. **전략 연동**: 실제 전략과 통합

## 문의

문제 발생 시:
- 로그 확인: `logs/live_trades/`
- API 상태 확인: Upbit 대시보드
- Kill-Switch 확인: `kill_switch.txt` 존재 여부
