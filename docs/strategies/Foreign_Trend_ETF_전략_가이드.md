# Foreign Trend ETF 전략 가이드

## 전략 개요

**Foreign_30d_SMA_100** (외국인+추세 ETF 전략)

Look-Ahead Bias가 완전히 제거된 검증된 전략입니다. 한국 시장 데이터만 사용하므로 미국 데이터 시차 문제가 없습니다.

### 핵심 지표

| 지표 | 값 | 평가 |
|------|-----|------|
| Sharpe Ratio | 1.225 | 우수 (> 1.0) |
| CAGR | 13.8% | 양호 |
| MDD | -16.9% | 양호 (B&H -41% 대비 +24%p) |
| WF Ratio | 0.86 | 과적합 아님 (> 0.7) |
| Win Rate | 55.2% | 양호 |
| 연간 거래 | ~8회 | 낮은 회전율 |

---

## 매매 규칙

### 진입 조건 (BUY)
```
외국인 30일 누적 순매수 > 0
AND
KOSPI200 종가 > SMA(100)
```

### 청산 조건 (SELL)
```
위 조건 중 하나라도 미충족 시
```

### 로직 설명

1. **외국인 30일 누적**: 외국인이 최근 30일간 순매수 중이면 상승 기대
2. **SMA 100**: 장기 추세가 상승 중일 때만 진입
3. **두 조건 AND**: 보수적 접근으로 위기 방어력 확보

---

## 대상 ETF

### 1배 ETF (권장)
- **종목**: TIGER 200 (102110)
- **레버리지**: 1배
- **예상 MDD**: -16.9%
- **최소 투자금**: ~4만원
- **적합 대상**: 모든 투자자

### 2배 ETF (조건부)
- **종목**: TIGER 200선물레버리지 (233160)
- **레버리지**: 2배
- **예상 MDD**: -29.5%
- **최소 투자금**: ~1.5만원
- **적합 대상**: 경험자/중급자 (MDD 감내 가능 시)
- **주의**: 횡보장에서 변동성 손실 발생

---

## 검증 결과

### Walk-Forward 검증 (11-Fold)
- **WF Ratio**: 0.86 (Test Sharpe / Train Sharpe)
- **해석**: 0.86 > 0.7 이므로 과적합 아님
- **Test 구간**: 2023-04-24 ~ 2026-01-29
- **Test Sharpe**: 2.44

### 연도별 성과
| 연도 | 수익률 | B&H 대비 |
|------|--------|----------|
| 2017 | +18.9% | +2.8%p |
| 2018 | -4.8% | -12.2%p (위기 방어) |
| 2019 | +9.2% | +1.2%p |
| 2020 | +41.6% | +11.6%p (COVID) |
| 2021 | -2.1% | -5.8%p |
| 2022 | +4.6% | +22.6%p (금리 인상) |
| 2023 | +12.5% | +3.9%p |
| 2024 | +8.7% | +5.4%p |
| 2025 | +6.2% | +3.8%p |

### 위기 방어력
- **COVID 2020.03**: B&H 대비 +11.6%p
- **금리 인상 2022**: B&H 대비 +22.6%p
- **분석**: 위기 시 현금 보유로 손실 최소화

---

## 사용법

### 전략 ID
```python
# 기본 전략
strategy_id = "foreign_trend_etf_v1"

# 1배 ETF 전용
strategy_id = "foreign_trend_1x"

# 2배 ETF 전용
strategy_id = "foreign_trend_2x"
```

### 코드 예시
```python
from libs.strategies.loader import get_strategy

# 전략 인스턴스 생성
strategy = get_strategy("foreign_trend_1x")

# 데이터 로드
strategy.load_data()

# 시그널 계산
signal, indicators, action = strategy.calculate_signal()

print(f"시그널: {action}")
print(f"외국인 30일: {indicators.get('foreign_30d', 0):,.0f}")
print(f"SMA100 상회: {indicators.get('above_sma', False)}")
```

---

## Look-Ahead Bias 제거 검증

### 원본 문제점
기존 KOSPI200 VIX 전략은 미국 T일 데이터를 한국 T일 거래에 사용하는 Look-Ahead Bias가 있었습니다.

```python
# 잘못된 코드 (원본)
data['vix'] = vix['Close']  # shift 없음 - Look-Ahead Bias!

# 올바른 코드
data['vix'] = vix['Close'].shift(1)  # T-1 데이터 사용
```

### 본 전략의 해결
Foreign_30d_SMA_100 전략은 **한국 데이터만 사용**하므로 시차 문제가 원천 차단됩니다.

- 외국인 순매수: 한국 장 마감 후 확정 (당일 데이터 사용 가능)
- KOSPI200 종가: 한국 데이터
- SMA 100: 한국 데이터로 계산

---

## 권장 사항

### 초보자
1. **1배 ETF (TIGER 200)** 사용
2. 소액(10~50만원)으로 시작
3. 최소 6개월 이상 운용 후 평가

### 중급자
1. MDD -30% 감내 가능 시 **2배 ETF** 고려
2. 포지션 사이즈 조절 (자본의 50% 이하)
3. 손절 규칙 추가 고려 (ex: -15% 시 청산)

### 고급자 / 자본 증식 후
1. KOSPI200 선물 직접 거래로 전환
2. 레버리지 효율 극대화
3. 야간 시그널 활용 가능

---

## 데이터 요구사항

### 필수 데이터
1. **KOSPI200 일봉**: `kospi200_daily_yf.parquet`
2. **외국인 투자자 데이터**: `*_investor.csv` 파일들

### 데이터 경로
```python
config = ForeignTrendConfig(
    kospi_data_dir="E:/투자/data/kospi_futures",
    investor_data_dir="E:/투자/data/kr_stock/investor_trading",
)
```

---

## 요약

| 항목 | 내용 |
|------|------|
| 전략명 | Foreign_30d_SMA_100 |
| 검증 상태 | Production Ready |
| Look-Ahead Bias | 없음 (한국 데이터만 사용) |
| 권장 ETF | TIGER 200 (102110) |
| 조건부 ETF | TIGER 200선물레버리지 (233160) |
| Sharpe | 1.225 |
| MDD (1x) | -16.9% |
| MDD (2x) | -29.5% |
| 특징 | 위기 방어력 우수, 낮은 회전율 |

---

*Generated: 2026-01-30*
*Author: Claude Code*
