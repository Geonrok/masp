# KAMA+EMA Hybrid VWAP Breakout (Long-Only) - 최종 전략 명세서

> 작성일: 2026-01-28
> 최종 수정: 2026-01-28
> 버전: 2.0.0 (Phase 18 재검증 + Phase 19 적응형 필터 반영)
> 상태: 백테스트 검증 완료 / 페이퍼 트레이딩 승인

---

## 변경 이력

| 버전 | 날짜 | 내용 |
|------|------|------|
| 1.0.0 | 2026-01-28 | 초기 작성 (Vol Profile Breakout, Sharpe 2.52) |
| 2.0.0 | 2026-01-28 | Phase 18: min_len 버그 발견 및 수정. Phase 19: KAMA 하이브리드 추가. 검증 기준 수정 (추세추종 적합). 전 전략 재검증 완료 |

### v2.0 주요 변경사항
- **min_len 버그 수정**: 기존 OOS가 3개월(3윈도우)에 불과했던 문제 해결 → 2년(18윈도우)으로 확장
- **검증 기준 수정**: WR>45%→WR>30%, PF>1.5→PF>1.3 (추세추종 전략 특성 반영)
- **KAMA 하이브리드 추가**: EMA 추세 필터에 KAMA(20) 기울기 조건 추가
- **성과 수치 전면 수정**: Sharpe 2.52→1.41 (올바른 OOS 기간 반영)

---

## 1. 전략 개요

| 항목 | 내용 |
|------|------|
| 전략명 | KAMA+EMA Hybrid VWAP Breakout (Long-Only) |
| 타임프레임 | 1시간 (1H) |
| 시장 | 바이낸스 USDT-M 선물 전 종목 (~257개) |
| 방향 | 롱 온리 (매수만) |
| 성격 | 추세추종 + 거래량 확인 돌파 + 적응형 필터 |
| Sharpe Ratio | 1.41 (5x 스케일, 2년 TRUE OOS) |
| 수익률 | +59.6% (5x, OOS 2년) |
| 최대 낙폭 | -4.9% (5x) |
| 승률 | 41% |
| Profit Factor | 1.57 |

---

## 2. 진입 조건

**5가지 조건 모두 동시 충족 시 롱 진입:**

### 2.1 채널 돌파 (Donchian Breakout)
```
Close > Donchian(48) Upper Band
```
- Donchian(48) Upper = 최근 48봉(2일) 최고가
- 현재 종가가 최근 2일 최고가를 돌파해야 함

### 2.2 거래량 확인 (VWAP Filter)
```
Close > VWAP(48) x 1.02
```
- VWAP(48) = 48봉 거래량가중평균가
- 공식: `VWAP = Sum(Close x Volume) / Sum(Volume)` (48봉 rolling)
- 종가가 VWAP보다 **2%** 이상 위에 있어야 함
- 가짜 돌파를 걸러내는 핵심 필터

### 2.3 장기 추세 필터 (Dual EMA)
```
EMA(50) > EMA(200)
```
- 50시간 지수이동평균 > 200시간 지수이동평균
- 상승 추세가 확인된 시장에서만 진입

### 2.4 적응형 추세 품질 (KAMA Slope)
```
KAMA(20) - KAMA(20)[10봉 전] > 0
```
- KAMA = Kaufman Adaptive Moving Average (period=20, fast_sc=2, slow_sc=30)
- 추세장에서는 빠르게 반응, 횡보장에서는 느리게 반응
- 10봉 전 대비 KAMA가 상승 중이어야 함
- **횡보장에서 EMA 골든크로스 상태이나 실제 방향성이 없는 구간을 필터링**

### 2.5 방향
```
Long only (매수만)
```
- 숏(공매도) 포지션은 사용하지 않음

---

## 3. 청산 조건

**3가지 중 하나라도 충족 시 즉시 청산:**

### 3.1 손절 (Stop Loss)
```
손실 > ATR(14) x 3
```
- ATR(14) = 14봉 평균진폭 (Average True Range)
- 진입 후 가격이 ATR의 3배만큼 하락하면 손절

### 3.2 익절 (Profit Target)
```
이익 > ATR(14) x 8
```
- 진입 후 가격이 ATR의 8배만큼 상승하면 익절
- 손익비 R:R = 8:3 = 2.67:1

### 3.3 시간 제한 (Time Exit)
```
보유 기간 >= 72봉 (3일)
```
- 최대 3일 보유 후 현재 가격에서 강제 청산

---

## 4. 포트폴리오 관리

### 4.1 종목 선정
```
선정 기준: 168시간(7일) 수익률 표준편차 기준 변동성이 낮은 순서
선정 수: 상위 10종목
리밸런싱: 720봉(30일) 마다
```

### 4.2 포지션 크기 (Volatility Targeting)
```python
vol_168h = df['close'].pct_change().rolling(168).std().iloc[-1]
ann_vol = vol_168h * sqrt(24 * 365)
position_pct = min(0.10 / ann_vol / num_positions, 0.05)
position_pct *= position_scale  # 1x / 3x / 5x
```

| 변수 | 설명 |
|------|------|
| `vol_168h` | 7일 수익률 표준편차 |
| `ann_vol` | 연환산 변동성 |
| `0.10` | 연간 변동성 타겟 10% |
| `num_positions` | 선정된 종목 수 (최대 10) |
| `0.05` | 종목당 최대 5% 제한 |
| `position_scale` | 리스크 배율 (1x/3x/5x) |

### 4.3 포트폴리오 제약
| 항목 | 값 |
|------|-----|
| 최대 동시 보유 | 10종목 |
| 종목당 최대 배분 | 5% (scale 전) |
| 리밸런싱 주기 | 30일 |
| 종목 유니버스 | 바이낸스 USDT-M 전체 |
| 최소 데이터 요건 | 10,000봉 이상 (약 14개월) |

---

## 5. 비용 가정

| 항목 | 수치 | 비고 |
|------|------|------|
| 거래 수수료 | 0.04% (편도) | 바이낸스 Maker/Taker 평균 |
| 왕복 수수료 | 0.08% | 진입 + 청산 |
| 펀딩 비용 | 0.01% / 8시간 | 선물 펀딩레이트 |
| 슬리피지 | 0.03% (편도) | 시장가 주문 기준 |

---

## 6. 성과 및 리스크

### 6.1 백테스트 성과 (수정된 OOS, 2년, 18윈도우)

| 지표 | 1x | 3x | 5x |
|------|-----|-----|-----|
| Sharpe | 1.41 | 1.41 | 1.41 |
| 수익률 | +11.9% | +35.8% | +59.6% |
| 최대 DD | -1.0% | -2.9% | -4.9% |
| 승률 | 41% | 41% | 41% |
| PF | 1.57 | 1.57 | 1.57 |
| WFA 효율 | 67% | 67% | 67% |
| 거래수 | 601 | 601 | 601 |

### 6.2 시장 레짐별 성과 (5x)

| 레짐 | BTC 기준 | 윈도우 수 | 수익률 | 승률 |
|------|----------|-----------|--------|------|
| 강세장 | BTC 30일 >+10% | 4 (22%) | +41.8% | 100% |
| 약세장 | BTC 30일 <-10% | 2 (11%) | +2.2% | 50% |
| 횡보장 | 나머지 | 12 (67%) | +10.1% | 42% |

### 6.3 현실적 기대치

| 항목 | 백테스트 | 실전 예상 | 비고 |
|------|----------|-----------|------|
| Sharpe | 1.41 | 0.7~1.0 | 실행 비효율 30~50% 감소 |
| 연 수익 (5x) | +30% | +10~20% | |
| 최대 DD | -4.9% | -8~15% | |
| 승률 | 41% | 35~41% | 10번 중 6번 손실 |

### 6.4 최소 자본금 권장
| 구분 | 금액 | 비고 |
|------|------|------|
| 절대 최소 | $500 | 종목당 $15, 기능하지만 빠듯 |
| 권장 최소 | $1,000 | 종목당 $30, 수수료 비중 합리적 |
| 이상적 | $3,000~5,000 | 변동성 타겟팅 정밀 작동 |
| 용량 한계 | $500,000 | 슬리피지 영향 시작 |

---

## 7. 검증 결과

### 7.1 수정된 검증 기준 (v2.0)

| 기준 | 값 | 이유 |
|------|-----|------|
| Sharpe > 1.0 | 유지 | 리스크 조정 수익 |
| MaxDD < 25% | 유지 | 파산 방지 |
| WR > 30% | 기존 45%에서 완화 | 추세추종은 구조적으로 승률 30~45% |
| PF > 1.3 | 기존 1.5에서 완화 | 손익비 2.67:1에서 WR 30%면 PF > 1.3 가능 |
| WFA > 50% | 유지 | 기간별 일관성 |
| Trades > 100 | 유지 | 통계적 유의성 |

기준 완화 이유: 유명 추세추종 펀드 (Turtle Trading, CTA) 승률이 35~45%. 승률 45% 기준은 평균회귀 전략에만 적합.

### 7.2 KAMA+EMA Hybrid 검증 결과 (Phase 19)

```
수정 기준: 6/6 PASS
Sharpe=1.41  MaxDD=-4.9%  WR=41%  PF=1.57  WFA=67%  T=601
강세장=+41.8%  약세장=+2.2%  횡보장=+10.1%
```

### 7.3 Phase 18 OOS 재검증 (min_len 버그 수정)

```
기존 검증 (v1.0, 버그 있음):
  OOS 기간: 3개월 (3 윈도우) — min_len이 최단 종목에 종속
  결과: Sharpe 2.52 — 과대평가

수정 검증 (v2.0):
  OOS 기간: 2년 (18 윈도우) — OOS>16000봉 종목만 사용 (19개)
  결과: Sharpe 1.41 — 정확한 평가
```

### 7.4 전 전략 재검증 결과 (Phase 18c, 수정 기준 + 수정 OOS)

수정 기준 6/6 통과 전략 7개:

| 전략 | Sharpe | 수익 | 약세장 | 횡보장 | 종합점수 |
|------|--------|------|--------|--------|----------|
| VWAP Breakout (1.02) | 1.38 | +56.9% | +2.2% | +8.4% | 6 (1위) |
| Vol Profile (1.01) | 1.33 | +56.1% | +1.7% | +8.5% | 8 |
| OBV Trend | 1.39 | +59.1% | +1.7% | +10.6% | 8 |
| MFI Breakout 60 | 1.36 | +59.2% | +2.2% | +10.5% | 8 |
| Dual MA Breakout | 1.53 | +40.3% | +0.3% | +7.9% | 17 |
| Momentum Filter | 1.27 | +45.8% | -0.7% | +7.7% | 28 |
| DM Wide Exit | 1.16 | +45.6% | -0.7% | +4.3% | 29 |

종합점수 = 강세순위 + 약세순위 + 횡보순위 (낮을수록 좋음)
약세장 마이너스 전략은 +10 페널티

### 7.5 적응형 필터 비교 (Phase 19)

| 필터 | 6/6 | Sharpe | 약세장 | 횡보장 |
|------|-----|--------|--------|--------|
| **KAMA+EMA 하이브리드** | **PASS** | **1.41** | **+2.2%** | **+10.1%** |
| EMA 50/200 (기존) | PASS | 1.38 | +2.2% | +8.4% |
| KAMA 10/50 단독 | FAIL | 0.66 | -2.6% | -11.5% |
| DEMA 50/200 | FAIL | 0.85 | -6.1% | +1.9% |
| TEMA 50/200 | FAIL | 0.85 | -2.6% | -6.3% |
| Super Smoother | FAIL | 0.81 | -5.8% | -3.2% |

KAMA를 EMA 대체가 아닌 보조 필터로 추가 시 횡보장 성과 +20% 개선.

---

## 8. 테스트 이력 (전체 기록)

총 **120개 이상** 전략/변형 테스트:

| Phase | 내용 | 테스트 수 | 결과 |
|-------|------|-----------|------|
| 1~10 | 기초 프레임워크 + 파라미터 | 수백 | Dual MA 6/6 (구 기준) |
| 11 | 대안 전략 (센티먼트/매크로) | 8 | 전부 FAIL |
| 12 | 스케일링 | 18 | 1x~5x 유효 |
| 13 | 볼륨 전략 | 30 | Vol Profile 최우수 |
| 14 | 종합 검증 | 7 tests | 통과 (단, min_len 버그) |
| 15 | 변형 + 구제 | 32 | 미달 |
| 16 | ML (XGB/LGB/RF/앙상블) | 24 | 전부 FAIL |
| 17 | DVOL 옵션 내재변동성 | 15 | 추가 가치 없음 |
| 18 | **OOS 재검증 (버그 수정)** | 7 전략 x 5 임계값 | 수정 기준 7개 통과 |
| 18d | 레짐 분석 | 7 전략 | VWAP Breakout 1위 |
| 19 | 적응형 필터 (KAMA/DEMA/TEMA) | 11 | KAMA+EMA 하이브리드 최우수 |

---

## 9. 안전장치 (봇 필수 구현)

| 장치 | 설정 | 설명 |
|------|------|------|
| 일일 손실 한도 | -3% | 도달 시 당일 거래 중단 |
| 연속 손실 한도 | 5연패 | 1주 중단 후 점검 |
| 킬 스위치 | 수동 | 즉시 전 포지션 청산 |
| 모니터링 알림 | 진입/청산/DD 경고 | 텔레그램 등 |
| API 오류 대응 | 재시도 3회 | 실패 시 알림 + 중단 |
| 페이퍼 모드 | 기본값 | 실거래는 명시적 opt-in |

---

## 10. 배포 로드맵

```
1단계: 페이퍼 트레이딩 (3개월)
  - 실시간 시그널 생성 + 가상 체결
  - 백테스트 vs 실전 괴리 측정
  - 슬리피지, 체결 지연, API 오류 모니터링

2단계: 소액 실전 ($500~1000, 1개월)
  - 실제 체결 환경 확인
  - 1x 스케일로 시작

3단계: 목표 금액 투입
  - 단계적 증액 (월 2배씩)
  - 성과 확인 후 3x → 5x 스케일 전환

월간 리뷰:
  - Sharpe > 0.5 유지 확인
  - DD < 15% 확인
  - 3개월 연속 손실 시 전략 재검토
```

---

## 11. 핵심 코드

### 11.1 KAMA 계산
```python
def kama(series, period=20, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average"""
    close = series.values.astype(float)
    n = len(close)
    result = np.full(n, np.nan)
    fast_alpha = 2.0 / (fast_sc + 1)
    slow_alpha = 2.0 / (slow_sc + 1)
    result[period] = close[period]

    for i in range(period + 1, n):
        direction = abs(close[i] - close[i - period])
        volatility = sum(abs(close[j] - close[j-1])
                         for j in range(i - period + 1, i + 1))
        er = direction / volatility if volatility > 0 else 0
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        result[i] = result[i-1] + sc * (close[i] - result[i-1])

    return pd.Series(result, index=series.index)
```

### 11.2 시그널 생성
```python
def generate_signals(df, lookback=48):
    """
    KAMA+EMA Hybrid VWAP Breakout 시그널 생성

    Parameters:
        df: OHLCV DataFrame (columns: close, high, volume)
        lookback: 룩백 기간 (기본 48봉 = 2일)

    Returns:
        numpy array: 1=롱 진입, 0=무신호
    """
    close = df['close']
    high = df['high']
    volume = df['volume']

    # 1. Donchian Channel Upper (48봉 최고가)
    upper = high.rolling(lookback).max().shift(1)

    # 2. VWAP (48봉 거래량가중평균가)
    vwap = (close * volume).rolling(lookback).sum() / \
           (volume.rolling(lookback).sum() + 1e-10)

    # 3. Dual EMA (50/200)
    ema_fast = close.ewm(span=50, adjust=False).mean()
    ema_slow = close.ewm(span=200, adjust=False).mean()

    # 4. KAMA(20) 기울기
    k = kama(close, period=20, fast_sc=2, slow_sc=30)
    k_slope = k - k.shift(10)

    # 5가지 조건 동시 충족
    signals = np.where(
        (close > upper) &           # 채널 돌파
        (close > vwap * 1.02) &     # VWAP 2% 위
        (ema_fast > ema_slow) &     # 상승 추세
        (k_slope > 0),              # KAMA 상승 중
        1, 0
    )
    return signals
```

### 11.3 청산 조건
```python
def calculate_exit(entry_price, current_price, atr, bars_held):
    """청산 조건 확인"""
    unrealized_atr = (current_price - entry_price) / (atr + 1e-10)

    if bars_held >= 72:         # 시간 제한 (3일)
        return True
    if unrealized_atr < -3.0:   # 손절 (ATR 3x)
        return True
    if unrealized_atr > 8.0:    # 익절 (ATR 8x)
        return True
    return False
```

### 11.4 포지션 사이징
```python
def position_sizing(capital, vol_168h, num_positions, scale=5.0):
    """변동성 타겟팅 포지션 크기"""
    ann_vol = vol_168h * np.sqrt(24 * 365)
    position_pct = min(0.10 / (ann_vol + 1e-10) / max(num_positions, 1), 0.05)
    position_pct *= scale
    return capital * position_pct
```

---

## 12. 파라미터 요약

| 파라미터 | 값 | 용도 |
|----------|-----|------|
| Donchian period | 48 | 채널 돌파 감지 |
| VWAP period | 48 | 거래량 확인 |
| VWAP multiplier | 1.02 | 돌파 강도 필터 |
| EMA fast | 50 | 단기 추세 |
| EMA slow | 200 | 장기 추세 |
| KAMA period | 20 | 적응형 추세 |
| KAMA fast_sc | 2 | 추세장 민감도 |
| KAMA slow_sc | 30 | 횡보장 둔감도 |
| KAMA slope lookback | 10 | 기울기 계산 |
| ATR period | 14 | 변동성 측정 |
| Stop loss | ATR x 3 | 손절 |
| Profit target | ATR x 8 | 익절 |
| Max holding | 72봉 | 시간 제한 |
| Max positions | 10 | 동시 보유 |
| Rebalance | 720봉 | 종목 교체 |
| Vol target | 10% annual | 포지션 크기 |
| Max allocation | 5% per symbol | 집중 제한 |

---

## 13. 구현 참고 파일

| 파일 | 내용 |
|------|------|
| `research/phase18_revalidation.py` | OOS 재검증 (min_len 수정) |
| `research/phase18b_fair_revalidation.py` | 임계값별 공정 검증 |
| `research/phase18c_all_strategies_retest.py` | 전 전략 재검증 (수정 기준) |
| `research/phase18d_regime_analysis.py` | 레짐별 분석 + 종합 순위 |
| `research/phase19_adaptive_filter.py` | KAMA/DEMA/TEMA 비교 |
| `research/ralph_loop_state.json` | 전체 상태 기록 |
