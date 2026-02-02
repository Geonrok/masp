# TIGER 200 ETF 전략 가이드

## 개요

KOSPI200 지수를 추종하는 **TIGER 200 ETF(102110)**에 검증된 VIX 기반 마켓 타이밍 전략을 적용합니다.

| 항목 | 내용 |
|------|------|
| 전략 ID | `tiger200_etf_v1` |
| 대상 ETF | TIGER 200 (102110) |
| 운용사 | 미래에셋자산운용 |
| 총보수 | 0.05% |
| 레버리지 | 1배 (원지수 추종) |
| 최소 투자금 | ~4만원 (1주) |

---

## 예상 성과

| 지표 | 값 | 비고 |
|------|-----|------|
| **Sharpe Ratio** | 2.37 | 우수 |
| **CAGR** | 23.1% | 연평균 수익률 |
| **MDD** | -11.5% | 최대 낙폭 |
| COVID 2020.03 | +1.8% | 위기 방어 |
| 금리인상 2022 | +8.5% | 하락장 방어 |

---

## 전략 구성

### 3개 서브 전략의 가중 조합

```
Composite Weight = Σ(전략별 시그널 × 가중치)
```

| 전략 | 가중치 | 조건 | 의미 |
|------|--------|------|------|
| VIX Below SMA20 | **50%** | VIX(T-1) < VIX_SMA20 | 시장 공포 평균 이하 |
| VIX Declining | **30%** | VIX(T-1) < VIX(T-2) | 공포 감소 중 |
| Semicon+Foreign | **20%** | 반도체↑ AND 외국인 순매수 | 팩터 강세 |

---

## 매매 규칙

### 시그널 판단

```python
if composite_weight >= 0.5:
    action = "BUY"      # 매수 (50% 이상)
elif composite_weight < 0.2:
    action = "SELL"     # 매도 (20% 미만)
else:
    action = "HOLD"     # 보유 유지
```

### 시그널 예시

| 상황 | VIX<SMA | VIX↓ | 반도체+외국인 | Composite | 행동 |
|------|---------|------|--------------|-----------|------|
| 전체 강세 | 1 | 1 | 1 | 100% | **BUY** |
| VIX 강세 | 1 | 1 | 0 | 80% | **BUY** |
| VIX만 SMA 하회 | 1 | 0 | 0 | 50% | **BUY** |
| VIX만 하락 | 0 | 1 | 0 | 30% | HOLD |
| 팩터만 강세 | 0 | 0 | 1 | 20% | HOLD |
| 전체 약세 | 0 | 0 | 0 | 0% | **SELL** |

---

## 서브 전략 상세

### 1. VIX Below SMA20 (가중치 50%)

**A+ 등급 전략** - 가장 높은 성과

```python
def signal():
    vix_t1 = vix.iloc[-2]           # T-1 VIX (어제 미국 종가)
    vix_sma = vix[:-1].rolling(20).mean().iloc[-1]  # T-1 기준 SMA

    return 1 if vix_t1 < vix_sma else 0
```

| 지표 | 값 |
|------|-----|
| Sharpe | 2.25 |
| CAGR | 27.8% |
| MDD | -12.0% |

**원리**: VIX가 평균보다 낮으면 시장 공포가 낮음 → 상승 가능성 높음

### 2. VIX Declining (가중치 30%)

**A+ 등급 전략** - 위기 방어 최고

```python
def signal():
    vix_t1 = vix.iloc[-2]    # T-1 VIX
    vix_t2 = vix.iloc[-3]    # T-2 VIX

    return 1 if vix_t1 < vix_t2 else 0
```

| 지표 | 값 |
|------|-----|
| Sharpe | 1.86 |
| CAGR | 19.1% |
| MDD | -13.3% |
| COVID 2020.03 | **+17.7%** |
| 금리인상 2022 | **+20.6%** |

**원리**: VIX가 하락 중이면 공포 해소 → 반등 시작

### 3. Semicon + Foreign (가중치 20%)

**A 등급 전략** - 팩터 기반

```python
def signal():
    semicon_bullish = semicon.iloc[-1] > semicon.rolling(20).mean().iloc[-1]
    foreign_bullish = foreign_flow.iloc[-20:].sum() > 0

    return 1 if (semicon_bullish and foreign_bullish) else 0
```

| 지표 | 값 |
|------|-----|
| Sharpe | 1.53 |
| CAGR | 16.6% |
| MDD | -14.8% |

**원리**: 반도체 강세 + 외국인 매수 = 한국 시장 상승 신호

---

## T-1 시차 처리 (중요)

### 왜 T-1 데이터를 사용하는가?

```
미국 시장 종가: 한국 시간 06:00 (다음날)
한국 시장 개장: 한국 시간 09:00

→ 한국 T일 거래 시점에 미국 T일 종가는 아직 없음
→ 미국 T-1일 종가(어제)를 사용해야 함
```

### 코드 구현

```python
# 올바른 구현 (Look-ahead bias 방지)
current_vix = self._vix_data.iloc[-2]  # T-1 VIX
prev_vix = self._vix_data.iloc[-3]      # T-2 VIX

# 잘못된 구현 (Look-ahead bias 발생)
# current_vix = self._vix_data.iloc[-1]  # 미래 데이터 사용!
```

---

## 사용 방법

### 기본 사용

```python
from libs.strategies.tiger200_etf import TIGER200Strategy

# 전략 초기화
strategy = TIGER200Strategy()

# 데이터 로드
strategy.load_data()

# 시그널 계산
composite, signals, action = strategy.calculate_signal()

print(f"Composite Weight: {composite:.0%}")
print(f"Individual Signals: {signals}")
print(f"Action: {action}")
```

### Strategy Loader 사용

```python
from libs.strategies.loader import get_strategy

# 기본 전략
strategy = get_strategy("tiger200_etf_v1")

# 안정형 (동일)
strategy = get_strategy("tiger200_stable")

# VIX 단일 전략
strategy = get_strategy("tiger200_vix_only")
```

### 시그널 생성

```python
signals = strategy.generate_signals(["102110"])

for signal in signals:
    print(f"Symbol: {signal.symbol}")
    print(f"Signal: {signal.signal}")  # BUY, SELL, HOLD
    print(f"Strength: {signal.strength:.0%}")
    print(f"Reason: {signal.reason}")
```

---

## 전략 변형

### 1. TIGER200StableStrategy (기본과 동일)

```python
strategy_id = "tiger200_stable"
가중치 = VIX 50% + VIX하락 30% + 반도체외국인 20%
```

### 2. TIGER200VIXOnlyStrategy

```python
strategy_id = "tiger200_vix_only"
가중치 = VIX_Below_SMA20 100%
```

가장 단순하고 성과가 좋은 단일 전략만 사용

---

## 거래 비용

| 항목 | 비율 | 비고 |
|------|------|------|
| 매수 수수료 | 0.015% | 증권사별 상이 |
| 매도 수수료 | 0.015% | 증권사별 상이 |
| 매도세 | **0%** | ETF 비과세 |
| 총 왕복 비용 | ~0.03% | 선물 대비 저렴 |

---

## KOSPI200 선물 vs TIGER 200 ETF

| 항목 | 선물 | ETF |
|------|------|-----|
| 최소 투자금 | ~1,500만원 | ~4만원 |
| 레버리지 | 조절 가능 | 1배 고정 |
| 증거금 | 필요 | 불필요 |
| 만기/롤오버 | 있음 | 없음 |
| 공매도 | 가능 | 불가 |
| 거래 비용 | 0.09% | 0.03% |
| 운용 난이도 | 높음 | 낮음 |

**권장**:
- 소액 투자자 → TIGER 200 ETF
- 레버리지/공매도 필요 → 선물
- 장기 보유 → ETF (롤오버 비용 없음)

---

## 주의사항

### 1. 데이터 의존성

전략 실행에 필요한 데이터:
- VIX 일봉 (Yahoo Finance)
- 반도체 지수 (SOXX 등)
- 외국인 투자자 매매 동향

데이터 갱신이 지연되면 시그널 정확도 저하

### 2. 시장 레짐 의존성

| 시장 상황 | 전략 성과 |
|----------|----------|
| 상승장 | 우수 (대부분 매수 유지) |
| 하락장 | 우수 (조기 매도) |
| 횡보장 | 보통 (잦은 시그널 변경) |

### 3. 과거 성과 ≠ 미래 성과

백테스트 결과는 참고용이며 실제 성과는 다를 수 있습니다.

---

## 파일 위치

```
libs/strategies/tiger200_etf.py          # 전략 구현
libs/strategies/loader.py                # 전략 로더 (등록됨)
docs/strategies/TIGER200_ETF_전략_가이드.md  # 본 문서
```

---

## 버전 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.0.0 | 2026-01-30 | 최초 작성 |

---

## 관련 전략

- `kospi200_stable_portfolio` - KOSPI200 선물용 동일 로직
- `kospi200_vix_below_sma20` - VIX 단일 전략 (선물)
- `kospi200_vix_declining` - VIX 하락 전략 (선물)
