# 전략 개발을 위한 데이터 소스 가이드

## 현재 보유 데이터

| 데이터 | 기간 | 위치 |
|--------|------|------|
| OHLCV (4H) | 2020-2026 | `binance_futures_4h/` |
| Fear & Greed | 2018-2025 | `FEAR_GREED_INDEX.csv` |
| Funding Rate | 2020-2026 | `binance_funding_rate/` |
| VIX | 장기 | `macro/VIX.csv` |
| DXY, GOLD, etc. | 장기 | `macro/` |

## 필요한 추가 데이터

### 1. Open Interest 장기 데이터 (HIGH PRIORITY)

**무료 소스:**
- **Coinglass**: https://www.coinglass.com/
  - 웹 스크래핑 또는 API ($0-99/월)
  - 2020년부터 일별 OI 제공

- **The Block Data**: https://www.theblock.co/data
  - 일부 무료 차트 제공

**수집 방법 (Coinglass):**
```python
# Selenium으로 스크래핑
from selenium import webdriver
# Coinglass에서 CSV 다운로드 자동화
```

### 2. Liquidation 데이터 (HIGH PRIORITY)

**무료 소스:**
- **Coinglass Liquidation**: https://www.coinglass.com/LiquidationData
- **Bybit API**: 청산 데이터 제공

**기대 효과:**
- 대규모 청산 = 공포 저점 포착
- Profit Factor +0.15 ~ +0.25 예상

### 3. 온체인 데이터 (MEDIUM PRIORITY)

**유료 소스 (권장):**
- **Glassnode** ($29/월): https://glassnode.com
  - Exchange Netflow
  - MVRV, SOPR
  - Active Addresses

- **CryptoQuant** ($29/월): https://cryptoquant.com
  - Exchange Reserve
  - Whale Ratio

**무료 대안:**
- **Blockchain.com**: https://www.blockchain.com/charts
  - 기본 온체인 메트릭

### 4. 센티먼트 데이터 (LOW PRIORITY)

**무료:**
- **Google Trends**: pytrends 라이브러리
```python
from pytrends.request import TrendReq
pytrends = TrendReq()
pytrends.build_payload(["bitcoin"])
df = pytrends.interest_over_time()
```

## 데이터 수집 우선순위

```
1단계 (무료, 즉시 가능):
├── Funding Rate (2020-2026) ✅ 완료
├── Google Trends (Bitcoin 검색량)
└── Coinglass 웹 스크래핑 (OI, Liquidation)

2단계 (무료 API):
├── Bybit Liquidation API
└── Alternative.me Fear & Greed 업데이트

3단계 (유료, 필요시):
├── Glassnode ($29/월) - 온체인 데이터
└── CryptoQuant ($29/월) - Exchange Flow
```

## 데이터 수집 스크립트 위치

```
scripts/
├── collect_binance_data.py    # Binance Futures 데이터
├── collect_coinglass.py       # Coinglass 스크래핑 (TODO)
├── collect_google_trends.py   # Google Trends (TODO)
└── collect_onchain.py         # 온체인 데이터 (TODO)
```

## 전략 개선 예상 효과

| 데이터 추가 | 현재 PF | 예상 PF |
|------------|---------|---------|
| Funding Rate만 | 1.05 | 1.05 |
| + OI 장기 데이터 | - | 1.15 |
| + Liquidation | - | 1.25 |
| + Exchange Flow | - | 1.35 |
| **목표** | - | **>1.3** |

## 다음 단계

1. Google Trends 수집 스크립트 작성
2. Coinglass 스크래퍼 작성 (OI + Liquidation)
3. 수집된 데이터로 v10 전략 개발
4. Paper Trading 6개월 검증
