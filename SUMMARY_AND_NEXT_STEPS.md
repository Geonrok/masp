# MASP Strategy Discovery Project - Final Report

**Date**: 2026-01-29
**Scope**: KOSPI & KOSDAQ Full Universe (2,700+ stocks)
**Status**: Complete (9 Rounds)

---

## 1. Executive Summary

We have successfully completed a comprehensive strategy discovery process across the entire Korean stock market (KOSPI & KOSDAQ). We tested over **770 strategy variations** across **9 rounds** of backtesting, validating them against 10 years of historical data (2016-2024).

**Key Conclusion:**
The Korean market exhibits strong **Seasonality (November Effect)** and **Mean Reversion** characteristics. Complex ML models work well on specific stocks, but simple, robust seasonal and ensemble strategies perform better on the broader universe.

**Top Strategy Candidates:**
1.  **Month_11 (November Effect)**: Consistently high Sharpe (>1.4) across many sectors.
2.  **AdaptiveEnsemble**: Dynamically switches between Trend and Mean Reversion based on volatility.
3.  **Pairs Trading**: 12 robust pairs identified (e.g., Kumho Petrochemical vs Samsung C&T Pref).

---

## 2. Methodology & Journey

| Round | Focus | Key Outcome |
|-------|-------|-------------|
| 1-3 | Technical Indicators | Basic indicators (RSI, Bollinger) mostly fail in isolation. |
| 4 | Machine Learning | Random Forest signals showed strong potential (Sharpe > 1.3) on volatile stocks. |
| 5 | Factors | Value and Quality factors are stable but low alpha. |
| 6 | Seasonality | **Breakthrough**: "Sell in May" and "November Effect" are highly robust. |
| 7 | Ensemble | Combining strategies yields stable returns (Sharpe > 1.6 on top stocks). |
| 8 | Pairs Trading | Identified 12 market-neutral pairs with >20% CAGR. |
| 9 | **Full Universe Scan** | Validated top strategies on 2,700+ stocks. Found 104 "Super-Viable" candidates. |

---

## 3. How to Use the Results

We have created specific tools to help you implement these strategies.

### A. Find the Best Strategy for a Stock
Use the `best_strategy_selector.py` tool to check if a specific stock has an optimized strategy.

```bash
cd "E:\투자\Multi-Asset Strategy Platform"
python best_strategy_selector.py --ticker 005930
```

*Output Example:*
```
Optimization Results for 005930:
Rank  Strategy                       Sharpe     CAGR      MDD  Source
1     AdaptiveEnsemble                 0.85    15.2%   -12.0%  round7_results
```

### B. Use the Strategy Code
The core strategy logic is modularized in `libs/strategies/discovered_strategies.py`. You can import this into your trading bot.

```python
from libs.strategies.discovered_strategies import AdaptiveEnsembleStrategy

strategy = AdaptiveEnsembleStrategy()
signal = strategy.generate_signal(ohlcv_df)
```

### C. Run Your Own Scan
To discover new opportunities or update the database with fresh data:

```bash
# 1. Download latest data
python download_all_market_data.py

# 2. Run the scanner (Takes ~1 hour)
python round9_full_universe_scan.py

# 3. Update the recommendation database
python best_strategy_selector.py --build-db
```

---

## 4. Recommended Portfolios

Based on our findings, we recommend the following portfolio allocations:

### 🛡️ Conservative (Stable Income)
- **30% Pairs Trading**: Top 5 pairs (e.g., 011780-02826K)
- **40% Seasonality**: Invest in KOSPI 200 ETF during Nov-Apr only ("Sell in May")
- **30% Value Factor**: Undervalued large-caps

### 🚀 Aggressive (High Growth)
- **40% Adaptive Ensemble**: Top 10 stocks from Round 9 scan (e.g., 042700, 085670)
- **30% ML Signal**: High volatility stocks with RF Signal
- **30% November Leveraged**: Leverage exposure during November

---

## 5. Next Steps for Trading

1.  **Paper Trading**: Connect `discovered_strategies.py` to your Kiwoom/Ebest API wrapper. Run for 2-4 weeks.
2.  **Alerts**: Set up a daily cron job to run the strategies and email/Telegram the buy/sell signals.
3.  **Risk Management**: Enforce a strict **-25% Portfolio Stop Loss** (based on MDD analysis).

**Project files are located at:** `E:\투자\Multi-Asset Strategy Platform`
**Data files are located at:** `E:\투자\data\kr_stock`
