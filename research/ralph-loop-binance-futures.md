# Ralph-Loop: Binance Futures Strategy Development
## Autonomous AI Strategy Research System

**Version**: 1.0.0  
**Created**: 2026-01-27  
**Target**: MASP (Multi-Asset Strategy Platform)  
**Methodology**: Geoffrey Huntley's Ralph-Loop (2025)

---

## 🎯 Mission Statement

You are an autonomous AI researcher tasked with discovering profitable, robust trading strategies for Binance Perpetual Futures. Your goal is to explore the complete strategy space without artificial restrictions, validate rigorously, and document all findings for human review.

**Core Principle**: Unrestricted exploration - limiting to validated factors prevents new discoveries. AI can handle large-scale search systematically.

---

## 📁 Data Environment

### Primary Data Location
```
E:\data\
├── crypto_ohlcv\
│   ├── binance_futures_1h\          # 1-hour OHLCV (primary)
│   ├── binance_futures_4h\          # 4-hour OHLCV
│   ├── binance_futures_1d\          # Daily OHLCV
│   ├── binance_futures_1w\          # Weekly OHLCV
│   ├── binance_funding_rate\        # Funding rates per symbol
│   ├── binance_open_interest\       # Open interest
│   ├── binance_long_short_ratio\    # Long/Short ratio
│   ├── binance_taker_volume\        # Taker buy/sell volume
│   ├── binance_spot_1h/4h/1d/       # Spot data (basis calculation)
│   ├── coingecko\                   # Market cap, volume
│   ├── coinglass\                   # Derivatives metrics
│   ├── defillama\                   # DeFi TVL
│   ├── macro\                       # Macro indicators
│   ├── onchain\                     # On-chain metrics
│   └── sentiment\                   # Fear/Greed, social
├── crypto_funding\binance_futures\  # Historical funding (legacy)
└── indicator_cache\                 # Pre-calculated indicators
```

### Expected CSV Format
```csv
timestamp,open,high,low,close,volume
2020-01-01 00:00:00,7200.5,7250.0,7180.0,7230.5,1234567.89
```

---

## 🔬 Research Protocol

### Phase 1: Data Discovery & Validation
```
□ 1.1 Enumerate all available symbols in binance_futures_1h
□ 1.2 Identify data start/end dates per symbol
□ 1.3 Check for gaps, anomalies, survivorship bias
□ 1.4 Calculate liquidity metrics (ADV, spread proxy)
□ 1.5 Create symbol universe tiers (Top 50, Top 100, All)
□ 1.6 Document data quality report → STATE_FILE
```

### Phase 2: Feature Engineering
```
□ 2.1 Price-based features
    - Returns (1h, 4h, 1d, 3d, 7d, 14d, 30d)
    - Volatility (realized, Parkinson, Garman-Klass)
    - ATR, Bollinger Band width
    - KAMA, DEMA, TEMA, Hull MA
    
□ 2.2 Volume-based features
    - VWAP deviation
    - Volume momentum (OBV, CMF, MFI)
    - Taker buy/sell imbalance
    - Volume profile levels
    
□ 2.3 Derivatives-specific features
    - Funding rate (raw, cumulative, z-score)
    - Open interest change rate
    - Long/short ratio deviation
    - Basis (futures vs spot)
    - Liquidation heatmap zones
    
□ 2.4 Cross-asset features
    - BTC correlation rolling
    - Sector momentum (DeFi, Layer1, Meme)
    - Market cap rank change
    
□ 2.5 Alternative data features
    - Fear/Greed regime
    - On-chain flow signals
    - Social sentiment scores
    
□ 2.6 Macro features
    - DXY momentum
    - US10Y yield change
    - SPX correlation regime
```

### Phase 3: Strategy Exploration (Unrestricted)
```
□ 3.1 Momentum Strategies
    - Time-series momentum (TSMOM)
    - Cross-sectional momentum (XSMOM)
    - KAMA-TSMOM hybrid
    - Dual momentum (absolute + relative)
    
□ 3.2 Mean Reversion Strategies
    - Funding rate mean reversion
    - Bollinger band reversals
    - RSI extremes with filters
    - Pairs trading (correlated assets)
    
□ 3.3 Breakout Strategies
    - Donchian channel breakouts
    - Volatility compression → expansion
    - Volume-confirmed breakouts
    - Support/resistance levels
    
□ 3.4 Carry Strategies
    - Funding rate carry (long positive, short negative)
    - Basis arbitrage signals
    - Roll yield extraction
    
□ 3.5 Sentiment Strategies
    - Funding rate crowding reversals
    - Open interest divergence
    - Long/short ratio extremes
    
□ 3.6 Machine Learning Strategies
    - Feature importance ranking
    - Ensemble classifiers
    - Regime detection (HMM, K-means)
    
□ 3.7 Hybrid/Composite Strategies
    - Multi-factor models
    - Strategy of strategies
    - Adaptive allocation
```

### Phase 4: Validation Framework
```
□ 4.1 In-Sample Backtest (IS)
    - Period: First 60% of data
    - Metrics: Sharpe, Sortino, Max DD, Win Rate
    - Slippage: 0.05% per trade
    - Commission: 0.04% (taker)
    
□ 4.2 Out-of-Sample Test (OOS)
    - Period: Next 20% of data
    - No parameter adjustment allowed
    - Record degradation ratio (OOS/IS)
    
□ 4.3 Walk-Forward Analysis (WFA)
    - Window: 180 days train, 30 days test
    - Rolling forward monthly
    - Calculate WFA efficiency ratio
    
□ 4.4 Stress Testing
    - COVID crash (March 2020)
    - May 2021 crash
    - FTX collapse (Nov 2022)
    - 2024 ETF rally
    - Recent 2025 volatility
    
□ 4.5 Statistical Tests
    - t-test for Sharpe > 0
    - Multiple testing correction (Bonferroni/FDR)
    - Monte Carlo permutation test
    - Bootstrap confidence intervals
```

### Phase 5: Production Readiness
```
□ 5.1 Position Sizing
    - Kelly criterion (fractional)
    - Volatility targeting
    - Maximum position limits
    
□ 5.2 Risk Management
    - Per-trade stop loss
    - Daily/weekly drawdown limits
    - Correlation-based position scaling
    
□ 5.3 Execution Considerations
    - Liquidity filtering (min ADV)
    - Entry/exit timing optimization
    - Order type selection (limit vs market)
    
□ 5.4 Live Trading Integration
    - Signal generation code
    - MASP StrategyRunner compatibility
    - Monitoring & alerting setup
```

---

## 📊 Success Criteria

### Minimum Requirements (MUST PASS)
| Metric | Threshold | Note |
|--------|-----------|------|
| Sharpe Ratio (OOS) | > 1.0 | After costs |
| Max Drawdown | < 25% | Any period |
| Win Rate | > 45% | For trend strategies |
| Profit Factor | > 1.5 | Gross profit / Gross loss |
| WFA Efficiency | > 50% | OOS vs IS performance |
| Trade Count | > 100 | Statistical significance |

### Desirable Characteristics
- Low correlation to BTC buy-and-hold
- Consistent performance across market regimes
- Reasonable turnover (< 2x daily for capacity)
- Scalable to $100K+ notional

---

## 🔄 State Management (Ralph-Loop Core)

### State File Location
```
E:\투자\Multi-Asset Strategy Platform\research\ralph_loop_state.json
```

### State Schema
```json
{
  "version": "1.0.0",
  "last_updated": "2026-01-27T12:00:00Z",
  "current_phase": "2",
  "current_task": "2.3",
  "completed_tasks": ["1.1", "1.2", "1.3"],
  "findings": {
    "data_quality": {...},
    "promising_strategies": [...],
    "failed_strategies": [...],
    "best_performers": [...]
  },
  "parameters": {
    "explored": {...},
    "optimal": {...}
  },
  "next_actions": ["Complete 2.3", "Start 2.4"],
  "blockers": [],
  "git_commits": ["abc1234", "def5678"]
}
```

### Git Workflow
```bash
# After each significant finding
git add -A
git commit -m "Ralph-Loop: [Phase X.Y] Description of progress"

# Tag promising strategies
git tag strategy-tsmom-v1 -m "TSMOM strategy OOS Sharpe 1.5"
```

---

## 🚨 Constraints & Guardrails

### DO NOT
- ❌ Optimize on full dataset (causes overfitting)
- ❌ Use future information (look-ahead bias)
- ❌ Ignore transaction costs
- ❌ Skip survivorship bias check
- ❌ Cherry-pick backtest periods
- ❌ Over-complicate (prefer simple, robust)

### ALWAYS
- ✅ Log all experiments to state file
- ✅ Use realistic slippage (0.05-0.1%)
- ✅ Validate on truly out-of-sample data
- ✅ Document failed strategies (learning value)
- ✅ Commit after each phase completion
- ✅ Consider capacity constraints

---

## 🛠 Technical Implementation

### Python Environment
```python
# Required packages
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Strategy framework
DATA_PATH = Path("E:/data/crypto_ohlcv")
STATE_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/ralph_loop_state.json")

def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"version": "1.0.0", "current_phase": "1", "completed_tasks": []}

def save_state(state):
    state["last_updated"] = datetime.now().isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))
```

### Backtest Template
```python
def backtest_strategy(signals: pd.DataFrame, 
                      prices: pd.DataFrame,
                      slippage: float = 0.0005,
                      commission: float = 0.0004) -> dict:
    """
    Standard backtest function for all strategies.
    
    Args:
        signals: DataFrame with columns [symbol, timestamp, signal] 
                 where signal in {-1, 0, 1}
        prices: DataFrame with columns [symbol, timestamp, close]
        slippage: Percentage slippage per trade
        commission: Percentage commission per trade
    
    Returns:
        dict with performance metrics
    """
    # Implementation here
    pass
```

---

## 📋 Execution Instructions

### For Claude Code / AI Agent
1. Read this entire prompt
2. Load or initialize state file
3. Check current_phase and current_task
4. Execute the next incomplete task
5. Update state with findings
6. Git commit progress
7. Loop until all phases complete OR blocker encountered

### For Human Review
1. Check `ralph_loop_state.json` for progress
2. Review git log for experiment history
3. Examine `promising_strategies` in state
4. Validate top performers manually
5. Approve or request additional exploration

---

## 🎬 Quick Start Command

```bash
# Initialize Ralph-Loop
cd "E:\투자\Multi-Asset Strategy Platform"
git checkout -b ralph-loop/binance-futures-v1

# Create research directory
mkdir -p research
cp ralph-loop-binance-futures.md research/

# Initialize state
python -c "
from pathlib import Path
import json
state = {
    'version': '1.0.0',
    'current_phase': '1',
    'current_task': '1.1',
    'completed_tasks': [],
    'findings': {},
    'next_actions': ['Enumerate symbols in binance_futures_1h']
}
Path('research/ralph_loop_state.json').write_text(json.dumps(state, indent=2))
print('State initialized!')
"

# Start autonomous loop
# (Run Claude Code or preferred AI agent with this prompt)
```

---

## 📝 Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-27 | Initial Ralph-Loop prompt |

---

**End of Ralph-Loop Prompt**

*This prompt is designed for autonomous execution. The AI agent should work through each phase systematically, documenting all findings, and only stopping when all phases are complete or a human-requiring blocker is encountered.*
