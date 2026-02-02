"""
Round 12: Price Action & Candlestick Patterns
=============================================
Detects pure price action patterns without lagging indicators.
Focuses on Reversal and Continuation patterns.

Patterns:
1. Bullish Engulfing (Reversal): Day 1 Red, Day 2 Green covering Day 1 body.
2. Hammer (Reversal): Long lower shadow at downtrend.
3. Gap Up Continuation: Open > Prev High, Close > Open.
4. Inside Bar Breakout: Volatility contraction followed by expansion.

Exit:
- Fixed Hold (3-5 days) or Trailing Stop.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("E:/투자/data/kr_stock")
KOSPI_DIR = DATA_DIR / "kospi_ohlcv"
KOSDAQ_DIR = DATA_DIR / "kosdaq_ohlcv"
OUTPUT_PATH = Path("E:/투자/data/round12_results")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

TOTAL_COST = (0.00015 + 0.0020 + 0.0005) * 2 # Round trip costs

def identify_patterns(df):
    """Adds boolean columns for patterns."""
    o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_h, prev_l = h.shift(1), l.shift(1)
    
    # Body Size
    body = np.abs(c - o)
    prev_body = np.abs(prev_c - prev_o)
    
    # 1. Bullish Engulfing
    # Prev: Red, Curr: Green
    # Curr Open < Prev Close, Curr Close > Prev Open (Strict)
    # Or simply covers the body
    engulfing = (prev_c < prev_o) & (c > o) & (c > prev_o) & (o < prev_c)
    
    # 2. Hammer
    # Small body, Long lower shadow (> 2x body), Small upper shadow
    lower_shadow = np.minimum(o, c) - l
    upper_shadow = h - np.maximum(o, c)
    is_hammer = (lower_shadow > 2 * body) & (upper_shadow < body * 0.5)
    # Trend filter: Close < MA(20)
    ma20 = c.rolling(20).mean()
    downtrend = c < ma20
    hammer = is_hammer & downtrend
    
    # 3. Gap Up (Strong Momentum)
    # Open > Prev High
    gap_up = (o > prev_h * 1.01) & (c > o) # 1% gap and close higher
    
    return engulfing, hammer, gap_up

def run_price_action(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 500: return []
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            return []
            
        df = df.sort_index()
        
        engulfing, hammer, gap_up = identify_patterns(df)
        
        patterns = [
            (engulfing, "Bullish_Engulfing"),
            (hammer, "Hammer"),
            (gap_up, "Gap_Up")
        ]
        
        results = []
        
        for signal_series, name in patterns:
            # Exit Strategy: Hold 3 days
            hold_days = 3
            
            # Vectorized Return Calculation
            # Enter at Close of Signal Day (or Next Open) -> Here Close of Signal Day
            # Realistically, pattern is confirmed at Close. So enter at Close.
            
            # Future return (3 days later Close / Signal Close - 1)
            future_ret = df['Close'].shift(-hold_days) / df['Close'] - 1
            
            # Filter signals (Test Period)
            test_signals = signal_series["2022-01-01":"2024-12-31"]
            test_returns = future_ret["2022-01-01":"2024-12-31"]
            
            triggered_rets = test_returns[test_signals]
            
            if len(triggered_rets) < 10: continue
            
            # Apply costs
            net_rets = triggered_rets - TOTAL_COST
            
            win_rate = (net_rets > 0).mean()
            avg_return = net_rets.mean()
            total_return = net_rets.sum() # Simple sum of trade returns
            
            # Profit Factor
            gross_profit = net_rets[net_rets > 0].sum()
            gross_loss = abs(net_rets[net_rets < 0].sum())
            pf = gross_profit / gross_loss if gross_loss > 0 else 999
            
            # Heuristic Score: WinRate * AvgRet * Sqrt(Trades)
            score = win_rate * avg_return * np.sqrt(len(net_rets))
            
            results.append({
                "ticker": file_path.stem,
                "pattern": name,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "total_return": total_return,
                "pf": pf,
                "trades": len(net_rets),
                "score": score
            })
            
        return results
        
    except Exception:
        return []

def main():
    print("="*80)
    print("Round 12: Price Action Patterns")
    print("="*80)
    
    all_files = list(KOSPI_DIR.glob("*.csv")) + list(KOSDAQ_DIR.glob("*.csv"))
    
    all_results = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_price_action, f) for f in all_files]
        
        for f in tqdm(as_completed(futures), total=len(futures)):
            res = f.result()
            if res:
                all_results.extend(res)
                
    if not all_results:
        print("No results.")
        return
        
    df = pd.DataFrame(all_results)
    
    # Filter Viable
    viable = df[
        (df['win_rate'] > 0.6) & 
        (df['pf'] > 1.5) & 
        (df['trades'] >= 10)
    ].sort_values('score', ascending=False)
    
    print(f"\nTotal Viable Patterns: {len(viable)}")
    print(f"\nTOP 20 Price Action Patterns:")
    print(viable.head(20))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viable.to_csv(OUTPUT_PATH / f"round12_viable_{timestamp}.csv", index=False)

if __name__ == "__main__":
    main()
