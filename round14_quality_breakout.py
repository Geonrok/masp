"""
Round 14: Quality Breakout (Techno-Fundamental)
===============================================
Filters stocks based on "Price Quality" to avoid fake breakouts in penny stocks.
Hypothesis: Breakouts work better on stocks that are already in a long-term uptrend (Stage 2).

Filters:
1. Long-Term Trend: Close > SMA(200)
2. Momentum: 6-Month Return > 0
3. Low Volatility: Volatility(20) < Median(252) (Optional, or just not extreme)

Strategy:
- Buy if filters met AND Price > Open + 0.5 * Range
- Exit at Close
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
OUTPUT_PATH = Path("E:/투자/data/round14_results")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

TOTAL_COST = 0.0035

def run_quality_breakout(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 500: return []
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            return []
            
        df = df.sort_index()
        
        # 1. Quality Filters
        df['SMA200'] = df['Close'].rolling(200).mean()
        df['Ret6M'] = df['Close'].pct_change(120) # Approx 6 months
        
        # Filter: Price > SMA200 AND 6M Momentum > 0
        quality_filter = (df['Close'] > df['SMA200']) & (df['Ret6M'] > 0)
        # Shift filter to avoid lookahead (Filter must be valid YESTERDAY or TODAY OPEN)
        # Using Prev Close for filter check to be safe
        filter_ready = quality_filter.shift(1).fillna(False)
        
        # 2. Breakout Signal
        df['Range'] = df['High'] - df['Low']
        df['PrevRange'] = df['Range'].shift(1)
        target = df['Open'] + df['PrevRange'] * 0.5
        
        breakout = df['High'] > target
        
        # Combined Signal
        signal = breakout & filter_ready
        
        # Returns
        # Buy at Target, Sell at Close
        daily_ret = (df['Close'] - target) / target - TOTAL_COST
        
        # Apply Signal
        strat_ret = daily_ret.where(signal, 0)
        
        # Test Period
        test_ret = strat_ret["2022-01-01":"2024-12-31"]
        if len(test_ret) == 0: return []
        
        trades = signal["2022-01-01":"2024-12-31"].sum()
        if trades < 20: return []
        
        cum_ret = (1 + test_ret).cumprod()
        total_ret = cum_ret.iloc[-1] - 1
        
        if test_ret.std() == 0: return []
        sharpe = (test_ret.mean() * 252) / (test_ret.std() * np.sqrt(252))
        mdd = (cum_ret / cum_ret.cummax() - 1).min()
        win_rate = (test_ret > 0).sum() / trades
        
        return [{
            "ticker": file_path.stem,
            "strategy": "Quality_Breakout",
            "sharpe": sharpe,
            "return": total_ret,
            "mdd": mdd,
            "trades": trades,
            "win_rate": win_rate
        }]
        
    except Exception:
        return []

def main():
    print("="*80)
    print("Round 14: Quality Breakout (Trend Filtered)")
    print("="*80)
    
    all_files = list(KOSPI_DIR.glob("*.csv")) + list(KOSDAQ_DIR.glob("*.csv"))
    all_results = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_quality_breakout, f) for f in all_files]
        for f in tqdm(as_completed(futures), total=len(futures)):
            res = f.result()
            if res: all_results.extend(res)
            
    if not all_results:
        print("No results.")
        return
        
    df = pd.DataFrame(all_results)
    
    # Strict viable criteria
    viable = df[
        (df['sharpe'] > 1.0) &
        (df['mdd'] > -0.25) &
        (df['trades'] > 30)
    ].sort_values('sharpe', ascending=False)
    
    print(f"\nViable Quality Strategies: {len(viable)}")
    print(viable.head(20))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viable.to_csv(OUTPUT_PATH / f"round14_viable_{timestamp}.csv", index=False)

if __name__ == "__main__":
    main()
