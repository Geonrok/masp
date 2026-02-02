"""
Round 9: Full Universe Strategy Scan (KOSPI & KOSDAQ)
=====================================================
Run the top performing strategies on the entire KOSPI and KOSDAQ universe (approx. 2700 stocks).

Target Strategies:
1. AdaptiveEnsemble_Long (Dynamic Weighting)
2. Month_11_Long (Seasonality)
3. RF_Signal_Long (ML Proxy)
4. VolTiming_Long (Low Volatility Anomaly)
5. MarketTiming_Trend_Long (Trend Following)

Criteria for Viable:
- Test Sharpe > 0.5
- Test CAGR > 10%
- Test MDD > -30%
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configuration
KOSPI_PATH = Path("E:/투자/data/kr_stock/kospi_ohlcv")
KOSDAQ_PATH = Path("E:/투자/data/kr_stock/kosdaq_ohlcv")
OUTPUT_PATH = Path("E:/투자/data/round9_results")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2016-01-01"
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2024-12-31"

TOTAL_COST = (0.00015 + 0.0020 + 0.0005) # Spot costs: Commission + Tax + Slippage ~ 0.265%

# =============================================================================
# Indicators
# =============================================================================
def sma(s, p): return s.rolling(p, min_periods=1).mean()
def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + g/(l + 1e-10)))
def volatility(s, p=20): return s.pct_change().rolling(p).std() * np.sqrt(252)
def momentum(s, p=60): return s / s.shift(p) - 1
def macd(s, fast=12, slow=26, signal=9):
    fast_ema = s.ewm(span=fast, adjust=False).mean()
    slow_ema = s.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# =============================================================================
# Strategies
# =============================================================================
def strategy_adaptive_ensemble(df):
    close = df['Close']
    vol = volatility(close, 20)
    vol_median = vol.rolling(252).median()
    high_vol = vol > vol_median
    trend_up = close > sma(close, 100)
    
    sig_trend = (close > sma(close, 50)).astype(float)
    sig_mom = (momentum(close, 30) > 0).astype(float)
    rsi_val = rsi(close, 14)
    sig_rsi = ((rsi_val > 35) & (rsi_val < 65)).astype(float)
    
    w_trend = pd.Series(0.4, index=df.index)
    w_mom = pd.Series(0.4, index=df.index)
    w_rsi = pd.Series(0.2, index=df.index)
    
    w_trend[high_vol] = 0.5
    w_mom[high_vol] = 0.3
    w_rsi[high_vol] = 0.2
    
    w_trend[~trend_up] = 0.6
    w_mom[~trend_up] = 0.2
    w_rsi[~trend_up] = 0.2
    
    score = (sig_trend * w_trend) + (sig_mom * w_mom) + (sig_rsi * w_rsi)
    return (score > 0.5).astype(int)

def strategy_month_11(df):
    signal = (df.index.month == 11).astype(int)
    return pd.Series(signal, index=df.index)

def strategy_rf_signal(df):
    close = df['Close']
    ret_5 = close.pct_change(5)
    ret_10 = close.pct_change(10)
    ret_20 = close.pct_change(20)
    sma_5_ratio = close / sma(close, 5) - 1
    sma_20_ratio = close / sma(close, 20) - 1
    macd_l, sig_l = macd(close)
    macd_hist = macd_l - sig_l
    rsi_val = rsi(close)
    
    score = (
        (ret_5 > 0).astype(float) * 0.15 +
        (ret_10 > 0).astype(float) * 0.15 +
        (ret_20 > 0).astype(float) * 0.20 +
        (sma_5_ratio > 0).astype(float) * 0.10 +
        (sma_20_ratio > 0).astype(float) * 0.15 +
        (macd_hist > 0).astype(float) * 0.15 +
        ((rsi_val > 30) & (rsi_val < 70)).astype(float) * 0.10
    )
    return (score > 0.5).astype(int)

def strategy_vol_timing(df):
    vol = volatility(df['Close'], 20)
    # Low vol anomaly: Invest when vol is low (below annual median)
    vol_median = vol.rolling(252).median()
    return (vol < vol_median).astype(int)

def strategy_market_timing_trend(df):
    # Simple 100-day trend following
    return (df['Close'] > sma(df['Close'], 100)).astype(int)

# =============================================================================
# Backtest Engine
# =============================================================================
def run_backtest(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) < 500: 
            # print(f"Skip {file_path.name}: too short {len(df)}")
            return []
        
        # Date parsing
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            # print(f"Skip {file_path.name}: no Date column {df.columns}")
            return []
            
        df = df.sort_index()
        ticker = file_path.stem
        
        # Split Data
        train_df = df[TRAIN_START:TRAIN_END]
        test_df = df[TEST_START:TEST_END]
        
        if len(test_df) < 200: return []
        
        results = []
        strategies = [
            (strategy_adaptive_ensemble, "AdaptiveEnsemble"),
            (strategy_month_11, "Month_11"),
            (strategy_rf_signal, "RF_Signal"),
            (strategy_vol_timing, "VolTiming"),
            (strategy_market_timing_trend, "Trend_100"),
        ]
        
        for func, name in strategies:
            # 1. Generate Signal on full data to avoid lookahead bias at boundaries
            signal = func(df)
            
            # 2. Slice for Train/Test
            for period, sub_df in [("Train", train_df), ("Test", test_df)]:
                if len(sub_df) == 0: continue
                
                sig = signal.loc[sub_df.index].fillna(0)
                pos = sig.shift(1).fillna(0) # Enter on next day open/close
                
                ret = sub_df['Close'].pct_change().fillna(0)
                strat_ret = pos * ret
                
                # Costs
                trades = pos.diff().abs().fillna(0)
                costs = trades * TOTAL_COST
                net_ret = strat_ret - costs
                
                cum_ret = (1 + net_ret).cumprod()
                total_ret = cum_ret.iloc[-1] - 1
                
                days = (sub_df.index[-1] - sub_df.index[0]).days
                cagr = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
                
                vol = net_ret.std() * np.sqrt(252)
                sharpe = (net_ret.mean() * 252) / (vol + 1e-10)
                mdd = (cum_ret / cum_ret.cummax() - 1).min()
                
                results.append({
                    "ticker": ticker,
                    "strategy": name,
                    "period": period,
                    "sharpe": sharpe,
                    "cagr": cagr,
                    "mdd": mdd,
                    "trades": trades.sum()
                })
                
        return results
        
    except Exception as e:
        print(f"Error {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

# =============================================================================
# Main
# =============================================================================
def main():
    print("="*80)
    print("Round 9: Full Universe Strategy Scan")
    print("="*80)
    
    # Collect all files
    kospi_files = list(KOSPI_PATH.glob("*.csv"))
    kosdaq_files = list(KOSDAQ_PATH.glob("*.csv"))
    all_files = kospi_files + kosdaq_files
    
    print(f"Total files to scan: {len(all_files)}")
    
    # Run Parallel
    all_results = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_backtest, f) for f in all_files]
        
        for f in tqdm(as_completed(futures), total=len(futures)):
            res = f.result()
            if res:
                all_results.extend(res)
                
    # Process Results
    if not all_results:
        print("No results generated.")
        return
        
    df_res = pd.DataFrame(all_results)
    
    # Pivot to have Train/Test columns
    # We need to reshape: ticker, strategy, period -> metrics
    # This is complex with pivot_table if we have multiple metrics
    
    # Let's just filter Test period for now and join Train Sharpe
    test_res = df_res[df_res['period'] == 'Test'].copy()
    train_res = df_res[df_res['period'] == 'Train'][['ticker', 'strategy', 'sharpe']].copy()
    train_res.columns = ['ticker', 'strategy', 'train_sharpe']
    
    final_df = pd.merge(test_res, train_res, on=['ticker', 'strategy'], how='left')
    
    # Filter Viable
    viable = final_df[
        (final_df['sharpe'] > 0.5) & 
        (final_df['cagr'] > 0.10) & 
        (final_df['mdd'] > -0.30) &
        (final_df['train_sharpe'] > 0.3)
    ].sort_values('sharpe', ascending=False)
    
    print(f"\nAnalysis Complete!")
    print(f"Total Strategies Tested: {len(test_res)}")
    print(f"Viable Strategies Found: {len(viable)}")
    
    if not viable.empty:
        print(f"\nTOP 20 Strategies:")
        print(viable[['ticker', 'strategy', 'sharpe', 'cagr', 'mdd', 'trades']].head(20))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viable.to_csv(OUTPUT_PATH / f"round9_viable_{timestamp}.csv", index=False)
        final_df.to_csv(OUTPUT_PATH / f"round9_all_{timestamp}.csv", index=False)
        print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
