"""
MASP Strategy Selector & Recommender (Final Update)
===================================================
Consolidates all discovered strategies from 12 rounds of backtesting.
Now includes Volatility Breakout and Price Action Patterns.

Usage:
    python best_strategy_selector.py --ticker 005930
    python best_strategy_selector.py --build-db
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
import sys

DATA_DIR = Path("E:/투자/data")
MASTER_DB_PATH = DATA_DIR / "master_strategy_map_v2.json"

def find_result_files():
    """Find all viable strategy result files."""
    files = []
    # Previous Rounds
    files.extend(DATA_DIR.glob("kospi_futures_backtest_results/*viable*.csv"))
    files.extend(DATA_DIR.glob("kospi_extended_backtest_results/*viable*.csv"))
    files.extend(DATA_DIR.glob("kospi_round3_results/*viable*.csv"))
    files.extend(DATA_DIR.glob("kospi_round4_results/*viable*.csv"))
    files.extend(DATA_DIR.glob("kospi_round5_results/*viable*.csv"))
    files.extend(DATA_DIR.glob("kospi_round6_results/*viable*.csv"))
    files.extend(DATA_DIR.glob("kospi_round7_results/*viable*.csv"))
    files.extend(DATA_DIR.glob("round9_results/*viable*.csv"))
    
    # New Rounds (11 & 12)
    files.extend(DATA_DIR.glob("round11_results/*viable*.csv"))
    files.extend(DATA_DIR.glob("round12_results/*viable*.csv"))
    
    return files

def build_master_db():
    print("Building Master Strategy Database V2...")
    files = find_result_files()
    print(f"Found {len(files)} result files.")
    
    all_strategies = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.lower() for c in df.columns]
            
            # Normalize column names for new rounds
            if 'return' in df.columns and 'cagr' not in df.columns:
                df['cagr'] = df['return'] # Proxy for simple sorting
            
            if 'pattern' in df.columns:
                df['strategy'] = df['pattern']
            
            if 'k' in df.columns:
                df['strategy'] = 'VolBreakout_k' + df['k'].astype(str)
                
            required = ['ticker', 'strategy', 'sharpe']
            if not all(c in df.columns for c in required):
                continue
                
            df['source_round'] = f.parent.name
            all_strategies.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not all_strategies:
        print("No strategies found!")
        return
        
    full_df = pd.concat(all_strategies, ignore_index=True)
    
    # Priority Logic
    # 1. Round 11 (Vol Breakout) - High Return
    # 2. Round 9 (Full Universe) - Validated
    # 3. Round 12 (Price Action) - High Win Rate
    
    full_df['priority'] = 0
    full_df.loc[full_df['source_round'].str.contains('round11'), 'priority'] = 4
    full_df.loc[full_df['source_round'].str.contains('round9'), 'priority'] = 3
    full_df.loc[full_df['source_round'].str.contains('round12'), 'priority'] = 2
    
    # Sort
    full_df = full_df.sort_values(['ticker', 'priority', 'sharpe'], ascending=[True, False, False])
    
    # Top 3 per ticker
    top_strategies = full_df.groupby('ticker').head(3)
    
    # Build Map
    strategy_map = {}
    for _, row in top_strategies.iterrows():
        ticker = str(row['ticker']).zfill(6)
        
        if ticker not in strategy_map:
            strategy_map[ticker] = []
            
        # Handle missing columns gracefully
        cagr = row.get('cagr', 0)
        mdd = row.get('mdd', 0)
        
        strategy_map[ticker].append({
            "strategy": row['strategy'],
            "sharpe": round(row['sharpe'], 2),
            "cagr_or_ret": round(cagr * 100, 1),
            "mdd": round(mdd * 100, 1),
            "source": row['source_round']
        })
        
    with open(MASTER_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(strategy_map, f, indent=2, ensure_ascii=False)
        
    print(f"Database V2 saved to {MASTER_DB_PATH}")
    print(f"Total Tickers Covered: {len(strategy_map)}")

def get_recommendation(ticker):
    if not MASTER_DB_PATH.exists():
        print("Database not found. Please run with --build-db first.")
        return
        
    with open(MASTER_DB_PATH, 'r', encoding='utf-8') as f:
        db = json.load(f)
        
    ticker = str(ticker).zfill(6)
    
    if ticker in db:
        print(f"\nOptimization Results for {ticker}:")
        print(f"{'Rank':<5} {'Strategy':<30} {'Sharpe':>8} {'Return':>8} {'MDD':>8} {'Source'}")
        print("-" * 90)
        
        for i, strat in enumerate(db[ticker]):
            print(f"{i+1:<5} {strat['strategy']:<30} {strat['sharpe']:>8.2f} {strat['cagr_or_ret']:>7.1f}% {strat['mdd']:>7.1f}% {strat['source']}")
            
        best = db[ticker][0]
        print(f"\n>>> Recommended: {best['strategy']} (Sharpe: {best['sharpe']})")
        
        if "VolBreakout" in best['strategy']:
            print("Note: This is an intraday strategy. Buy Stop Order = Open + (PrevRange * k)")
    else:
        print(f"\nNo specific optimized strategy found for {ticker}.")
        print("Recommendation: Use 'AdaptiveEnsemble' or 'Month_11' as robust defaults.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--build-db':
        build_master_db()
    elif len(sys.argv) > 2 and sys.argv[1] == '--ticker':
        get_recommendation(sys.argv[2])
    else:
        # Default behavior
        build_master_db()
