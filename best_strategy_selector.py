"""
MASP Strategy Selector & Recommender
====================================
Consolidates all discovered strategies from 9 rounds of backtesting.
Provides optimal strategy recommendations for any KOSPI/KOSDAQ ticker.

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
MASTER_DB_PATH = DATA_DIR / "master_strategy_map.json"

def find_result_files():
    """Find all viable strategy result files."""
    files = []
    # Round 1-3
    files.extend(DATA_DIR.glob("kospi_futures_backtest_results/*viable*.csv"))
    files.extend(DATA_DIR.glob("kospi_extended_backtest_results/*viable*.csv"))
    files.extend(DATA_DIR.glob("kospi_round3_results/*viable*.csv"))
    
    # Round 4-7
    files.extend(DATA_DIR.glob("kospi_round4_results/*viable*.csv"))
    files.extend(DATA_DIR.glob("kospi_round5_results/*viable*.csv"))
    files.extend(DATA_DIR.glob("kospi_round6_results/*viable*.csv"))
    files.extend(DATA_DIR.glob("kospi_round7_results/*viable*.csv"))
    
    # Round 8 (Pairs) - Handled separately usually, but included if structure matches
    # Round 9 (Full Universe)
    files.extend(DATA_DIR.glob("round9_results/*viable*.csv"))
    
    return files

def build_master_db():
    print("Building Master Strategy Database...")
    files = find_result_files()
    print(f"Found {len(files)} result files.")
    
    all_strategies = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            # Standardize column names
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure required columns exist
            required = ['ticker', 'strategy', 'sharpe', 'cagr', 'mdd']
            if not all(c in df.columns for c in required):
                continue
                
            # Add source file info for traceability
            df['source_round'] = f.parent.name
            
            all_strategies.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not all_strategies:
        print("No strategies found!")
        return
        
    full_df = pd.concat(all_strategies, ignore_index=True)
    
    # Filter valid tickers (numeric mostly for KR)
    # Some might be futures codes, keep them all
    
    # Sort by Sharpe and pick best per ticker
    # We want robust strategies.
    # Preference priority: Round 9 (Full Universe validated) > Round 7 (Ensemble) > Others
    
    full_df['priority'] = 0
    full_df.loc[full_df['source_round'].str.contains('round9'), 'priority'] = 3
    full_df.loc[full_df['source_round'].str.contains('round7'), 'priority'] = 2
    full_df.loc[full_df['source_round'].str.contains('round6'), 'priority'] = 1 # Seasonality is good
    
    # Sort: Priority (desc), Sharpe (desc)
    full_df = full_df.sort_values(['ticker', 'priority', 'sharpe'], ascending=[True, False, False])
    
    # Top 3 strategies per ticker
    top_strategies = full_df.groupby('ticker').head(3)
    
    # Convert to dictionary
    strategy_map = {}
    for _, row in top_strategies.iterrows():
        ticker = str(row['ticker']).zfill(6) # Ensure 6 digit
        if ticker not in strategy_map:
            strategy_map[ticker] = []
            
        strategy_map[ticker].append({
            "strategy": row['strategy'],
            "sharpe": round(row['sharpe'], 2),
            "cagr": round(row['cagr'] * 100, 1),
            "mdd": round(row['mdd'] * 100, 1),
            "source": row['source_round']
        })
        
    # Save
    with open(MASTER_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(strategy_map, f, indent=2, ensure_ascii=False)
        
    print(f"Database saved to {MASTER_DB_PATH}")
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
        print(f"{'Rank':<5} {'Strategy':<30} {'Sharpe':>8} {'CAGR':>8} {'MDD':>8} {'Source'}")
        print("-" * 80)
        
        for i, strat in enumerate(db[ticker]):
            print(f"{i+1:<5} {strat['strategy']:<30} {strat['sharpe']:>8.2f} {strat['cagr']:>7.1f}% {strat['mdd']:>7.1f}% {strat['source']}")
            
        best = db[ticker][0]
        print(f"\n>>> Recommended: {best['strategy']} (Sharpe: {best['sharpe']})")
    else:
        print(f"\nNo specific optimized strategy found for {ticker}.")
        print("Recommendation: Use 'AdaptiveEnsemble' or 'Month_11' as robust defaults.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--build-db':
        build_master_db()
    elif len(sys.argv) > 2 and sys.argv[1] == '--ticker':
        get_recommendation(sys.argv[2])
    else:
        print(__doc__)
        # Auto-build if arguments missing but we want to be helpful
        # build_master_db() 
