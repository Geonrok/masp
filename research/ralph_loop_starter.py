"""
Ralph-Loop Starter Script
Binance Futures Strategy Research System

Usage:
    python ralph_loop_starter.py --phase 1
    
This script initializes and manages the Ralph-Loop autonomous research cycle.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import glob
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("E:/data/crypto_ohlcv")
PROJECT_PATH = Path("E:/투자/Multi-Asset Strategy Platform")
STATE_PATH = PROJECT_PATH / "research" / "ralph_loop_state.json"
RESULTS_PATH = PROJECT_PATH / "research" / "results"

# Ensure directories exist
(PROJECT_PATH / "research").mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================
# STATE MANAGEMENT
# ============================================================

def load_state() -> Dict:
    """Load Ralph-Loop state from file."""
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding='utf-8'))
    return initialize_state()

def save_state(state: Dict) -> None:
    """Save Ralph-Loop state to file."""
    state["last_updated"] = datetime.now().isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[STATE] Saved to {STATE_PATH}")

def initialize_state() -> Dict:
    """Initialize fresh Ralph-Loop state."""
    return {
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "current_phase": "1",
        "current_task": "1.1",
        "completed_tasks": [],
        "findings": {
            "data_quality": {},
            "symbols": {},
            "promising_strategies": [],
            "failed_strategies": [],
            "best_performers": []
        },
        "parameters": {
            "explored": {},
            "optimal": {}
        },
        "next_actions": ["Enumerate symbols in binance_futures_1h"],
        "blockers": [],
        "git_commits": []
    }

# ============================================================
# PHASE 1: DATA DISCOVERY
# ============================================================

def task_1_1_enumerate_symbols() -> Dict:
    """Task 1.1: Enumerate all available symbols."""
    print("\n" + "="*60)
    print("[TASK 1.1] Enumerating Binance Futures Symbols")
    print("="*60)
    
    futures_path = DATA_PATH / "binance_futures_1h"
    
    if not futures_path.exists():
        return {"error": f"Path not found: {futures_path}"}
    
    # Find all CSV files
    csv_files = list(futures_path.glob("*.csv"))
    symbols = [f.stem.replace("_1h", "").replace("USDT", "").upper() + "USDT" 
               for f in csv_files]
    
    # Normalize symbol names
    symbol_info = {}
    for f in csv_files:
        # Try to extract symbol from filename
        name = f.stem
        # Common patterns: BTCUSDT_1h, BTCUSDT, etc.
        symbol = name.replace("_1h", "").replace("_4h", "").replace("_1d", "").upper()
        if not symbol.endswith("USDT"):
            symbol = symbol + "USDT"
        
        # Get file info
        stat = f.stat()
        symbol_info[symbol] = {
            "file": str(f),
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    
    result = {
        "total_symbols": len(symbol_info),
        "symbols": sorted(symbol_info.keys()),
        "symbol_details": symbol_info
    }
    
    print(f"Found {result['total_symbols']} symbols")
    print(f"Sample: {result['symbols'][:10]}...")
    
    return result

def task_1_2_analyze_date_ranges() -> Dict:
    """Task 1.2: Identify data start/end dates per symbol."""
    print("\n" + "="*60)
    print("[TASK 1.2] Analyzing Date Ranges")
    print("="*60)
    
    futures_path = DATA_PATH / "binance_futures_1h"
    date_ranges = {}
    
    for csv_file in sorted(futures_path.glob("*.csv"))[:50]:  # Sample first 50
        try:
            # Read only first and last rows for efficiency
            df = pd.read_csv(csv_file, usecols=['timestamp'], parse_dates=['timestamp'])
            
            if len(df) > 0:
                symbol = csv_file.stem.replace("_1h", "").upper()
                date_ranges[symbol] = {
                    "start": df['timestamp'].min().isoformat(),
                    "end": df['timestamp'].max().isoformat(),
                    "rows": len(df),
                    "days": (df['timestamp'].max() - df['timestamp'].min()).days
                }
        except Exception as e:
            print(f"  Error reading {csv_file.name}: {e}")
    
    # Calculate statistics
    if date_ranges:
        days_list = [v['days'] for v in date_ranges.values()]
        stats = {
            "analyzed_symbols": len(date_ranges),
            "min_history_days": min(days_list),
            "max_history_days": max(days_list),
            "median_history_days": int(np.median(days_list)),
            "symbols_with_3yr_plus": sum(1 for d in days_list if d >= 1095)
        }
    else:
        stats = {}
    
    result = {
        "date_ranges": date_ranges,
        "statistics": stats
    }
    
    print(f"Analyzed {len(date_ranges)} symbols")
    if stats:
        print(f"History range: {stats['min_history_days']} - {stats['max_history_days']} days")
        print(f"Symbols with 3+ years: {stats['symbols_with_3yr_plus']}")
    
    return result

def task_1_3_check_data_quality() -> Dict:
    """Task 1.3: Check for gaps, anomalies, survivorship bias."""
    print("\n" + "="*60)
    print("[TASK 1.3] Checking Data Quality")
    print("="*60)
    
    futures_path = DATA_PATH / "binance_futures_1h"
    quality_report = {}
    
    # Sample 20 major symbols
    major_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
                     'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT']
    
    for symbol in major_symbols:
        # Find file (could be named differently)
        pattern = f"*{symbol.lower()}*" if symbol.lower() in str(futures_path).lower() else f"*{symbol}*"
        matches = list(futures_path.glob(f"*{symbol.replace('USDT', '')}*"))
        
        if not matches:
            matches = list(futures_path.glob(f"*{symbol.lower().replace('usdt', '')}*"))
        
        if matches:
            try:
                df = pd.read_csv(matches[0], parse_dates=['timestamp'])
                df = df.sort_values('timestamp')
                
                # Check for gaps (missing hours)
                df['expected_next'] = df['timestamp'] + pd.Timedelta(hours=1)
                gaps = (df['timestamp'].shift(-1) - df['timestamp']) > pd.Timedelta(hours=1)
                gap_count = gaps.sum()
                
                # Check for anomalies
                returns = df['close'].pct_change()
                extreme_moves = (returns.abs() > 0.5).sum()  # >50% moves
                
                # Check for zero/negative values
                zero_close = (df['close'] <= 0).sum()
                zero_volume = (df['volume'] <= 0).sum()
                
                quality_report[symbol] = {
                    "rows": len(df),
                    "gap_count": int(gap_count),
                    "extreme_moves": int(extreme_moves),
                    "zero_close": int(zero_close),
                    "zero_volume": int(zero_volume),
                    "quality_score": "GOOD" if gap_count < 100 and extreme_moves < 50 else "CHECK"
                }
            except Exception as e:
                quality_report[symbol] = {"error": str(e)}
    
    # Summary
    good_count = sum(1 for v in quality_report.values() if v.get('quality_score') == 'GOOD')
    
    result = {
        "symbols_checked": len(quality_report),
        "good_quality": good_count,
        "needs_review": len(quality_report) - good_count,
        "details": quality_report
    }
    
    print(f"Quality check: {good_count}/{len(quality_report)} symbols passed")
    
    return result

def task_1_4_calculate_liquidity() -> Dict:
    """Task 1.4: Calculate liquidity metrics."""
    print("\n" + "="*60)
    print("[TASK 1.4] Calculating Liquidity Metrics")
    print("="*60)
    
    futures_path = DATA_PATH / "binance_futures_1h"
    liquidity_data = {}
    
    for csv_file in sorted(futures_path.glob("*.csv")):
        try:
            # Read recent data only (last 30 days)
            df = pd.read_csv(csv_file, parse_dates=['timestamp'])
            df = df.sort_values('timestamp').tail(720)  # ~30 days of hourly
            
            if len(df) >= 100:
                symbol = csv_file.stem.replace("_1h", "").upper()
                
                # Average Daily Volume (in quote currency)
                daily_volume = df.groupby(df['timestamp'].dt.date)['volume'].sum()
                adv = daily_volume.mean()
                
                # Volatility (annualized)
                returns = df['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(24 * 365)
                
                liquidity_data[symbol] = {
                    "adv_usdt": float(adv),
                    "volatility_annual": float(volatility),
                    "last_price": float(df['close'].iloc[-1]),
                    "adv_notional_millions": float(adv * df['close'].iloc[-1] / 1e6)
                }
        except Exception as e:
            pass  # Skip files with issues
    
    # Rank by ADV
    sorted_by_adv = sorted(liquidity_data.items(), 
                          key=lambda x: x[1]['adv_notional_millions'], 
                          reverse=True)
    
    result = {
        "total_symbols": len(liquidity_data),
        "top_20_by_adv": [s[0] for s in sorted_by_adv[:20]],
        "liquidity_tiers": {
            "tier1_100m_plus": [s[0] for s in sorted_by_adv if s[1]['adv_notional_millions'] >= 100],
            "tier2_10m_plus": [s[0] for s in sorted_by_adv if 10 <= s[1]['adv_notional_millions'] < 100],
            "tier3_under_10m": [s[0] for s in sorted_by_adv if s[1]['adv_notional_millions'] < 10]
        },
        "details": dict(sorted_by_adv[:50])  # Top 50 details
    }
    
    print(f"Liquidity tiers:")
    print(f"  Tier 1 ($100M+ ADV): {len(result['liquidity_tiers']['tier1_100m_plus'])} symbols")
    print(f"  Tier 2 ($10-100M ADV): {len(result['liquidity_tiers']['tier2_10m_plus'])} symbols")
    print(f"  Tier 3 (<$10M ADV): {len(result['liquidity_tiers']['tier3_under_10m'])} symbols")
    
    return result

# ============================================================
# MAIN EXECUTION
# ============================================================

def run_phase_1():
    """Execute Phase 1: Data Discovery."""
    state = load_state()
    
    tasks = [
        ("1.1", task_1_1_enumerate_symbols),
        ("1.2", task_1_2_analyze_date_ranges),
        ("1.3", task_1_3_check_data_quality),
        ("1.4", task_1_4_calculate_liquidity),
    ]
    
    for task_id, task_func in tasks:
        if task_id not in state["completed_tasks"]:
            print(f"\n{'='*60}")
            print(f"Executing Task {task_id}")
            print(f"{'='*60}")
            
            try:
                result = task_func()
                
                # Store findings
                state["findings"]["data_quality"][task_id] = result
                state["completed_tasks"].append(task_id)
                state["current_task"] = task_id
                
                # Save after each task
                save_state(state)
                print(f"[OK] Task {task_id} completed")
                
            except Exception as e:
                state["blockers"].append({
                    "task": task_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                save_state(state)
                print(f"[ERROR] Task {task_id} failed: {e}")
                break
    
    # Phase 1 complete?
    phase_1_tasks = ["1.1", "1.2", "1.3", "1.4"]
    if all(t in state["completed_tasks"] for t in phase_1_tasks):
        state["current_phase"] = "2"
        state["current_task"] = "2.1"
        state["next_actions"] = ["Start Phase 2: Feature Engineering"]
        save_state(state)
        print("\n" + "="*60)
        print("PHASE 1 COMPLETE - Ready for Phase 2")
        print("="*60)
    
    return state

def print_status():
    """Print current Ralph-Loop status."""
    state = load_state()
    
    print("\n" + "="*60)
    print("RALPH-LOOP STATUS")
    print("="*60)
    print(f"Version: {state.get('version', 'N/A')}")
    print(f"Current Phase: {state.get('current_phase', 'N/A')}")
    print(f"Current Task: {state.get('current_task', 'N/A')}")
    print(f"Completed Tasks: {len(state.get('completed_tasks', []))}")
    print(f"Blockers: {len(state.get('blockers', []))}")
    print(f"Last Updated: {state.get('last_updated', 'Never')}")
    print(f"Next Actions: {state.get('next_actions', [])}")
    print("="*60)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ralph-Loop Autonomous Research")
    parser.add_argument("--phase", type=int, default=1, help="Phase to execute (1-5)")
    parser.add_argument("--status", action="store_true", help="Print status only")
    parser.add_argument("--reset", action="store_true", help="Reset state to beginning")
    
    args = parser.parse_args()
    
    if args.status:
        print_status()
    elif args.reset:
        state = initialize_state()
        save_state(state)
        print("State reset to initial values")
    elif args.phase == 1:
        run_phase_1()
    else:
        print(f"Phase {args.phase} not yet implemented")
        print("Available: --phase 1, --status, --reset")
