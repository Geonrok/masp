#!/usr/bin/env python3
"""
Ralph-Loop Phase 1: Data Discovery & Validation
===============================================
Task 1.1 ~ 1.6: Complete data audit for Binance Futures
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Paths
DATA_ROOT = Path("E:/data/crypto_ohlcv")
STATE_PATH = Path(
    "E:/투자/Multi-Asset Strategy Platform/research/ralph_loop_state.json"
)
RESULTS_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/results")

RESULTS_PATH.mkdir(exist_ok=True)


def load_state():
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state):
    state["last_updated"] = datetime.now().isoformat()
    STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def task_1_1_enumerate_symbols():
    """Task 1.1: Enumerate all available symbols"""
    print("=" * 70)
    print("TASK 1.1: Enumerate Symbols")
    print("=" * 70)

    results = {}

    # Check all timeframes
    timeframes = ["binance_futures_1h", "binance_futures_4h", "binance_futures_1d"]

    for tf in timeframes:
        tf_path = DATA_ROOT / tf
        if tf_path.exists():
            files = list(tf_path.glob("*.csv"))
            symbols = sorted([f.stem for f in files if f.stem.endswith("USDT")])
            results[tf] = {"count": len(symbols), "symbols": symbols}
            print(f"  {tf}: {len(symbols)} symbols")
        else:
            print(f"  {tf}: NOT FOUND")
            results[tf] = {"count": 0, "symbols": []}

    # Additional data sources
    extra_sources = [
        "binance_funding_rate",
        "binance_open_interest",
        "binance_long_short_ratio",
        "binance_taker_volume",
        "macro",
        "coinglass",
    ]

    for src in extra_sources:
        src_path = DATA_ROOT / src
        if src_path.exists():
            files = list(src_path.glob("*.csv"))
            results[src] = {"count": len(files), "available": True}
            print(f"  {src}: {len(files)} files")
        else:
            results[src] = {"count": 0, "available": False}
            print(f"  {src}: NOT FOUND")

    return results


def task_1_2_date_ranges():
    """Task 1.2: Identify data start/end dates per symbol"""
    print("\n" + "=" * 70)
    print("TASK 1.2: Date Ranges")
    print("=" * 70)

    # Use 4h data (more likely to have)
    tf_path = DATA_ROOT / "binance_futures_4h"
    if not tf_path.exists():
        tf_path = DATA_ROOT / "binance_futures_1h"

    results = {}
    files = sorted(tf_path.glob("*.csv"))

    for f in files:
        symbol = f.stem
        try:
            df = pd.read_csv(f, nrows=5)  # First few rows
            for col in ["datetime", "timestamp", "date"]:
                if col in df.columns:
                    start_date = pd.to_datetime(df[col].iloc[0])
                    break
            else:
                continue

            # Read last rows
            df_tail = pd.read_csv(f)
            for col in ["datetime", "timestamp", "date"]:
                if col in df_tail.columns:
                    df_tail[col] = pd.to_datetime(df_tail[col])
                    end_date = df_tail[col].max()
                    rows = len(df_tail)
                    break

            results[symbol] = {
                "start": str(start_date.date()),
                "end": str(end_date.date()),
                "rows": rows,
                "days": (end_date - start_date).days,
            }
        except Exception as e:
            results[symbol] = {"error": str(e)}

    # Summary stats
    valid = {k: v for k, v in results.items() if "days" in v}
    if valid:
        avg_days = np.mean([v["days"] for v in valid.values()])
        min_days = min(v["days"] for v in valid.values())
        max_days = max(v["days"] for v in valid.values())
        print(f"  Symbols with data: {len(valid)}")
        print(f"  Average history: {avg_days:.0f} days")
        print(f"  Range: {min_days} - {max_days} days")

    return results


def task_1_3_check_gaps():
    """Task 1.3: Check for gaps and anomalies"""
    print("\n" + "=" * 70)
    print("TASK 1.3: Data Quality Check")
    print("=" * 70)

    tf_path = DATA_ROOT / "binance_futures_4h"
    if not tf_path.exists():
        print("  4h data not found, using 1h")
        tf_path = DATA_ROOT / "binance_futures_1h"

    results = {"gap_analysis": [], "anomalies": []}
    sample_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

    for symbol in sample_symbols:
        fp = tf_path / f"{symbol}.csv"
        if not fp.exists():
            continue

        df = pd.read_csv(fp)
        for col in ["datetime", "timestamp", "date"]:
            if col in df.columns:
                df["dt"] = pd.to_datetime(df[col])
                break

        if "dt" not in df.columns:
            continue

        df = df.sort_values("dt")

        # Check gaps (expected 4h interval)
        df["gap"] = df["dt"].diff()
        expected_gap = pd.Timedelta(hours=4)
        gaps = df[df["gap"] > expected_gap * 1.5]

        # Check price anomalies
        if "close" in df.columns:
            df["ret"] = df["close"].pct_change()
            anomalies = df[abs(df["ret"]) > 0.5]  # >50% moves

            results["gap_analysis"].append(
                {
                    "symbol": symbol,
                    "total_rows": len(df),
                    "gaps_found": len(gaps),
                    "max_gap_hours": (
                        gaps["gap"].max().total_seconds() / 3600 if len(gaps) > 0 else 0
                    ),
                    "anomalies": len(anomalies),
                }
            )

            print(
                f"  {symbol}: {len(df)} rows, {len(gaps)} gaps, {len(anomalies)} anomalies"
            )

    return results


def task_1_4_liquidity_metrics():
    """Task 1.4: Calculate liquidity metrics (ADV, spread proxy)"""
    print("\n" + "=" * 70)
    print("TASK 1.4: Liquidity Metrics")
    print("=" * 70)

    tf_path = DATA_ROOT / "binance_futures_4h"
    if not tf_path.exists():
        tf_path = DATA_ROOT / "binance_futures_1h"

    results = []
    files = sorted(tf_path.glob("*USDT.csv"))

    for f in files:
        symbol = f.stem
        try:
            df = pd.read_csv(f)
            if "volume" not in df.columns or "close" not in df.columns:
                continue

            # Use last 90 days
            df = df.tail(90 * 6)  # 6 bars per day for 4h

            # Average Daily Volume (in USD)
            adv_usd = (df["volume"] * df["close"]).mean() * 6  # Scale to daily

            # Spread proxy: High-Low range relative to close
            spread_proxy = ((df["high"] - df["low"]) / df["close"]).mean() * 100

            # Volatility (daily)
            returns = df["close"].pct_change()
            volatility = returns.std() * np.sqrt(6) * 100  # Annualized approx

            results.append(
                {
                    "symbol": symbol,
                    "adv_usd": adv_usd,
                    "spread_proxy_pct": spread_proxy,
                    "volatility_pct": volatility,
                }
            )
        except Exception:
            pass

    # Sort by ADV
    results = sorted(results, key=lambda x: x["adv_usd"], reverse=True)

    print(f"  Total symbols analyzed: {len(results)}")
    print("\n  Top 10 by ADV:")
    for r in results[:10]:
        print(
            f"    {r['symbol']:<12} ADV=${r['adv_usd']/1e6:.1f}M  Spread={r['spread_proxy_pct']:.2f}%"
        )

    return results


def task_1_5_create_tiers():
    """Task 1.5: Create symbol universe tiers"""
    print("\n" + "=" * 70)
    print("TASK 1.5: Symbol Universe Tiers")
    print("=" * 70)

    # Get liquidity data
    liquidity = task_1_4_liquidity_metrics()

    # Create tiers based on ADV
    tier_1 = [r["symbol"] for r in liquidity[:20]]  # Top 20
    tier_2 = [r["symbol"] for r in liquidity[20:50]]  # 21-50
    tier_3 = [r["symbol"] for r in liquidity[50:100]]  # 51-100

    results = {
        "tier_1_top20": tier_1,
        "tier_2_top50": tier_1 + tier_2,
        "tier_3_top100": tier_1 + tier_2 + tier_3,
        "all": [r["symbol"] for r in liquidity],
    }

    print(f"\n  Tier 1 (Top 20): {len(tier_1)} symbols")
    print(f"  Tier 2 (Top 50): {len(tier_1) + len(tier_2)} symbols")
    print(f"  Tier 3 (Top 100): {len(tier_1) + len(tier_2) + len(tier_3)} symbols")
    print(f"  All: {len(liquidity)} symbols")

    print(f"\n  Tier 1 symbols: {', '.join(tier_1[:10])}...")

    return results


def task_1_6_document_report():
    """Task 1.6: Generate final data quality report"""
    print("\n" + "=" * 70)
    print("TASK 1.6: Data Quality Report")
    print("=" * 70)

    # Run all tasks and compile
    symbols = task_1_1_enumerate_symbols()
    date_ranges = task_1_2_date_ranges()
    gaps = task_1_3_check_gaps()
    liquidity = task_1_4_liquidity_metrics()
    tiers = task_1_5_create_tiers()

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_symbols": symbols.get("binance_futures_4h", {}).get("count", 0),
            "avg_history_days": np.mean(
                [v["days"] for v in date_ranges.values() if "days" in v]
            ),
            "data_sources": {
                k: v.get("available", v.get("count", 0) > 0) for k, v in symbols.items()
            },
        },
        "tiers": tiers,
        "gap_analysis": gaps,
        "liquidity_ranking": liquidity[:50],  # Top 50
    }

    # Save report
    report_path = RESULTS_PATH / "phase1_data_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n  Report saved: {report_path}")

    return report


def main():
    print("=" * 70)
    print("RALPH-LOOP PHASE 1: DATA DISCOVERY & VALIDATION")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    state = load_state()

    # Run all Phase 1 tasks
    report = task_1_6_document_report()

    # Update state
    state["current_phase"] = "2"
    state["current_task"] = "2.1"
    state["completed_tasks"] = ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6"]
    state["findings"]["data_quality"] = {
        "total_symbols": report["summary"]["total_symbols"],
        "avg_history_days": report["summary"]["avg_history_days"],
        "tier_1_symbols": report["tiers"]["tier_1_top20"],
        "tier_2_symbols": report["tiers"]["tier_2_top50"],
    }
    state["next_actions"] = ["2.1: Price-based features", "2.2: Volume-based features"]

    save_state(state)

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"  Total symbols: {report['summary']['total_symbols']}")
    print(f"  Avg history: {report['summary']['avg_history_days']:.0f} days")
    print(f"  Tier 1 symbols: {len(report['tiers']['tier_1_top20'])}")
    print("  Next: Phase 2 - Feature Engineering")

    return report


if __name__ == "__main__":
    report = main()
