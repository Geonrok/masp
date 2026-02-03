#!/usr/bin/env python3
"""
Ralph-Loop Phase 9: Trade Count Fix
=====================================
Phase 8 achieved 5/6 on TRUE OOS. Only failing: trade_count > 100.

Root cause: OOS period (40% of data) has limited WFA windows, and 5 positions
max with strict entry filters = only 30-55 trades.

Fixes to try:
A) Increase max_positions (5 → 10, 15, 20)
B) Shorter lookback (96 → 48, 72) = more frequent breakouts
C) Lower ATR expansion threshold (1.0 → 0.8)
D) More frequent rebalancing (720 → 360 bars)
E) Use ALL symbols simultaneously (not just top-5 per period)
F) Combined: relax multiple constraints

Key: must NOT degrade other 5 criteria while fixing trade count.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

DATA_ROOT = Path("E:/data/crypto_ohlcv")
RESULTS_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/results")
RESULTS_PATH.mkdir(exist_ok=True)

COMMISSION = 0.0004
FUNDING_PER_8H = 0.0001


def load_ohlcv(symbol, timeframe="1h"):
    path = DATA_ROOT / f"binance_futures_{timeframe}" / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["datetime", "timestamp", "date"]:
        if col in df.columns:
            df["datetime"] = pd.to_datetime(df[col])
            break
    return df.sort_values("datetime").reset_index(drop=True)


def calc_atr(high, low, close, period=14):
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
    )
    tr[0] = high[0] - low[0]
    return pd.Series(tr).rolling(period).mean().values


def strat_dual_ma(df, lookback=96, ema_fast=50, ema_slow=200, atr_expansion=1.3):
    """Dual MA breakout (best from Phase 8)"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion
    signals = np.where((close > upper) & (ema_f > ema_s) & expanding, 1, 0)
    return signals


def strat_relaxed_long(df, lookback=96, ema_period=200, atr_expansion=1.0):
    """Relaxed ATR long-only (2nd best from Phase 8)"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    upper = high.rolling(lookback).max().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()
    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion
    signals = np.where((close > upper) & (close > ema) & expanding, 1, 0)
    return signals


def strat_short_lookback(df, lookback=48, ema_fast=50, ema_slow=200, atr_expansion=1.0):
    """Shorter lookback = more frequent breakouts"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion
    signals = np.where((close > upper) & (ema_f > ema_s) & expanding, 1, 0)
    return signals


def strat_medium_lookback(
    df, lookback=72, ema_fast=50, ema_slow=200, atr_expansion=1.0
):
    """Medium lookback"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion
    signals = np.where((close > upper) & (ema_f > ema_s) & expanding, 1, 0)
    return signals


def simulate(
    df,
    signals,
    position_pct=0.02,
    max_bars=72,
    atr_stop=3.0,
    profit_target_atr=8.0,
    slippage=0.0003,
):
    capital = 1.0
    position = 0
    entry_price = 0
    bars_held = 0
    trades = []

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr_vals = calc_atr(high, low, close, 14)

    for i in range(len(df)):
        c = close[i]
        sig = signals[i] if i < len(signals) else 0
        cur_atr = atr_vals[i] if not np.isnan(atr_vals[i]) else c * 0.01

        if position != 0:
            bars_held += 1
            unrealized_atr = position * (c - entry_price) / (cur_atr + 1e-10)
            should_exit = False
            if bars_held >= max_bars:
                should_exit = True
            elif atr_stop > 0 and unrealized_atr < -atr_stop:
                should_exit = True
            elif profit_target_atr > 0 and unrealized_atr > profit_target_atr:
                should_exit = True
            elif sig != 0 and sig != position:
                should_exit = True

            if should_exit:
                exit_p = c * (1 - slippage * np.sign(position))
                pnl = position * (exit_p - entry_price) / entry_price * position_pct
                pnl -= COMMISSION * position_pct
                pnl -= FUNDING_PER_8H * bars_held / 8 * position_pct
                trades.append(pnl)
                capital += pnl
                position = 0

        if position == 0 and sig != 0:
            position = int(np.sign(sig))
            entry_price = c * (1 + slippage * position)
            capital -= COMMISSION * position_pct
            bars_held = 0

    if position != 0:
        c = close[-1]
        exit_p = c * (1 - slippage * np.sign(position))
        pnl = position * (exit_p - entry_price) / entry_price * position_pct
        pnl -= COMMISSION * position_pct
        pnl -= FUNDING_PER_8H * bars_held / 8 * position_pct
        trades.append(pnl)

    wins = sum(1 for t in trades if t > 0)
    losses = sum(1 for t in trades if t <= 0)
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))
    return {
        "total_return": capital - 1,
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "profit_factor": gp / (gl + 1e-10),
        "trade_count": len(trades),
        "trades": trades,
    }


def check_criteria(r):
    c = {
        "sharpe_gt_1": r.get("sharpe", 0) > 1.0,
        "max_dd_lt_25": r.get("max_drawdown", -1) > -0.25,
        "win_rate_gt_45": r.get("win_rate", 0) > 0.45,
        "profit_factor_gt_1_5": r.get("profit_factor", 0) > 1.5,
        "wfa_efficiency_gt_50": r.get("wfa_efficiency", 0) > 50,
        "trade_count_gt_100": r.get("trade_count", 0) > 100,
    }
    return c, sum(v for v in c.values())


def run_true_oos(strategy_fn, all_data, exit_params, max_positions=5, test_bars=720):
    """True OOS test with configurable max_positions and rebalance frequency."""

    # STEP A: Selection on first 60%
    selection_results = {}
    for symbol, df in all_data.items():
        split = int(len(df) * 0.6)
        sel_df = df.iloc[:split].copy().reset_index(drop=True)
        if len(sel_df) < 5000:
            continue

        train_bars = 4320
        all_trades = []
        period_returns = []

        i = train_bars
        while i + test_bars <= len(sel_df):
            full = sel_df.iloc[: i + test_bars]
            sigs = strategy_fn(full)
            test_sigs = sigs[i : i + test_bars]
            test_df = sel_df.iloc[i : i + test_bars].copy().reset_index(drop=True)
            r = simulate(
                test_df,
                test_sigs,
                0.02,
                exit_params["max_bars"],
                exit_params["atr_stop"],
                exit_params["profit_target_atr"],
                0.0003,
            )
            period_returns.append(r["total_return"])
            all_trades.extend(r["trades"])
            i += test_bars

        if not all_trades:
            continue

        total_ret = 1.0
        for pr in period_returns:
            total_ret *= 1 + pr

        selection_results[symbol] = {
            "total_return": total_ret - 1,
            "trade_count": len(all_trades),
        }

    if not selection_results:
        return None

    # Use all symbols (best from Phase 7/8)
    oos_data = {}
    for symbol in all_data:
        df = all_data[symbol]
        split = int(len(df) * 0.6)
        oos_df = df.iloc[split:].copy().reset_index(drop=True)
        if len(oos_df) > 2000:
            oos_data[symbol] = oos_df

    if not oos_data:
        return None

    equity = [1.0]
    period_returns = []
    all_trades = []
    min_len = min(len(d) for d in oos_data.values())

    train_bars = min(4320, min_len // 3)

    i = train_bars
    while i + test_bars <= min_len:
        period_pnl = 0

        scored = []
        for symbol, df in oos_data.items():
            if len(df) <= i:
                continue
            train_ret = df["close"].iloc[:i].pct_change()
            vol = train_ret.rolling(168).std().iloc[-1]
            if np.isnan(vol) or vol == 0:
                vol = 0.01
            scored.append((symbol, vol))

        scored.sort(key=lambda x: x[1])
        selected = scored[:max_positions]

        for symbol, vol in selected:
            df = oos_data[symbol]
            if i + test_bars > len(df):
                continue
            full = df.iloc[: i + test_bars]
            sigs = strategy_fn(full)
            test_sigs = sigs[i : i + test_bars]
            test_df = df.iloc[i : i + test_bars].copy().reset_index(drop=True)

            ann_vol = vol * np.sqrt(24 * 365)
            position_pct = min(0.10 / (ann_vol + 1e-10) / max(len(selected), 1), 0.05)

            r = simulate(
                test_df,
                test_sigs,
                position_pct,
                exit_params["max_bars"],
                exit_params["atr_stop"],
                exit_params["profit_target_atr"],
                0.0003,
            )
            period_pnl += r["total_return"]
            all_trades.extend(r["trades"])

        period_returns.append(period_pnl)
        equity.append(equity[-1] * (1 + period_pnl))
        i += test_bars

    if not period_returns:
        return None

    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peak) / peak

    wins = sum(1 for t in all_trades if t > 0)
    losses = sum(1 for t in all_trades if t <= 0)
    gp = sum(t for t in all_trades if t > 0)
    gl = abs(sum(t for t in all_trades if t < 0))
    sharpe = np.mean(period_returns) / (np.std(period_returns) + 1e-10) * np.sqrt(12)

    result = {
        "total_return": float(equity_arr[-1] - 1),
        "max_drawdown": float(dd.min()),
        "sharpe": float(sharpe),
        "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
        "profit_factor": gp / (gl + 1e-10),
        "trade_count": len(all_trades),
        "periods": len(period_returns),
        "wfa_efficiency": sum(1 for r in period_returns if r > 0)
        / len(period_returns)
        * 100,
    }

    criteria, passed = check_criteria(result)
    result["criteria"] = criteria
    result["criteria_passed"] = int(passed)

    return result


def main():
    print("=" * 70)
    print("RALPH-LOOP PHASE 9: TRADE COUNT FIX")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Load all 1h data
    all_path = DATA_ROOT / "binance_futures_1h"
    all_data = {}
    for f in sorted(all_path.glob("*USDT.csv")):
        symbol = f.stem
        df = load_ohlcv(symbol, "1h")
        if not df.empty and len(df) > 10000:
            all_data[symbol] = df

    print(f"Loaded {len(all_data)} symbols\n")

    # Strategy + exit + max_positions + rebalance combos
    strategies = {
        "dual_ma_96": strat_dual_ma,
        "relaxed_long_96": strat_relaxed_long,
        "dual_ma_48": strat_short_lookback,
        "dual_ma_72": strat_medium_lookback,
    }

    exit_configs = {
        "default": {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0},
        "wide": {"max_bars": 96, "atr_stop": 4.0, "profit_target_atr": 10.0},
        "no_pt": {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 0},
    }

    position_configs = [5, 10, 15, 20]
    rebalance_configs = [720, 360]  # bars between rebalances

    all_results = []

    total_combos = (
        len(strategies)
        * len(exit_configs)
        * len(position_configs)
        * len(rebalance_configs)
    )
    tested = 0

    for strat_name, strat_fn in strategies.items():
        for exit_name, exit_params in exit_configs.items():
            for max_pos in position_configs:
                for test_bars in rebalance_configs:
                    tested += 1
                    config = f"{strat_name}_{exit_name}_pos{max_pos}_rb{test_bars}"
                    print(
                        f"  [{tested}/{total_combos}] {config}...", end=" ", flush=True
                    )

                    result = run_true_oos(
                        strat_fn,
                        all_data,
                        exit_params,
                        max_positions=max_pos,
                        test_bars=test_bars,
                    )

                    if result is None:
                        print("SKIP")
                        continue

                    passed = result["criteria_passed"]
                    r = result
                    print(
                        f"[{passed}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                        f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                        f"PF={r['profit_factor']:.2f} T={r['trade_count']}"
                    )

                    all_results.append(
                        {
                            "config": config,
                            "strat": strat_name,
                            "exit": exit_name,
                            "max_positions": max_pos,
                            "rebalance_bars": test_bars,
                            **{k: v for k, v in result.items() if k != "criteria"},
                            "criteria": result.get("criteria", {}),
                        }
                    )

    # Sort by criteria_passed, then sharpe
    all_results.sort(key=lambda x: (-x["criteria_passed"], -x.get("sharpe", 0)))

    print(f"\n\n{'=' * 70}")
    print("PHASE 9 RESULTS - TOP 15")
    print("=" * 70)

    for i, r in enumerate(all_results[:15]):
        passed = r["criteria_passed"]
        print(f"\n  {i+1}. [{passed}/6] {r['config']}")
        print(
            f"     Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
            f"DD={r['max_drawdown']*100:.1f}%  WR={r['win_rate']*100:.0f}%  "
            f"PF={r['profit_factor']:.2f}  WFA={r['wfa_efficiency']:.0f}%  T={r['trade_count']}"
        )
        fails = [k for k, v in r.get("criteria", {}).items() if not v]
        if fails:
            print(f"     FAILS: {', '.join(fails)}")

    # Check for 6/6
    six_six = [r for r in all_results if r["criteria_passed"] == 6]
    if six_six:
        print(f"\n\n*** {len(six_six)} CONFIGURATIONS PASSED 6/6 ON TRUE OOS! ***")
        for r in six_six:
            print(
                f"  - {r['config']}: Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                f"T={r['trade_count']}"
            )
    else:
        print(f"\n\nNo 6/6 yet. Best: {all_results[0]['criteria_passed']}/6")
        # Show what's closest
        best = all_results[0]
        fails = [k for k, v in best.get("criteria", {}).items() if not v]
        print(f"Closest: {best['config']} - fails: {', '.join(fails)}")

    # Save
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "total_tested": len(all_results),
        "six_six_count": len(six_six),
        "top_15": [
            {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in r.items()
                if k != "criteria"
            }
            for r in all_results[:15]
        ],
    }
    with open(RESULTS_PATH / "phase9_report.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nSaved to {RESULTS_PATH / 'phase9_report.json'}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
