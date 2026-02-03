#!/usr/bin/env python3
"""
Ralph-Loop Phase 13: Strategies from New Data Sources
======================================================
Available new data:
1. Fear & Greed full history (2018-2026, 2914 days) - already tested in Phase 11
2. Binance aggregate trades (2024-02 to 2024-05, BTC) - build order flow features
3. Deribit DVOL (if downloaded)

Strategy ideas from trade data:
A) Order Flow Imbalance: buy_volume >> sell_volume → bullish
B) Large Trade Detection: whale trades → follow the whale
C) Trade Count Spike: unusual activity → breakout imminent
D) Volume-Weighted Buy Pressure: VWAP crossover with buy/sell bias

Since trade data is only 4 months (insufficient for TRUE OOS),
we'll use the existing 1H OHLCV data with volume-based features instead.
These features approximate order flow using volume + price action.

NEW APPROACH: Volume-enhanced strategies on full 1H dataset (257 symbols, multi-year)
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


# =============================================================================
# STRATEGY VARIANTS: Volume-Enhanced
# =============================================================================


def strat_volume_breakout(df, lookback=48, vol_mult=2.0):
    """Breakout + volume spike: price breaks high AND volume > 2x average"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    vol_avg = vol.rolling(lookback).mean()
    vol_spike = vol > vol_avg * vol_mult

    signals = np.where((close > upper) & (ema_f > ema_s) & vol_spike, 1, 0)
    return signals


def strat_obv_trend(df, lookback=48):
    """On-Balance Volume trend: OBV making new highs = bullish"""
    close = df["close"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

    # OBV
    obv = (np.sign(close.diff()) * vol).cumsum()
    obv_upper = obv.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    # Price breakout + OBV breakout
    price_upper = close.rolling(lookback).max().shift(1) * close.rolling(
        lookback
    ).max().shift(1)
    upper = df["high"].rolling(lookback).max().shift(1)

    signals = np.where((close > upper) & (obv > obv_upper) & (ema_f > ema_s), 1, 0)
    return signals


def strat_vwap_breakout(df, lookback=48):
    """VWAP breakout: price > VWAP + breakout"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

    # Rolling VWAP
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    signals = np.where((close > upper) & (close > vwap) & (ema_f > ema_s), 1, 0)
    return signals


def strat_mfi_breakout(df, lookback=48, mfi_thresh=80):
    """Money Flow Index breakout: MFI confirms buying pressure"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

    # Typical price
    tp = (high + low + close) / 3
    rmf = tp * vol

    # Positive/negative money flow
    tp_diff = tp.diff()
    pos_mf = rmf.where(tp_diff > 0, 0).rolling(14).sum()
    neg_mf = rmf.where(tp_diff <= 0, 0).rolling(14).sum()
    mfi = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-10))

    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    # ATR expansion
    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * 1.0

    signals = np.where(
        (close > upper) & (mfi > mfi_thresh) & (ema_f > ema_s) & expanding, 1, 0
    )
    return signals


def strat_volume_profile(df, lookback=48):
    """Volume Profile: breakout above high-volume price level"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

    # Estimate POC (Point of Control) as VWAP
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)

    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    # Breakout above both Donchian high AND VWAP
    signals = np.where((close > upper) & (close > vwap * 1.01) & (ema_f > ema_s), 1, 0)
    return signals


def strat_cmf_breakout(df, lookback=48):
    """Chaikin Money Flow breakout: CMF > 0.1 = accumulation"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

    # CMF
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
    mfv = mfm * vol
    cmf = mfv.rolling(20).sum() / (vol.rolling(20).sum() + 1e-10)

    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    signals = np.where((close > upper) & (cmf > 0.1) & (ema_f > ema_s), 1, 0)
    return signals


def strat_relative_volume(df, lookback=48, rvol_thresh=1.5):
    """Relative Volume: trade when volume is 1.5x+ above average"""
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    # Use same-hour-of-day average volume for better comparison
    vol_avg = vol.rolling(lookback * 7).mean()  # long-term average
    rvol = vol / (vol_avg + 1e-10)

    signals = np.where((close > upper) & (rvol > rvol_thresh) & (ema_f > ema_s), 1, 0)
    return signals


# =============================================================================
# SIMULATION + VALIDATION (same as Phase 9/10)
# =============================================================================


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


def run_true_oos(
    strategy_fn,
    all_data,
    max_positions=10,
    test_bars=720,
    exit_params=None,
    position_scale=3.0,
):
    if exit_params is None:
        exit_params = {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0}

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
            vol = df["close"].iloc[:i].pct_change().rolling(168).std().iloc[-1]
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
            position_pct *= position_scale

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

    return {
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


def main():
    print("=" * 70)
    print("RALPH-LOOP PHASE 13: VOLUME-ENHANCED STRATEGIES")
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

    exit_params = {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0}

    # Test all volume-enhanced strategies
    strategies = {
        "vol_breakout_2x": lambda df: strat_volume_breakout(df, 48, 2.0),
        "vol_breakout_1.5x": lambda df: strat_volume_breakout(df, 48, 1.5),
        "obv_trend": lambda df: strat_obv_trend(df, 48),
        "vwap_breakout": lambda df: strat_vwap_breakout(df, 48),
        "mfi_breakout_80": lambda df: strat_mfi_breakout(df, 48, 80),
        "mfi_breakout_60": lambda df: strat_mfi_breakout(df, 48, 60),
        "vol_profile": lambda df: strat_volume_profile(df, 48),
        "cmf_breakout": lambda df: strat_cmf_breakout(df, 48),
        "rvol_1.5x": lambda df: strat_relative_volume(df, 48, 1.5),
        "rvol_2.0x": lambda df: strat_relative_volume(df, 48, 2.0),
    }

    # Test with different position configs
    configs = [
        {"max_pos": 10, "test_bars": 720, "scale": 3.0},
        {"max_pos": 15, "test_bars": 360, "scale": 3.0},
        {"max_pos": 10, "test_bars": 720, "scale": 5.0},
    ]

    all_results = []

    for strat_name, strat_fn in strategies.items():
        for cfg in configs:
            config_name = (
                f"{strat_name}_pos{cfg['max_pos']}_rb{cfg['test_bars']}_s{cfg['scale']}"
            )
            print(f"  {config_name}...", end=" ", flush=True)

            r = run_true_oos(
                strat_fn,
                all_data,
                max_positions=cfg["max_pos"],
                test_bars=cfg["test_bars"],
                exit_params=exit_params,
                position_scale=cfg["scale"],
            )
            if r:
                c, p = check_criteria(r)
                r["criteria_passed"] = p
                all_results.append((config_name, p, r))
                print(
                    f"[{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                    f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                    f"PF={r['profit_factor']:.2f} T={r['trade_count']}"
                )
            else:
                print("SKIP")

    # Sort and display
    all_results.sort(key=lambda x: (-x[1], -x[2].get("sharpe", 0)))

    print(f"\n\n{'=' * 70}")
    print("PHASE 13 RESULTS - TOP 15")
    print("=" * 70)

    for i, (name, passed, r) in enumerate(all_results[:15]):
        c, _ = check_criteria(r)
        print(f"\n  {i+1}. [{passed}/6] {name}")
        print(
            f"     Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
            f"DD={r['max_drawdown']*100:.1f}%  WR={r['win_rate']*100:.0f}%  "
            f"PF={r['profit_factor']:.2f}  WFA={r['wfa_efficiency']:.0f}%  T={r['trade_count']}"
        )
        fails = [k for k, v in c.items() if not v]
        if fails:
            print(f"     FAILS: {', '.join(fails)}")

    six_six = [(n, r) for n, p, r in all_results if p == 6]
    if six_six:
        print(f"\n*** {len(six_six)} configs passed 6/6! ***")
        for name, r in six_six:
            print(
                f"  {name}: Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}%"
            )
    else:
        print(f"\nNo 6/6. Best: {all_results[0][1]}/6" if all_results else "No results")

    # Compare with baseline (Dual MA Breakout)
    print(f"\n{'=' * 60}")
    print("COMPARISON WITH BASELINE")
    print("=" * 60)
    print("  Baseline Dual MA Breakout 3x: [6/6] Sharpe=2.39 Ret=+8.8%")
    if all_results:
        best = all_results[0]
        print(f"  Best new: [{best[1]}/6] {best[0]}")
        print(
            f"    Sharpe={best[2]['sharpe']:.2f}  Ret={best[2]['total_return']*100:+.1f}%"
        )
        if best[1] >= 6:
            print("  → New strategy matches or beats baseline!")
        else:
            print("  → Baseline Dual MA Breakout remains best.")

    # Save
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "strategies_tested": len(strategies),
        "configs_per_strategy": len(configs),
        "total_tested": len(all_results),
        "six_six_count": len(six_six) if six_six else 0,
        "top_10": [
            {
                "config": n,
                "passed": int(p),
                "sharpe": float(r.get("sharpe", 0)),
                "return": float(r.get("total_return", 0)),
                "max_dd": float(r.get("max_drawdown", 0)),
                "win_rate": float(r.get("win_rate", 0)),
                "pf": float(r.get("profit_factor", 0)),
                "trades": int(r.get("trade_count", 0)),
            }
            for n, p, r in all_results[:10]
        ],
    }
    with open(RESULTS_PATH / "phase13_report.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nSaved to {RESULTS_PATH / 'phase13_report.json'}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
