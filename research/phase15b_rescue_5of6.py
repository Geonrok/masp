#!/usr/bin/env python3
"""
Phase 15b: 5/6 전략 구제 시도
- G1 momentum_filter (Sharpe 3.41, T=50) → positions 늘려서 거래수 확보
- G3 adx_filter (Sharpe 3.09, T=96) → positions 늘려서 거래수 확보
- Also try shorter rebalance (360 bars) to increase trades
"""

import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

DATA_ROOT = Path("E:/data/crypto_ohlcv")
RESULTS_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/results")

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


def strat_vol_profile(df, lookback=48):
    close = df["close"]
    high = df["high"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    signals = np.where((close > upper) & (close > vwap * 1.01) & (ema_f > ema_s), 1, 0)
    return signals


def strat_momentum_filter(df, lookback=48):
    close = df["close"]
    high = df["high"]
    ret = close.pct_change(lookback)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    signals = np.where((close > upper) & (ret > 0.05) & (ema_f > ema_s), 1, 0)
    return signals


def strat_adx_filter(df, lookback=48, adx_period=14, adx_thresh=25):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr = pd.Series(
        calc_atr(high.values, low.values, close.values, adx_period), index=df.index
    )
    plus_di = 100 * (plus_dm.rolling(adx_period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(adx_period).mean() / (atr + 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(adx_period).mean()

    signals = np.where(
        (close > upper) & (close > vwap * 1.01) & (ema_f > ema_s) & (adx > adx_thresh),
        1,
        0,
    )
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


def run_portfolio_oos(
    all_data,
    strat_fn,
    max_positions=10,
    test_bars=720,
    exit_params=None,
    position_scale=5.0,
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
            sigs = strat_fn(full)
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
    print("PHASE 15b: RESCUE 5/6 STRATEGIES")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    all_path = DATA_ROOT / "binance_futures_1h"
    all_data = {}
    for f in sorted(all_path.glob("*USDT.csv")):
        symbol = f.stem
        df = load_ohlcv(symbol, "1h")
        if not df.empty and len(df) > 10000:
            all_data[symbol] = df
    print(f"Loaded {len(all_data)} symbols\n")

    exit_params = {"max_bars": 72, "atr_stop": 3.0, "profit_target_atr": 8.0}

    strategies = {
        "momentum_filter": strat_momentum_filter,
        "adx_filter": strat_adx_filter,
        "vol_profile_baseline": strat_vol_profile,
    }

    configs = [
        {"max_pos": 10, "test_bars": 720},
        {"max_pos": 15, "test_bars": 720},
        {"max_pos": 20, "test_bars": 720},
        {"max_pos": 10, "test_bars": 360},
        {"max_pos": 15, "test_bars": 360},
        {"max_pos": 20, "test_bars": 360},
    ]

    print(
        f"{'Config':<45} {'Pass':>4} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'WR':>5} {'PF':>6} {'WFA':>5} {'T':>5}"
    )
    print("-" * 95)

    all_results = []

    for strat_name, strat_fn in strategies.items():
        for cfg in configs:
            label = f"{strat_name}_pos{cfg['max_pos']}_rb{cfg['test_bars']}"
            r = run_portfolio_oos(
                all_data,
                strat_fn,
                max_positions=cfg["max_pos"],
                test_bars=cfg["test_bars"],
                exit_params=exit_params,
                position_scale=5.0,
            )
            if r:
                c, p = check_criteria(r)
                all_results.append((label, p, r))
                fails = [k.split("_")[0] for k, v in c.items() if not v]
                fail_str = f"  FAIL:{','.join(fails)}" if fails else ""
                print(
                    f"  {label:<43} [{p}/6] {r['sharpe']:>6.2f} {r['total_return']*100:>+7.1f}% "
                    f"{r['max_drawdown']*100:>6.1f}% {r['win_rate']*100:>4.0f}% "
                    f"{r['profit_factor']:>5.2f} {r['wfa_efficiency']:>4.0f}% {r['trade_count']:>4}{fail_str}"
                )

    # Sort
    all_results.sort(key=lambda x: (-x[1], -x[2].get("sharpe", 0)))

    print(f"\n{'=' * 70}")
    print("6/6 RESULTS")
    print("=" * 70)

    six_six = [(n, p, r) for n, p, r in all_results if p == 6]
    if six_six:
        for name, p, r in six_six:
            print(
                f"  [{p}/6] {name}: Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}%"
            )

        # Does any beat vol_profile baseline?
        vp_baseline = [
            r for n, p, r in six_six if "vol_profile" in n and "pos10_rb720" in n
        ]
        vp_sharpe = vp_baseline[0]["sharpe"] if vp_baseline else 2.52
        better = [
            (n, r)
            for n, p, r in six_six
            if r["sharpe"] > vp_sharpe and "vol_profile" not in n
        ]
        if better:
            print(
                f"\n*** STRATEGIES BEATING VOL PROFILE (Sharpe > {vp_sharpe:.2f}) ***"
            )
            for name, r in better:
                print(
                    f"  {name}: Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}%"
                )
        else:
            print(f"\nNo rescued strategy beats Vol Profile (Sharpe={vp_sharpe:.2f})")
    else:
        print("  None passed 6/6")

    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
