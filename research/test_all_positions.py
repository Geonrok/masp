#!/usr/bin/env python3
"""Test vol_profile with varying max_positions: 10, 30, 50, 100, ALL"""
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

DATA_ROOT = Path("E:/data/crypto_ohlcv")
COMMISSION = 0.0004
FUNDING_PER_8H = 0.0001


def load_ohlcv(symbol, timeframe="1h"):
    path = DATA_ROOT / f"binance_futures_{timeframe}" / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ['datetime', 'timestamp', 'date']:
        if col in df.columns:
            df['datetime'] = pd.to_datetime(df[col])
            break
    return df.sort_values('datetime').reset_index(drop=True)


def calc_atr(high, low, close, period=14):
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return pd.Series(tr).rolling(period).mean().values


def vol_profile_strategy(df, lookback=48, vwap_mult=1.01, ema_fast=50, ema_slow=200):
    close = df['close']
    high = df['high']
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    signals = np.where((close > upper) & (close > vwap * vwap_mult) & (ema_f > ema_s), 1, 0)
    return signals


def simulate(df, signals, position_pct=0.02, max_bars=72,
             atr_stop=3.0, profit_target_atr=8.0, slippage=0.0003):
    capital = 1.0
    position = 0
    entry_price = 0
    bars_held = 0
    trades = []

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
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
        'total_return': capital - 1,
        'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
        'profit_factor': gp / (gl + 1e-10),
        'trade_count': len(trades),
        'trades': trades,
    }


def check_criteria(r):
    c = {
        'sharpe_gt_1': r.get('sharpe', 0) > 1.0,
        'max_dd_lt_25': r.get('max_drawdown', -1) > -0.25,
        'win_rate_gt_45': r.get('win_rate', 0) > 0.45,
        'profit_factor_gt_1_5': r.get('profit_factor', 0) > 1.5,
        'wfa_efficiency_gt_50': r.get('wfa_efficiency', 0) > 50,
        'trade_count_gt_100': r.get('trade_count', 0) > 100,
    }
    return c, sum(v for v in c.values())


def run_portfolio_oos(all_data, max_positions, test_bars=720, position_scale=5.0):
    exit_params = {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0}

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
            vol = df['close'].iloc[:i].pct_change().rolling(168).std().iloc[-1]
            if np.isnan(vol) or vol == 0:
                vol = 0.01
            scored.append((symbol, vol))
        scored.sort(key=lambda x: x[1])
        selected = scored[:max_positions]

        for symbol, vol in selected:
            df = oos_data[symbol]
            if i + test_bars > len(df):
                continue
            full = df.iloc[:i + test_bars]
            sigs = vol_profile_strategy(full)
            test_sigs = sigs[i:i + test_bars]
            test_df = df.iloc[i:i + test_bars].copy().reset_index(drop=True)
            ann_vol = vol * np.sqrt(24 * 365)
            position_pct = min(0.10 / (ann_vol + 1e-10) / max(len(selected), 1), 0.05)
            position_pct *= position_scale
            r = simulate(test_df, test_sigs, position_pct,
                       exit_params['max_bars'], exit_params['atr_stop'],
                       exit_params['profit_target_atr'], 0.0003)
            period_pnl += r['total_return']
            all_trades.extend(r['trades'])

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
        'total_return': float(equity_arr[-1] - 1),
        'max_drawdown': float(dd.min()),
        'sharpe': float(sharpe),
        'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
        'profit_factor': gp / (gl + 1e-10),
        'trade_count': len(all_trades),
        'periods': len(period_returns),
        'wfa_efficiency': sum(1 for r in period_returns if r > 0) / len(period_returns) * 100,
    }


def main():
    print("=" * 70)
    print("VOL PROFILE: MAX POSITIONS COMPARISON (5x scale)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    all_path = DATA_ROOT / "binance_futures_1h"
    all_data = {}
    for f in sorted(all_path.glob("*USDT.csv")):
        symbol = f.stem
        df = load_ohlcv(symbol, "1h")
        if not df.empty and len(df) > 10000:
            all_data[symbol] = df
    total_symbols = len(all_data)
    print(f"Total symbols: {total_symbols}\n")

    print(f"{'Positions':<12} {'Pass':>4} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'WR':>5} {'PF':>6} {'WFA':>5} {'Trades':>7}")
    print("-" * 70)

    for max_pos in [5, 10, 20, 30, 50, 100, total_symbols]:
        label = f"{max_pos}" if max_pos < total_symbols else f"ALL({total_symbols})"
        r = run_portfolio_oos(all_data, max_pos, test_bars=720, position_scale=5.0)
        if r:
            c, p = check_criteria(r)
            print(f"{label:<12} [{p}/6] {r['sharpe']:>6.2f} {r['total_return']*100:>+7.1f}% "
                  f"{r['max_drawdown']*100:>6.1f}% {r['win_rate']*100:>4.0f}% "
                  f"{r['profit_factor']:>5.2f} {r['wfa_efficiency']:>4.0f}% {r['trade_count']:>6}")
            fails = [k for k, v in c.items() if not v]
            if fails:
                print(f"             FAILS: {', '.join(fails)}")

    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
