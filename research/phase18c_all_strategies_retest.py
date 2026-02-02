#!/usr/bin/env python3
"""
Phase 18c: 수정된 기준 + 수정된 OOS로 전 전략 재검증

수정 기준 (추세추종 적합):
- Sharpe > 1.0
- MaxDD < 25%
- WR > 30% (기존 45% → 완화)
- PF > 1.3 (기존 1.5 → 완화)
- WFA > 50%
- Trades > 100

OOS: min_oos_bars=16000 (장기 종목, 18윈도우, ~2년)
"""
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


# ============================================================
# ALL STRATEGIES
# ============================================================

def strat_vol_profile(df, lookback=48):
    close = df['close']
    high = df['high']
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    return np.where((close > upper) & (close > vwap * 1.01) & (ema_f > ema_s), 1, 0)


def strat_dual_ma(df, lookback=48):
    close = df['close']
    high = df['high']
    atr = pd.Series(calc_atr(high.values, df['low'].values, close.values, 14), index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    return np.where((close > upper) & (ema_f > ema_s) & (atr > atr_avg), 1, 0)


def strat_momentum_filter(df, lookback=48):
    close = df['close']
    high = df['high']
    ret = close.pct_change(lookback)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    return np.where((close > upper) & (ret > 0.05) & (ema_f > ema_s), 1, 0)


def strat_adx_filter(df, lookback=48, adx_period=14, adx_thresh=25):
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr = pd.Series(calc_atr(high.values, low.values, close.values, adx_period), index=df.index)
    plus_di = 100 * (plus_dm.rolling(adx_period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(adx_period).mean() / (atr + 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(adx_period).mean()
    return np.where((close > upper) & (close > vwap * 1.01) & (ema_f > ema_s) & (adx > adx_thresh), 1, 0)


def strat_obv_trend(df, lookback=48):
    close = df['close']
    high = df['high']
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    obv = (np.sign(close.diff()) * vol).cumsum()
    obv_ma = obv.rolling(lookback).mean()
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    return np.where((close > upper) & (obv > obv_ma) & (ema_f > ema_s), 1, 0)


def strat_mfi_breakout(df, lookback=48, mfi_thresh=60):
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    tp = (high + low + close) / 3
    mf = tp * vol
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = mf.where(tp <= tp.shift(1), 0).rolling(14).sum()
    mfi = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-10))
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    return np.where((close > upper) & (mfi > mfi_thresh) & (ema_f > ema_s), 1, 0)


def strat_vwap_breakout(df, lookback=48):
    close = df['close']
    high = df['high']
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    return np.where((close > upper) & (close > vwap * 1.02) & (ema_f > ema_s), 1, 0)


# ============================================================
# SIMULATION
# ============================================================

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
            position = 1
            entry_price = c * (1 + slippage)
            capital -= COMMISSION * position_pct
            bars_held = 0

    if position != 0:
        c = close[-1]
        exit_p = c * (1 - slippage)
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


def check_criteria_original(r):
    c = {
        'sharpe_gt_1': r.get('sharpe', 0) > 1.0,
        'max_dd_lt_25': r.get('max_drawdown', -1) > -0.25,
        'win_rate_gt_45': r.get('win_rate', 0) > 0.45,
        'profit_factor_gt_1_5': r.get('profit_factor', 0) > 1.5,
        'wfa_efficiency_gt_50': r.get('wfa_efficiency', 0) > 50,
        'trade_count_gt_100': r.get('trade_count', 0) > 100,
    }
    return c, sum(v for v in c.values())


def check_criteria_revised(r):
    c = {
        'sharpe_gt_1': r.get('sharpe', 0) > 1.0,
        'max_dd_lt_25': r.get('max_drawdown', -1) > -0.25,
        'win_rate_gt_30': r.get('win_rate', 0) > 0.30,
        'profit_factor_gt_1_3': r.get('profit_factor', 0) > 1.3,
        'wfa_efficiency_gt_50': r.get('wfa_efficiency', 0) > 50,
        'trade_count_gt_100': r.get('trade_count', 0) > 100,
    }
    return c, sum(v for v in c.values())


def run_portfolio_oos(all_data, strat_fn, min_oos_bars=16000,
                      max_positions=10, test_bars=720,
                      position_scale=5.0, train_bars=4320,
                      exit_params=None):
    if exit_params is None:
        exit_params = {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0}

    oos_data = {}
    for symbol in all_data:
        df = all_data[symbol]
        split = int(len(df) * 0.6)
        oos_df = df.iloc[split:].copy().reset_index(drop=True)
        if len(oos_df) > min_oos_bars:
            oos_data[symbol] = oos_df

    if not oos_data:
        return None

    min_len = min(len(d) for d in oos_data.values())

    equity = [1.0]
    period_returns = []
    all_trades = []

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
            sigs = strat_fn(full)
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
        'period_returns': period_returns,
    }


def main():
    print("=" * 80)
    print("PHASE 18c: ALL STRATEGIES RE-TEST (CORRECTED OOS + REVISED CRITERIA)")
    print("=" * 80)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Load data
    all_path = DATA_ROOT / "binance_futures_1h"
    all_data = {}
    for f in sorted(all_path.glob("*USDT.csv")):
        symbol = f.stem
        df = load_ohlcv(symbol, "1h")
        if not df.empty and len(df) > 10000:
            all_data[symbol] = df
    print(f"Total symbols loaded: {len(all_data)}")

    # Count qualifying symbols
    qualifying = sum(1 for df in all_data.values() if int(len(df) * 0.4) > 16000)
    print(f"Symbols with OOS > 16000: {qualifying}\n")

    # Define all strategies to test
    strategies = {
        # Phase 10: Baseline
        'dual_ma_breakout': {
            'fn': strat_dual_ma,
            'exit': {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0},
        },
        # Phase 13: Volume strategies
        'vol_profile': {
            'fn': strat_vol_profile,
            'exit': {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0},
        },
        'obv_trend': {
            'fn': strat_obv_trend,
            'exit': {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0},
        },
        'mfi_breakout_60': {
            'fn': lambda df: strat_mfi_breakout(df, mfi_thresh=60),
            'exit': {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0},
        },
        'mfi_breakout_80': {
            'fn': lambda df: strat_mfi_breakout(df, mfi_thresh=80),
            'exit': {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0},
        },
        'vwap_breakout': {
            'fn': strat_vwap_breakout,
            'exit': {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0},
        },
        # Phase 15: Variants
        'momentum_filter': {
            'fn': strat_momentum_filter,
            'exit': {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0},
        },
        'adx_filter': {
            'fn': strat_adx_filter,
            'exit': {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0},
        },
        # Exit variants
        'vp_tight_exit': {
            'fn': strat_vol_profile,
            'exit': {'max_bars': 48, 'atr_stop': 2.0, 'profit_target_atr': 6.0},
        },
        'vp_wide_exit': {
            'fn': strat_vol_profile,
            'exit': {'max_bars': 96, 'atr_stop': 4.0, 'profit_target_atr': 10.0},
        },
        'dm_tight_exit': {
            'fn': strat_dual_ma,
            'exit': {'max_bars': 48, 'atr_stop': 2.0, 'profit_target_atr': 6.0},
        },
        'dm_wide_exit': {
            'fn': strat_dual_ma,
            'exit': {'max_bars': 96, 'atr_stop': 4.0, 'profit_target_atr': 10.0},
        },
    }

    print(f"Testing {len(strategies)} strategies at 5x scale, OOS>16000 (18 windows, ~2yr)\n")

    header = (f"{'Strategy':<22} {'Orig':>4} {'Rev':>4} {'Sharpe':>7} {'Return':>8} "
              f"{'MaxDD':>7} {'WR':>5} {'PF':>6} {'WFA':>5} {'T':>5} {'W':>3}")
    print(header)
    print("-" * 90)

    results = []
    for name, cfg in strategies.items():
        r = run_portfolio_oos(all_data, cfg['fn'],
                              min_oos_bars=16000,
                              max_positions=10,
                              test_bars=720,
                              position_scale=5.0,
                              exit_params=cfg['exit'])
        if r:
            c_orig, p_orig = check_criteria_original(r)
            c_rev, p_rev = check_criteria_revised(r)
            results.append((name, p_orig, p_rev, r))

            fails_orig = [k.split('_')[0] for k, v in c_orig.items() if not v]
            fails_rev = [k.split('_')[0] for k, v in c_rev.items() if not v]

            print(f"  {name:<20} [{p_orig}/6] [{p_rev}/6] {r['sharpe']:>6.2f} "
                  f"{r['total_return']*100:>+7.1f}% {r['max_drawdown']*100:>6.1f}% "
                  f"{r['win_rate']*100:>4.0f}% {r['profit_factor']:>5.2f} "
                  f"{r['wfa_efficiency']:>4.0f}% {r['trade_count']:>4} {r['periods']:>3}")
            if fails_rev:
                print(f"    Rev fails: {', '.join(fails_rev)}")
        else:
            print(f"  {name:<20} NO DATA")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)

    print("\nOriginal criteria (WR>45%, PF>1.5):")
    orig_pass = [(n, p, r) for n, p, _, r in results if p == 6]
    if orig_pass:
        for n, p, r in orig_pass:
            print(f"  PASS: {n} Sharpe={r['sharpe']:.2f}")
    else:
        print("  NONE PASSED 6/6")

    print("\nRevised criteria (WR>30%, PF>1.3):")
    rev_pass = [(n, p, r) for n, _, p, r in results if p == 6]
    if rev_pass:
        for n, p, r in sorted(rev_pass, key=lambda x: -x[2]['sharpe']):
            print(f"  PASS: {n} Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                  f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                  f"PF={r['profit_factor']:.2f} T={r['trade_count']}")
    else:
        print("  NONE PASSED 6/6")

    # 5/6 in revised
    print("\nRevised 5/6 (near-pass):")
    rev_5 = [(n, p, r) for n, _, p, r in results if p == 5]
    for n, p, r in sorted(rev_5, key=lambda x: -x[2]['sharpe']):
        print(f"  [{p}/6] {n} Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}%")

    # Yearly breakdown for top strategies
    print(f"\n{'=' * 80}")
    print("YEARLY BREAKDOWN (top strategies)")
    print("=" * 80)

    top = sorted(results, key=lambda x: -x[2])[:5]  # top 5 by revised score
    for name, p_orig, p_rev, r in top:
        if r.get('period_returns'):
            prs = r['period_returns']
            # 18 windows, assume roughly 720 bars = 30 days each
            # First window starts after train_bars in OOS
            n = len(prs)
            mid = n // 2
            first_half = prs[:mid]
            second_half = prs[mid:]
            r1 = 1.0
            for p in first_half:
                r1 *= (1 + p)
            r2 = 1.0
            for p in second_half:
                r2 *= (1 + p)
            w1 = sum(1 for p in first_half if p > 0) / len(first_half) * 100
            w2 = sum(1 for p in second_half if p > 0) / len(second_half) * 100
            print(f"\n  {name} [{p_rev}/6 rev]:")
            print(f"    First half ({len(first_half)}W): Ret={( r1-1)*100:+.1f}% WinRate={w1:.0f}%")
            print(f"    Second half ({len(second_half)}W): Ret={(r2-1)*100:+.1f}% WinRate={w2:.0f}%")

    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
