#!/usr/bin/env python3
"""
Phase 17: DVOL (Deribit Volatility Index) 기반 전략 테스트
- BTC DVOL 일봉 데이터 (2021-03~2026-01) 활용
- TRUE OOS 프레임워크 (60/40 시간 분할, 6개 기준)

전략 유형:
A. DVOL 레벨 기반: 고변동성 회피, 저변동성 진입
B. DVOL 변화율 기반: 급등/급락 시 역행
C. DVOL + Vol Profile 복합: 기존 전략에 DVOL 필터 추가
D. DVOL 레짐 전환: 변동성 수축→확장 전환점 포착
"""
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

DATA_ROOT = Path("E:/data/crypto_ohlcv")
DVOL_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/data/btc_dvol.csv")
RESULTS_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/results")

COMMISSION = 0.0004
FUNDING_PER_8H = 0.0001


def load_dvol():
    """Load BTC DVOL daily data"""
    df = pd.read_csv(DVOL_PATH, skiprows=1)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.rename(columns={'close': 'dvol_close', 'open': 'dvol_open',
                            'high': 'dvol_high', 'low': 'dvol_low'})
    return df[['date', 'dvol_open', 'dvol_high', 'dvol_low', 'dvol_close']]


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


def merge_dvol_to_hourly(df_hourly, dvol_daily):
    """Merge daily DVOL into hourly data by date"""
    df_hourly = df_hourly.copy()
    df_hourly['date'] = df_hourly['datetime'].dt.date
    dvol_daily = dvol_daily.copy()
    dvol_daily['date'] = dvol_daily['date'].dt.date
    merged = df_hourly.merge(dvol_daily, on='date', how='left')
    # Forward fill DVOL for dates without data
    for col in ['dvol_close', 'dvol_open', 'dvol_high', 'dvol_low']:
        if col in merged.columns:
            merged[col] = merged[col].ffill()
    return merged


# ============================================================
# Strategy A: DVOL Level-based (low vol = buy opportunity)
# ============================================================
def strat_dvol_low_vol_entry(df, dvol_thresh_low=40, lookback=48):
    """Enter long when DVOL < threshold (low fear = trending market)"""
    close = df['close']
    high = df['high']
    dvol = df['dvol_close'] if 'dvol_close' in df.columns else pd.Series(50, index=df.index)

    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    signals = np.where(
        (close > upper) & (ema_f > ema_s) & (dvol < dvol_thresh_low),
        1, 0
    )
    return signals


def strat_dvol_high_vol_avoid(df, dvol_thresh_high=70, lookback=48):
    """Vol Profile but skip entries when DVOL > threshold (too volatile)"""
    close = df['close']
    high = df['high']
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    dvol = df['dvol_close'] if 'dvol_close' in df.columns else pd.Series(50, index=df.index)

    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    signals = np.where(
        (close > upper) & (close > vwap * 1.01) & (ema_f > ema_s) & (dvol < dvol_thresh_high),
        1, 0
    )
    return signals


# ============================================================
# Strategy B: DVOL Change-based (contrarian on vol spikes)
# ============================================================
def strat_dvol_spike_contrarian(df, dvol_change_thresh=-5, lookback=48):
    """Enter long after DVOL drops sharply (vol crush = rally signal)"""
    close = df['close']
    high = df['high']
    dvol = df['dvol_close'] if 'dvol_close' in df.columns else pd.Series(50, index=df.index)
    dvol_change = dvol - dvol.shift(24)  # 24h change in DVOL

    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    signals = np.where(
        (close > upper) & (ema_f > ema_s) & (dvol_change < dvol_change_thresh),
        1, 0
    )
    return signals


def strat_dvol_mean_reversion(df, dvol_zscore_thresh=2.0, lookback=48):
    """Enter long when DVOL z-score is extremely high (fear = buy opportunity)"""
    close = df['close']
    dvol = df['dvol_close'] if 'dvol_close' in df.columns else pd.Series(50, index=df.index)

    # DVOL z-score over 30-day rolling window (30*24 = 720 bars)
    dvol_ma = dvol.rolling(720).mean()
    dvol_std = dvol.rolling(720).std()
    dvol_z = (dvol - dvol_ma) / (dvol_std + 1e-10)

    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    # High DVOL z-score = extreme fear → contrarian long
    signals = np.where(
        (dvol_z > dvol_zscore_thresh) & (ema_f > ema_s),
        1, 0
    )
    return signals


# ============================================================
# Strategy C: DVOL + Vol Profile Composite
# ============================================================
def strat_vol_profile_dvol_filter(df, dvol_max=80, lookback=48):
    """Vol Profile Breakout + DVOL filter (skip extreme vol)"""
    close = df['close']
    high = df['high']
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    dvol = df['dvol_close'] if 'dvol_close' in df.columns else pd.Series(50, index=df.index)

    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    signals = np.where(
        (close > upper) & (close > vwap * 1.01) & (ema_f > ema_s) & (dvol < dvol_max),
        1, 0
    )
    return signals


def strat_vol_profile_dvol_regime(df, lookback=48):
    """Vol Profile + DVOL regime: tighter stops in high vol, wider in low vol"""
    close = df['close']
    high = df['high']
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    dvol = df['dvol_close'] if 'dvol_close' in df.columns else pd.Series(50, index=df.index)

    vwap = (close * vol).rolling(lookback).sum() / (vol.rolling(lookback).sum() + 1e-10)
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    # Signal = 1 (low vol regime) or 2 (high vol regime) for different exit params
    base = (close > upper) & (close > vwap * 1.01) & (ema_f > ema_s)
    signals = np.where(base & (dvol < 50), 1,
              np.where(base & (dvol >= 50), 2, 0))
    return signals


# ============================================================
# Strategy D: DVOL Regime Transition
# ============================================================
def strat_dvol_contraction_breakout(df, lookback=48):
    """Enter when DVOL contracts to low then price breaks out (vol squeeze)"""
    close = df['close']
    high = df['high']
    dvol = df['dvol_close'] if 'dvol_close' in df.columns else pd.Series(50, index=df.index)

    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    # DVOL percentile over 60 days (60*24 = 1440 bars)
    dvol_pct = dvol.rolling(1440).rank(pct=True)

    # Low DVOL percentile (<30%) + breakout = vol squeeze breakout
    signals = np.where(
        (close > upper) & (ema_f > ema_s) & (dvol_pct < 0.3),
        1, 0
    )
    return signals


def strat_dvol_falling(df, lookback=48):
    """Enter when DVOL is falling (5-day MA declining) + breakout"""
    close = df['close']
    high = df['high']
    dvol = df['dvol_close'] if 'dvol_close' in df.columns else pd.Series(50, index=df.index)

    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    dvol_ma5 = dvol.rolling(120).mean()  # 5-day MA of DVOL
    dvol_falling = dvol_ma5 < dvol_ma5.shift(24)  # declining over 24h

    signals = np.where(
        (close > upper) & (ema_f > ema_s) & dvol_falling,
        1, 0
    )
    return signals


# ============================================================
# Simulation & Portfolio
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
            position = 1  # long only
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


def run_portfolio_oos(all_data, dvol_daily, strat_fn, strat_name,
                      max_positions=10, test_bars=720,
                      exit_params=None, position_scale=5.0):
    if exit_params is None:
        exit_params = {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0}

    # Prepare OOS data with DVOL merged
    oos_data = {}
    for symbol in all_data:
        df = all_data[symbol]
        split = int(len(df) * 0.6)
        oos_df = df.iloc[split:].copy().reset_index(drop=True)
        if len(oos_df) > 2000:
            # Merge DVOL
            oos_merged = merge_dvol_to_hourly(oos_df, dvol_daily)
            if oos_merged['dvol_close'].notna().sum() > 1000:
                oos_data[symbol] = oos_merged

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
    }


def main():
    print("=" * 70)
    print("PHASE 17: DVOL (Deribit Volatility Index) STRATEGIES")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Load DVOL
    dvol = load_dvol()
    print(f"DVOL data: {dvol['date'].min()} to {dvol['date'].max()} ({len(dvol)} days)")
    print(f"DVOL range: {dvol['dvol_close'].min():.1f} - {dvol['dvol_close'].max():.1f}")
    print(f"DVOL mean: {dvol['dvol_close'].mean():.1f}, median: {dvol['dvol_close'].median():.1f}\n")

    # Load hourly data
    all_path = DATA_ROOT / "binance_futures_1h"
    all_data = {}
    for f in sorted(all_path.glob("*USDT.csv")):
        symbol = f.stem
        df = load_ohlcv(symbol, "1h")
        if not df.empty and len(df) > 10000:
            all_data[symbol] = df
    print(f"Loaded {len(all_data)} symbols\n")

    exit_params = {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0}

    strategies = {
        # A: Level-based
        'A1_low_vol_entry_40': lambda df: strat_dvol_low_vol_entry(df, dvol_thresh_low=40),
        'A2_low_vol_entry_50': lambda df: strat_dvol_low_vol_entry(df, dvol_thresh_low=50),
        'A3_low_vol_entry_60': lambda df: strat_dvol_low_vol_entry(df, dvol_thresh_low=60),
        'A4_high_vol_avoid_70': lambda df: strat_dvol_high_vol_avoid(df, dvol_thresh_high=70),
        'A5_high_vol_avoid_80': lambda df: strat_dvol_high_vol_avoid(df, dvol_thresh_high=80),
        'A6_high_vol_avoid_90': lambda df: strat_dvol_high_vol_avoid(df, dvol_thresh_high=90),
        # B: Change-based
        'B1_spike_contrarian_m5': lambda df: strat_dvol_spike_contrarian(df, dvol_change_thresh=-5),
        'B2_spike_contrarian_m3': lambda df: strat_dvol_spike_contrarian(df, dvol_change_thresh=-3),
        'B3_mean_reversion_z2': lambda df: strat_dvol_mean_reversion(df, dvol_zscore_thresh=2.0),
        'B4_mean_reversion_z1.5': lambda df: strat_dvol_mean_reversion(df, dvol_zscore_thresh=1.5),
        # C: Composite with Vol Profile
        'C1_vp_dvol_max70': lambda df: strat_vol_profile_dvol_filter(df, dvol_max=70),
        'C2_vp_dvol_max80': lambda df: strat_vol_profile_dvol_filter(df, dvol_max=80),
        'C3_vp_dvol_max90': lambda df: strat_vol_profile_dvol_filter(df, dvol_max=90),
        # D: Regime transition
        'D1_contraction_bkout': lambda df: strat_dvol_contraction_breakout(df),
        'D2_dvol_falling': lambda df: strat_dvol_falling(df),
    }

    print(f"{'Strategy':<30} {'Pass':>4} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'WR':>5} {'PF':>6} {'WFA':>5} {'T':>5}")
    print("-" * 85)

    all_results = []
    for name, fn in strategies.items():
        r = run_portfolio_oos(all_data, dvol, fn, name,
                              max_positions=10, test_bars=720,
                              exit_params=exit_params, position_scale=5.0)
        if r:
            c, p = check_criteria(r)
            all_results.append((name, p, r))
            fails = [k for k, v in c.items() if not v]
            print(f"  {name:<28} [{p}/6] {r['sharpe']:>6.2f} {r['total_return']*100:>+7.1f}% "
                  f"{r['max_drawdown']*100:>6.1f}% {r['win_rate']*100:>4.0f}% "
                  f"{r['profit_factor']:>5.2f} {r['wfa_efficiency']:>4.0f}% {r['trade_count']:>4}")
            if fails:
                print(f"    FAILS: {', '.join(fails)}")
        else:
            print(f"  {name:<28} NO DATA (DVOL date range mismatch)")

    # Summary
    all_results.sort(key=lambda x: (-x[1], -x[2].get('sharpe', 0)))

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    six_six = [(n, p, r) for n, p, r in all_results if p == 6]
    if six_six:
        print("\n6/6 PASSED:")
        for name, p, r in six_six:
            print(f"  {name}: Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                  f"DD={r['max_drawdown']*100:.1f}%")

        # Compare vs Vol Profile baseline (Sharpe 2.52)
        better = [(n, r) for n, p, r in six_six if r['sharpe'] > 2.52]
        if better:
            print(f"\n*** BEATS VOL PROFILE (Sharpe > 2.52) ***")
            for name, r in better:
                print(f"  {name}: Sharpe={r['sharpe']:.2f}")
        else:
            print(f"\nNo DVOL strategy beats Vol Profile (Sharpe=2.52)")
    else:
        print("\nNo strategy passed 6/6")
        best = all_results[0] if all_results else None
        if best:
            print(f"Best: {best[0]} [{best[1]}/6] Sharpe={best[2]['sharpe']:.2f}")

    print(f"\nVol Profile baseline reminder: Sharpe=2.52, 6/6 PASS")
    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
