#!/usr/bin/env python3
"""
Ralph-Loop Phase 8: Bear Market Fix & Strategy Improvement
============================================================
Phase 7 revealed:
1. Bear market: PF=0.14, -163% → CATASTROPHIC
2. True OOS top_30: 2/6, all_257: 4/6
3. WR consistently ~31-43%, needs >45%

Approach:
A) Add regime filter: reduce/skip trades in bear regime
B) Improve short signal quality (bear market shorts are failing)
C) Test alternative strategy variants that are more regime-aware
D) Use full universe (257 symbols) since diversification helps
E) Relax entry: test multiple ATR expansion thresholds
"""
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

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


# =============================================================================
# STRATEGY VARIANTS
# =============================================================================

def strat_base(df, lookback=96, ema_period=200, atr_expansion=1.3):
    """Original ATR Expansion Breakout (Phase 5/6/7 strategy)"""
    close = df['close']
    high = df['high']
    low = df['low']
    upper = high.rolling(lookback).max().shift(1)
    lower = low.rolling(lookback).min().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()
    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion
    signals = np.where((close > upper) & (close > ema) & expanding, 1,
              np.where((close < lower) & (close < ema) & expanding, -1, 0))
    return signals


def strat_long_only(df, lookback=96, ema_period=200, atr_expansion=1.3):
    """Long-only: eliminate catastrophic bear-market shorts"""
    close = df['close']
    high = df['high']
    low = df['low']
    upper = high.rolling(lookback).max().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()
    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion
    signals = np.where((close > upper) & (close > ema) & expanding, 1, 0)
    return signals


def strat_regime_filter(df, lookback=96, ema_period=200, atr_expansion=1.3):
    """Regime-aware: long in bull/sideways, short only in strong bear"""
    close = df['close']
    high = df['high']
    low = df['low']
    upper = high.rolling(lookback).max().shift(1)
    lower = low.rolling(lookback).min().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()
    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion

    # Regime: 30-day return of price
    ret_30d = close.pct_change(720)
    bull = ret_30d > 0.05
    bear = ret_30d < -0.15  # Only short in strong bear

    signals = np.where((close > upper) & (close > ema) & expanding & ~bear, 1,
              np.where((close < lower) & (close < ema) & expanding & bear, -1, 0))
    return signals.astype(int)


def strat_regime_v2(df, lookback=96, ema_period=200, atr_expansion=1.3):
    """Regime v2: No shorts at all, use regime for position sizing signal"""
    close = df['close']
    high = df['high']
    low = df['low']
    upper = high.rolling(lookback).max().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()
    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion

    # Only long when medium-term trend is not deeply negative
    ret_30d = close.pct_change(720)
    not_crash = ret_30d > -0.20

    signals = np.where((close > upper) & (close > ema) & expanding & not_crash, 1, 0)
    return signals


def strat_dual_ma(df, lookback=96, ema_fast=50, ema_slow=200, atr_expansion=1.3):
    """Dual MA filter: breakout + fast EMA > slow EMA for trend confirmation"""
    close = df['close']
    high = df['high']
    low = df['low']
    upper = high.rolling(lookback).max().shift(1)
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion

    # Long: breakout + fast > slow (uptrend confirmed)
    signals = np.where((close > upper) & (ema_f > ema_s) & expanding, 1, 0)
    return signals


def strat_volume_confirm(df, lookback=96, ema_period=200, atr_expansion=1.3):
    """Volume confirmation: breakout + above-average volume"""
    close = df['close']
    high = df['high']
    low = df['low']
    upper = high.rolling(lookback).max().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()
    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion

    vol = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)
    vol_avg = vol.rolling(lookback).mean()
    vol_high = vol > vol_avg * 1.5

    signals = np.where((close > upper) & (close > ema) & expanding & vol_high, 1, 0)
    return signals


def strat_relaxed_atr(df, lookback=96, ema_period=200, atr_expansion=1.0):
    """Relaxed ATR expansion (1.0x instead of 1.3x) - more trades"""
    close = df['close']
    high = df['high']
    low = df['low']
    upper = high.rolling(lookback).max().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()
    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion
    signals = np.where((close > upper) & (close > ema) & expanding, 1, 0)
    return signals


def strat_momentum_filter(df, lookback=96, ema_period=200, atr_expansion=1.3):
    """Momentum confirmation: breakout + positive momentum (ROC > 0)"""
    close = df['close']
    high = df['high']
    low = df['low']
    upper = high.rolling(lookback).max().shift(1)
    ema = close.ewm(span=ema_period, adjust=False).mean()
    atr_vals = calc_atr(high.values, low.values, close.values, 14)
    atr = pd.Series(atr_vals, index=df.index)
    atr_avg = atr.rolling(lookback).mean()
    expanding = atr > atr_avg * atr_expansion

    roc = close.pct_change(48)  # 2-day momentum
    momentum_pos = roc > 0

    signals = np.where((close > upper) & (close > ema) & expanding & momentum_pos, 1, 0)
    return signals


# =============================================================================
# SIMULATION
# =============================================================================
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


# =============================================================================
# MAIN TEST: True OOS with each variant
# =============================================================================
def run_true_oos_test(strategy_fn, strategy_name, all_data, exit_params=None):
    """Run true out-of-sample test for a strategy variant."""
    if exit_params is None:
        exit_params = {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0}

    # STEP A: Selection on first 60%
    selection_results = {}
    for symbol, df in all_data.items():
        split = int(len(df) * 0.6)
        sel_df = df.iloc[:split].copy().reset_index(drop=True)
        if len(sel_df) < 5000:
            continue

        train_bars = 4320
        test_bars = 720
        all_trades = []
        period_returns = []

        i = train_bars
        while i + test_bars <= len(sel_df):
            full = sel_df.iloc[:i + test_bars]
            sigs = strategy_fn(full)
            test_sigs = sigs[i:i + test_bars]
            test_df = sel_df.iloc[i:i + test_bars].copy().reset_index(drop=True)
            r = simulate(test_df, test_sigs, 0.02,
                        exit_params['max_bars'], exit_params['atr_stop'],
                        exit_params['profit_target_atr'], 0.0003)
            period_returns.append(r['total_return'])
            all_trades.extend(r['trades'])
            i += test_bars

        if not all_trades:
            continue

        total_ret = 1.0
        for pr in period_returns:
            total_ret *= (1 + pr)

        selection_results[symbol] = {
            'total_return': total_ret - 1,
            'trade_count': len(all_trades),
        }

    if not selection_results:
        return None

    # STEP B: TRUE OOS on last 40%
    # Test universes: all symbols (since all_257 was best in Phase 7)
    universes = {}

    # All symbols (best in Phase 7)
    universes['all'] = list(all_data.keys())

    # Top 30 by selection period
    ranked = sorted(selection_results.items(), key=lambda x: x[1]['total_return'], reverse=True)
    universes['top_30'] = [s for s, _ in ranked[:30]]
    universes['top_50'] = [s for s, _ in ranked[:50]]

    results = {}
    for univ_name, univ_symbols in universes.items():
        oos_data = {}
        for symbol in univ_symbols:
            if symbol not in all_data:
                continue
            df = all_data[symbol]
            split = int(len(df) * 0.6)
            oos_df = df.iloc[split:].copy().reset_index(drop=True)
            if len(oos_df) > 2000:
                oos_data[symbol] = oos_df

        if not oos_data:
            continue

        equity = [1.0]
        period_returns = []
        all_trades = []
        min_len = min(len(d) for d in oos_data.values())

        train_bars = min(4320, min_len // 3)
        test_bars = 720
        max_positions = 5

        i = train_bars
        while i + test_bars <= min_len:
            period_pnl = 0

            scored = []
            for symbol, df in oos_data.items():
                if len(df) <= i:
                    continue
                train_ret = df['close'].iloc[:i].pct_change()
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
                full = df.iloc[:i + test_bars]
                sigs = strategy_fn(full)
                test_sigs = sigs[i:i + test_bars]
                test_df = df.iloc[i:i + test_bars].copy().reset_index(drop=True)

                ann_vol = vol * np.sqrt(24 * 365)
                position_pct = min(0.10 / (ann_vol + 1e-10) / max(len(selected), 1), 0.05)

                r = simulate(test_df, test_sigs, position_pct,
                           exit_params['max_bars'], exit_params['atr_stop'],
                           exit_params['profit_target_atr'], 0.0003)
                period_pnl += r['total_return']
                all_trades.extend(r['trades'])

            period_returns.append(period_pnl)
            equity.append(equity[-1] * (1 + period_pnl))
            i += test_bars

        if not period_returns:
            continue

        equity_arr = np.array(equity)
        peak = np.maximum.accumulate(equity_arr)
        dd = (equity_arr - peak) / peak

        wins = sum(1 for t in all_trades if t > 0)
        losses = sum(1 for t in all_trades if t <= 0)
        gp = sum(t for t in all_trades if t > 0)
        gl = abs(sum(t for t in all_trades if t < 0))
        sharpe = np.mean(period_returns) / (np.std(period_returns) + 1e-10) * np.sqrt(12)

        result = {
            'total_return': float(equity_arr[-1] - 1),
            'max_drawdown': float(dd.min()),
            'sharpe': float(sharpe),
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
            'profit_factor': gp / (gl + 1e-10),
            'trade_count': len(all_trades),
            'periods': len(period_returns),
            'wfa_efficiency': sum(1 for r in period_returns if r > 0) / len(period_returns) * 100,
        }

        criteria, passed = check_criteria(result)
        results[univ_name] = {**result, 'criteria': criteria, 'criteria_passed': int(passed)}

    return results


def run_regime_test(strategy_fn, strategy_name, btc_data, test_symbols_data):
    """Test strategy in different market regimes."""
    btc_close = btc_data['close']
    ret_30d = btc_close.pct_change(720)
    btc_data = btc_data.copy()
    btc_data['regime'] = np.where(ret_30d > 0.10, 'bull',
                          np.where(ret_30d < -0.10, 'bear', 'sideways'))

    regime_results = {}
    for regime in ['bull', 'bear', 'sideways']:
        regime_mask = btc_data['regime'] == regime
        regime_dates = set(btc_data.loc[regime_mask, 'datetime'])

        if len(regime_dates) < 1000:
            continue

        all_trades = []
        for symbol, df in test_symbols_data.items():
            regime_df = df[df['datetime'].isin(regime_dates)].copy().reset_index(drop=True)
            if len(regime_df) < 500:
                continue
            sigs = strategy_fn(regime_df)
            r = simulate(regime_df, sigs, 0.02, 72, 3.0, 8.0, 0.0003)
            all_trades.extend(r['trades'])

        if all_trades:
            wins = sum(1 for t in all_trades if t > 0)
            losses = sum(1 for t in all_trades if t <= 0)
            gp = sum(t for t in all_trades if t > 0)
            gl = abs(sum(t for t in all_trades if t < 0))
            regime_results[regime] = {
                'trades': len(all_trades),
                'wr': wins / (wins + losses) if (wins + losses) > 0 else 0,
                'pf': gp / (gl + 1e-10),
                'net': sum(all_trades),
            }

    return regime_results


def main():
    print("=" * 70)
    print("RALPH-LOOP PHASE 8: BEAR MARKET FIX & STRATEGY IMPROVEMENT")
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

    print(f"Loaded {len(all_data)} symbols with >10000 bars\n")

    # ==========================================================================
    # PART 1: Test all strategy variants with TRUE OOS
    # ==========================================================================
    print("=" * 60)
    print("PART 1: Strategy Variants - True Out-of-Sample")
    print("=" * 60)

    strategies = {
        'base': strat_base,
        'long_only': strat_long_only,
        'regime_filter': strat_regime_filter,
        'regime_v2': strat_regime_v2,
        'dual_ma': strat_dual_ma,
        'volume_confirm': strat_volume_confirm,
        'relaxed_atr': strat_relaxed_atr,
        'momentum_filter': strat_momentum_filter,
    }

    # Also test different exit parameters
    exit_configs = {
        'default': {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0},
        'tight_stop': {'max_bars': 48, 'atr_stop': 2.0, 'profit_target_atr': 6.0},
        'wide_stop': {'max_bars': 96, 'atr_stop': 4.0, 'profit_target_atr': 10.0},
        'no_pt': {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 0},
    }

    all_results = {}
    best_score = 0
    best_config = None

    for strat_name, strat_fn in strategies.items():
        for exit_name, exit_params in exit_configs.items():
            config_name = f"{strat_name}_{exit_name}"
            print(f"\n  Testing: {config_name}...", end=" ", flush=True)

            results = run_true_oos_test(strat_fn, strat_name, all_data, exit_params)
            if results is None:
                print("SKIP")
                continue

            all_results[config_name] = results

            # Print best universe result
            best_univ = max(results.items(), key=lambda x: x[1].get('criteria_passed', 0))
            univ_name, r = best_univ
            passed = r.get('criteria_passed', 0)
            print(f"[{passed}/6] best={univ_name} Sharpe={r['sharpe']:.2f} "
                  f"Ret={r['total_return']*100:+.1f}% DD={r['max_drawdown']*100:.1f}% "
                  f"WR={r['win_rate']*100:.0f}% PF={r['profit_factor']:.2f} T={r['trade_count']}")

            if passed > best_score:
                best_score = passed
                best_config = config_name

    # ==========================================================================
    # PART 2: Regime Analysis for top strategies
    # ==========================================================================
    print(f"\n\n{'=' * 60}")
    print("PART 2: Regime Analysis for Top Strategies")
    print("=" * 60)

    btc = load_ohlcv('BTCUSDT', '1h')
    test_symbols = {}
    for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'ADAUSDT']:
        df = load_ohlcv(s, '1h')
        if not df.empty:
            test_symbols[s] = df

    # Sort strategies by best criteria_passed
    ranked_configs = []
    for config_name, results in all_results.items():
        best_univ = max(results.items(), key=lambda x: x[1].get('criteria_passed', 0))
        ranked_configs.append((config_name, best_univ[1].get('criteria_passed', 0),
                               best_univ[0], best_univ[1]))
    ranked_configs.sort(key=lambda x: (-x[1], -x[3].get('sharpe', 0)))

    print(f"\nTop 10 configs by criteria passed:")
    for i, (config_name, passed, univ, r) in enumerate(ranked_configs[:10]):
        print(f"  {i+1}. [{passed}/6] {config_name} ({univ}) "
              f"Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
              f"WR={r['win_rate']*100:.0f}% PF={r['profit_factor']:.2f} T={r['trade_count']}")
        criteria = r.get('criteria', {})
        fails = [k for k, v in criteria.items() if not v]
        if fails:
            print(f"     FAILS: {', '.join(fails)}")

    # Regime test for top 5
    print(f"\nRegime analysis for top 5:")
    for config_name, passed, univ, r in ranked_configs[:5]:
        strat_name = config_name.rsplit('_', 1)[0]
        # Find the strategy function
        for sn, sf in strategies.items():
            if config_name.startswith(sn):
                regime_results = run_regime_test(sf, sn, btc, test_symbols)
                print(f"\n  {config_name}:")
                for regime, rr in regime_results.items():
                    status = "OK" if rr['net'] > 0 else "BAD"
                    print(f"    {regime}: {rr['trades']} trades  WR={rr['wr']:.0%}  "
                          f"PF={rr['pf']:.2f}  Net={rr['net']*100:+.2f}%  [{status}]")
                break

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print(f"\n\n{'=' * 70}")
    print("PHASE 8 SUMMARY")
    print("=" * 70)

    if best_config:
        best_r = all_results[best_config]
        best_univ = max(best_r.items(), key=lambda x: x[1].get('criteria_passed', 0))
        univ_name, r = best_univ
        criteria, passed = check_criteria(r)

        print(f"\nBest configuration: {best_config}")
        print(f"Universe: {univ_name}")
        print(f"Criteria passed: {passed}/6")
        print(f"  Sharpe: {r['sharpe']:.2f}  {'PASS' if criteria['sharpe_gt_1'] else 'FAIL'}")
        print(f"  Max DD: {r['max_drawdown']*100:.1f}%  {'PASS' if criteria['max_dd_lt_25'] else 'FAIL'}")
        print(f"  Win Rate: {r['win_rate']*100:.0f}%  {'PASS' if criteria['win_rate_gt_45'] else 'FAIL'}")
        print(f"  Profit Factor: {r['profit_factor']:.2f}  {'PASS' if criteria['profit_factor_gt_1_5'] else 'FAIL'}")
        print(f"  WFA Eff: {r['wfa_efficiency']:.0f}%  {'PASS' if criteria['wfa_efficiency_gt_50'] else 'FAIL'}")
        print(f"  Trade Count: {r['trade_count']}  {'PASS' if criteria['trade_count_gt_100'] else 'FAIL'}")

        if passed == 6:
            print(f"\n*** 6/6 PASSED ON TRUE OUT-OF-SAMPLE! ***")
            print(f"Strategy is a candidate for paper trading.")
        else:
            print(f"\nStill {6-passed} criteria failing. More work needed.")
    else:
        print("No valid results found.")

    # Save results
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'total_configs_tested': len(all_results),
        'best_config': best_config,
        'best_score': int(best_score),
        'top_10': [
            {
                'config': c[0],
                'passed': int(c[1]),
                'universe': c[2],
                'sharpe': float(c[3].get('sharpe', 0)),
                'return': float(c[3].get('total_return', 0)),
                'max_dd': float(c[3].get('max_drawdown', 0)),
                'win_rate': float(c[3].get('win_rate', 0)),
                'pf': float(c[3].get('profit_factor', 0)),
                'trades': int(c[3].get('trade_count', 0)),
            }
            for c in ranked_configs[:10]
        ],
    }

    with open(RESULTS_PATH / "phase8_report.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH / 'phase8_report.json'}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
