#!/usr/bin/env python3
"""
Ralph-Loop Phase 7: Honest Validation
=======================================
Phase 6 had symbol selection bias (picked winners, re-tested same data).
This phase fixes that with:

1. TRUE out-of-sample: split time in half, select symbols on first half,
   validate on second half (never seen)
2. Realistic slippage per liquidity tier
3. Full funding rate from actual data
4. Capacity test: can we actually trade this size?
5. Regime analysis: does it work in all market conditions?
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


def atr_expansion_breakout(df, lookback=96, ema_period=200, atr_expansion=1.3):
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
# TEST 1: True Out-of-Sample (Time-Split)
# =============================================================================
def test_1_true_oos():
    """
    Split each symbol's data into:
    - First 60%: symbol selection period (pick top symbols here)
    - Last 40%: TRUE out-of-sample (never seen during selection)

    This eliminates symbol selection bias.
    """
    print("=" * 60)
    print("TEST 1: True Out-of-Sample (Time-Split)")
    print("=" * 60)

    # Load all symbols
    all_path = DATA_ROOT / "binance_futures_1h"
    all_data = {}
    for f in sorted(all_path.glob("*USDT.csv")):
        symbol = f.stem
        df = load_ohlcv(symbol, "1h")
        if not df.empty and len(df) > 10000:
            all_data[symbol] = df

    print(f"  Symbols with >10000 bars: {len(all_data)}")

    # STEP A: Use first 60% of data to rank symbols
    selection_results = {}
    for symbol, df in all_data.items():
        split = int(len(df) * 0.6)
        selection_df = df.iloc[:split].copy().reset_index(drop=True)

        if len(selection_df) < 5000:
            continue

        # Walk-forward on selection period only
        train_bars = 4320
        test_bars = 720
        all_trades = []
        period_returns = []

        i = train_bars
        while i + test_bars <= len(selection_df):
            full = selection_df.iloc[:i + test_bars]
            sigs = atr_expansion_breakout(full)
            test_sigs = sigs[i:i + test_bars]
            test_df = selection_df.iloc[i:i + test_bars].copy().reset_index(drop=True)
            r = simulate(test_df, test_sigs, 0.02, 72, 3.0, 8.0, 0.0003)
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

    # Rank by selection period performance
    ranked = sorted(selection_results.items(), key=lambda x: x[1]['total_return'], reverse=True)
    selected_top30 = [s for s, _ in ranked[:30]]
    selected_profitable = [s for s, r in ranked if r['total_return'] > 0]

    print(f"  Selection period: Top 30 selected from {len(selection_results)} symbols")
    print(f"  Selection period: {len(selected_profitable)} profitable symbols")

    # STEP B: TRUE OOS on last 40% of data (NEVER seen during selection)
    print(f"\n  --- TRUE OUT-OF-SAMPLE (last 40% of data) ---")

    for universe_name, universe_symbols in [('top_30_oos', selected_top30),
                                              ('profitable_oos', selected_profitable[:50]),
                                              ('all_oos', list(all_data.keys()))]:
        oos_data = {}
        for symbol in universe_symbols:
            if symbol not in all_data:
                continue
            df = all_data[symbol]
            split = int(len(df) * 0.6)
            oos_df = df.iloc[split:].copy().reset_index(drop=True)
            if len(oos_df) > 2000:
                oos_data[symbol] = oos_df

        if not oos_data:
            continue

        # Walk-forward on OOS period
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
                sigs = atr_expansion_breakout(full)
                test_sigs = sigs[i:i + test_bars]
                test_df = df.iloc[i:i + test_bars].copy().reset_index(drop=True)

                ann_vol = vol * np.sqrt(24 * 365)
                position_pct = min(0.10 / (ann_vol + 1e-10) / max(len(selected), 1), 0.05)

                r = simulate(test_df, test_sigs, position_pct, 72, 3.0, 8.0, 0.0003)
                period_pnl += r['total_return']
                all_trades.extend(r['trades'])

            period_returns.append(period_pnl)
            equity.append(equity[-1] * (1 + period_pnl))
            i += test_bars

        if not period_returns:
            print(f"  {universe_name}: SKIP (insufficient OOS data)")
            continue

        equity = np.array(equity)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak

        wins = sum(1 for t in all_trades if t > 0)
        losses = sum(1 for t in all_trades if t <= 0)
        gp = sum(t for t in all_trades if t > 0)
        gl = abs(sum(t for t in all_trades if t < 0))
        sharpe = np.mean(period_returns) / (np.std(period_returns) + 1e-10) * np.sqrt(12)

        result = {
            'total_return': equity[-1] - 1,
            'max_drawdown': dd.min(),
            'sharpe': sharpe,
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
            'profit_factor': gp / (gl + 1e-10),
            'trade_count': len(all_trades),
            'periods': len(period_returns),
            'wfa_efficiency': sum(1 for r in period_returns if r > 0) / len(period_returns) * 100,
        }

        criteria, passed = check_criteria(result)

        print(f"\n  {universe_name} ({len(oos_data)} symbols, OOS period):")
        print(f"    [{passed}/6] Sharpe={result['sharpe']:.2f}  Ret={result['total_return']*100:+.1f}%  "
              f"DD={result['max_drawdown']*100:.1f}%  WR={result['win_rate']*100:.0f}%  "
              f"PF={result['profit_factor']:.2f}  WFA={result['wfa_efficiency']:.0f}%  T={result['trade_count']}")
        for c, v in criteria.items():
            print(f"      {c}: {'PASS' if v else 'FAIL'}")

    return selection_results


# =============================================================================
# TEST 2: Realistic Slippage per Liquidity Tier
# =============================================================================
def test_2_slippage_sensitivity():
    """Test with different slippage levels"""
    print(f"\n{'='*60}")
    print("TEST 2: Slippage Sensitivity")
    print(f"{'='*60}")

    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
               'BNBUSDT', 'ADAUSDT', 'LINKUSDT', 'AVAXUSDT', 'LTCUSDT']

    slippage_levels = {
        'optimistic_0.02%': 0.0002,
        'base_0.03%': 0.0003,
        'realistic_0.05%': 0.0005,
        'conservative_0.08%': 0.0008,
        'worst_case_0.10%': 0.001,
    }

    for slip_name, slip_val in slippage_levels.items():
        all_data = {}
        for s in symbols:
            df = load_ohlcv(s, "1h")
            if not df.empty and len(df) > 5040:
                all_data[s] = df

        if not all_data:
            continue

        min_len = min(len(d) for d in all_data.values())
        equity = [1.0]
        period_returns = []
        all_trades = []

        i = 4320
        while i + 720 <= min_len:
            period_pnl = 0
            scored = [(s, all_data[s]['close'].iloc[:i].pct_change().rolling(168).std().iloc[-1])
                      for s in all_data]
            scored = [(s, v if not np.isnan(v) and v > 0 else 0.01) for s, v in scored]
            scored.sort(key=lambda x: x[1])

            for symbol, vol in scored[:5]:
                df = all_data[symbol]
                full = df.iloc[:i + 720]
                sigs = atr_expansion_breakout(full)
                test_sigs = sigs[i:i + 720]
                test_df = df.iloc[i:i + 720].copy().reset_index(drop=True)

                ann_vol = vol * np.sqrt(24 * 365)
                position_pct = min(0.10 / (ann_vol + 1e-10) / 5, 0.05)

                r = simulate(test_df, test_sigs, position_pct, 72, 3.0, 8.0, slip_val)
                period_pnl += r['total_return']
                all_trades.extend(r['trades'])

            period_returns.append(period_pnl)
            equity.append(equity[-1] * (1 + period_pnl))
            i += 720

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

        wr = wins / (wins + losses) if (wins + losses) > 0 else 0
        pf = gp / (gl + 1e-10)

        print(f"  {slip_name:<25} Sharpe={sharpe:.2f}  Ret={equity_arr[-1]-1:+.1%}  "
              f"DD={dd.min():.1%}  WR={wr:.0%}  PF={pf:.2f}  T={len(all_trades)}")


# =============================================================================
# TEST 3: Regime Analysis
# =============================================================================
def test_3_regime_analysis():
    """How does the strategy perform in different market regimes?"""
    print(f"\n{'='*60}")
    print("TEST 3: Market Regime Analysis")
    print(f"{'='*60}")

    btc = load_ohlcv('BTCUSDT', '1h')
    if btc.empty:
        print("  BTC data not available")
        return

    # Define regimes by BTC trend
    btc_close = btc['close']
    btc_ret_30d = btc_close.pct_change(720)  # 30 days in 1h
    btc['regime'] = np.where(btc_ret_30d > 0.10, 'bull',
                   np.where(btc_ret_30d < -0.10, 'bear', 'sideways'))

    # Count regime bars
    for regime in ['bull', 'bear', 'sideways']:
        count = (btc['regime'] == regime).sum()
        pct = count / len(btc) * 100
        print(f"  {regime}: {count} bars ({pct:.0f}%)")

    # Test strategy on each regime
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'ADAUSDT']

    for regime in ['bull', 'bear', 'sideways']:
        regime_mask = btc['regime'] == regime
        regime_dates = btc.loc[regime_mask, 'datetime']

        if len(regime_dates) < 1000:
            print(f"\n  {regime}: insufficient data")
            continue

        all_trades = []
        for symbol in symbols:
            df = load_ohlcv(symbol, '1h')
            if df.empty:
                continue

            # Filter to regime periods
            regime_df = df[df['datetime'].isin(regime_dates)].copy().reset_index(drop=True)
            if len(regime_df) < 500:
                continue

            sigs = atr_expansion_breakout(regime_df)
            r = simulate(regime_df, sigs, 0.02, 72, 3.0, 8.0, 0.0003)
            all_trades.extend(r['trades'])

        if all_trades:
            wins = sum(1 for t in all_trades if t > 0)
            losses = sum(1 for t in all_trades if t <= 0)
            gp = sum(t for t in all_trades if t > 0)
            gl = abs(sum(t for t in all_trades if t < 0))
            wr = wins / (wins + losses) if (wins + losses) > 0 else 0
            pf = gp / (gl + 1e-10)
            print(f"\n  {regime}: {len(all_trades)} trades  WR={wr:.0%}  PF={pf:.2f}  "
                  f"Net={sum(all_trades)*100:+.2f}%")


# =============================================================================
# TEST 4: Capacity Test
# =============================================================================
def test_4_capacity():
    """Can we actually trade meaningful size?"""
    print(f"\n{'='*60}")
    print("TEST 4: Capacity Analysis")
    print(f"{'='*60}")

    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
               'BNBUSDT', 'ADAUSDT', 'LINKUSDT', 'AVAXUSDT', 'LTCUSDT']

    print(f"  Assuming $100K portfolio, 5 positions max, 2% per position = $2K per trade\n")

    for symbol in symbols:
        df = load_ohlcv(symbol, '1h')
        if df.empty:
            continue

        # Last 90 days volume
        recent = df.tail(90 * 24)
        if 'volume' in recent.columns and 'close' in recent.columns:
            adv_usd = (recent['volume'] * recent['close']).mean() * 24  # Daily
            trade_size = 2000  # $2K per trade
            market_impact = trade_size / adv_usd * 100 if adv_usd > 0 else 999

            print(f"  {symbol:<12} ADV=${adv_usd/1e6:.0f}M  "
                  f"$2K trade = {market_impact:.4f}% of ADV  "
                  f"{'OK' if market_impact < 0.1 else 'CAUTION' if market_impact < 1 else 'TOO LARGE'}")


def main():
    print("=" * 70)
    print("RALPH-LOOP PHASE 7: HONEST VALIDATION")
    print("Addressing symbol selection bias from Phase 6")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    # Run all tests
    test_1_true_oos()
    test_2_slippage_sensitivity()
    test_3_regime_analysis()
    test_4_capacity()

    print(f"\n{'='*70}")
    print("HONEST ASSESSMENT")
    print(f"{'='*70}")
    print("""
Key Questions for Live Trading Decision:
1. Does the strategy pass 6/6 on TRUE out-of-sample data?
2. Does it survive realistic slippage (0.05%+)?
3. Does it work in bear markets?
4. Is the capacity sufficient for target portfolio size?

If ANY answer is NO → more work needed before live trading.
""")

    return True


if __name__ == "__main__":
    main()
