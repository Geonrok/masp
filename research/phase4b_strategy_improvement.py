#!/usr/bin/env python3
"""
Ralph-Loop Phase 4B: Strategy Improvement
==========================================
Targeted improvements to address Phase 4 failures:
- Win Rate < 45% → Add regime filter, reduce whipsaw
- Sharpe < 1.0 → Better exits, signal quality
- PF < 1.5 → Asymmetric risk/reward, trailing stops
"""
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

DATA_ROOT = Path("E:/data/crypto_ohlcv")
STATE_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/ralph_loop_state.json")
RESULTS_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/results")

SLIPPAGE = 0.0005
COMMISSION = 0.0004
FUNDING_PER_8H = 0.0001

def load_ohlcv(symbol, timeframe="4h"):
    tf_map = {"1h": "binance_futures_1h", "4h": "binance_futures_4h", "1d": "binance_futures_1d"}
    path = DATA_ROOT / tf_map.get(timeframe, "binance_futures_4h") / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ['datetime', 'timestamp', 'date']:
        if col in df.columns:
            df['datetime'] = pd.to_datetime(df[col])
            break
    return df.sort_values('datetime').reset_index(drop=True)


def simulate_with_stops(df, signals, position_pct=0.02, max_bars=48,
                        atr_stop_mult=0.0, trailing_atr_mult=0.0):
    """
    Improved simulation with:
    - ATR-based stop loss
    - Trailing stop
    - Better exit logic
    """
    equity = [1.0]
    capital = 1.0
    position = 0
    entry_price = 0
    bars_held = 0
    trades = []
    highest_since_entry = 0
    lowest_since_entry = float('inf')

    # Pre-calculate ATR
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    tr = np.maximum(high - low,
         np.maximum(abs(high - np.roll(close, 1)),
                    abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(14).mean().values

    for i in range(len(df)):
        c = close[i]
        sig = signals[i] if i < len(signals) else 0
        current_atr = atr[i] if not np.isnan(atr[i]) else c * 0.02

        if position != 0:
            bars_held += 1

            # Track extremes for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, c)
            else:
                lowest_since_entry = min(lowest_since_entry, c)

            should_exit = False
            exit_reason = ''

            # Max hold time
            if bars_held >= max_bars:
                should_exit = True
                exit_reason = 'max_bars'

            # ATR stop loss
            elif atr_stop_mult > 0:
                if position == 1 and c < entry_price - atr_stop_mult * current_atr:
                    should_exit = True
                    exit_reason = 'atr_stop'
                elif position == -1 and c > entry_price + atr_stop_mult * current_atr:
                    should_exit = True
                    exit_reason = 'atr_stop'

            # Trailing stop
            elif trailing_atr_mult > 0:
                if position == 1 and c < highest_since_entry - trailing_atr_mult * current_atr:
                    should_exit = True
                    exit_reason = 'trailing_stop'
                elif position == -1 and c > lowest_since_entry + trailing_atr_mult * current_atr:
                    should_exit = True
                    exit_reason = 'trailing_stop'

            # Signal reversal
            elif sig != 0 and sig != position:
                should_exit = True
                exit_reason = 'reversal'
            elif sig == 0:
                should_exit = True
                exit_reason = 'flat'

            if should_exit:
                exit_price = c * (1 - SLIPPAGE * np.sign(position))
                pnl = position * (exit_price - entry_price) / entry_price * position_pct
                pnl -= COMMISSION * position_pct
                pnl -= FUNDING_PER_8H * bars_held / 2 * position_pct
                trades.append(pnl)
                capital += pnl
                position = 0

        if position == 0 and sig != 0:
            position = int(np.sign(sig))
            entry_price = c * (1 + SLIPPAGE * position)
            capital -= COMMISSION * position_pct
            bars_held = 0
            highest_since_entry = c
            lowest_since_entry = c

        equity.append(capital)

    # Close open position
    if position != 0:
        c = close[-1]
        exit_price = c * (1 - SLIPPAGE * np.sign(position))
        pnl = position * (exit_price - entry_price) / entry_price * position_pct
        pnl -= COMMISSION * position_pct
        pnl -= FUNDING_PER_8H * bars_held / 2 * position_pct
        trades.append(pnl)

    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak

    wins = sum(1 for t in trades if t > 0)
    losses = sum(1 for t in trades if t <= 0)
    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))

    return {
        'total_return': capital - 1,
        'max_drawdown': dd.min(),
        'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
        'profit_factor': gross_profit / (gross_loss + 1e-10),
        'trade_count': len(trades),
        'trades': trades,
    }


# =============================================================================
# Improvement 1: TSMOM + Regime Filter
# =============================================================================
def strategy_tsmom_regime(df, lookback=84):
    """TSMOM with regime filter: only trade when volatility is moderate"""
    close = df['close']
    ret = close.pct_change(lookback)

    # Regime: volatility filter
    vol = close.pct_change().rolling(lookback).std()
    vol_median = vol.rolling(lookback * 3).median()

    # Only trade when vol is 0.5x-2x median (avoid low-vol chop and extreme vol)
    vol_ok = (vol > vol_median * 0.5) & (vol < vol_median * 2.0)

    # Trend strength filter: require strong enough trend
    abs_ret = abs(ret)
    ret_threshold = abs_ret.rolling(lookback).quantile(0.3)  # Top 70% of moves

    signals = np.where(vol_ok & (abs_ret > ret_threshold), np.sign(ret), 0)
    return pd.Series(signals, index=df.index).fillna(0).astype(int)


# =============================================================================
# Improvement 2: TSMOM + Confirmation (multi-timeframe)
# =============================================================================
def strategy_tsmom_confirmed(df, short_lb=42, long_lb=84):
    """TSMOM with dual-timeframe confirmation: both must agree"""
    close = df['close']
    ret_short = close.pct_change(short_lb)
    ret_long = close.pct_change(long_lb)

    # Both timeframes must agree
    signal_short = np.sign(ret_short)
    signal_long = np.sign(ret_long)

    # Only enter when both agree
    signals = np.where(signal_short == signal_long, signal_long, 0)
    return pd.Series(signals, index=df.index).fillna(0).astype(int)


# =============================================================================
# Improvement 3: Cross-Sectional Momentum (XSMOM)
# =============================================================================
def strategy_xsmom(all_data, lookback=84, top_n=3):
    """
    Cross-sectional momentum: long top N, short bottom N by past returns.
    Returns signals per symbol per bar.
    """
    # Align all closes
    closes = {}
    min_len = float('inf')
    for symbol, df in all_data.items():
        closes[symbol] = df['close'].values
        min_len = min(min_len, len(df))

    min_len = int(min_len)
    for symbol in closes:
        closes[symbol] = closes[symbol][:min_len]

    signals = {s: np.zeros(min_len) for s in closes}

    for i in range(lookback, min_len):
        # Rank symbols by past return
        rets = {}
        for symbol, c in closes.items():
            if c[i - lookback] > 0:
                rets[symbol] = c[i] / c[i - lookback] - 1
            else:
                rets[symbol] = 0

        ranked = sorted(rets.items(), key=lambda x: x[1], reverse=True)

        # Long top N
        for symbol, _ in ranked[:top_n]:
            signals[symbol][i] = 1
        # Short bottom N
        for symbol, _ in ranked[-top_n:]:
            signals[symbol][i] = -1

    return signals


# =============================================================================
# Improvement 4: TSMOM + Multi-Factor Combined
# =============================================================================
def strategy_tsmom_multifactor(df, lookback=84):
    """Combine TSMOM with multi-factor scoring for higher conviction"""
    close = df['close']
    high = df['high']
    low = df['low']

    # TSMOM signal
    ret = close.pct_change(lookback)
    tsmom = np.sign(ret)

    # Factor 1: EMA trend
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=50, adjust=False).mean()
    trend = np.where(ema_fast > ema_slow, 1, -1)

    # Factor 2: Donchian breakout
    upper = high.rolling(24).max().shift(1)
    lower = low.rolling(24).min().shift(1)
    breakout = np.where(close > upper, 1, np.where(close < lower, -1, 0))

    # Factor 3: Volume confirmation
    vol = df['volume']
    vol_sma = vol.rolling(24).mean()
    vol_confirm = np.where(vol > vol_sma * 1.2, 1, 0)  # Above avg volume

    # Combined score: TSMOM + at least 1 confirming factor
    score = tsmom.values + trend + breakout
    # Require TSMOM direction + at least 1 confirmation
    signals = np.where((score >= 2) & (tsmom > 0), 1,
              np.where((score <= -2) & (tsmom < 0), -1, 0))

    # Volume filter: only enter on above-avg volume
    signals = np.where(vol_confirm, signals, 0)

    return pd.Series(signals, index=df.index).fillna(0).astype(int)


# =============================================================================
# Improvement 5: Adaptive TSMOM (switch lookback based on regime)
# =============================================================================
def strategy_adaptive_tsmom(df):
    """Adaptively choose lookback based on market regime"""
    close = df['close']

    # Multiple lookbacks
    ret_42 = close.pct_change(42)
    ret_84 = close.pct_change(84)
    ret_168 = close.pct_change(168)

    # Regime: use short lookback in high vol, long lookback in low vol
    vol = close.pct_change().rolling(42).std()
    vol_median = vol.rolling(168).median()

    signals = np.zeros(len(df))

    for i in range(168, len(df)):
        if pd.isna(vol.iloc[i]) or pd.isna(vol_median.iloc[i]):
            continue

        if vol.iloc[i] > vol_median.iloc[i] * 1.5:
            # High vol → short lookback (adapt quickly)
            sig = np.sign(ret_42.iloc[i])
        elif vol.iloc[i] < vol_median.iloc[i] * 0.7:
            # Low vol → long lookback (patient)
            sig = np.sign(ret_168.iloc[i])
        else:
            # Normal → medium lookback
            sig = np.sign(ret_84.iloc[i])

        signals[i] = sig

    return pd.Series(signals, index=df.index).fillna(0).astype(int)


# =============================================================================
# Walk-Forward Portfolio Backtest
# =============================================================================
def portfolio_walk_forward(symbols, signal_func_name, train_bars=1080, test_bars=180,
                           target_vol=0.10, max_positions=5,
                           atr_stop=0.0, trailing_stop=0.0):
    """Portfolio-level walk-forward with improvements"""
    all_data = {}
    for symbol in symbols:
        df = load_ohlcv(symbol, "4h")
        if not df.empty and len(df) > train_bars + test_bars:
            all_data[symbol] = df

    if not all_data:
        return {}

    min_len = min(len(df) for df in all_data.values())
    equity = [1.0]
    period_returns = []
    all_trades = []

    i = train_bars
    while i + test_bars <= min_len:
        period_pnl = 0

        if signal_func_name == 'xsmom':
            # Cross-sectional: compute rankings across all symbols
            train_data = {s: df.iloc[:i] for s, df in all_data.items()}
            xsmom_signals = strategy_xsmom(train_data, lookback=84, top_n=min(3, len(symbols)//2))

            for symbol, df in all_data.items():
                test = df.iloc[i:i+test_bars].copy().reset_index(drop=True)
                sig_array = xsmom_signals[symbol][i:i+test_bars]

                # Volatility-based sizing
                train_ret = df['close'].iloc[:i].pct_change()
                vol = train_ret.rolling(84).std().iloc[-1]
                if np.isnan(vol) or vol == 0:
                    vol = 0.02
                position_pct = min(target_vol / (vol * np.sqrt(6*365) + 1e-10) / len(symbols), 0.05)

                r = simulate_with_stops(test, sig_array, position_pct, 48, atr_stop, trailing_stop)
                period_pnl += r['total_return']
                all_trades.extend(r['trades'])

        else:
            # Per-symbol signal generation
            scored = []
            for symbol, df in all_data.items():
                train = df.iloc[:i]

                if signal_func_name == 'tsmom_regime':
                    sigs = strategy_tsmom_regime(train)
                elif signal_func_name == 'tsmom_confirmed':
                    sigs = strategy_tsmom_confirmed(train)
                elif signal_func_name == 'tsmom_multifactor':
                    sigs = strategy_tsmom_multifactor(train)
                elif signal_func_name == 'adaptive_tsmom':
                    sigs = strategy_adaptive_tsmom(train)
                else:
                    continue

                last_signal = sigs.iloc[-1] if len(sigs) > 0 else 0

                train_ret = df['close'].iloc[:i].pct_change()
                vol = train_ret.rolling(84).std().iloc[-1]
                if np.isnan(vol) or vol == 0:
                    vol = 0.02

                scored.append((symbol, last_signal, vol))

            # Select positions
            active = [(s, sig, v) for s, sig, v in scored if sig != 0]
            active.sort(key=lambda x: 1/x[2])  # Sort by inverse vol
            selected = active[:max_positions]

            for symbol, signal, vol in selected:
                df = all_data[symbol]
                # Generate full signals for test period
                full = df.iloc[:i+test_bars]
                if signal_func_name == 'tsmom_regime':
                    full_sigs = strategy_tsmom_regime(full)
                elif signal_func_name == 'tsmom_confirmed':
                    full_sigs = strategy_tsmom_confirmed(full)
                elif signal_func_name == 'tsmom_multifactor':
                    full_sigs = strategy_tsmom_multifactor(full)
                elif signal_func_name == 'adaptive_tsmom':
                    full_sigs = strategy_adaptive_tsmom(full)
                else:
                    continue

                test = df.iloc[i:i+test_bars].copy().reset_index(drop=True)
                test_sigs = full_sigs.iloc[i:i+test_bars].values

                position_pct = min(target_vol / (vol * np.sqrt(6*365) + 1e-10) / max(len(selected), 1), 0.05)

                r = simulate_with_stops(test, test_sigs, position_pct, 48, atr_stop, trailing_stop)
                period_pnl += r['total_return']
                all_trades.extend(r['trades'])

        period_returns.append(period_pnl)
        equity.append(equity[-1] * (1 + period_pnl))
        i += test_bars

    if not period_returns:
        return {}

    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak

    wins = sum(1 for t in all_trades if t > 0)
    losses = sum(1 for t in all_trades if t <= 0)
    gross_profit = sum(t for t in all_trades if t > 0)
    gross_loss = abs(sum(t for t in all_trades if t < 0))

    sharpe = np.mean(period_returns) / (np.std(period_returns) + 1e-10) * np.sqrt(12) if len(period_returns) > 1 else 0

    return {
        'total_return': equity[-1] - 1,
        'max_drawdown': dd.min(),
        'sharpe': sharpe,
        'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
        'profit_factor': gross_profit / (gross_loss + 1e-10),
        'trade_count': len(all_trades),
        'periods': len(period_returns),
        'wfa_efficiency': sum(1 for r in period_returns if r > 0) / len(period_returns) * 100,
        'period_returns': period_returns,
    }


def check_criteria(result):
    criteria = {
        'sharpe_gt_1': result.get('sharpe', 0) > 1.0,
        'max_dd_lt_25': result.get('max_drawdown', -1) > -0.25,
        'win_rate_gt_45': result.get('win_rate', 0) > 0.45,
        'profit_factor_gt_1_5': result.get('profit_factor', 0) > 1.5,
        'wfa_efficiency_gt_50': result.get('wfa_efficiency', 0) > 50,
        'trade_count_gt_100': result.get('trade_count', 0) > 100,
    }
    return criteria, sum(v for v in criteria.values())


def main():
    print("=" * 70)
    print("RALPH-LOOP PHASE 4B: STRATEGY IMPROVEMENT")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    symbols_10 = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
                   'BNBUSDT', 'ADAUSDT', 'LINKUSDT', 'AVAXUSDT', 'LTCUSDT']

    # Test matrix: strategy × stop config
    strategies = {
        'tsmom_regime': 'TSMOM + Regime Filter',
        'tsmom_confirmed': 'TSMOM + Dual Timeframe',
        'tsmom_multifactor': 'TSMOM + Multi-Factor',
        'adaptive_tsmom': 'Adaptive TSMOM',
        'xsmom': 'Cross-Sectional Momentum',
    }

    stop_configs = {
        'no_stop': {'atr_stop': 0, 'trailing_stop': 0},
        'atr_2x': {'atr_stop': 2.0, 'trailing_stop': 0},
        'atr_3x': {'atr_stop': 3.0, 'trailing_stop': 0},
        'trail_2x': {'atr_stop': 0, 'trailing_stop': 2.0},
        'trail_3x': {'atr_stop': 0, 'trailing_stop': 3.0},
    }

    all_results = {}

    for strat_key, strat_name in strategies.items():
        for stop_key, stop_cfg in stop_configs.items():
            key = f"{strat_key}_{stop_key}"
            print(f"Testing: {strat_name} + {stop_key}...", end=" ", flush=True)

            result = portfolio_walk_forward(
                symbols_10, strat_key,
                target_vol=0.10, max_positions=5,
                **stop_cfg
            )

            if not result:
                print("SKIP")
                continue

            criteria, passed = check_criteria(result)
            result['criteria'] = criteria
            result['criteria_passed'] = passed
            all_results[key] = result

            print(f"Sharpe={result['sharpe']:.2f}  Ret={result['total_return']*100:+.1f}%  "
                  f"DD={result['max_drawdown']*100:.1f}%  WR={result['win_rate']*100:.0f}%  "
                  f"PF={result['profit_factor']:.2f}  WFA={result['wfa_efficiency']:.0f}%  "
                  f"Criteria={passed}/6")

    # Also test aggressive vol target
    print(f"\n--- Aggressive vol target (15%) ---")
    for strat_key in ['tsmom_regime', 'tsmom_multifactor', 'adaptive_tsmom', 'xsmom']:
        for stop_key in ['no_stop', 'trail_3x']:
            stop_cfg = stop_configs[stop_key]
            key = f"{strat_key}_{stop_key}_aggr"
            strat_name = strategies[strat_key]
            print(f"Testing: {strat_name} + {stop_key} (15% vol)...", end=" ", flush=True)

            result = portfolio_walk_forward(
                symbols_10, strat_key,
                target_vol=0.15, max_positions=5,
                **stop_cfg
            )

            if not result:
                print("SKIP")
                continue

            criteria, passed = check_criteria(result)
            result['criteria'] = criteria
            result['criteria_passed'] = passed
            all_results[key] = result

            print(f"Sharpe={result['sharpe']:.2f}  Ret={result['total_return']*100:+.1f}%  "
                  f"DD={result['max_drawdown']*100:.1f}%  WR={result['win_rate']*100:.0f}%  "
                  f"PF={result['profit_factor']:.2f}  WFA={result['wfa_efficiency']:.0f}%  "
                  f"Criteria={passed}/6")

    # Final ranking
    print(f"\n{'='*70}")
    print("IMPROVEMENT RANKING (by criteria passed, then Sharpe)")
    print(f"{'='*70}")

    ranked = sorted(all_results.items(),
                    key=lambda x: (x[1]['criteria_passed'], x[1]['sharpe']),
                    reverse=True)

    for rank, (key, res) in enumerate(ranked[:15], 1):
        marker = "***" if res['criteria_passed'] >= 5 else ("**" if res['criteria_passed'] >= 4 else "")
        print(f"  {rank:2d}. {key:<40} {res['criteria_passed']}/6  "
              f"Sharpe={res['sharpe']:.2f}  Ret={res['total_return']*100:+.1f}%  "
              f"DD={res['max_drawdown']*100:.1f}%  WR={res['win_rate']*100:.0f}%  "
              f"PF={res['profit_factor']:.2f}  {marker}")

    # Detail on best
    if ranked:
        best_key, best = ranked[0]
        print(f"\n{'='*70}")
        print(f"BEST: {best_key}")
        print(f"{'='*70}")
        for c, v in best['criteria'].items():
            print(f"  {c}: {'PASS' if v else 'FAIL'}")

        # Compare vs Phase 4 baseline
        print(f"\n  vs Phase 4 baseline (top_10_aggressive: Sharpe=0.64, 4/6):")
        print(f"  Sharpe: 0.64 → {best['sharpe']:.2f} ({'improved' if best['sharpe'] > 0.64 else 'worse'})")
        print(f"  Criteria: 4/6 → {best['criteria_passed']}/6 ({'improved' if best['criteria_passed'] > 4 else 'same or worse'})")

    # Save report
    report = {
        'generated_at': datetime.now().isoformat(),
        'configs_tested': len(all_results),
        'ranking': [{'rank': i+1, 'key': k,
                     'criteria_passed': int(v['criteria_passed']),
                     'sharpe': float(v['sharpe']),
                     'total_return': float(v['total_return']),
                     'max_drawdown': float(v['max_drawdown']),
                     'win_rate': float(v['win_rate']),
                     'profit_factor': float(v['profit_factor']),
                     'wfa_efficiency': float(v['wfa_efficiency']),
                     'trade_count': int(v['trade_count']),
                     }
                    for i, (k, v) in enumerate(ranked)],
        'best': best_key if ranked else None,
    }

    report_path = RESULTS_PATH / "phase4b_improvement_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"\n  Report saved: {report_path}")

    return report


if __name__ == "__main__":
    report = main()
