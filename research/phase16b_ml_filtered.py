#!/usr/bin/env python3
"""
Phase 16b: ML-Filtered Vol Profile + Alternative Labels
Memory-efficient version: process one symbol at a time, no .copy()
"""
import json, gc
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

DATA_ROOT = Path("E:/data/crypto_ohlcv")
RESULTS_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/results")

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


def build_features_array(df):
    """Build features as numpy array to save memory."""
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    vol = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.ones(len(df))
    n = len(df)

    features = []
    names = []

    # Returns
    for p in [1, 4, 12, 24, 48]:
        ret = np.full(n, np.nan)
        ret[p:] = close[p:] / close[:-p] - 1
        features.append(ret)
        names.append(f'ret_{p}')

    # MA ratios
    cs = pd.Series(close)
    for p in [20, 50, 200]:
        ma = cs.rolling(p).mean().values
        features.append(close / (ma + 1e-10) - 1)
        names.append(f'ma_{p}')

    # EMA trend
    ema_f = cs.ewm(span=50, adjust=False).mean().values
    ema_s = cs.ewm(span=200, adjust=False).mean().values
    features.append(ema_f / (ema_s + 1e-10) - 1)
    names.append('ema_ratio')

    # Donchian position
    hs = pd.Series(high)
    ls = pd.Series(low)
    h48 = hs.rolling(48).max().values
    l48 = ls.rolling(48).min().values
    features.append((close - l48) / (h48 - l48 + 1e-10))
    names.append('donchian_pos')

    # ATR ratio
    atr = calc_atr(high, low, close, 14)
    features.append(atr / (close + 1e-10))
    names.append('atr_ratio')
    atr_avg = pd.Series(atr).rolling(48).mean().values
    features.append(atr / (atr_avg + 1e-10))
    names.append('atr_exp')

    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    features.append(100 - 100 / (1 + rs))
    names.append('rsi')

    # Bollinger %B
    ma20 = cs.rolling(20).mean().values
    std20 = cs.rolling(20).std().values
    features.append((close - (ma20 - 2*std20)) / (4*std20 + 1e-10))
    names.append('bb_pctb')

    # Volume z-score
    vs = pd.Series(vol)
    vol_ma = vs.rolling(48).mean().values
    vol_std = vs.rolling(48).std().values
    features.append((vol - vol_ma) / (vol_std + 1e-10))
    names.append('vol_z')

    # VWAP ratio
    vwap = pd.Series(close * vol).rolling(48).sum().values / (pd.Series(vol).rolling(48).sum().values + 1e-10)
    features.append(close / (vwap + 1e-10) - 1)
    names.append('vwap_ratio')

    # Realized vol ratio
    rets = pd.Series(close).pct_change()
    rv24 = rets.rolling(24).std().values
    rv168 = rets.rolling(168).std().values
    features.append(rv24 / (rv168 + 1e-10))
    names.append('vol_ratio')

    X = np.column_stack(features)
    return X, names


def ml_filtered_signals(df, model_type='xgb', train_bars=4320, test_bars=720,
                          forward_bars=24, threshold=0.005, prob_threshold=0.50):
    """Vol Profile + ML filter. Memory efficient."""
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    vol = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(1.0, index=df.index)

    # Rule-based signals
    vwap = (close * vol).rolling(48).sum() / (vol.rolling(48).sum() + 1e-10)
    upper = high.rolling(48).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    rule_signals = ((close > upper) & (close > vwap * 1.01) & (ema_f > ema_s)).values

    # Build features
    X, feat_names = build_features_array(df)
    n = len(df)

    # Labels
    fwd_ret = np.full(n, np.nan)
    fwd_ret[:n-forward_bars] = close.values[forward_bars:] / close.values[:n-forward_bars] - 1
    labels = (fwd_ret > threshold).astype(int)

    signals = np.zeros(n)

    i = train_bars
    while i + test_bars <= n:
        # Valid training indices (no NaN in features or labels)
        train_mask = np.all(np.isfinite(X[:i]), axis=1) & np.isfinite(fwd_ret[:i])
        train_mask[:200] = False  # skip warmup
        train_mask[max(0,i-forward_bars):i] = False  # no future leak
        train_idx = np.where(train_mask)[0]

        if len(train_idx) < 500:
            # Fallback to rule-based
            for j in range(i, min(i+test_bars, n)):
                if rule_signals[j]:
                    signals[j] = 1
            i += test_bars
            continue

        X_train = X[train_idx]
        y_train = labels[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        pos = y_train.sum()
        neg = len(y_train) - pos
        if pos == 0 or neg == 0:
            i += test_bars
            continue
        spw = neg / pos

        try:
            if model_type == 'xgb' and HAS_XGB:
                model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=spw, random_state=42, verbosity=0,
                    use_label_encoder=False, eval_metric='logloss')
            elif model_type == 'lgb' and HAS_LGB:
                model = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=spw, random_state=42, verbosity=-1)
            else:
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=6,
                    class_weight='balanced', random_state=42, n_jobs=-1)

            model.fit(X_train_s, y_train)

            # Test
            test_end = min(i + test_bars, n)
            test_mask = np.all(np.isfinite(X[i:test_end]), axis=1)
            test_idx = np.where(test_mask)[0] + i

            if len(test_idx) > 0:
                X_test = X[test_idx]
                X_test_s = scaler.transform(X_test)
                probs = model.predict_proba(X_test_s)[:, 1]

                for idx, prob in zip(test_idx, probs):
                    if rule_signals[idx] and prob > prob_threshold:
                        signals[idx] = 1

            del model, X_train_s
            gc.collect()

        except Exception:
            for j in range(i, min(i+test_bars, n)):
                if rule_signals[j]:
                    signals[j] = 1

        i += test_bars

    return signals


def pure_ml_signals(df, model_type='xgb', train_bars=4320, test_bars=720,
                      forward_bars=24, threshold=0.005, prob_threshold=0.55):
    """Pure ML signals (no rule-based filter)."""
    close = df['close'].astype(float)
    X, _ = build_features_array(df)
    n = len(df)

    fwd_ret = np.full(n, np.nan)
    fwd_ret[:n-forward_bars] = close.values[forward_bars:] / close.values[:n-forward_bars] - 1
    labels = (fwd_ret > threshold).astype(int)

    signals = np.zeros(n)

    i = train_bars
    while i + test_bars <= n:
        train_mask = np.all(np.isfinite(X[:i]), axis=1) & np.isfinite(fwd_ret[:i])
        train_mask[:200] = False
        train_mask[max(0,i-forward_bars):i] = False
        train_idx = np.where(train_mask)[0]

        if len(train_idx) < 500:
            i += test_bars
            continue

        X_train = X[train_idx]
        y_train = labels[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        pos = y_train.sum()
        neg = len(y_train) - pos
        if pos == 0 or neg == 0:
            i += test_bars
            continue
        spw = neg / pos

        try:
            if model_type == 'xgb' and HAS_XGB:
                model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=spw, random_state=42, verbosity=0,
                    use_label_encoder=False, eval_metric='logloss')
            elif model_type == 'lgb' and HAS_LGB:
                model = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=spw, random_state=42, verbosity=-1)
            else:
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=6,
                    class_weight='balanced', random_state=42, n_jobs=-1)

            model.fit(X_train_s, y_train)

            test_end = min(i + test_bars, n)
            test_mask = np.all(np.isfinite(X[i:test_end]), axis=1)
            test_idx = np.where(test_mask)[0] + i

            if len(test_idx) > 0:
                X_test_s = scaler.transform(X[test_idx])
                probs = model.predict_proba(X_test_s)[:, 1]
                for idx, prob in zip(test_idx, probs):
                    if prob > prob_threshold:
                        signals[idx] = 1

            del model, X_train_s
            gc.collect()
        except:
            pass

        i += test_bars

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


def run_portfolio(all_data, signal_func, signal_kwargs, max_positions=10,
                   test_bars=720, position_scale=5.0):
    exit_params = {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0}

    oos_data = {}
    for symbol in all_data:
        df = all_data[symbol]
        split = int(len(df) * 0.6)
        oos_df = df.iloc[split:].reset_index(drop=True)
        if len(oos_df) > 2000:
            oos_data[symbol] = oos_df

    if not oos_data:
        return None

    # Compute signals per symbol
    print(f"    Signals for {len(oos_data)} symbols...", end=" ", flush=True)
    symbol_signals = {}
    for idx, (symbol, df) in enumerate(oos_data.items()):
        tb = min(4320, len(df) // 3)
        sigs = signal_func(df, train_bars=tb, test_bars=test_bars, **signal_kwargs)
        symbol_signals[symbol] = sigs
        if (idx + 1) % 50 == 0:
            print(f"{idx+1}", end=" ", flush=True)
        gc.collect()
    print("done")

    # Portfolio sim
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
            test_sigs = symbol_signals[symbol][i:i + test_bars]
            test_df = df.iloc[i:i + test_bars].reset_index(drop=True)
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

    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    wins = sum(1 for t in all_trades if t > 0)
    losses = sum(1 for t in all_trades if t <= 0)
    gp = sum(t for t in all_trades if t > 0)
    gl = abs(sum(t for t in all_trades if t < 0))
    sharpe = np.mean(period_returns) / (np.std(period_returns) + 1e-10) * np.sqrt(12)

    return {
        'total_return': float(eq[-1] - 1),
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
    print("PHASE 16b: ML-FILTERED VOL PROFILE + ALTERNATIVE LABELS")
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

    all_results = []

    # -----------------------------------------------------------------
    # C. ML-filtered Vol Profile
    # -----------------------------------------------------------------
    print("=" * 60)
    print("C. ML-Filtered Vol Profile (rule + ML confirmation)")
    print("=" * 60)

    configs = [
        ('C_xgb_p045', 'xgb', 0.45),
        ('C_xgb_p050', 'xgb', 0.50),
        ('C_xgb_p055', 'xgb', 0.55),
        ('C_lgb_p045', 'lgb', 0.45),
        ('C_lgb_p050', 'lgb', 0.50),
        ('C_lgb_p055', 'lgb', 0.55),
        ('C_rf_p045', 'rf', 0.45),
        ('C_rf_p050', 'rf', 0.50),
    ]

    for name, model_type, prob_thresh in configs:
        if model_type == 'xgb' and not HAS_XGB:
            continue
        if model_type == 'lgb' and not HAS_LGB:
            continue

        print(f"\n  {name}:")
        r = run_portfolio(all_data, ml_filtered_signals,
                          {'model_type': model_type, 'prob_threshold': prob_thresh},
                          position_scale=5.0)
        if r:
            c, p = check_criteria(r)
            all_results.append((name, p, r))
            fails = [k for k, v in c.items() if not v]
            print(f"    [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                  f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                  f"PF={r['profit_factor']:.2f} T={r['trade_count']}")
            if fails:
                print(f"    FAILS: {', '.join(fails)}")
        gc.collect()

    # -----------------------------------------------------------------
    # D. Alternative labels with XGBoost
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("D. Alternative label definitions (XGBoost)")
    print("=" * 60)

    if HAS_XGB:
        label_configs = [
            ('D_fwd12_thr003', 12, 0.003, 0.55),
            ('D_fwd48_thr010', 48, 0.010, 0.55),
            ('D_fwd24_thr002', 24, 0.002, 0.55),
            ('D_fwd24_thr010', 24, 0.010, 0.55),
        ]

        for name, fwd, thr, prob in label_configs:
            print(f"\n  {name} (fwd={fwd}h, thr={thr*100:.1f}%):")
            r = run_portfolio(all_data, pure_ml_signals,
                              {'model_type': 'xgb', 'forward_bars': fwd,
                               'threshold': thr, 'prob_threshold': prob},
                              position_scale=5.0)
            if r:
                c, p = check_criteria(r)
                all_results.append((name, p, r))
                fails = [k for k, v in c.items() if not v]
                print(f"    [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                      f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                      f"PF={r['profit_factor']:.2f} T={r['trade_count']}")
                if fails:
                    print(f"    FAILS: {', '.join(fails)}")
            gc.collect()

        # Also try ML-filtered with alt labels
        print(f"\n{'=' * 60}")
        print("E. ML-Filtered Vol Profile with alternative labels")
        print("=" * 60)

        alt_configs = [
            ('E_xgb_filt_fwd12', 12, 0.003, 0.50),
            ('E_xgb_filt_fwd48', 48, 0.010, 0.50),
        ]

        for name, fwd, thr, prob in alt_configs:
            print(f"\n  {name}:")
            r = run_portfolio(all_data, ml_filtered_signals,
                              {'model_type': 'xgb', 'forward_bars': fwd,
                               'threshold': thr, 'prob_threshold': prob},
                              position_scale=5.0)
            if r:
                c, p = check_criteria(r)
                all_results.append((name, p, r))
                fails = [k for k, v in c.items() if not v]
                print(f"    [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                      f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                      f"PF={r['profit_factor']:.2f} T={r['trade_count']}")
                if fails:
                    print(f"    FAILS: {', '.join(fails)}")
            gc.collect()

    # -----------------------------------------------------------------
    # FINAL
    # -----------------------------------------------------------------
    all_results.sort(key=lambda x: (-x[1], -x[2].get('sharpe', 0)))

    print(f"\n\n{'=' * 70}")
    print("FINAL RANKING - ALL ML STRATEGIES (Phase 16a + 16b)")
    print("=" * 70)
    print("\nPhase 16a results (Pure ML):")
    print("  All 1-3/6. Best: RF prob0.6 [3/6] Sharpe=2.45")
    print("  XGB, LGB, RF, LR, Ensemble: all failed\n")

    print("Phase 16b results:")
    for i, (name, passed, r) in enumerate(all_results):
        c, _ = check_criteria(r)
        fails = [k for k, v in c.items() if not v]
        fail_str = f"  FAILS: {', '.join(fails)}" if fails else ""
        print(f"  {i+1}. [{passed}/6] {name}")
        print(f"     Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
              f"DD={r['max_drawdown']*100:.1f}%  WR={r['win_rate']*100:.0f}%  "
              f"PF={r['profit_factor']:.2f}  WFA={r['wfa_efficiency']:.0f}%  T={r['trade_count']}{fail_str}")

    six_six = [(n, r) for n, p, r in all_results if p == 6]
    if six_six:
        print(f"\n*** {len(six_six)} ML configs passed 6/6! ***")
        for name, r in six_six:
            print(f"  {name}: Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}%")
        best = max(six_six, key=lambda x: x[1]['sharpe'])
        if best[1]['sharpe'] > 2.52:
            print(f"\n  → ML BEATS Vol Profile baseline (Sharpe {best[1]['sharpe']:.2f} > 2.52)!")
        else:
            print(f"\n  → Vol Profile baseline still better (Sharpe 2.52)")
    else:
        print(f"\n*** No ML config passed 6/6. Vol Profile (Sharpe=2.52) remains best. ***")

    save_data = {
        'timestamp': datetime.now().isoformat(),
        'total_tested': len(all_results),
        'six_six_count': len(six_six) if six_six else 0,
        'results': [
            {'name': n, 'passed': int(p),
             'sharpe': float(r.get('sharpe', 0)),
             'return': float(r.get('total_return', 0)),
             'max_dd': float(r.get('max_drawdown', 0)),
             'win_rate': float(r.get('win_rate', 0)),
             'pf': float(r.get('profit_factor', 0)),
             'trades': int(r.get('trade_count', 0))}
            for n, p, r in all_results
        ],
        'phase16a_summary': 'All pure ML failed (1-3/6). XGB/LGB/RF/LR/Ensemble tested.',
    }
    with open(RESULTS_PATH / "phase16b_ml_report.json", 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nSaved to {RESULTS_PATH / 'phase16b_ml_report.json'}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
