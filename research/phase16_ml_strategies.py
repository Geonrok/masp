#!/usr/bin/env python3
"""
Ralph-Loop Phase 16: ML-Based Strategy Exploration
====================================================
All ML approaches using existing 1H OHLCV data (257 symbols).
Same TRUE OOS framework: 60/40 time split, 6 criteria.

ML Models:
1. XGBoost classifier (binary: up/flat)
2. LightGBM classifier
3. Random Forest classifier
4. Logistic Regression (baseline ML)
5. XGBoost with expanded feature set
6. Ensemble (voting of top models)

Feature Engineering:
- Price-based: returns, MA ratios, Donchian position, ATR ratio
- Volume-based: VWAP ratio, volume z-score, OBV slope
- Momentum: RSI, ROC, MACD proxy
- Volatility: Bollinger %B, ATR expansion, realized vol ratio

CRITICAL: To prevent lookahead bias:
- Features computed on training window only
- Model trained on training window, predict on test window
- Walk-forward: retrain every 720 bars (30 days)
"""
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

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
# FEATURE ENGINEERING
# =============================================================================

def build_features(df):
    """Build ML features from OHLCV data. No future information."""
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    vol = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(1.0, index=df.index)

    feat = pd.DataFrame(index=df.index)

    # Returns
    for p in [1, 4, 12, 24, 48, 72, 168]:
        feat[f'ret_{p}'] = close.pct_change(p)

    # MA ratios
    for p in [10, 20, 50, 100, 200]:
        ma = close.rolling(p).mean()
        feat[f'ma_ratio_{p}'] = close / (ma + 1e-10) - 1

    # EMA ratios
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()
    feat['ema_50_200_ratio'] = ema_f / (ema_s + 1e-10) - 1
    feat['ema_trend'] = (ema_f > ema_s).astype(int)

    # Donchian position
    for p in [24, 48, 72]:
        h = high.rolling(p).max()
        l = low.rolling(p).min()
        feat[f'donchian_pos_{p}'] = (close - l) / (h - l + 1e-10)

    # ATR-based
    atr_vals = pd.Series(calc_atr(high.values, low.values, close.values, 14), index=df.index)
    feat['atr_ratio'] = atr_vals / (close + 1e-10)
    atr_avg = atr_vals.rolling(48).mean()
    feat['atr_expansion'] = atr_vals / (atr_avg + 1e-10)

    # Volatility
    for p in [24, 48, 168]:
        feat[f'vol_realized_{p}'] = close.pct_change().rolling(p).std()
    feat['vol_ratio'] = feat['vol_realized_24'] / (feat['vol_realized_168'] + 1e-10)

    # RSI
    for rsi_p in [14, 28]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_p).mean()
        rs = gain / (loss + 1e-10)
        feat[f'rsi_{rsi_p}'] = 100 - 100 / (1 + rs)

    # Bollinger %B
    for p in [20, 50]:
        ma = close.rolling(p).mean()
        std = close.rolling(p).std()
        feat[f'bb_pct_b_{p}'] = (close - (ma - 2 * std)) / (4 * std + 1e-10)

    # MACD proxy
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat['macd_hist'] = (macd - signal) / (close + 1e-10)

    # Volume features
    vol_ma = vol.rolling(48).mean()
    feat['vol_zscore'] = (vol - vol_ma) / (vol.rolling(48).std() + 1e-10)
    vwap = (close * vol).rolling(48).sum() / (vol.rolling(48).sum() + 1e-10)
    feat['vwap_ratio'] = close / (vwap + 1e-10) - 1

    # OBV slope
    obv = (np.sign(close.diff()) * vol).cumsum()
    feat['obv_slope'] = obv.diff(24) / (vol_ma * 24 + 1e-10)

    # ROC (Rate of Change)
    for p in [12, 24, 48]:
        feat[f'roc_{p}'] = close.pct_change(p)

    # High-Low range ratio
    feat['hl_ratio'] = (high - low) / (close + 1e-10)
    feat['hl_ratio_ma'] = feat['hl_ratio'].rolling(48).mean()

    # Candle body ratio
    opn = df['open'].astype(float) if 'open' in df.columns else close.shift(1)
    feat['body_ratio'] = (close - opn) / (high - low + 1e-10)

    return feat


def build_labels(df, forward_bars=24, threshold=0.005):
    """
    Label: 1 if forward return > threshold (go long), 0 otherwise.
    Using 24-bar (1 day) forward return.
    """
    close = df['close'].astype(float)
    fwd_ret = close.shift(-forward_bars) / close - 1
    labels = (fwd_ret > threshold).astype(int)
    return labels


# =============================================================================
# ML SIGNAL GENERATION (Walk-Forward)
# =============================================================================

def ml_walk_forward_signals(df, model_type='xgb', train_bars=4320, test_bars=720,
                             forward_bars=24, threshold=0.005, prob_threshold=0.55):
    """
    Walk-forward ML signal generation.
    Train on [0:i], predict on [i:i+test_bars], slide forward.
    Returns signals array aligned with df.
    """
    features = build_features(df)
    labels = build_labels(df, forward_bars, threshold)

    signals = np.zeros(len(df))
    feature_cols = features.columns.tolist()

    i = train_bars
    while i + test_bars <= len(df):
        # Training data
        train_feat = features.iloc[:i].copy()
        train_labels = labels.iloc[:i].copy()

        # Remove NaN rows
        valid = train_feat.dropna().index.intersection(train_labels.dropna().index)
        # Also remove last forward_bars rows (labels use future data)
        valid = valid[valid < i - forward_bars]

        if len(valid) < 500:
            i += test_bars
            continue

        X_train = train_feat.loc[valid, feature_cols].values
        y_train = train_labels.loc[valid].values

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Handle class imbalance
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        if pos_count == 0 or neg_count == 0:
            i += test_bars
            continue
        scale_pos_weight = neg_count / pos_count

        # Train model
        try:
            if model_type == 'xgb' and HAS_XGB:
                model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42, verbosity=0,
                    use_label_encoder=False, eval_metric='logloss'
                )
            elif model_type == 'lgb' and HAS_LGB:
                model = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42, verbosity=-1
                )
            elif model_type == 'rf':
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=6,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
            elif model_type == 'lr':
                model = LogisticRegression(
                    class_weight='balanced', random_state=42, max_iter=1000
                )
            else:
                i += test_bars
                continue

            model.fit(X_train_scaled, y_train)

            # Predict on test window
            test_feat = features.iloc[i:i + test_bars].copy()
            valid_test = test_feat.dropna().index
            if len(valid_test) > 0:
                X_test = test_feat.loc[valid_test, feature_cols].values
                X_test_scaled = scaler.transform(X_test)
                probs = model.predict_proba(X_test_scaled)[:, 1]

                for idx, prob in zip(valid_test, probs):
                    if prob > prob_threshold:
                        signals[idx] = 1

        except Exception as e:
            pass

        i += test_bars

    return signals


def ml_ensemble_signals(df, train_bars=4320, test_bars=720,
                         forward_bars=24, threshold=0.005, prob_threshold=0.55):
    """
    Ensemble: average probabilities from XGB + LGB + RF, then threshold.
    """
    features = build_features(df)
    labels = build_labels(df, forward_bars, threshold)

    signals = np.zeros(len(df))
    feature_cols = features.columns.tolist()

    i = train_bars
    while i + test_bars <= len(df):
        train_feat = features.iloc[:i].copy()
        train_labels = labels.iloc[:i].copy()

        valid = train_feat.dropna().index.intersection(train_labels.dropna().index)
        valid = valid[valid < i - forward_bars]

        if len(valid) < 500:
            i += test_bars
            continue

        X_train = train_feat.loc[valid, feature_cols].values
        y_train = train_labels.loc[valid].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        if pos_count == 0 or neg_count == 0:
            i += test_bars
            continue
        spw = neg_count / pos_count

        models = []
        if HAS_XGB:
            models.append(xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=spw, random_state=42, verbosity=0,
                use_label_encoder=False, eval_metric='logloss'))
        if HAS_LGB:
            models.append(lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=spw, random_state=42, verbosity=-1))
        models.append(RandomForestClassifier(
            n_estimators=100, max_depth=6,
            class_weight='balanced', random_state=42, n_jobs=-1))

        try:
            all_probs = []
            for model in models:
                model.fit(X_train_scaled, y_train)

                test_feat = features.iloc[i:i + test_bars].copy()
                valid_test = test_feat.dropna().index
                if len(valid_test) > 0:
                    X_test = test_feat.loc[valid_test, feature_cols].values
                    X_test_scaled = scaler.transform(X_test)
                    probs = model.predict_proba(X_test_scaled)[:, 1]
                    all_probs.append((valid_test, probs))

            if all_probs:
                # Average probabilities
                for j, idx in enumerate(all_probs[0][0]):
                    avg_prob = np.mean([p[1][j] for p in all_probs if j < len(p[1])])
                    if avg_prob > prob_threshold:
                        signals[idx] = 1

        except Exception:
            pass

        i += test_bars

    return signals


# =============================================================================
# SIMULATION (same as previous phases)
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
# PORTFOLIO TRUE OOS
# =============================================================================

def run_ml_portfolio_oos(all_data, model_type='xgb', max_positions=10,
                          test_bars=720, position_scale=5.0,
                          forward_bars=24, threshold=0.005, prob_threshold=0.55):
    """
    TRUE OOS portfolio test with ML signals.
    ML models are trained per-symbol using walk-forward.
    """
    exit_params = {'max_bars': 72, 'atr_stop': 3.0, 'profit_target_atr': 8.0}

    # OOS on last 40%
    oos_data = {}
    for symbol in all_data:
        df = all_data[symbol]
        split = int(len(df) * 0.6)
        oos_df = df.iloc[split:].copy().reset_index(drop=True)
        if len(oos_df) > 2000:
            oos_data[symbol] = oos_df

    if not oos_data:
        return None

    # Pre-compute ML signals for each symbol (walk-forward within OOS)
    print(f"    Computing ML signals for {len(oos_data)} symbols...", end=" ", flush=True)
    symbol_signals = {}
    for sym_idx, (symbol, df) in enumerate(oos_data.items()):
        if model_type == 'ensemble':
            sigs = ml_ensemble_signals(df, train_bars=min(4320, len(df) // 3),
                                        test_bars=test_bars,
                                        forward_bars=forward_bars,
                                        threshold=threshold,
                                        prob_threshold=prob_threshold)
        else:
            sigs = ml_walk_forward_signals(df, model_type=model_type,
                                            train_bars=min(4320, len(df) // 3),
                                            test_bars=test_bars,
                                            forward_bars=forward_bars,
                                            threshold=threshold,
                                            prob_threshold=prob_threshold)
        symbol_signals[symbol] = sigs
        if (sym_idx + 1) % 50 == 0:
            print(f"{sym_idx+1}", end=" ", flush=True)

    print("done")

    # Portfolio simulation using pre-computed signals
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
        'all_trades': all_trades,
    }


# =============================================================================
# ML-ENHANCED RULE-BASED: Use ML as filter on top of Vol Profile
# =============================================================================

def ml_filtered_vol_profile_signals(df, model_type='xgb', train_bars=4320,
                                      test_bars=720, prob_threshold=0.50):
    """
    Vol Profile breakout entry, but only if ML model agrees (prob > threshold).
    This combines rule-based signal quality with ML confirmation.
    """
    close = df['close']
    high = df['high']
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    vwap = (close * vol).rolling(48).sum() / (vol.rolling(48).sum() + 1e-10)
    upper = high.rolling(48).max().shift(1)
    ema_f = close.ewm(span=50, adjust=False).mean()
    ema_s = close.ewm(span=200, adjust=False).mean()

    rule_signals = (close > upper) & (close > vwap * 1.01) & (ema_f > ema_s)

    features = build_features(df)
    labels = build_labels(df, 24, 0.005)
    feature_cols = features.columns.tolist()

    signals = np.zeros(len(df))

    i = train_bars
    while i + test_bars <= len(df):
        train_feat = features.iloc[:i].copy()
        train_labels = labels.iloc[:i].copy()
        valid = train_feat.dropna().index.intersection(train_labels.dropna().index)
        valid = valid[valid < i - 24]

        if len(valid) < 500:
            # Fall back to pure rule-based
            for j in range(i, min(i + test_bars, len(df))):
                if rule_signals.iloc[j]:
                    signals[j] = 1
            i += test_bars
            continue

        X_train = train_feat.loc[valid, feature_cols].values
        y_train = train_labels.loc[valid].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        if pos_count == 0 or neg_count == 0:
            i += test_bars
            continue
        spw = neg_count / pos_count

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

            model.fit(X_train_scaled, y_train)

            test_feat = features.iloc[i:i + test_bars].copy()
            valid_test = test_feat.dropna().index
            if len(valid_test) > 0:
                X_test = test_feat.loc[valid_test, feature_cols].values
                X_test_scaled = scaler.transform(X_test)
                probs = model.predict_proba(X_test_scaled)[:, 1]

                for idx, prob in zip(valid_test, probs):
                    # Only signal if BOTH rule-based AND ML agree
                    if rule_signals.iloc[idx] and prob > prob_threshold:
                        signals[idx] = 1

        except Exception:
            # Fallback to rule-based
            for j in range(i, min(i + test_bars, len(df))):
                if rule_signals.iloc[j]:
                    signals[j] = 1

        i += test_bars

    return signals


def run_ml_filtered_portfolio(all_data, model_type='xgb', max_positions=10,
                               test_bars=720, position_scale=5.0, prob_threshold=0.50):
    """Portfolio OOS with ML-filtered Vol Profile signals."""
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

    print(f"    Computing ML-filtered signals for {len(oos_data)} symbols...", end=" ", flush=True)
    symbol_signals = {}
    for sym_idx, (symbol, df) in enumerate(oos_data.items()):
        sigs = ml_filtered_vol_profile_signals(df, model_type=model_type,
                                                 train_bars=min(4320, len(df) // 3),
                                                 test_bars=test_bars,
                                                 prob_threshold=prob_threshold)
        symbol_signals[symbol] = sigs
        if (sym_idx + 1) % 50 == 0:
            print(f"{sym_idx+1}", end=" ", flush=True)
    print("done")

    # Same portfolio framework
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
        'all_trades': all_trades,
    }


def main():
    print("=" * 70)
    print("RALPH-LOOP PHASE 16: ML-BASED STRATEGIES")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"XGBoost: {'YES' if HAS_XGB else 'NO'}")
    print(f"LightGBM: {'YES' if HAS_LGB else 'NO'}")
    print()

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
    # PART A: Pure ML strategies (ML generates all signals)
    # -----------------------------------------------------------------
    print("=" * 60)
    print("PART A: Pure ML Strategies")
    print("=" * 60)

    ml_models = []
    if HAS_XGB:
        ml_models.append(('xgb', 'XGBoost'))
    if HAS_LGB:
        ml_models.append(('lgb', 'LightGBM'))
    ml_models.extend([('rf', 'Random Forest'), ('lr', 'Logistic Regression')])

    for model_key, model_name in ml_models:
        for prob_thresh in [0.55, 0.60]:
            config_name = f"A_{model_key}_prob{prob_thresh}"
            print(f"\n  {config_name} ({model_name}):")

            r = run_ml_portfolio_oos(all_data, model_type=model_key,
                                      prob_threshold=prob_thresh)
            if r:
                c, p = check_criteria(r)
                all_results.append((config_name, p, r))
                fails = [k for k, v in c.items() if not v]
                print(f"    [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                      f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                      f"PF={r['profit_factor']:.2f} T={r['trade_count']}")
                if fails:
                    print(f"    FAILS: {', '.join(fails)}")
            else:
                print(f"    SKIP")

    # -----------------------------------------------------------------
    # PART B: ML as ensemble
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PART B: ML Ensemble (XGB+LGB+RF voting)")
    print("=" * 60)

    for prob_thresh in [0.55, 0.60]:
        config_name = f"B_ensemble_prob{prob_thresh}"
        print(f"\n  {config_name}:")

        r = run_ml_portfolio_oos(all_data, model_type='ensemble',
                                  prob_threshold=prob_thresh)
        if r:
            c, p = check_criteria(r)
            all_results.append((config_name, p, r))
            fails = [k for k, v in c.items() if not v]
            print(f"    [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                  f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                  f"PF={r['profit_factor']:.2f} T={r['trade_count']}")
            if fails:
                print(f"    FAILS: {', '.join(fails)}")

    # -----------------------------------------------------------------
    # PART C: ML-filtered Vol Profile (hybrid rule + ML)
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PART C: ML-Filtered Vol Profile (rule-based + ML confirmation)")
    print("=" * 60)

    filter_models = []
    if HAS_XGB:
        filter_models.append(('xgb', 'XGBoost'))
    if HAS_LGB:
        filter_models.append(('lgb', 'LightGBM'))
    filter_models.append(('rf', 'Random Forest'))

    for model_key, model_name in filter_models:
        for prob_thresh in [0.45, 0.50, 0.55]:
            config_name = f"C_{model_key}_filter_prob{prob_thresh}"
            print(f"\n  {config_name} ({model_name}):")

            r = run_ml_filtered_portfolio(all_data, model_type=model_key,
                                           prob_threshold=prob_thresh)
            if r:
                c, p = check_criteria(r)
                all_results.append((config_name, p, r))
                fails = [k for k, v in c.items() if not v]
                print(f"    [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                      f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                      f"PF={r['profit_factor']:.2f} T={r['trade_count']}")
                if fails:
                    print(f"    FAILS: {', '.join(fails)}")

    # -----------------------------------------------------------------
    # PART D: Different label definitions
    # -----------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PART D: Alternative label definitions")
    print("=" * 60)

    if HAS_XGB:
        # Shorter forward window (12 bars = 12 hours)
        config_name = "D_xgb_fwd12_thr003"
        print(f"\n  {config_name}:")
        r = run_ml_portfolio_oos(all_data, model_type='xgb',
                                  forward_bars=12, threshold=0.003, prob_threshold=0.55)
        if r:
            c, p = check_criteria(r)
            all_results.append((config_name, p, r))
            print(f"    [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                  f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                  f"PF={r['profit_factor']:.2f} T={r['trade_count']}")

        # Longer forward window (48 bars = 2 days)
        config_name = "D_xgb_fwd48_thr010"
        print(f"\n  {config_name}:")
        r = run_ml_portfolio_oos(all_data, model_type='xgb',
                                  forward_bars=48, threshold=0.010, prob_threshold=0.55)
        if r:
            c, p = check_criteria(r)
            all_results.append((config_name, p, r))
            print(f"    [{p}/6] Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}% "
                  f"DD={r['max_drawdown']*100:.1f}% WR={r['win_rate']*100:.0f}% "
                  f"PF={r['profit_factor']:.2f} T={r['trade_count']}")

    # -----------------------------------------------------------------
    # FINAL RANKING
    # -----------------------------------------------------------------
    all_results.sort(key=lambda x: (-x[1], -x[2].get('sharpe', 0)))

    print(f"\n\n{'=' * 70}")
    print("FINAL RANKING - ALL ML STRATEGIES")
    print("=" * 70)

    for i, (name, passed, r) in enumerate(all_results):
        c, _ = check_criteria(r)
        fails = [k for k, v in c.items() if not v]
        fail_str = f"  FAILS: {', '.join(fails)}" if fails else ""
        print(f"  {i+1}. [{passed}/6] {name}")
        print(f"     Sharpe={r['sharpe']:.2f}  Ret={r['total_return']*100:+.1f}%  "
              f"DD={r['max_drawdown']*100:.1f}%  WR={r['win_rate']*100:.0f}%  "
              f"PF={r['profit_factor']:.2f}  WFA={r['wfa_efficiency']:.0f}%  T={r['trade_count']}{fail_str}")

    six_six = [(n, r) for n, p, r in all_results if p == 6]
    print(f"\n*** {len(six_six)} ML configs passed 6/6 ***")
    if six_six:
        for name, r in six_six:
            print(f"  {name}: Sharpe={r['sharpe']:.2f} Ret={r['total_return']*100:+.1f}%")

        best_ml = max(six_six, key=lambda x: x[1]['sharpe'])
        print(f"\n  Best ML: {best_ml[0]} Sharpe={best_ml[1]['sharpe']:.2f}")
        print(f"  Baseline Vol Profile: Sharpe=2.52")
        if best_ml[1]['sharpe'] > 2.52:
            print("  → ML BEATS BASELINE!")
        else:
            print("  → Baseline still better")
    else:
        print("  None. ML does not beat rule-based Vol Profile.")
        best_result = all_results[0] if all_results else None
        if best_result:
            print(f"  Best ML: [{best_result[1]}/6] {best_result[0]} Sharpe={best_result[2]['sharpe']:.2f}")

    # Correlation check for ensemble potential
    if six_six:
        print(f"\n{'=' * 60}")
        print("Correlation with Vol Profile baseline")
        print("=" * 60)
        # Would need baseline period returns for comparison

    # Save
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'total_tested': len(all_results),
        'six_six_count': len(six_six),
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
    }
    with open(RESULTS_PATH / "phase16_ml_report.json", 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nSaved to {RESULTS_PATH / 'phase16_ml_report.json'}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
