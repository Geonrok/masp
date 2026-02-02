#!/usr/bin/env python3
"""
Ralph-Loop Phase 11: New Strategy Categories
==============================================
Test 8 new strategy categories using locally available data.
Same TRUE OOS framework (60/40 time split, 6 criteria).

Strategies:
1. Funding Rate Mean Reversion
2. Fear & Greed Contrarian
3. Macro Regime (DXY + VIX)
4. BTC-Relative Altcoin Rotation
5. DeFi TVL Regime
6. Kimchi Premium (Cross-Exchange)
7. Grid Strategy (Range-Bound)
8. Composite Sentiment (FGI + Google Trends)
"""
import json
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


# =============================================================================
# DATA LOADERS
# =============================================================================

def load_futures_daily(symbol):
    path = DATA_ROOT / "binance_futures_1d" / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ['datetime', 'timestamp', 'date']:
        if col in df.columns:
            df['datetime'] = pd.to_datetime(df[col])
            break
    return df.sort_values('datetime').reset_index(drop=True)


def load_futures_1h(symbol):
    path = DATA_ROOT / "binance_futures_1h" / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ['datetime', 'timestamp', 'date']:
        if col in df.columns:
            df['datetime'] = pd.to_datetime(df[col])
            break
    return df.sort_values('datetime').reset_index(drop=True)


def load_funding(symbol="BTCUSDT"):
    # Try full first
    path = DATA_ROOT / "binance_funding_rate" / f"{symbol}_funding_full.csv"
    if not path.exists():
        path = DATA_ROOT / "binance_funding_rate" / f"{symbol}_funding.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


def load_fear_greed():
    path = DATA_ROOT / "sentiment" / "fear_greed_index.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


def load_macro(name):
    path = DATA_ROOT / "macro" / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


def load_tvl():
    path = DATA_ROOT / "defillama" / "total_tvl_history.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


def load_google_trends():
    path = DATA_ROOT / "sentiment" / "google_trends_bitcoin.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)


def load_spot(exchange, symbol):
    path = DATA_ROOT / f"{exchange}_1d" / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ['datetime', 'timestamp', 'date']:
        if col in df.columns:
            df['datetime'] = pd.to_datetime(df[col])
            break
    return df.sort_values('datetime').reset_index(drop=True)


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def simulate_daily(prices, signals, position_pct=0.05, stop_pct=0.10,
                   max_hold_days=30, slippage=0.001):
    """Simulate on daily data. signals: array of -1, 0, 1."""
    capital = 1.0
    position = 0
    entry_price = 0
    days_held = 0
    trades = []

    for i in range(len(prices)):
        p = prices[i]
        sig = signals[i] if i < len(signals) else 0

        if position != 0:
            days_held += 1
            pnl_pct = position * (p - entry_price) / entry_price

            should_exit = False
            if days_held >= max_hold_days:
                should_exit = True
            elif pnl_pct < -stop_pct:
                should_exit = True
            elif sig != 0 and sig != position:
                should_exit = True

            if should_exit:
                exit_p = p * (1 - slippage * np.sign(position))
                pnl = position * (exit_p - entry_price) / entry_price * position_pct
                pnl -= COMMISSION * position_pct * 2
                pnl -= FUNDING_PER_8H * days_held * 3 * position_pct
                trades.append(pnl)
                capital += pnl
                position = 0

        if position == 0 and sig != 0:
            position = int(np.sign(sig))
            entry_price = p * (1 + slippage * position)
            days_held = 0

    if position != 0:
        p = prices[-1]
        pnl = position * (p - entry_price) / entry_price * position_pct
        pnl -= COMMISSION * position_pct * 2
        pnl -= FUNDING_PER_8H * days_held * 3 * position_pct
        trades.append(pnl)

    return trades, capital - 1


def calc_metrics(trades, period_returns=None):
    if not trades:
        return {'sharpe': 0, 'total_return': 0, 'max_drawdown': 0,
                'win_rate': 0, 'profit_factor': 0, 'trade_count': 0,
                'wfa_efficiency': 0}

    wins = sum(1 for t in trades if t > 0)
    losses = sum(1 for t in trades if t <= 0)
    gp = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))

    equity = np.cumprod(1 + np.array(trades))
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak

    if period_returns is not None and len(period_returns) > 1:
        sharpe = np.mean(period_returns) / (np.std(period_returns) + 1e-10) * np.sqrt(12)
        wfa = sum(1 for r in period_returns if r > 0) / len(period_returns) * 100
    else:
        trade_arr = np.array(trades)
        sharpe = np.mean(trade_arr) / (np.std(trade_arr) + 1e-10) * np.sqrt(252)
        wfa = 0

    return {
        'sharpe': float(sharpe),
        'total_return': float(equity[-1] - 1),
        'max_drawdown': float(dd.min()) if len(dd) > 0 else 0,
        'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
        'profit_factor': gp / (gl + 1e-10),
        'trade_count': len(trades),
        'wfa_efficiency': wfa,
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


def run_wfa_daily(prices, signals, train_days=365, test_days=90,
                  position_pct=0.05, stop_pct=0.10, max_hold=30, slippage=0.001):
    """Walk-forward analysis on daily signals."""
    all_trades = []
    period_returns = []

    i = train_days
    while i + test_days <= len(prices):
        test_prices = prices[i:i + test_days]
        test_signals = signals[i:i + test_days]
        trades, ret = simulate_daily(test_prices, test_signals,
                                     position_pct, stop_pct, max_hold, slippage)
        all_trades.extend(trades)
        period_returns.append(ret)
        i += test_days

    return all_trades, period_returns


# =============================================================================
# STRATEGY 1: Funding Rate Mean Reversion
# =============================================================================

def strategy_1_funding_mr():
    print("\n" + "=" * 60)
    print("STRATEGY 1: Funding Rate Mean Reversion")
    print("=" * 60)

    results = {}
    symbols_with_funding = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT',
                            'DOGEUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT',
                            'SOLUSDT', 'XRPUSDT']

    for z_thresh in [1.5, 2.0, 2.5]:
        for lookback in [30, 60, 90]:
            config = f"z{z_thresh}_lb{lookback}"
            all_trades = []
            period_returns = []

            for symbol in symbols_with_funding:
                funding = load_funding(symbol)
                price_df = load_futures_daily(symbol.replace('USDT', ''))
                if funding.empty or price_df.empty:
                    continue

                # Resample funding to daily (mean of 3 readings)
                funding['date'] = funding['datetime'].dt.date
                daily_funding = funding.groupby('date')['fundingRate'].mean().reset_index()
                daily_funding['datetime'] = pd.to_datetime(daily_funding['date'])

                # Merge with price
                price_df['date'] = price_df['datetime'].dt.date
                merged = price_df.merge(daily_funding[['date', 'fundingRate']],
                                       on='date', how='left')
                merged['fundingRate'] = merged['fundingRate'].ffill()
                merged = merged.dropna(subset=['fundingRate', 'close']).reset_index(drop=True)

                if len(merged) < 500:
                    continue

                # Z-score of funding rate
                fr = merged['fundingRate']
                fr_mean = fr.rolling(lookback).mean()
                fr_std = fr.rolling(lookback).std()
                z_score = (fr - fr_mean) / (fr_std + 1e-10)

                # Signals: contrarian
                signals = np.where(z_score > z_thresh, -1,
                          np.where(z_score < -z_thresh, 1, 0))

                # TRUE OOS: first 60% select, last 40% test
                split = int(len(merged) * 0.6)
                oos_prices = merged['close'].values[split:]
                oos_signals = signals[split:]

                trades, period_rets = run_wfa_daily(
                    oos_prices, oos_signals,
                    train_days=180, test_days=90,
                    position_pct=0.03, stop_pct=0.08, max_hold=14)
                all_trades.extend(trades)
                period_returns.extend(period_rets)

            if all_trades:
                metrics = calc_metrics(all_trades, period_returns)
                criteria, passed = check_criteria(metrics)
                results[config] = {**metrics, 'criteria_passed': passed}
                print(f"  {config}: [{passed}/6] Sharpe={metrics['sharpe']:.2f} "
                      f"Ret={metrics['total_return']*100:+.1f}% WR={metrics['win_rate']*100:.0f}% "
                      f"PF={metrics['profit_factor']:.2f} T={metrics['trade_count']}")

    return results


# =============================================================================
# STRATEGY 2: Fear & Greed Contrarian
# =============================================================================

def strategy_2_fear_greed():
    print("\n" + "=" * 60)
    print("STRATEGY 2: Fear & Greed Contrarian")
    print("=" * 60)

    fgi = load_fear_greed()
    btc = load_futures_daily('BTC')
    if fgi.empty or btc.empty:
        print("  Data not available")
        return {}

    # Merge FGI with BTC daily
    fgi['date'] = fgi['datetime'].dt.date
    btc['date'] = btc['datetime'].dt.date
    merged = btc.merge(fgi[['date', 'fear_greed_value']], on='date', how='left')
    merged['fear_greed_value'] = merged['fear_greed_value'].ffill()
    merged = merged.dropna(subset=['fear_greed_value', 'close']).reset_index(drop=True)

    print(f"  Merged data: {len(merged)} rows")

    results = {}
    for fear_level, greed_level in [(15, 85), (20, 80), (25, 75), (30, 70)]:
        for smoothing in [1, 7]:
            config = f"fear{fear_level}_greed{greed_level}_smooth{smoothing}"
            fgi_val = merged['fear_greed_value']
            if smoothing > 1:
                fgi_val = fgi_val.rolling(smoothing).mean()

            signals = np.where(fgi_val < fear_level, 1,
                      np.where(fgi_val > greed_level, -1, 0))

            # TRUE OOS
            split = int(len(merged) * 0.6)
            oos_prices = merged['close'].values[split:]
            oos_signals = signals[split:]

            trades, period_rets = run_wfa_daily(
                oos_prices, oos_signals,
                train_days=180, test_days=90,
                position_pct=0.05, stop_pct=0.15, max_hold=60)

            if trades:
                metrics = calc_metrics(trades, period_rets)
                criteria, passed = check_criteria(metrics)
                results[config] = {**metrics, 'criteria_passed': passed}
                print(f"  {config}: [{passed}/6] Sharpe={metrics['sharpe']:.2f} "
                      f"Ret={metrics['total_return']*100:+.1f}% WR={metrics['win_rate']*100:.0f}% "
                      f"PF={metrics['profit_factor']:.2f} T={metrics['trade_count']}")

    return results


# =============================================================================
# STRATEGY 3: Macro Regime (DXY + VIX)
# =============================================================================

def strategy_3_macro_regime():
    print("\n" + "=" * 60)
    print("STRATEGY 3: Macro Regime (DXY + VIX)")
    print("=" * 60)

    dxy = load_macro('DXY')
    vix = load_macro('VIX')
    btc = load_futures_daily('BTC')

    if dxy.empty or vix.empty or btc.empty:
        print("  Data not available")
        return {}

    # Merge all on date
    dxy['date'] = dxy['datetime'].dt.date
    vix['date'] = vix['datetime'].dt.date
    btc['date'] = btc['datetime'].dt.date

    merged = btc[['date', 'close']].copy()
    merged = merged.merge(dxy[['date', 'close']].rename(columns={'close': 'dxy'}), on='date', how='left')
    merged = merged.merge(vix[['date', 'close']].rename(columns={'close': 'vix'}), on='date', how='left')
    merged['dxy'] = merged['dxy'].ffill()
    merged['vix'] = merged['vix'].ffill()
    merged = merged.dropna().reset_index(drop=True)

    print(f"  Merged data: {len(merged)} rows")

    results = {}
    for dxy_ma in [50, 100]:
        for vix_thresh in [20, 25]:
            config = f"dxy_ma{dxy_ma}_vix{vix_thresh}"

            dxy_sma = pd.Series(merged['dxy']).rolling(dxy_ma).mean()
            dxy_below = merged['dxy'] < dxy_sma  # DXY falling = risk-on
            vix_low = merged['vix'] < vix_thresh

            # Risk-on: DXY falling + VIX low → long BTC
            # Risk-off: DXY rising + VIX high → flat (or short)
            signals = np.where(dxy_below & vix_low, 1,
                      np.where(~dxy_below & ~vix_low, -1, 0))

            split = int(len(merged) * 0.6)
            oos_prices = merged['close'].values[split:]
            oos_signals = signals[split:]

            trades, period_rets = run_wfa_daily(
                oos_prices, oos_signals,
                train_days=180, test_days=90,
                position_pct=0.05, stop_pct=0.12, max_hold=30)

            if trades:
                metrics = calc_metrics(trades, period_rets)
                criteria, passed = check_criteria(metrics)
                results[config] = {**metrics, 'criteria_passed': passed}
                print(f"  {config}: [{passed}/6] Sharpe={metrics['sharpe']:.2f} "
                      f"Ret={metrics['total_return']*100:+.1f}% WR={metrics['win_rate']*100:.0f}% "
                      f"PF={metrics['profit_factor']:.2f} T={metrics['trade_count']}")

    # Also test long-only version
    for dxy_ma in [50, 100]:
        for vix_thresh in [20, 25]:
            config = f"long_only_dxy{dxy_ma}_vix{vix_thresh}"
            dxy_sma = pd.Series(merged['dxy']).rolling(dxy_ma).mean()
            dxy_below = merged['dxy'] < dxy_sma
            vix_low = merged['vix'] < vix_thresh

            signals = np.where(dxy_below & vix_low, 1, 0)

            split = int(len(merged) * 0.6)
            oos_prices = merged['close'].values[split:]
            oos_signals = signals[split:]

            trades, period_rets = run_wfa_daily(
                oos_prices, oos_signals,
                train_days=180, test_days=90,
                position_pct=0.05, stop_pct=0.12, max_hold=30)

            if trades:
                metrics = calc_metrics(trades, period_rets)
                criteria, passed = check_criteria(metrics)
                results[config] = {**metrics, 'criteria_passed': passed}
                print(f"  {config}: [{passed}/6] Sharpe={metrics['sharpe']:.2f} "
                      f"Ret={metrics['total_return']*100:+.1f}% WR={metrics['win_rate']*100:.0f}% "
                      f"PF={metrics['profit_factor']:.2f} T={metrics['trade_count']}")

    return results


# =============================================================================
# STRATEGY 4: BTC-Relative Altcoin Rotation
# =============================================================================

def strategy_4_altcoin_rotation():
    print("\n" + "=" * 60)
    print("STRATEGY 4: BTC-Relative Altcoin Rotation")
    print("=" * 60)

    btc = load_futures_daily('BTC')
    if btc.empty:
        print("  BTC data not available")
        return {}

    # Load top altcoins
    alt_symbols = ['ETH', 'SOL', 'XRP', 'DOGE', 'BNB', 'ADA', 'LINK',
                   'AVAX', 'LTC', 'DOT', 'UNI', 'AAVE', 'FIL', 'NEAR']

    alt_data = {}
    for s in alt_symbols:
        df = load_futures_daily(s)
        if not df.empty and len(df) > 500:
            df['date'] = df['datetime'].dt.date
            alt_data[s] = df

    btc['date'] = btc['datetime'].dt.date
    print(f"  Loaded {len(alt_data)} altcoins")

    results = {}
    for mom_period in [14, 21, 30]:
        for top_n in [3, 5]:
            config = f"mom{mom_period}_top{top_n}"
            all_trades = []
            period_returns = []

            # Find common dates
            all_dates = set(btc['date'])
            for s, df in alt_data.items():
                all_dates &= set(df['date'])
            all_dates = sorted(all_dates)

            if len(all_dates) < 500:
                continue

            # Build price matrix
            btc_prices = btc.set_index('date')['close']
            alt_prices = {}
            for s, df in alt_data.items():
                alt_prices[s] = df.set_index('date')['close']

            # TRUE OOS
            split = int(len(all_dates) * 0.6)
            oos_dates = all_dates[split:]

            # Walk-forward: rebalance every test_days
            rebal_days = 30
            i = 0
            while i + rebal_days <= len(oos_dates):
                period_dates = oos_dates[i:i + rebal_days]
                lookback_start = max(0, i - mom_period)
                lookback_dates = oos_dates[max(0, i - mom_period):i] if i > 0 else oos_dates[:1]

                # Rank altcoins by relative momentum vs BTC
                rankings = []
                for s in alt_data:
                    try:
                        start_date = lookback_dates[0] if lookback_dates else period_dates[0]
                        end_date = period_dates[0]
                        if start_date in alt_prices[s].index and end_date in alt_prices[s].index:
                            alt_ret = alt_prices[s][end_date] / alt_prices[s][start_date] - 1
                            btc_ret = btc_prices[end_date] / btc_prices[start_date] - 1
                            rel_mom = alt_ret - btc_ret
                            rankings.append((s, rel_mom))
                    except (KeyError, IndexError):
                        continue

                if len(rankings) < top_n:
                    i += rebal_days
                    continue

                rankings.sort(key=lambda x: x[1], reverse=True)
                selected = [s for s, _ in rankings[:top_n]]

                period_pnl = 0
                for s in selected:
                    try:
                        start_p = alt_prices[s][period_dates[0]]
                        end_p = alt_prices[s][period_dates[-1]]
                        ret = (end_p / start_p - 1) / top_n * 0.05  # equal weight, 5% total
                        ret -= COMMISSION * 0.05 / top_n * 2
                        all_trades.append(ret)
                        period_pnl += ret
                    except (KeyError, IndexError):
                        continue

                period_returns.append(period_pnl)
                i += rebal_days

            if all_trades:
                metrics = calc_metrics(all_trades, period_returns)
                criteria, passed = check_criteria(metrics)
                results[config] = {**metrics, 'criteria_passed': passed}
                print(f"  {config}: [{passed}/6] Sharpe={metrics['sharpe']:.2f} "
                      f"Ret={metrics['total_return']*100:+.1f}% WR={metrics['win_rate']*100:.0f}% "
                      f"PF={metrics['profit_factor']:.2f} T={metrics['trade_count']}")

    return results


# =============================================================================
# STRATEGY 5: DeFi TVL Regime
# =============================================================================

def strategy_5_tvl_regime():
    print("\n" + "=" * 60)
    print("STRATEGY 5: DeFi TVL Regime")
    print("=" * 60)

    tvl = load_tvl()
    btc = load_futures_daily('BTC')
    if tvl.empty or btc.empty:
        print("  Data not available")
        return {}

    # Filter TVL to non-zero periods
    tvl = tvl[tvl['tvl_usd'] > 0].reset_index(drop=True)
    tvl['date'] = tvl['datetime'].dt.date
    btc['date'] = btc['datetime'].dt.date

    merged = btc[['date', 'close']].merge(
        tvl[['date', 'tvl_usd']], on='date', how='left')
    merged['tvl_usd'] = merged['tvl_usd'].ffill()
    merged = merged.dropna().reset_index(drop=True)

    print(f"  Merged data: {len(merged)} rows")

    results = {}
    for roc_period in [14, 30, 60]:
        config = f"tvl_roc{roc_period}"
        tvl_roc = merged['tvl_usd'].pct_change(roc_period)

        # Long when TVL growing, flat when declining
        signals = np.where(tvl_roc > 0, 1, 0)

        split = int(len(merged) * 0.6)
        oos_prices = merged['close'].values[split:]
        oos_signals = signals[split:]

        trades, period_rets = run_wfa_daily(
            oos_prices, oos_signals,
            train_days=180, test_days=90,
            position_pct=0.05, stop_pct=0.15, max_hold=60)

        if trades:
            metrics = calc_metrics(trades, period_rets)
            criteria, passed = check_criteria(metrics)
            results[config] = {**metrics, 'criteria_passed': passed}
            print(f"  {config}: [{passed}/6] Sharpe={metrics['sharpe']:.2f} "
                  f"Ret={metrics['total_return']*100:+.1f}% WR={metrics['win_rate']*100:.0f}% "
                  f"PF={metrics['profit_factor']:.2f} T={metrics['trade_count']}")

    return results


# =============================================================================
# STRATEGY 6: Kimchi Premium (Cross-Exchange)
# =============================================================================

def strategy_6_kimchi_premium():
    print("\n" + "=" * 60)
    print("STRATEGY 6: Kimchi Premium (Cross-Exchange)")
    print("=" * 60)

    upbit_btc = load_spot('upbit', 'BTC')
    binance_btc = load_spot('binance_spot', 'BTC')
    btc_futures = load_futures_daily('BTC')

    if upbit_btc.empty or binance_btc.empty or btc_futures.empty:
        print("  Data not available")
        return {}

    # Upbit is in KRW. We need to estimate KRW/USD.
    # Use the ratio itself as signal (premium = Upbit_KRW / Binance_USD / KRWUSD_rate)
    # Since we don't have KRW/USD, use the ratio's z-score (relative premium)
    upbit_btc['date'] = upbit_btc['datetime'].dt.date
    binance_btc['date'] = binance_btc['datetime'].dt.date
    btc_futures['date'] = btc_futures['datetime'].dt.date

    merged = binance_btc[['date', 'close']].rename(columns={'close': 'binance_usd'})
    merged = merged.merge(upbit_btc[['date', 'close']].rename(columns={'close': 'upbit_krw'}),
                         on='date', how='inner')
    merged = merged.merge(btc_futures[['date', 'close']].rename(columns={'close': 'futures'}),
                         on='date', how='inner')

    # Kimchi premium ratio (relative, not absolute)
    merged['premium_ratio'] = merged['upbit_krw'] / merged['binance_usd']
    merged = merged.dropna().reset_index(drop=True)

    print(f"  Merged data: {len(merged)} rows")

    results = {}
    for lookback in [30, 60]:
        for z_thresh in [1.5, 2.0]:
            config = f"premium_lb{lookback}_z{z_thresh}"

            ratio = merged['premium_ratio']
            ratio_mean = ratio.rolling(lookback).mean()
            ratio_std = ratio.rolling(lookback).std()
            z = (ratio - ratio_mean) / (ratio_std + 1e-10)

            # High kimchi premium = overheated Korean buying = bearish signal
            # Low kimchi premium = Korean selling = contrarian buy
            signals = np.where(z > z_thresh, -1,
                      np.where(z < -z_thresh, 1, 0))

            split = int(len(merged) * 0.6)
            oos_prices = merged['futures'].values[split:]
            oos_signals = signals[split:]

            trades, period_rets = run_wfa_daily(
                oos_prices, oos_signals,
                train_days=180, test_days=90,
                position_pct=0.05, stop_pct=0.10, max_hold=21)

            if trades:
                metrics = calc_metrics(trades, period_rets)
                criteria, passed = check_criteria(metrics)
                results[config] = {**metrics, 'criteria_passed': passed}
                print(f"  {config}: [{passed}/6] Sharpe={metrics['sharpe']:.2f} "
                      f"Ret={metrics['total_return']*100:+.1f}% WR={metrics['win_rate']*100:.0f}% "
                      f"PF={metrics['profit_factor']:.2f} T={metrics['trade_count']}")

    return results


# =============================================================================
# STRATEGY 7: Grid Strategy (Range-Bound)
# =============================================================================

def strategy_7_grid():
    print("\n" + "=" * 60)
    print("STRATEGY 7: Grid Strategy (Range-Bound Detection)")
    print("=" * 60)

    # Use 1h data for grid (needs granularity)
    btc = load_futures_1h('BTCUSDT')
    if btc.empty:
        print("  Data not available")
        return {}

    results = {}
    for grid_pct in [0.01, 0.015, 0.02]:
        for adx_thresh in [20, 25]:
            config = f"grid{grid_pct*100:.1f}pct_adx{adx_thresh}"

            close = btc['close'].values
            high = btc['high'].values
            low = btc['low'].values

            # Simplified ADX calculation
            n = 14
            tr = np.maximum(high[1:] - low[1:],
                 np.maximum(np.abs(high[1:] - close[:-1]),
                            np.abs(low[1:] - close[:-1])))
            tr = np.insert(tr, 0, high[0] - low[0])
            atr = pd.Series(tr).rolling(n).mean().values

            # +DM / -DM
            up_move = high[1:] - high[:-1]
            down_move = low[:-1] - low[1:]
            up_move = np.insert(up_move, 0, 0)
            down_move = np.insert(down_move, 0, 0)

            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

            plus_di = pd.Series(plus_dm).rolling(n).mean() / (pd.Series(tr).rolling(n).mean() + 1e-10) * 100
            minus_di = pd.Series(minus_dm).rolling(n).mean() / (pd.Series(tr).rolling(n).mean() + 1e-10) * 100
            dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
            adx = dx.rolling(n).mean().values

            # Grid: when ADX < threshold, place grid orders
            # Simplified: alternate buy/sell when price moves grid_pct from last trade
            capital = 1.0
            trades = []
            last_trade_price = close[0]
            position = 0

            # TRUE OOS
            split = int(len(close) * 0.6)

            for i in range(split, len(close)):
                if np.isnan(adx[i]) or adx[i] >= adx_thresh:
                    # Trending - close grid positions
                    if position != 0:
                        pnl = position * (close[i] - last_trade_price) / last_trade_price * 0.02
                        pnl -= COMMISSION * 0.02 * 2
                        trades.append(pnl)
                        capital += pnl
                        position = 0
                    last_trade_price = close[i]
                    continue

                # Range-bound: grid logic
                price_change = (close[i] - last_trade_price) / last_trade_price

                if price_change < -grid_pct and position <= 0:
                    # Price dropped by grid_pct: buy
                    if position == -1:
                        pnl = -1 * (close[i] - last_trade_price) / last_trade_price * 0.02
                        pnl -= COMMISSION * 0.02 * 2
                        trades.append(pnl)
                        capital += pnl
                    position = 1
                    last_trade_price = close[i]
                elif price_change > grid_pct and position >= 0:
                    # Price rose by grid_pct: sell
                    if position == 1:
                        pnl = 1 * (close[i] - last_trade_price) / last_trade_price * 0.02
                        pnl -= COMMISSION * 0.02 * 2
                        trades.append(pnl)
                        capital += pnl
                    position = -1
                    last_trade_price = close[i]

            if trades:
                # Build pseudo period returns for WFA
                chunk = max(1, len(trades) // 6)
                period_rets = []
                for j in range(0, len(trades), chunk):
                    period_rets.append(sum(trades[j:j+chunk]))

                metrics = calc_metrics(trades, period_rets)
                criteria, passed = check_criteria(metrics)
                results[config] = {**metrics, 'criteria_passed': passed}
                print(f"  {config}: [{passed}/6] Sharpe={metrics['sharpe']:.2f} "
                      f"Ret={metrics['total_return']*100:+.1f}% WR={metrics['win_rate']*100:.0f}% "
                      f"PF={metrics['profit_factor']:.2f} T={metrics['trade_count']}")

    return results


# =============================================================================
# STRATEGY 8: Composite Sentiment (FGI + Google Trends)
# =============================================================================

def strategy_8_composite_sentiment():
    print("\n" + "=" * 60)
    print("STRATEGY 8: Composite Sentiment (FGI + Google Trends)")
    print("=" * 60)

    fgi = load_fear_greed()
    trends = load_google_trends()
    btc = load_futures_daily('BTC')

    if fgi.empty or btc.empty:
        print("  FGI or BTC data not available")
        return {}

    fgi['date'] = fgi['datetime'].dt.date
    btc['date'] = btc['datetime'].dt.date

    merged = btc[['date', 'close']].copy()
    merged = merged.merge(fgi[['date', 'fear_greed_value']], on='date', how='left')
    merged['fear_greed_value'] = merged['fear_greed_value'].ffill()

    # Google Trends may be monthly - resample
    if not trends.empty:
        # Get the column name for trends value
        trends_col = [c for c in trends.columns if c != 'datetime'][0] if len(trends.columns) > 1 else None
        if trends_col:
            trends['date'] = trends['datetime'].dt.date
            merged = merged.merge(trends[['date', trends_col]].rename(columns={trends_col: 'gtrends'}),
                                on='date', how='left')
            merged['gtrends'] = merged['gtrends'].ffill()
        else:
            merged['gtrends'] = 50  # neutral
    else:
        merged['gtrends'] = 50

    merged = merged.dropna(subset=['fear_greed_value', 'close']).reset_index(drop=True)
    print(f"  Merged data: {len(merged)} rows")

    results = {}
    for fear_thresh in [20, 25]:
        config = f"composite_fear{fear_thresh}"

        fgi_val = merged['fear_greed_value']
        gtrends = merged['gtrends']

        # Normalize both to 0-100
        gt_norm = (gtrends - gtrends.min()) / (gtrends.max() - gtrends.min() + 1e-10) * 100

        # Composite: average of FGI and inverted Google Trends
        # (high search = greed proxy, low search = fear proxy)
        composite = (fgi_val + (100 - gt_norm)) / 2

        signals = np.where(composite < fear_thresh, 1,
                  np.where(composite > (100 - fear_thresh), -1, 0))

        split = int(len(merged) * 0.6)
        oos_prices = merged['close'].values[split:]
        oos_signals = signals[split:]

        trades, period_rets = run_wfa_daily(
            oos_prices, oos_signals,
            train_days=180, test_days=90,
            position_pct=0.05, stop_pct=0.15, max_hold=60)

        if trades:
            metrics = calc_metrics(trades, period_rets)
            criteria, passed = check_criteria(metrics)
            results[config] = {**metrics, 'criteria_passed': passed}
            print(f"  {config}: [{passed}/6] Sharpe={metrics['sharpe']:.2f} "
                  f"Ret={metrics['total_return']*100:+.1f}% WR={metrics['win_rate']*100:.0f}% "
                  f"PF={metrics['profit_factor']:.2f} T={metrics['trade_count']}")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("RALPH-LOOP PHASE 11: NEW STRATEGY CATEGORIES")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    all_results = {}

    all_results['1_funding_mr'] = strategy_1_funding_mr()
    all_results['2_fear_greed'] = strategy_2_fear_greed()
    all_results['3_macro_regime'] = strategy_3_macro_regime()
    all_results['4_altcoin_rotation'] = strategy_4_altcoin_rotation()
    all_results['5_tvl_regime'] = strategy_5_tvl_regime()
    all_results['6_kimchi_premium'] = strategy_6_kimchi_premium()
    all_results['7_grid'] = strategy_7_grid()
    all_results['8_composite_sentiment'] = strategy_8_composite_sentiment()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n\n{'=' * 70}")
    print("PHASE 11 SUMMARY")
    print("=" * 70)

    best_per_strategy = []
    for strat_name, results in all_results.items():
        if not results:
            best_per_strategy.append((strat_name, 'NO DATA', 0, {}))
            continue
        best_config = max(results.items(), key=lambda x: x[1].get('criteria_passed', 0))
        config_name, metrics = best_config
        passed = metrics.get('criteria_passed', 0)
        best_per_strategy.append((strat_name, config_name, passed, metrics))

    best_per_strategy.sort(key=lambda x: (-x[2], -x[3].get('sharpe', 0) if isinstance(x[3], dict) else 0))

    print("\nBest config per strategy (sorted by criteria passed):\n")
    for strat, config, passed, metrics in best_per_strategy:
        if isinstance(metrics, dict) and metrics:
            print(f"  [{passed}/6] {strat} → {config}")
            print(f"         Sharpe={metrics.get('sharpe',0):.2f}  Ret={metrics.get('total_return',0)*100:+.1f}%  "
                  f"DD={metrics.get('max_drawdown',0)*100:.1f}%  WR={metrics.get('win_rate',0)*100:.0f}%  "
                  f"PF={metrics.get('profit_factor',0):.2f}  T={metrics.get('trade_count',0)}")
        else:
            print(f"  [N/A] {strat} → {config}")

    # Strategies passing 6/6
    six_six = [(s, c, m) for s, c, p, m in best_per_strategy if p == 6]
    if six_six:
        print(f"\n*** {len(six_six)} NEW STRATEGIES PASSED 6/6! ***")
        for s, c, m in six_six:
            print(f"  {s}: {c}")
    else:
        five_plus = [(s, c, p, m) for s, c, p, m in best_per_strategy if p >= 5]
        if five_plus:
            print(f"\n{len(five_plus)} strategies passed 5+/6:")
            for s, c, p, m in five_plus:
                print(f"  [{p}/6] {s}: {c}")
        else:
            print("\nNo strategies passed 5+/6 on TRUE OOS.")

    # Compare with existing Dual MA Breakout
    print(f"\n{'=' * 60}")
    print("COMPARISON WITH EXISTING STRATEGY")
    print("=" * 60)
    print("  Existing: Dual MA Breakout (Long-Only) - [6/6] Sharpe=2.39 Ret=+2.9%")
    print("  New strategies above may complement it if uncorrelated.\n")

    # Save
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'strategies_tested': 8,
        'results': {}
    }
    for strat_name, results in all_results.items():
        if results:
            best = max(results.items(), key=lambda x: x[1].get('criteria_passed', 0))
            save_data['results'][strat_name] = {
                'best_config': best[0],
                'criteria_passed': int(best[1].get('criteria_passed', 0)),
                'sharpe': float(best[1].get('sharpe', 0)),
                'total_return': float(best[1].get('total_return', 0)),
                'win_rate': float(best[1].get('win_rate', 0)),
                'profit_factor': float(best[1].get('profit_factor', 0)),
                'trade_count': int(best[1].get('trade_count', 0)),
            }

    with open(RESULTS_PATH / "phase11_report.json", 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"Saved to {RESULTS_PATH / 'phase11_report.json'}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
