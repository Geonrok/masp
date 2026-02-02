#!/usr/bin/env python3
"""
Ralph-Loop Phase 2: Feature Engineering
========================================
Task 2.1 ~ 2.6: Create comprehensive feature set for strategy development
"""
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Paths
DATA_ROOT = Path("E:/data/crypto_ohlcv")
STATE_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/ralph_loop_state.json")
RESULTS_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/results")
FEATURE_PATH = Path("E:/투자/Multi-Asset Strategy Platform/research/features")

RESULTS_PATH.mkdir(exist_ok=True)
FEATURE_PATH.mkdir(exist_ok=True)

def load_state():
    return json.loads(STATE_PATH.read_text(encoding='utf-8'))

def save_state(state):
    state["last_updated"] = datetime.now().isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')

def load_ohlcv(symbol: str, timeframe: str = "4h") -> pd.DataFrame:
    """Load OHLCV data for a symbol"""
    tf_map = {"1h": "binance_futures_1h", "4h": "binance_futures_4h", "1d": "binance_futures_1d"}
    path = DATA_ROOT / tf_map.get(timeframe, "binance_futures_4h") / f"{symbol}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    for col in ['datetime', 'timestamp', 'date']:
        if col in df.columns:
            df['datetime'] = pd.to_datetime(df[col])
            break

    df = df.sort_values('datetime').reset_index(drop=True)
    return df

# =============================================================================
# Task 2.1: Price-based Features
# =============================================================================
def task_2_1_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price-based features:
    - Returns (various periods)
    - Volatility measures
    - Moving averages
    - ATR, Bollinger Bands
    """
    features = pd.DataFrame(index=df.index)

    close = df['close']
    high = df['high']
    low = df['low']

    # === Returns (multiple periods) ===
    periods = [1, 4, 6, 12, 18, 24, 42, 84, 168]  # 4h bars: 1=4h, 6=1d, 42=7d, 168=28d
    for p in periods:
        features[f'ret_{p}'] = close.pct_change(p)

    # === Volatility Measures ===
    # Realized volatility (rolling std of returns)
    for window in [6, 24, 42]:  # 1d, 4d, 7d
        features[f'vol_realized_{window}'] = close.pct_change().rolling(window).std()

    # Parkinson volatility (uses high-low range)
    log_hl = np.log(high / low)
    for window in [6, 24, 42]:
        features[f'vol_parkinson_{window}'] = np.sqrt(
            (1 / (4 * np.log(2))) * (log_hl ** 2).rolling(window).mean()
        )

    # Garman-Klass volatility
    open_price = df['open']
    log_hl_sq = (np.log(high / low)) ** 2
    log_co_sq = (np.log(close / open_price)) ** 2
    for window in [6, 24, 42]:
        gk = 0.5 * log_hl_sq - (2 * np.log(2) - 1) * log_co_sq
        features[f'vol_gk_{window}'] = np.sqrt(gk.rolling(window).mean())

    # === Moving Averages ===
    # SMA
    for window in [6, 12, 24, 50, 100, 200]:
        features[f'sma_{window}'] = close.rolling(window).mean()
        features[f'sma_{window}_dist'] = (close - features[f'sma_{window}']) / features[f'sma_{window}']

    # EMA
    for window in [6, 12, 24, 50, 100, 200]:
        features[f'ema_{window}'] = close.ewm(span=window, adjust=False).mean()
        features[f'ema_{window}_dist'] = (close - features[f'ema_{window}']) / features[f'ema_{window}']

    # DEMA (Double EMA)
    for window in [12, 24, 50]:
        ema1 = close.ewm(span=window, adjust=False).mean()
        ema2 = ema1.ewm(span=window, adjust=False).mean()
        features[f'dema_{window}'] = 2 * ema1 - ema2

    # TEMA (Triple EMA)
    for window in [12, 24, 50]:
        ema1 = close.ewm(span=window, adjust=False).mean()
        ema2 = ema1.ewm(span=window, adjust=False).mean()
        ema3 = ema2.ewm(span=window, adjust=False).mean()
        features[f'tema_{window}'] = 3 * ema1 - 3 * ema2 + ema3

    # Hull MA
    for window in [12, 24, 50]:
        half_window = window // 2
        sqrt_window = int(np.sqrt(window))
        wma_half = close.rolling(half_window).apply(
            lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
        )
        wma_full = close.rolling(window).apply(
            lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
        )
        hull_raw = 2 * wma_half - wma_full
        features[f'hull_{window}'] = hull_raw.rolling(sqrt_window).apply(
            lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
        )

    # KAMA (Kaufman Adaptive MA)
    for window in [12, 24]:
        change = abs(close - close.shift(window))
        volatility = abs(close - close.shift(1)).rolling(window).sum()
        er = change / (volatility + 1e-10)  # Efficiency Ratio

        fast_sc = 2 / (2 + 1)
        slow_sc = 2 / (30 + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[window] = close.iloc[window]
        for i in range(window + 1, len(close)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
        features[f'kama_{window}'] = kama

    # === ATR (Average True Range) ===
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)

    for window in [6, 14, 24]:
        features[f'atr_{window}'] = tr.rolling(window).mean()
        features[f'atr_{window}_pct'] = features[f'atr_{window}'] / close

    # === Bollinger Bands ===
    for window in [12, 24, 50]:
        sma = close.rolling(window).mean()
        std = close.rolling(window).std()
        features[f'bb_upper_{window}'] = sma + 2 * std
        features[f'bb_lower_{window}'] = sma - 2 * std
        features[f'bb_width_{window}'] = (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}']) / sma
        features[f'bb_pct_{window}'] = (close - features[f'bb_lower_{window}']) / (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}'] + 1e-10)

    # === Donchian Channel ===
    for window in [12, 24, 50]:
        features[f'donchian_high_{window}'] = high.rolling(window).max()
        features[f'donchian_low_{window}'] = low.rolling(window).min()
        features[f'donchian_mid_{window}'] = (features[f'donchian_high_{window}'] + features[f'donchian_low_{window}']) / 2
        features[f'donchian_pct_{window}'] = (close - features[f'donchian_low_{window}']) / (features[f'donchian_high_{window}'] - features[f'donchian_low_{window}'] + 1e-10)

    # === RSI ===
    for window in [6, 14, 24]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-10)
        features[f'rsi_{window}'] = 100 - (100 / (1 + rs))

    # === MACD ===
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']

    # === Momentum ===
    for period in [6, 12, 24]:
        features[f'momentum_{period}'] = close / close.shift(period) - 1

    # === Rate of Change ===
    for period in [6, 12, 24]:
        features[f'roc_{period}'] = (close - close.shift(period)) / close.shift(period)

    return features

# =============================================================================
# Task 2.2: Volume-based Features
# =============================================================================
def task_2_2_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate volume-based features:
    - VWAP deviation
    - Volume momentum (OBV, CMF, MFI)
    - Volume profile
    """
    features = pd.DataFrame(index=df.index)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # === VWAP (Volume Weighted Average Price) ===
    typical_price = (high + low + close) / 3
    for window in [6, 24, 42]:
        cum_vol = volume.rolling(window).sum()
        cum_tp_vol = (typical_price * volume).rolling(window).sum()
        features[f'vwap_{window}'] = cum_tp_vol / (cum_vol + 1e-10)
        features[f'vwap_{window}_dist'] = (close - features[f'vwap_{window}']) / features[f'vwap_{window}']

    # === OBV (On-Balance Volume) ===
    obv = pd.Series(0, index=df.index, dtype=float)
    obv_sign = np.sign(close.diff())
    obv = (obv_sign * volume).cumsum()
    features['obv'] = obv

    # OBV momentum
    for window in [6, 12, 24]:
        features[f'obv_momentum_{window}'] = obv.pct_change(window)
        features[f'obv_sma_{window}'] = obv.rolling(window).mean()

    # === CMF (Chaikin Money Flow) ===
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
    mfv = mfm * volume
    for window in [12, 24]:
        features[f'cmf_{window}'] = mfv.rolling(window).sum() / (volume.rolling(window).sum() + 1e-10)

    # === MFI (Money Flow Index) ===
    for window in [6, 14, 24]:
        raw_mf = typical_price * volume
        mf_sign = np.sign(typical_price.diff())
        pos_mf = raw_mf.where(mf_sign > 0, 0).rolling(window).sum()
        neg_mf = raw_mf.where(mf_sign < 0, 0).rolling(window).sum()
        mfr = pos_mf / (neg_mf + 1e-10)
        features[f'mfi_{window}'] = 100 - (100 / (1 + mfr))

    # === Volume momentum ===
    for window in [6, 12, 24]:
        vol_sma = volume.rolling(window).mean()
        features[f'volume_ratio_{window}'] = volume / (vol_sma + 1e-10)
        features[f'volume_std_{window}'] = volume.rolling(window).std() / (vol_sma + 1e-10)

    # === A/D Line (Accumulation/Distribution) ===
    ad = ((close - low) - (high - close)) / (high - low + 1e-10) * volume
    features['ad_line'] = ad.cumsum()
    for window in [6, 12, 24]:
        features[f'ad_momentum_{window}'] = features['ad_line'].pct_change(window)

    # === Force Index ===
    for window in [6, 13, 24]:
        features[f'force_index_{window}'] = (close.diff() * volume).ewm(span=window, adjust=False).mean()

    # === Ease of Movement ===
    dm = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    br = volume / (high - low + 1e-10)
    eom = dm / (br + 1e-10)
    for window in [6, 14, 24]:
        features[f'eom_{window}'] = eom.rolling(window).mean()

    return features

# =============================================================================
# Task 2.3: Derivatives-specific Features
# =============================================================================
def task_2_3_derivatives_features(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derivatives-specific features:
    - Funding rate
    - Open interest
    - Long/short ratio
    - Basis (futures vs spot)
    """
    features = pd.DataFrame(index=df.index)
    dt_col = df['datetime']

    # === Funding Rate ===
    fr_path = DATA_ROOT / "binance_funding_rate" / f"{symbol}.csv"
    if fr_path.exists():
        try:
            fr_df = pd.read_csv(fr_path)
            for col in ['datetime', 'timestamp', 'fundingTime']:
                if col in fr_df.columns:
                    fr_df['datetime'] = pd.to_datetime(fr_df[col])
                    break

            # Find funding rate column
            fr_col = None
            for col in ['fundingRate', 'funding_rate', 'rate']:
                if col in fr_df.columns:
                    fr_col = col
                    break

            if fr_col:
                fr_df = fr_df.sort_values('datetime')
                fr_df = fr_df.set_index('datetime')

                # Merge with OHLCV
                merged = df.set_index('datetime')
                merged['funding_rate'] = fr_df[fr_col].reindex(merged.index, method='ffill')

                features['funding_rate'] = merged['funding_rate'].values

                # Funding rate features
                fr = features['funding_rate']
                for window in [8, 24, 42]:  # 8 = ~1 day (3 funding periods), 24 = ~3 days
                    features[f'fr_cumulative_{window}'] = fr.rolling(window).sum()
                    features[f'fr_zscore_{window}'] = (fr - fr.rolling(window).mean()) / (fr.rolling(window).std() + 1e-10)
                    features[f'fr_abs_mean_{window}'] = fr.abs().rolling(window).mean()
        except Exception as e:
            pass

    # === Open Interest ===
    oi_path = DATA_ROOT / "binance_open_interest" / f"{symbol}.csv"
    if oi_path.exists():
        try:
            oi_df = pd.read_csv(oi_path)
            for col in ['datetime', 'timestamp']:
                if col in oi_df.columns:
                    oi_df['datetime'] = pd.to_datetime(oi_df[col])
                    break

            oi_col = None
            for col in ['sumOpenInterest', 'openInterest', 'oi']:
                if col in oi_df.columns:
                    oi_col = col
                    break

            if oi_col:
                oi_df = oi_df.sort_values('datetime')
                oi_df = oi_df.set_index('datetime')

                merged = df.set_index('datetime')
                merged['open_interest'] = oi_df[oi_col].reindex(merged.index, method='ffill')

                features['open_interest'] = merged['open_interest'].values

                oi = features['open_interest']
                for window in [6, 24, 42]:
                    features[f'oi_change_{window}'] = oi.pct_change(window)
                    features[f'oi_zscore_{window}'] = (oi - oi.rolling(window).mean()) / (oi.rolling(window).std() + 1e-10)
        except Exception as e:
            pass

    # === Long/Short Ratio ===
    lsr_path = DATA_ROOT / "binance_long_short_ratio" / f"{symbol}.csv"
    if lsr_path.exists():
        try:
            lsr_df = pd.read_csv(lsr_path)
            for col in ['datetime', 'timestamp']:
                if col in lsr_df.columns:
                    lsr_df['datetime'] = pd.to_datetime(lsr_df[col])
                    break

            lsr_col = None
            for col in ['longShortRatio', 'ratio']:
                if col in lsr_df.columns:
                    lsr_col = col
                    break

            if lsr_col:
                lsr_df = lsr_df.sort_values('datetime')
                lsr_df = lsr_df.set_index('datetime')

                merged = df.set_index('datetime')
                merged['long_short_ratio'] = lsr_df[lsr_col].reindex(merged.index, method='ffill')

                features['long_short_ratio'] = merged['long_short_ratio'].values

                lsr = features['long_short_ratio']
                for window in [6, 24, 42]:
                    features[f'lsr_zscore_{window}'] = (lsr - lsr.rolling(window).mean()) / (lsr.rolling(window).std() + 1e-10)
                    features[f'lsr_change_{window}'] = lsr.pct_change(window)
        except Exception as e:
            pass

    # === Taker Volume ===
    tv_path = DATA_ROOT / "binance_taker_volume" / f"{symbol}.csv"
    if tv_path.exists():
        try:
            tv_df = pd.read_csv(tv_path)
            for col in ['datetime', 'timestamp']:
                if col in tv_df.columns:
                    tv_df['datetime'] = pd.to_datetime(tv_df[col])
                    break

            if 'buyVol' in tv_df.columns and 'sellVol' in tv_df.columns:
                tv_df['taker_ratio'] = tv_df['buyVol'] / (tv_df['buyVol'] + tv_df['sellVol'] + 1e-10)
            elif 'takerBuyVol' in tv_df.columns:
                tv_df['taker_ratio'] = tv_df['takerBuyVol'] / (tv_df['takerBuyVol'] + tv_df.get('takerSellVol', 0) + 1e-10)

            if 'taker_ratio' in tv_df.columns:
                tv_df = tv_df.sort_values('datetime')
                tv_df = tv_df.set_index('datetime')

                merged = df.set_index('datetime')
                merged['taker_ratio'] = tv_df['taker_ratio'].reindex(merged.index, method='ffill')

                features['taker_ratio'] = merged['taker_ratio'].values

                tr = features['taker_ratio']
                for window in [6, 24, 42]:
                    features[f'taker_imbalance_{window}'] = (tr - 0.5).rolling(window).mean()
                    features[f'taker_zscore_{window}'] = (tr - tr.rolling(window).mean()) / (tr.rolling(window).std() + 1e-10)
        except Exception as e:
            pass

    # === Basis (Futures vs Spot) ===
    spot_path = DATA_ROOT / "binance_spot_4h" / f"{symbol}.csv"
    if spot_path.exists():
        try:
            spot_df = pd.read_csv(spot_path)
            for col in ['datetime', 'timestamp', 'date']:
                if col in spot_df.columns:
                    spot_df['datetime'] = pd.to_datetime(spot_df[col])
                    break

            spot_df = spot_df.sort_values('datetime')
            spot_df = spot_df.set_index('datetime')

            merged = df.set_index('datetime')
            merged['spot_close'] = spot_df['close'].reindex(merged.index, method='ffill')

            features['basis'] = ((df['close'].values - merged['spot_close'].values) / merged['spot_close'].values)

            basis = features['basis']
            for window in [6, 24, 42]:
                features[f'basis_zscore_{window}'] = (basis - basis.rolling(window).mean()) / (basis.rolling(window).std() + 1e-10)
                features[f'basis_momentum_{window}'] = basis.diff(window)
        except Exception as e:
            pass

    return features

# =============================================================================
# Task 2.4: Cross-asset Features
# =============================================================================
def task_2_4_cross_asset_features(symbol: str, df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cross-asset features:
    - BTC correlation
    - Market beta
    - Relative strength
    """
    features = pd.DataFrame(index=df.index)

    if btc_df.empty:
        return features

    # Align BTC data to symbol's datetime index
    df_aligned = df.set_index('datetime')
    btc_aligned = btc_df.set_index('datetime')
    btc_close_aligned = btc_aligned['close'].reindex(df_aligned.index, method='ffill')

    if btc_close_aligned.isna().all():
        return features

    # Reset index for consistent integer indexing
    close = df['close'].reset_index(drop=True)
    btc_close = btc_close_aligned.reset_index(drop=True)

    # === BTC Correlation ===
    for window in [24, 42, 84]:
        ret_symbol = close.pct_change()
        ret_btc = btc_close.pct_change()
        features[f'btc_corr_{window}'] = ret_symbol.rolling(window).corr(ret_btc).values

    # === Beta to BTC ===
    for window in [24, 42, 84]:
        ret_symbol = close.pct_change()
        ret_btc = btc_close.pct_change()
        cov = ret_symbol.rolling(window).cov(ret_btc)
        var_btc = ret_btc.rolling(window).var()
        features[f'btc_beta_{window}'] = (cov / (var_btc + 1e-10)).values

    # === Relative Strength vs BTC ===
    for period in [6, 24, 42]:
        ret_symbol = close.pct_change(period)
        ret_btc = btc_close.pct_change(period)
        features[f'rel_strength_btc_{period}'] = (ret_symbol - ret_btc).values

    # === Spread (log ratio) ===
    log_ratio = np.log(close / (btc_close + 1e-10))
    features['btc_spread'] = log_ratio.values
    for window in [24, 42]:
        features[f'btc_spread_zscore_{window}'] = (
            (log_ratio - log_ratio.rolling(window).mean()) / (log_ratio.rolling(window).std() + 1e-10)
        ).values

    return features

# =============================================================================
# Task 2.5: Alternative Data Features
# =============================================================================
def task_2_5_alternative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate alternative data features:
    - Fear/Greed index
    - On-chain metrics
    - Social sentiment
    """
    features = pd.DataFrame(index=df.index)
    dt_col = df['datetime']

    # === Fear & Greed Index ===
    fg_path = DATA_ROOT / "sentiment" / "fear_greed.csv"
    if fg_path.exists():
        try:
            fg_df = pd.read_csv(fg_path)
            for col in ['datetime', 'timestamp', 'date']:
                if col in fg_df.columns:
                    fg_df['datetime'] = pd.to_datetime(fg_df[col])
                    break

            fg_col = None
            for col in ['value', 'fear_greed', 'index']:
                if col in fg_df.columns:
                    fg_col = col
                    break

            if fg_col:
                fg_df = fg_df.sort_values('datetime')
                fg_df = fg_df.set_index('datetime')

                merged = df.set_index('datetime')
                merged['fear_greed'] = fg_df[fg_col].reindex(merged.index, method='ffill')

                features['fear_greed'] = merged['fear_greed'].values

                fg = features['fear_greed']
                features['fg_regime'] = pd.cut(fg, bins=[0, 25, 45, 55, 75, 100],
                                               labels=[1, 2, 3, 4, 5]).astype(float)

                for window in [6, 24, 42]:
                    features[f'fg_zscore_{window}'] = (fg - fg.rolling(window).mean()) / (fg.rolling(window).std() + 1e-10)
                    features[f'fg_momentum_{window}'] = fg.diff(window)
        except Exception as e:
            pass

    # === DeFi TVL ===
    tvl_path = DATA_ROOT / "defillama" / "total_tvl.csv"
    if tvl_path.exists():
        try:
            tvl_df = pd.read_csv(tvl_path)
            for col in ['datetime', 'timestamp', 'date']:
                if col in tvl_df.columns:
                    tvl_df['datetime'] = pd.to_datetime(tvl_df[col])
                    break

            tvl_col = None
            for col in ['tvl', 'totalLiquidity', 'value']:
                if col in tvl_df.columns:
                    tvl_col = col
                    break

            if tvl_col:
                tvl_df = tvl_df.sort_values('datetime')
                tvl_df = tvl_df.set_index('datetime')

                merged = df.set_index('datetime')
                merged['defi_tvl'] = tvl_df[tvl_col].reindex(merged.index, method='ffill')

                features['defi_tvl'] = merged['defi_tvl'].values

                tvl = features['defi_tvl']
                for window in [6, 24, 42]:
                    features[f'tvl_change_{window}'] = tvl.pct_change(window)
        except Exception as e:
            pass

    return features

# =============================================================================
# Task 2.6: Macro Features
# =============================================================================
def task_2_6_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate macro features:
    - DXY momentum
    - US10Y yield
    - SPX correlation regime
    """
    features = pd.DataFrame(index=df.index)

    macro_path = DATA_ROOT / "macro"

    # === DXY (Dollar Index) ===
    dxy_path = macro_path / "DXY.csv"
    if dxy_path.exists():
        try:
            dxy_df = pd.read_csv(dxy_path)
            for col in ['datetime', 'timestamp', 'date', 'Date']:
                if col in dxy_df.columns:
                    dxy_df['datetime'] = pd.to_datetime(dxy_df[col])
                    break

            close_col = None
            for col in ['close', 'Close', 'value']:
                if col in dxy_df.columns:
                    close_col = col
                    break

            if close_col:
                dxy_df = dxy_df.sort_values('datetime')
                dxy_df = dxy_df.set_index('datetime')

                merged = df.set_index('datetime')
                merged['dxy'] = dxy_df[close_col].reindex(merged.index, method='ffill')

                features['dxy'] = merged['dxy'].values

                dxy = features['dxy']
                for window in [6, 24, 42]:
                    features[f'dxy_momentum_{window}'] = dxy.pct_change(window)
                    features[f'dxy_zscore_{window}'] = (dxy - dxy.rolling(window).mean()) / (dxy.rolling(window).std() + 1e-10)
        except Exception as e:
            pass

    # === US10Y Yield ===
    us10y_path = macro_path / "US10Y.csv"
    if us10y_path.exists():
        try:
            us10y_df = pd.read_csv(us10y_path)
            for col in ['datetime', 'timestamp', 'date', 'Date']:
                if col in us10y_df.columns:
                    us10y_df['datetime'] = pd.to_datetime(us10y_df[col])
                    break

            close_col = None
            for col in ['close', 'Close', 'value', 'yield']:
                if col in us10y_df.columns:
                    close_col = col
                    break

            if close_col:
                us10y_df = us10y_df.sort_values('datetime')
                us10y_df = us10y_df.set_index('datetime')

                merged = df.set_index('datetime')
                merged['us10y'] = us10y_df[close_col].reindex(merged.index, method='ffill')

                features['us10y'] = merged['us10y'].values

                us10y = features['us10y']
                for window in [6, 24, 42]:
                    features[f'us10y_change_{window}'] = us10y.diff(window)
        except Exception as e:
            pass

    # === SPX (S&P 500) ===
    spx_path = macro_path / "SPX.csv"
    if not spx_path.exists():
        spx_path = macro_path / "SP500.csv"

    if spx_path.exists():
        try:
            spx_df = pd.read_csv(spx_path)
            for col in ['datetime', 'timestamp', 'date', 'Date']:
                if col in spx_df.columns:
                    spx_df['datetime'] = pd.to_datetime(spx_df[col])
                    break

            close_col = None
            for col in ['close', 'Close', 'value']:
                if col in spx_df.columns:
                    close_col = col
                    break

            if close_col:
                spx_df = spx_df.sort_values('datetime')
                spx_df = spx_df.set_index('datetime')

                merged = df.set_index('datetime')
                merged['spx'] = spx_df[close_col].reindex(merged.index, method='ffill')

                features['spx'] = merged['spx'].values

                spx = features['spx']
                close = df['close']

                for window in [24, 42, 84]:
                    ret_close = close.pct_change()
                    ret_spx = spx.pct_change()
                    features[f'spx_corr_{window}'] = ret_close.rolling(window).corr(ret_spx)
                    features[f'spx_momentum_{window}'] = spx.pct_change(window)
        except Exception as e:
            pass

    return features

# =============================================================================
# Main Feature Engineering Pipeline
# =============================================================================
def generate_features_for_symbol(symbol: str, timeframe: str = "4h") -> pd.DataFrame:
    """Generate all features for a single symbol"""
    df = load_ohlcv(symbol, timeframe)
    if df.empty:
        return pd.DataFrame()

    # Load BTC for cross-asset features
    btc_df = load_ohlcv("BTCUSDT", timeframe)

    # Generate all feature sets
    price_feat = task_2_1_price_features(df)
    volume_feat = task_2_2_volume_features(df)
    deriv_feat = task_2_3_derivatives_features(symbol, df)
    cross_feat = task_2_4_cross_asset_features(symbol, df, btc_df)
    alt_feat = task_2_5_alternative_features(df)
    macro_feat = task_2_6_macro_features(df)

    # Combine all features
    all_features = pd.concat([
        df[['datetime', 'open', 'high', 'low', 'close', 'volume']],
        price_feat,
        volume_feat,
        deriv_feat,
        cross_feat,
        alt_feat,
        macro_feat
    ], axis=1)

    return all_features

def main():
    print("=" * 70)
    print("RALPH-LOOP PHASE 2: FEATURE ENGINEERING")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    state = load_state()

    # Get Tier 1 symbols from state
    tier_1_symbols = state.get('findings', {}).get('data_quality', {}).get('tier_1_symbols', [])
    if not tier_1_symbols:
        tier_1_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
                         'BNBUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'LTCUSDT']

    print(f"\nProcessing {len(tier_1_symbols)} Tier 1 symbols...")

    feature_counts = {}
    sample_features = None

    for symbol in tier_1_symbols:
        print(f"\n  Processing {symbol}...")

        features = generate_features_for_symbol(symbol)

        if features.empty:
            print(f"    Skipped (no data)")
            continue

        # Count features by category
        price_cols = [c for c in features.columns if any(x in c for x in ['ret_', 'vol_', 'sma_', 'ema_', 'atr_', 'bb_', 'rsi_', 'macd', 'momentum', 'roc_', 'donchian', 'dema', 'tema', 'hull', 'kama'])]
        volume_cols = [c for c in features.columns if any(x in c for x in ['vwap', 'obv', 'cmf', 'mfi', 'volume_', 'ad_', 'force', 'eom'])]
        deriv_cols = [c for c in features.columns if any(x in c for x in ['funding', 'fr_', 'open_interest', 'oi_', 'long_short', 'lsr_', 'taker', 'basis'])]
        cross_cols = [c for c in features.columns if any(x in c for x in ['btc_corr', 'btc_beta', 'rel_strength', 'btc_spread'])]
        alt_cols = [c for c in features.columns if any(x in c for x in ['fear_greed', 'fg_', 'defi', 'tvl'])]
        macro_cols = [c for c in features.columns if any(x in c for x in ['dxy', 'us10y', 'spx'])]

        feature_counts[symbol] = {
            'total': len(features.columns),
            'price': len(price_cols),
            'volume': len(volume_cols),
            'derivatives': len(deriv_cols),
            'cross_asset': len(cross_cols),
            'alternative': len(alt_cols),
            'macro': len(macro_cols),
            'rows': len(features),
            'non_null_pct': (features.notna().sum().sum() / (len(features) * len(features.columns))) * 100
        }

        print(f"    Features: {feature_counts[symbol]['total']} total, {feature_counts[symbol]['rows']} rows")
        print(f"    Non-null: {feature_counts[symbol]['non_null_pct']:.1f}%")

        # Save features
        output_path = FEATURE_PATH / f"{symbol}_features.parquet"
        features.to_parquet(output_path, index=False)
        print(f"    Saved: {output_path.name}")

        if sample_features is None:
            sample_features = features

    # Generate summary report
    summary = {
        'generated_at': datetime.now().isoformat(),
        'symbols_processed': len(feature_counts),
        'feature_counts': feature_counts,
        'feature_categories': {
            'price_based': list(set([c for c in sample_features.columns if any(x in c for x in ['ret_', 'vol_', 'sma_', 'ema_', 'atr_', 'bb_', 'rsi_', 'macd', 'momentum', 'roc_'])])),
            'volume_based': list(set([c for c in sample_features.columns if any(x in c for x in ['vwap', 'obv', 'cmf', 'mfi', 'volume_'])])),
            'derivatives': list(set([c for c in sample_features.columns if any(x in c for x in ['funding', 'fr_', 'oi_', 'lsr_', 'taker', 'basis'])])),
        }
    }

    report_path = RESULTS_PATH / "phase2_feature_report.json"
    report_path.write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')

    # Update state
    state['current_phase'] = '3'
    state['current_task'] = '3.1'
    state['completed_tasks'].extend(['2.1', '2.2', '2.3', '2.4', '2.5', '2.6'])
    state['findings']['feature_engineering'] = {
        'total_features': sum(fc['total'] for fc in feature_counts.values()) // len(feature_counts) if feature_counts else 0,
        'categories': ['price', 'volume', 'derivatives', 'cross_asset', 'alternative', 'macro'],
        'symbols_processed': len(feature_counts),
        'avg_non_null_pct': sum(fc['non_null_pct'] for fc in feature_counts.values()) / len(feature_counts) if feature_counts else 0
    }
    state['next_actions'] = ['3.1: Momentum Strategies', '3.2: Mean Reversion Strategies']

    save_state(state)

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"  Symbols processed: {len(feature_counts)}")
    if feature_counts:
        avg_features = sum(fc['total'] for fc in feature_counts.values()) // len(feature_counts)
        print(f"  Average features per symbol: {avg_features}")
        print(f"  Feature categories: 6")
    print(f"  Report saved: {report_path}")
    print(f"  Next: Phase 3 - Strategy Exploration")

    return summary


if __name__ == "__main__":
    summary = main()
