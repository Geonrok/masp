#!/usr/bin/env python3
"""
Realistic Backtest Validation
- Slippage simulation
- Funding rate costs
- Liquidity constraints
- Stress testing (drawdown scenarios)
- Market regime analysis

Also validates MF + OnChain + ML strategy with same rigor as MF + ML
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

DATA_ROOT = Path("E:/data/crypto_ohlcv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================

def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()
def calc_sma(s, p): return s.rolling(p).mean()
def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


# ============================================================================
# Realistic Cost Model
# ============================================================================

class RealisticCostModel:
    """Realistic trading cost simulation"""

    def __init__(self,
                 base_fee: float = 0.0004,      # 0.04% taker fee
                 slippage_base: float = 0.001,  # 0.1% base slippage
                 funding_rate: float = 0.0001,  # 0.01% per 8 hours avg
                 position_size_impact: float = 0.5):  # Size impact factor
        self.base_fee = base_fee
        self.slippage_base = slippage_base
        self.funding_rate = funding_rate
        self.position_size_impact = position_size_impact

    def calc_entry_cost(self, price: float, volume_24h: float,
                        position_value: float) -> float:
        """Calculate entry slippage and fees"""
        # Fee
        fee = self.base_fee

        # Slippage increases with position size relative to volume
        size_ratio = position_value / (volume_24h * price + 1e-10)
        slippage = self.slippage_base * (1 + size_ratio * self.position_size_impact * 100)
        slippage = min(slippage, 0.02)  # Cap at 2%

        return fee + slippage

    def calc_exit_cost(self, price: float, volume_24h: float,
                       position_value: float) -> float:
        """Calculate exit slippage and fees"""
        return self.calc_entry_cost(price, volume_24h, position_value)

    def calc_funding_cost(self, bars_held: int, timeframe_hours: int = 4) -> float:
        """Calculate cumulative funding rate cost"""
        # Funding is paid every 8 hours
        hours_held = bars_held * timeframe_hours
        funding_periods = hours_held / 8
        return self.funding_rate * funding_periods


# ============================================================================
# Signal Generators (Same as combined strategy)
# ============================================================================

class MultiFactorSignal:
    def generate(self, df: pd.DataFrame, btc_df: pd.DataFrame,
                 fear_greed: pd.DataFrame, macro: pd.DataFrame) -> pd.Series:
        scores = pd.Series(0.0, index=df.index)

        # Technical (weight: 1.5)
        ema20, ema50, ema200 = calc_ema(df['close'], 20), calc_ema(df['close'], 50), calc_ema(df['close'], 200)
        tech = np.where((df['close'] > ema20) & (ema20 > ema50) & (ema50 > ema200), 2,
               np.where((df['close'] > ema50), 1,
               np.where((df['close'] < ema20) & (ema20 < ema50) & (ema50 < ema200), -2,
               np.where((df['close'] < ema50), -1, 0))))

        rsi = calc_rsi(df['close'], 14)
        tech_rsi = np.where(rsi < 30, 1.5, np.where(rsi < 40, 0.5,
                   np.where(rsi > 70, -1.5, np.where(rsi > 60, -0.5, 0))))

        scores += (pd.Series(tech, index=df.index) + pd.Series(tech_rsi, index=df.index)) / 2 * 1.5

        # Fear & Greed (weight: 1.2)
        if not fear_greed.empty:
            fg = fear_greed['fear_greed'].reindex(df.index, method='ffill')
            fg_score = np.where(fg < 20, 2, np.where(fg < 35, 1,
                       np.where(fg > 80, -2, np.where(fg > 65, -1, 0))))
            scores += pd.Series(fg_score, index=df.index).fillna(0) * 1.2

        # BTC Correlation (weight: 1.0)
        if not btc_df.empty:
            btc_close = btc_df['close'].reindex(df.index, method='ffill')
            btc_ret = btc_close.pct_change(20)
            btc_ema50 = calc_ema(btc_close, 50)
            btc_ema200 = calc_ema(btc_close, 200)
            btc_uptrend = (btc_close > btc_ema50) & (btc_ema50 > btc_ema200)
            btc_downtrend = (btc_close < btc_ema50) & (btc_ema50 < btc_ema200)
            btc_score = np.where(btc_uptrend & (btc_ret > 0.1), 2,
                        np.where(btc_uptrend & (btc_ret > 0.03), 1,
                        np.where(btc_downtrend & (btc_ret < -0.1), -2,
                        np.where(btc_downtrend & (btc_ret < -0.03), -1, 0))))
            scores += pd.Series(btc_score, index=df.index).fillna(0) * 1.0

        # Macro (weight: 1.0)
        if not macro.empty:
            macro_r = macro.reindex(df.index, method='ffill')
            macro_score = pd.Series(0.0, index=df.index)
            if 'dxy' in macro_r.columns:
                dxy = macro_r['dxy']
                dxy_ma = calc_sma(dxy, 50)
                macro_score += np.where(dxy < dxy_ma * 0.98, 1, np.where(dxy > dxy_ma * 1.02, -1, 0))
            if 'vix' in macro_r.columns:
                vix = macro_r['vix']
                macro_score += np.where(vix > 30, 1, np.where(vix > 25, 0.5, np.where(vix < 15, -0.5, 0)))
            scores += macro_score.fillna(0) * 1.0

        return scores


class OnChainSignal:
    def generate(self, df: pd.DataFrame) -> pd.Series:
        scores = pd.Series(0.0, index=df.index)
        volume = df['volume']
        close = df['close']

        # Whale Activity
        vol_zscore = (volume - volume.rolling(50).mean()) / volume.rolling(50).std()
        price_change = close.pct_change().abs()
        whale_activity = vol_zscore * (1 + price_change * 10)

        price_below_ma = close < close.rolling(20).mean()
        whale_buy = (whale_activity > 2) & price_below_ma
        whale_sell = (whale_activity > 2) & ~price_below_ma

        scores += np.where(whale_buy, 2, np.where(whale_sell, -1, 0)) * 1.5

        # Exchange Flow
        price_chg = close.pct_change()
        vol_chg = volume.pct_change()
        flow = -price_chg * 10 + vol_chg
        flow_ema = flow.ewm(span=5).mean()

        scores += np.where(flow_ema < -0.5, 2,
                  np.where(flow_ema < -0.2, 1,
                  np.where(flow_ema > 0.5, -2,
                  np.where(flow_ema > 0.2, -1, 0)))) * 2.0

        # Accumulation
        price_ma = close.rolling(20).mean()
        vol_ma = volume.rolling(20).mean()
        accumulation = ((close > price_ma) & (volume < vol_ma)).astype(int) - \
                      ((close < price_ma) & (volume > vol_ma)).astype(int)
        accumulation = accumulation.rolling(10).sum()

        scores += np.where(accumulation > 5, 1.5,
                  np.where(accumulation > 2, 0.5,
                  np.where(accumulation < -5, -1.5,
                  np.where(accumulation < -2, -0.5, 0)))) * 1.0

        return scores.fillna(0)


class MLSignal:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        close = df['close']
        volume = df['volume']

        for p in [5, 10, 20, 50]:
            features[f'ret_{p}'] = close.pct_change(p)
            features[f'vol_ret_{p}'] = volume.pct_change(p)

        for p in [10, 20]:
            features[f'volatility_{p}'] = close.pct_change().rolling(p).std()

        features['rsi_14'] = calc_rsi(close, 14)

        for p in [20, 50]:
            features[f'ema_pos_{p}'] = close / calc_ema(close, p) - 1

        features['vol_zscore'] = (volume - volume.rolling(50).mean()) / volume.rolling(50).std()

        return features.replace([np.inf, -np.inf], np.nan).fillna(0)

    def train_and_predict(self, df: pd.DataFrame, train_end_idx: int) -> pd.Series:
        """Train on data up to train_end_idx, predict on all"""
        features = self._create_features(df)

        future_ret = df['close'].shift(-6) / df['close'] - 1
        target = np.where(future_ret > 0.02, 1, np.where(future_ret < -0.02, -1, 0))
        target = pd.Series(target, index=df.index)

        valid = ~(features.isna().any(axis=1) | target.isna())
        features_valid = features[valid]
        target_valid = target[valid]

        if len(features_valid) < 300:
            return pd.Series(0.0, index=df.index)

        # Train only on data before train_end_idx
        train_mask = features_valid.index <= df.index[train_end_idx]
        X_train = features_valid[train_mask]
        y_train = target_valid[train_mask]

        if len(X_train) < 200:
            return pd.Series(0.0, index=df.index)

        self.feature_names = features.columns.tolist()

        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        X_scaled = self.scaler.transform(features)
        predictions = self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)
        confidence = np.max(proba, axis=1)

        result = pd.Series(0.0, index=df.index)
        result.loc[features.index] = predictions * confidence * 3
        return result


# ============================================================================
# Realistic Backtester
# ============================================================================

class RealisticBacktester:
    def __init__(self, threshold: float = 4.0,
                 cost_model: RealisticCostModel = None,
                 max_position_pct: float = 0.2,
                 leverage: float = 2.0):
        self.threshold = threshold
        self.cost_model = cost_model or RealisticCostModel()
        self.max_position_pct = max_position_pct
        self.leverage = leverage

    def backtest(self, df: pd.DataFrame, scores: pd.Series,
                 symbol: str, test_start_idx: int) -> Dict:
        """Backtest with realistic costs, only on test period"""
        if len(df) < 200 or test_start_idx >= len(df) - 50:
            return None

        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]

        # Calculate 24h volume proxy (6 bars for 4h data)
        vol_24h = df['volume'].rolling(6).sum()

        for i in range(test_start_idx, len(df)):
            price = df['close'].iloc[i]
            score = scores.iloc[i] if i < len(scores) else 0
            if pd.isna(score):
                score = 0

            current_vol_24h = vol_24h.iloc[i] if i < len(vol_24h) else df['volume'].iloc[i] * 6

            if position:
                bars_held = i - position['entry_idx']

                # Exit conditions
                should_exit = (position['dir'] == 1 and score < 0) or \
                             (position['dir'] == -1 and score > 0) or \
                             bars_held >= 36  # Max 6 days hold

                if should_exit:
                    # Calculate PnL with realistic costs
                    gross_pnl_pct = (price / position['entry'] - 1) * position['dir']

                    # Exit costs
                    exit_cost = self.cost_model.calc_exit_cost(
                        price, current_vol_24h, position['value']
                    )

                    # Funding costs
                    funding_cost = self.cost_model.calc_funding_cost(bars_held)

                    # Net PnL
                    net_pnl_pct = gross_pnl_pct - position['entry_cost'] - exit_cost - funding_cost
                    pnl = position['value'] * net_pnl_pct

                    trades.append({
                        'pnl': pnl,
                        'gross_pnl_pct': gross_pnl_pct,
                        'entry_cost': position['entry_cost'],
                        'exit_cost': exit_cost,
                        'funding_cost': funding_cost,
                        'bars_held': bars_held,
                        'direction': position['dir']
                    })
                    capital += pnl
                    position = None

            if not position and abs(score) >= self.threshold and capital > 0:
                direction = 1 if score > 0 else -1
                mult = min(abs(score) / self.threshold, 2.0)
                position_value = capital * self.max_position_pct * self.leverage * mult

                # Entry cost
                entry_cost = self.cost_model.calc_entry_cost(
                    price, current_vol_24h, position_value
                )

                position = {
                    'entry': price,
                    'dir': direction,
                    'mult': mult,
                    'entry_idx': i,
                    'value': position_value,
                    'entry_cost': entry_cost
                }

            equity.append(max(capital, 0))

        # Close any remaining position
        if position:
            price = df['close'].iloc[-1]
            bars_held = len(df) - 1 - position['entry_idx']
            gross_pnl_pct = (price / position['entry'] - 1) * position['dir']
            exit_cost = self.cost_model.calc_exit_cost(price, vol_24h.iloc[-1], position['value'])
            funding_cost = self.cost_model.calc_funding_cost(bars_held)
            net_pnl_pct = gross_pnl_pct - position['entry_cost'] - exit_cost - funding_cost
            trades.append({
                'pnl': position['value'] * net_pnl_pct,
                'gross_pnl_pct': gross_pnl_pct,
                'entry_cost': position['entry_cost'],
                'exit_cost': exit_cost,
                'funding_cost': funding_cost,
                'bars_held': bars_held,
                'direction': position['dir']
            })
            capital += trades[-1]['pnl']

        if len(trades) < 3:
            return None

        # Calculate metrics
        pnls = [t['pnl'] for t in trades]
        equity_s = pd.Series(equity)
        mdd = ((equity_s - equity_s.expanding().max()) / equity_s.expanding().max()).min() * 100

        wins = sum(1 for p in pnls if p > 0)
        gp = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)

        # Cost breakdown
        total_entry_cost = sum(t['entry_cost'] for t in trades)
        total_exit_cost = sum(t['exit_cost'] for t in trades)
        total_funding_cost = sum(t['funding_cost'] for t in trades)
        total_gross_pnl = sum(t['gross_pnl_pct'] for t in trades)

        return {
            'symbol': symbol,
            'pf': min(pf, 999),
            'ret': (capital / init - 1) * 100,
            'wr': wins / len(trades) * 100,
            'mdd': mdd,
            'trades': len(trades),
            'avg_entry_cost': total_entry_cost / len(trades) * 100,
            'avg_exit_cost': total_exit_cost / len(trades) * 100,
            'avg_funding_cost': total_funding_cost / len(trades) * 100,
            'total_cost_impact': (total_entry_cost + total_exit_cost + total_funding_cost) / len(trades) * 100,
            'gross_return': total_gross_pnl / len(trades) * 100
        }


# ============================================================================
# Data Loader
# ============================================================================

class DataLoader:
    def __init__(self):
        self.root = DATA_ROOT
        self._cache = {}

    def load_ohlcv(self, symbol: str) -> pd.DataFrame:
        if symbol in self._cache:
            return self._cache[symbol].copy()

        filepath = self.root / "binance_futures_4h" / f"{symbol}.csv"
        if not filepath.exists():
            return pd.DataFrame()

        df = pd.read_csv(filepath)
        for col in ['datetime', 'timestamp', 'date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col).sort_index()
                break

        self._cache[symbol] = df
        return df.copy()

    def load_fear_greed(self) -> pd.DataFrame:
        for fn in ["FEAR_GREED_INDEX_updated.csv", "FEAR_GREED_INDEX.csv"]:
            fp = self.root / fn
            if fp.exists():
                df = pd.read_csv(fp)
                for col in ["datetime", "timestamp"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col)
                        break
                if "value" in df.columns:
                    return df[["value"]].rename(columns={"value": "fear_greed"})
                if "close" in df.columns:
                    return df[["close"]].rename(columns={"close": "fear_greed"})
        return pd.DataFrame()

    def load_macro(self) -> pd.DataFrame:
        dfs = []
        for fn, col in [("DXY.csv", "dxy"), ("SP500.csv", "sp500"), ("VIX.csv", "vix")]:
            fp = self.root / "macro" / fn
            if fp.exists():
                df = pd.read_csv(fp)
                for c in ["datetime", "timestamp"]:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c])
                        df = df.set_index(c)
                        break
                if "close" in df.columns:
                    dfs.append(df[["close"]].rename(columns={"close": col}))
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()


# ============================================================================
# Market Regime Analysis
# ============================================================================

def classify_market_regime(df: pd.DataFrame) -> pd.Series:
    """Classify market into regimes: bull, bear, sideways"""
    close = df['close']

    # 50-day return and volatility
    ret_50 = close.pct_change(50)
    vol_20 = close.pct_change().rolling(20).std()
    vol_mean = vol_20.rolling(100).mean()

    regime = pd.Series('sideways', index=df.index)
    regime[ret_50 > 0.15] = 'bull'
    regime[ret_50 < -0.15] = 'bear'
    regime[(vol_20 > vol_mean * 1.5) & (regime == 'sideways')] = 'high_vol'

    return regime


# ============================================================================
# Main Validation
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("REALISTIC BACKTEST VALIDATION")
    logger.info("=" * 70)
    logger.info("Testing: MF+ML and MF+OnChain+ML with realistic costs")
    logger.info("Costs: Slippage, Funding Rate, Liquidity Impact")
    logger.info("=" * 70)

    loader = DataLoader()
    fear_greed = loader.load_fear_greed()
    macro = loader.load_macro()
    btc_df = loader.load_ohlcv("BTCUSDT")

    ohlcv_dir = DATA_ROOT / "binance_futures_4h"
    all_symbols = sorted([f.stem for f in ohlcv_dir.glob("*.csv") if f.stem.endswith('USDT')])

    mf_signal = MultiFactorSignal()
    oc_signal = OnChainSignal()

    # Cost scenarios
    cost_scenarios = {
        'optimistic': RealisticCostModel(base_fee=0.0002, slippage_base=0.0005, funding_rate=0.00005),
        'realistic': RealisticCostModel(base_fee=0.0004, slippage_base=0.001, funding_rate=0.0001),
        'pessimistic': RealisticCostModel(base_fee=0.0006, slippage_base=0.002, funding_rate=0.0002),
    }

    results = {
        'MF_ML': {scenario: [] for scenario in cost_scenarios},
        'MF_OnChain_ML': {scenario: [] for scenario in cost_scenarios},
    }

    regime_results = {
        'MF_ML': {'bull': [], 'bear': [], 'sideways': [], 'high_vol': []},
        'MF_OnChain_ML': {'bull': [], 'bear': [], 'sideways': [], 'high_vol': []},
    }

    logger.info(f"\nTesting {len(all_symbols)} symbols...")

    tested = 0
    for symbol in all_symbols:
        df = loader.load_ohlcv(symbol)
        if df.empty:
            continue

        df = df[df.index >= "2022-01-01"]
        if len(df) < 800:
            continue

        tested += 1

        # Train/Test split: 70/30
        train_end_idx = int(len(df) * 0.7)

        # Generate signals
        mf_scores = mf_signal.generate(df, btc_df, fear_greed, macro)
        oc_scores = oc_signal.generate(df)

        # ML signal (trained only on train period)
        ml_signal = MLSignal()
        ml_scores = ml_signal.train_and_predict(df, train_end_idx)

        # Combined scores
        mf_ml_scores = mf_scores + ml_scores * 0.8
        mf_oc_ml_scores = mf_scores + oc_scores * 0.5 + ml_scores * 0.5

        # Market regime classification
        regimes = classify_market_regime(df)
        test_regimes = regimes.iloc[train_end_idx:]
        dominant_regime = test_regimes.mode().iloc[0] if len(test_regimes) > 0 else 'sideways'

        # Test each cost scenario
        for scenario, cost_model in cost_scenarios.items():
            backtester = RealisticBacktester(threshold=4.0, cost_model=cost_model)

            # MF + ML
            result = backtester.backtest(df, mf_ml_scores, symbol, train_end_idx)
            if result and result['trades'] >= 5:
                results['MF_ML'][scenario].append(result)
                if scenario == 'realistic':
                    regime_results['MF_ML'][dominant_regime].append(result)

            # MF + OnChain + ML
            result = backtester.backtest(df, mf_oc_ml_scores, symbol, train_end_idx)
            if result and result['trades'] >= 5:
                results['MF_OnChain_ML'][scenario].append(result)
                if scenario == 'realistic':
                    regime_results['MF_OnChain_ML'][dominant_regime].append(result)

        if tested % 50 == 0:
            logger.info(f"  Progress: {tested} symbols")

    # =========================================================================
    # Results Summary
    # =========================================================================

    logger.info(f"\n{'=' * 70}")
    logger.info("COST SENSITIVITY ANALYSIS")
    logger.info("=" * 70)

    for strategy in ['MF_ML', 'MF_OnChain_ML']:
        logger.info(f"\n[{strategy}]")
        logger.info(f"{'Scenario':<15} {'Symbols':>8} {'Profitable':>12} {'Avg PF':>10} {'Avg Ret':>10} {'Avg Cost':>10}")
        logger.info("-" * 70)

        for scenario in ['optimistic', 'realistic', 'pessimistic']:
            res_list = results[strategy][scenario]
            if not res_list:
                continue

            df_r = pd.DataFrame(res_list)
            profitable = (df_r['pf'] > 1.0).sum()
            avg_pf = df_r[df_r['pf'] < 999]['pf'].mean()
            avg_ret = df_r['ret'].mean()
            avg_cost = df_r['total_cost_impact'].mean()

            logger.info(f"{scenario:<15} {len(res_list):>8} {profitable/len(res_list)*100:>11.1f}% "
                       f"{avg_pf:>10.2f} {avg_ret:>9.1f}% {avg_cost:>9.2f}%")

    # =========================================================================
    # Market Regime Analysis
    # =========================================================================

    logger.info(f"\n{'=' * 70}")
    logger.info("MARKET REGIME PERFORMANCE (Realistic Costs)")
    logger.info("=" * 70)

    for strategy in ['MF_ML', 'MF_OnChain_ML']:
        logger.info(f"\n[{strategy}]")
        logger.info(f"{'Regime':<12} {'Symbols':>8} {'Profitable':>12} {'Avg PF':>10} {'Avg Ret':>10}")
        logger.info("-" * 55)

        for regime in ['bull', 'bear', 'sideways', 'high_vol']:
            res_list = regime_results[strategy][regime]
            if not res_list:
                continue

            df_r = pd.DataFrame(res_list)
            profitable = (df_r['pf'] > 1.0).sum()
            avg_pf = df_r[df_r['pf'] < 999]['pf'].mean()
            avg_ret = df_r['ret'].mean()

            logger.info(f"{regime:<12} {len(res_list):>8} {profitable/len(res_list)*100:>11.1f}% "
                       f"{avg_pf:>10.2f} {avg_ret:>9.1f}%")

    # =========================================================================
    # Paired Comparison: MF+ML vs MF+OnChain+ML
    # =========================================================================

    logger.info(f"\n{'=' * 70}")
    logger.info("PAIRED COMPARISON: MF+ML vs MF+OnChain+ML (Realistic Costs)")
    logger.info("=" * 70)

    # Find symbols that have results in both strategies
    mf_ml_symbols = {r['symbol'] for r in results['MF_ML']['realistic']}
    mf_oc_ml_symbols = {r['symbol'] for r in results['MF_OnChain_ML']['realistic']}
    common_symbols = mf_ml_symbols & mf_oc_ml_symbols

    mf_ml_dict = {r['symbol']: r for r in results['MF_ML']['realistic']}
    mf_oc_ml_dict = {r['symbol']: r for r in results['MF_OnChain_ML']['realistic']}

    paired_comparison = []
    for symbol in common_symbols:
        mf_ml = mf_ml_dict[symbol]
        mf_oc_ml = mf_oc_ml_dict[symbol]
        paired_comparison.append({
            'symbol': symbol,
            'mf_ml_pf': mf_ml['pf'],
            'mf_ml_ret': mf_ml['ret'],
            'mf_oc_ml_pf': mf_oc_ml['pf'],
            'mf_oc_ml_ret': mf_oc_ml['ret'],
            'pf_diff': mf_oc_ml['pf'] - mf_ml['pf'],
            'ret_diff': mf_oc_ml['ret'] - mf_ml['ret'],
        })

    if paired_comparison:
        df_paired = pd.DataFrame(paired_comparison)

        logger.info(f"\nPaired samples: {len(df_paired)}")
        logger.info(f"\n{'Metric':<25} {'MF+ML':>15} {'MF+OnChain+ML':>15}")
        logger.info("-" * 60)

        mf_ml_profitable = (df_paired['mf_ml_pf'] > 1.0).sum()
        mf_oc_ml_profitable = (df_paired['mf_oc_ml_pf'] > 1.0).sum()

        logger.info(f"{'Profitable Rate':<25} {mf_ml_profitable/len(df_paired)*100:>14.1f}% "
                   f"{mf_oc_ml_profitable/len(df_paired)*100:>14.1f}%")

        mf_ml_avg_pf = df_paired[df_paired['mf_ml_pf'] < 999]['mf_ml_pf'].mean()
        mf_oc_ml_avg_pf = df_paired[df_paired['mf_oc_ml_pf'] < 999]['mf_oc_ml_pf'].mean()

        logger.info(f"{'Avg PF':<25} {mf_ml_avg_pf:>15.2f} {mf_oc_ml_avg_pf:>15.2f}")
        logger.info(f"{'Avg Return':<25} {df_paired['mf_ml_ret'].mean():>14.1f}% "
                   f"{df_paired['mf_oc_ml_ret'].mean():>14.1f}%")

        # Win comparison
        mf_oc_ml_wins = (df_paired['pf_diff'] > 0).sum()
        logger.info(f"\nMF+OnChain+ML wins: {mf_oc_ml_wins}/{len(df_paired)} "
                   f"({mf_oc_ml_wins/len(df_paired)*100:.1f}%)")

        # Top improvements
        logger.info("\nTop 10 symbols where OnChain improves performance:")
        top_improved = df_paired.nlargest(10, 'pf_diff')
        for _, r in top_improved.iterrows():
            logger.info(f"  {r['symbol']:<14} MF+ML PF={r['mf_ml_pf']:.2f} -> "
                       f"MF+OC+ML PF={r['mf_oc_ml_pf']:.2f} (+{r['pf_diff']:.2f})")

    # =========================================================================
    # Final Assessment
    # =========================================================================

    logger.info(f"\n{'=' * 70}")
    logger.info("FINAL ASSESSMENT (Realistic Costs)")
    logger.info("=" * 70)

    criteria = []

    # Check MF+ML realistic
    mf_ml_res = results['MF_ML']['realistic']
    if mf_ml_res:
        df_r = pd.DataFrame(mf_ml_res)
        profitable_rate = (df_r['pf'] > 1.0).sum() / len(df_r) * 100
        avg_pf = df_r[df_r['pf'] < 999]['pf'].mean()
        avg_ret = df_r['ret'].mean()

        c1 = profitable_rate >= 50
        c2 = avg_pf >= 1.2
        c3 = avg_ret >= 0

        logger.info(f"\n[MF+ML with Realistic Costs]")
        logger.info(f"  [{'PASS' if c1 else 'FAIL'}] Profitable >= 50%: {profitable_rate:.1f}%")
        logger.info(f"  [{'PASS' if c2 else 'FAIL'}] Avg PF >= 1.2: {avg_pf:.2f}")
        logger.info(f"  [{'PASS' if c3 else 'FAIL'}] Avg Return >= 0%: {avg_ret:.1f}%")

        criteria.extend([c1, c2, c3])

    # Check MF+OnChain+ML realistic
    mf_oc_ml_res = results['MF_OnChain_ML']['realistic']
    if mf_oc_ml_res:
        df_r = pd.DataFrame(mf_oc_ml_res)
        profitable_rate = (df_r['pf'] > 1.0).sum() / len(df_r) * 100
        avg_pf = df_r[df_r['pf'] < 999]['pf'].mean()
        avg_ret = df_r['ret'].mean()

        c4 = profitable_rate >= 50
        c5 = avg_pf >= 1.2
        c6 = avg_ret >= 0

        logger.info(f"\n[MF+OnChain+ML with Realistic Costs]")
        logger.info(f"  [{'PASS' if c4 else 'FAIL'}] Profitable >= 50%: {profitable_rate:.1f}%")
        logger.info(f"  [{'PASS' if c5 else 'FAIL'}] Avg PF >= 1.2: {avg_pf:.2f}")
        logger.info(f"  [{'PASS' if c6 else 'FAIL'}] Avg Return >= 0%: {avg_ret:.1f}%")

        criteria.extend([c4, c5, c6])

    passed = sum(criteria)
    total = len(criteria)

    logger.info(f"\nPassed: {passed}/{total}")
    if passed == total:
        logger.info("\n>>> ALL CRITERIA PASSED - STRATEGIES VALIDATED FOR REALISTIC TRADING <<<")
    elif passed >= total * 0.7:
        logger.info("\n>>> PARTIAL PASS - STRATEGIES MAY BE VIABLE WITH ADJUSTMENTS <<<")
    else:
        logger.info("\n>>> STRATEGIES NEED IMPROVEMENT FOR REALISTIC TRADING <<<")

    # Save results
    all_results = []
    for strategy in ['MF_ML', 'MF_OnChain_ML']:
        for scenario, res_list in results[strategy].items():
            for r in res_list:
                r['strategy'] = strategy
                r['cost_scenario'] = scenario
                all_results.append(r)

    if all_results:
        df_all = pd.DataFrame(all_results)
        output_path = DATA_ROOT / "realistic_backtest_results.csv"
        df_all.to_csv(output_path, index=False)
        logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = main()
