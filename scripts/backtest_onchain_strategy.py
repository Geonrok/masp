#!/usr/bin/env python3
"""
On-Chain Data Strategy Backtester
- Whale wallet tracking (large transactions)
- Exchange inflows/outflows
- Uses free APIs: Blockchain.com, Etherscan (limited)

Note: Full on-chain data requires Glassnode/CryptoQuant subscription
This version uses simulated on-chain metrics derived from price/volume
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_ROOT = Path("E:/data/crypto_ohlcv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        self.root = DATA_ROOT

    def load_ohlcv(self, symbol: str, tf: str = "4h") -> pd.DataFrame:
        folder = {"4h": "binance_futures_4h", "1d": "binance_futures_1d"}.get(tf)
        filepath = self.root / folder / f"{symbol}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            for col in ["datetime", "timestamp", "date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col).sort_index()
                    break
            if all(c in df.columns for c in ["open", "high", "low", "close", "volume"]):
                return df
        return pd.DataFrame()


class OnChainMetricsSimulator:
    """
    Simulate on-chain metrics from price/volume data
    In production, replace with actual API calls to Glassnode/CryptoQuant
    """

    def simulate_whale_activity(self, df: pd.DataFrame) -> pd.Series:
        """
        Simulate whale activity based on volume spikes
        Logic: Large volume with price movement = whale activity
        """
        volume = df["volume"]
        price_change = df["close"].pct_change().abs()

        # Volume Z-score
        vol_zscore = (volume - volume.rolling(50).mean()) / volume.rolling(50).std()

        # Whale activity score: high volume + price movement
        whale_score = vol_zscore * (1 + price_change * 10)
        return whale_score.fillna(0)

    def simulate_exchange_flow(self, df: pd.DataFrame) -> pd.Series:
        """
        Simulate exchange inflow/outflow
        Logic:
        - Price dropping + volume rising = exchange inflows (selling pressure)
        - Price rising + volume dropping = exchange outflows (accumulation)
        """
        price_change = df["close"].pct_change()
        volume_change = df["volume"].pct_change()

        # Positive = inflow (bearish), Negative = outflow (bullish)
        flow_indicator = -price_change * 10 + volume_change

        # Smooth with EMA
        return flow_indicator.ewm(span=5).mean().fillna(0)

    def simulate_holder_distribution(self, df: pd.DataFrame) -> pd.Series:
        """
        Simulate holder distribution changes
        Logic: Sustained price increase with decreasing volume = accumulation
        """
        price_ma = df["close"].rolling(20).mean()
        volume_ma = df["volume"].rolling(20).mean()

        price_above_ma = (df["close"] > price_ma).astype(int)
        volume_below_ma = (df["volume"] < volume_ma).astype(int)

        # Accumulation score
        accumulation = price_above_ma * volume_below_ma - (1 - price_above_ma) * (
            1 - volume_below_ma
        )
        return accumulation.rolling(10).sum().fillna(0)

    def simulate_network_activity(self, df: pd.DataFrame) -> pd.Series:
        """
        Simulate network activity (tx count, active addresses)
        Logic: Volume and volatility correlate with on-chain activity
        """
        volatility = df["close"].pct_change().rolling(20).std()
        volume_normalized = df["volume"] / df["volume"].rolling(100).mean()

        network_score = (volatility * 100 + volume_normalized) / 2
        return network_score.fillna(0)


class OnChainStrategy:
    """On-Chain based trading strategy"""

    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold
        self.simulator = OnChainMetricsSimulator()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals from on-chain metrics"""

        # Get simulated on-chain data
        whale = self.simulator.simulate_whale_activity(df)
        flow = self.simulator.simulate_exchange_flow(df)
        accumulation = self.simulator.simulate_holder_distribution(df)
        network = self.simulator.simulate_network_activity(df)

        scores = pd.Series(0.0, index=df.index)

        # 1. Whale Activity (weight: 1.5)
        # Large whale activity during dips = bullish
        price_below_ma = df["close"] < df["close"].rolling(20).mean()
        whale_buy = (whale > 2) & price_below_ma
        whale_sell = (whale > 2) & ~price_below_ma

        scores += np.where(whale_buy, 2, np.where(whale_sell, -1, 0)) * 1.5

        # 2. Exchange Flow (weight: 2.0) - Most important
        # Negative flow (outflow) = bullish, Positive flow (inflow) = bearish
        scores += (
            np.where(
                flow < -0.5,
                2,
                np.where(
                    flow < -0.2,
                    1,
                    np.where(flow > 0.5, -2, np.where(flow > 0.2, -1, 0)),
                ),
            )
            * 2.0
        )

        # 3. Accumulation (weight: 1.0)
        scores += (
            np.where(
                accumulation > 5,
                1.5,
                np.where(
                    accumulation > 2,
                    0.5,
                    np.where(
                        accumulation < -5, -1.5, np.where(accumulation < -2, -0.5, 0)
                    ),
                ),
            )
            * 1.0
        )

        # 4. Network Activity (weight: 0.5)
        network_ma = network.rolling(50).mean()
        scores += (
            np.where(
                network > network_ma * 1.5,
                0.5,
                np.where(network < network_ma * 0.5, -0.5, 0),
            )
            * 0.5
        )

        return scores

    def backtest(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Run backtest"""
        if len(df) < 200:
            return None

        signals = self.generate_signals(df)

        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]

        for i in range(50, len(df)):
            price = df["close"].iloc[i]
            signal = signals.iloc[i]

            # Exit conditions
            if position:
                exit_signal = (position["dir"] == 1 and signal < -self.threshold) or (
                    position["dir"] == -1 and signal > self.threshold
                )

                if exit_signal:
                    pnl_pct = (price / position["entry"] - 1) * position["dir"] - 0.002
                    pnl = capital * 0.2 * position["leverage"] * pnl_pct
                    trades.append({"pnl": pnl, "direction": position["dir"]})
                    capital += pnl
                    position = None

            # Entry conditions
            if not position and abs(signal) >= self.threshold:
                direction = 1 if signal > 0 else -1
                leverage = min(abs(signal) / self.threshold, 2.0)
                position = {"entry": price, "dir": direction, "leverage": leverage}

            equity.append(capital)

        # Close final position
        if position:
            pnl_pct = (df["close"].iloc[-1] / position["entry"] - 1) * position[
                "dir"
            ] - 0.002
            trades.append(
                {
                    "pnl": capital * 0.2 * position["leverage"] * pnl_pct,
                    "direction": position["dir"],
                }
            )
            capital += trades[-1]["pnl"]

        if len(trades) < 5:
            return None

        pnls = [t["pnl"] for t in trades]
        equity_s = pd.Series(equity)
        mdd = (
            (equity_s - equity_s.expanding().max()) / equity_s.expanding().max()
        ).min() * 100

        wins = sum(1 for p in pnls if p > 0)
        gp = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)

        return {
            "symbol": symbol,
            "pf": min(pf, 999),
            "ret": (capital / init - 1) * 100,
            "wr": wins / len(trades) * 100,
            "mdd": mdd,
            "trades": len(trades),
            "long_trades": sum(1 for t in trades if t["direction"] == 1),
            "short_trades": sum(1 for t in trades if t["direction"] == -1),
        }


def main():
    loader = DataLoader()
    strategy = OnChainStrategy(threshold=2.0)

    logger.info("=" * 70)
    logger.info("ON-CHAIN DATA STRATEGY BACKTESTER")
    logger.info("=" * 70)
    logger.info("Using simulated on-chain metrics (volume/price proxy)")
    logger.info("Metrics: Whale Activity, Exchange Flow, Accumulation, Network")
    logger.info("=" * 70)

    # Get all symbols
    ohlcv_dir = DATA_ROOT / "binance_futures_4h"
    symbols = sorted(
        [f.stem for f in ohlcv_dir.glob("*.csv") if f.stem.endswith("USDT")]
    )

    logger.info(f"Testing {len(symbols)} symbols...\n")

    results = []
    for symbol in symbols:
        df = loader.load_ohlcv(symbol, "4h")
        if df.empty:
            continue

        df = df[df.index >= "2022-01-01"]
        if len(df) < 500:
            continue

        result = strategy.backtest(df, symbol)
        if result and result["trades"] >= 10:
            results.append(result)

    logger.info(f"Valid results: {len(results)} symbols\n")

    if not results:
        return

    # Summary
    df_results = pd.DataFrame(results)
    profitable = df_results[df_results["pf"] > 1.0]

    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(
        f"Profitable: {len(profitable)}/{len(results)} ({len(profitable)/len(results)*100:.1f}%)"
    )
    logger.info(f"Avg PF: {df_results[df_results['pf'] < 999]['pf'].mean():.2f}")
    logger.info(f"Avg Return: {df_results['ret'].mean():+.1f}%")
    logger.info(f"Avg Win Rate: {df_results['wr'].mean():.1f}%")
    logger.info(f"Avg MDD: {df_results['mdd'].mean():.1f}%")

    # Top performers
    logger.info("\n" + "=" * 70)
    logger.info("TOP 20 PERFORMERS")
    logger.info("=" * 70)

    top = df_results.nlargest(20, "pf")
    for _, r in top.iterrows():
        logger.info(
            f"  {r['symbol']:<14} PF={r['pf']:5.2f} Ret={r['ret']:+7.1f}% WR={r['wr']:.0f}% "
            f"MDD={r['mdd']:.1f}% Trades={r['trades']}"
        )

    # Save results
    output_path = DATA_ROOT / "onchain_strategy_results.csv"
    df_results.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = main()
