#!/usr/bin/env python3
"""
Event-Based Trading Strategy Backtester
- Listing announcements (new coin listings)
- Airdrops / Token unlocks
- Network upgrades / Hard forks
- Major partnership announcements

Note: This uses simulated events based on price/volume anomalies
Real implementation requires event data feeds (CoinGecko, CryptoRank, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import warnings
from datetime import timedelta

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


class EventDetector:
    """
    Detect potential events from price/volume patterns
    In production, replace with actual event API data
    """

    def detect_listing_pattern(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect patterns similar to new listings:
        - Sudden volume spike (10x+)
        - High volatility
        - Often at the start of data
        """
        volume = df["volume"]
        vol_mean = volume.rolling(50).mean()
        vol_spike = volume / vol_mean

        close = df["close"]
        volatility = close.pct_change().rolling(10).std()
        vol_mean_hist = volatility.rolling(50).mean()

        # Listing pattern: massive volume spike + high volatility
        listing_score = np.where(
            (vol_spike > 5) & (volatility > vol_mean_hist * 2), vol_spike, 0
        )
        return pd.Series(listing_score, index=df.index)

    def detect_unlock_pattern(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect patterns similar to token unlocks:
        - Gradual price decline before event
        - Volume increase
        - Sharp drop on unlock day
        """
        close = df["close"]
        volume = df["volume"]

        # Pre-unlock: declining price with increasing volume
        price_trend = close.pct_change(20)
        vol_trend = volume.pct_change(20)

        # Unlock pattern: price down, volume up
        unlock_score = np.where(
            (price_trend < -0.1) & (vol_trend > 0.5), abs(price_trend) * vol_trend, 0
        )
        return pd.Series(unlock_score, index=df.index)

    def detect_upgrade_pattern(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect patterns similar to network upgrades:
        - Gradual price increase (anticipation)
        - Volume spike on upgrade day
        - Consolidation after
        """
        close = df["close"]
        volume = df["volume"]

        # Pre-upgrade anticipation
        price_trend = close.pct_change(30)
        vol_spike = volume / volume.rolling(30).mean()

        # Upgrade pattern: positive price trend + volume spike
        upgrade_score = np.where(
            (price_trend > 0.15) & (vol_spike > 3), price_trend * vol_spike, 0
        )
        return pd.Series(upgrade_score, index=df.index)

    def detect_news_pattern(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect patterns similar to major news/partnerships:
        - Sudden price movement
        - Volume spike
        - Gap up/down
        """
        close = df["close"]
        open_price = df["open"]
        volume = df["volume"]

        # Gap detection
        gap = abs(open_price - close.shift(1)) / close.shift(1)

        # Volume spike
        vol_spike = volume / volume.rolling(20).mean()

        # News pattern: gap + volume spike
        news_score = np.where((gap > 0.03) & (vol_spike > 2), gap * vol_spike * 10, 0)
        return pd.Series(news_score, index=df.index)


class EventStrategy:
    """Event-based trading strategies"""

    def __init__(self):
        self.detector = EventDetector()

    def backtest_listing_strategy(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Listing strategy:
        - Enter on volume spike detection
        - Hold for fixed period (anticipate pump)
        - Exit before the dump
        """
        if len(df) < 200:
            return None

        listing_score = self.detector.detect_listing_pattern(df)

        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]

        for i in range(50, len(df)):
            price = df["close"].iloc[i]
            score = listing_score.iloc[i]

            if position:
                bars_held = i - position["entry_idx"]
                # Exit after 12 bars (2 days for 4h) or if loss > 10%
                pnl_pct = price / position["entry"] - 1
                should_exit = bars_held >= 12 or pnl_pct < -0.10

                if should_exit:
                    pnl = capital * 0.15 * pnl_pct - capital * 0.002
                    trades.append(
                        {"pnl": pnl, "bars_held": bars_held, "type": "listing"}
                    )
                    capital += pnl
                    position = None

            if not position and score > 3:
                position = {"entry": price, "entry_idx": i, "score": score}

            equity.append(capital)

        return self._calc_result(trades, equity, capital, init, symbol, "listing")

    def backtest_unlock_strategy(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Token unlock strategy:
        - Detect unlock pattern (price down, volume up)
        - Short before unlock
        - Cover after dump
        """
        if len(df) < 200:
            return None

        unlock_score = self.detector.detect_unlock_pattern(df)

        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]

        for i in range(50, len(df)):
            price = df["close"].iloc[i]
            score = unlock_score.iloc[i]

            if position:
                bars_held = i - position["entry_idx"]
                pnl_pct = position["entry"] / price - 1  # Short
                should_exit = bars_held >= 6 or pnl_pct > 0.15 or pnl_pct < -0.10

                if should_exit:
                    pnl = capital * 0.15 * pnl_pct - capital * 0.002
                    trades.append(
                        {"pnl": pnl, "bars_held": bars_held, "type": "unlock"}
                    )
                    capital += pnl
                    position = None

            if not position and score > 0.05:
                position = {"entry": price, "entry_idx": i, "direction": -1}  # Short

            equity.append(capital)

        return self._calc_result(trades, equity, capital, init, symbol, "unlock")

    def backtest_upgrade_strategy(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Network upgrade strategy:
        - Enter during anticipation phase
        - Exit on upgrade day or shortly after
        """
        if len(df) < 200:
            return None

        upgrade_score = self.detector.detect_upgrade_pattern(df)

        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]

        for i in range(50, len(df)):
            price = df["close"].iloc[i]
            score = upgrade_score.iloc[i]

            if position:
                bars_held = i - position["entry_idx"]
                pnl_pct = price / position["entry"] - 1
                # Exit on big spike (sell the news) or timeout
                vol_spike = df["volume"].iloc[i] / df["volume"].iloc[i - 10 : i].mean()
                should_exit = (
                    bars_held >= 18
                    or (pnl_pct > 0.10 and vol_spike > 3)
                    or pnl_pct < -0.10
                )

                if should_exit:
                    pnl = capital * 0.2 * pnl_pct - capital * 0.002
                    trades.append(
                        {"pnl": pnl, "bars_held": bars_held, "type": "upgrade"}
                    )
                    capital += pnl
                    position = None

            if not position and score > 0.5:
                position = {"entry": price, "entry_idx": i}

            equity.append(capital)

        return self._calc_result(trades, equity, capital, init, symbol, "upgrade")

    def backtest_news_strategy(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        News/Partnership strategy:
        - Enter on positive gap with volume
        - Quick exit (momentum trade)
        """
        if len(df) < 200:
            return None

        news_score = self.detector.detect_news_pattern(df)

        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]

        for i in range(50, len(df)):
            price = df["close"].iloc[i]
            score = news_score.iloc[i]
            price_change = (price / df["close"].iloc[i - 1] - 1) if i > 0 else 0

            if position:
                bars_held = i - position["entry_idx"]
                pnl_pct = (price / position["entry"] - 1) * position["dir"]
                # Quick exit for news trades
                should_exit = bars_held >= 6 or pnl_pct > 0.08 or pnl_pct < -0.05

                if should_exit:
                    pnl = capital * 0.15 * pnl_pct - capital * 0.002
                    trades.append({"pnl": pnl, "bars_held": bars_held, "type": "news"})
                    capital += pnl
                    position = None

            if not position and score > 1:
                direction = 1 if price_change > 0 else -1
                position = {"entry": price, "entry_idx": i, "dir": direction}

            equity.append(capital)

        return self._calc_result(trades, equity, capital, init, symbol, "news")

    def _calc_result(
        self,
        trades: List,
        equity: List,
        capital: float,
        init: float,
        symbol: str,
        strategy: str,
    ) -> Dict:
        if len(trades) < 3:
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
            "strategy": strategy,
            "pf": min(pf, 999),
            "ret": (capital / init - 1) * 100,
            "wr": wins / len(trades) * 100,
            "mdd": mdd,
            "trades": len(trades),
            "avg_bars_held": np.mean([t["bars_held"] for t in trades]),
        }


def main():
    logger.info("=" * 70)
    logger.info("EVENT-BASED STRATEGY BACKTESTER")
    logger.info("=" * 70)
    logger.info("Strategies: Listing, Unlock, Upgrade, News")
    logger.info("Note: Using simulated events from price/volume patterns")
    logger.info("=" * 70)

    # Load all symbols
    ohlcv_dir = DATA_ROOT / "binance_futures_4h"
    symbols = sorted(
        [f.stem for f in ohlcv_dir.glob("*.csv") if f.stem.endswith("USDT")]
    )

    strategy = EventStrategy()
    all_results = []

    strategies = [
        ("Listing", strategy.backtest_listing_strategy),
        ("Unlock", strategy.backtest_unlock_strategy),
        ("Upgrade", strategy.backtest_upgrade_strategy),
        ("News", strategy.backtest_news_strategy),
    ]

    for strat_name, strat_func in strategies:
        logger.info(f"\n[{strat_name.upper()} STRATEGY]")
        logger.info("-" * 50)

        results = []
        for symbol in symbols:
            filepath = ohlcv_dir / f"{symbol}.csv"
            df = pd.read_csv(filepath)
            for col in ["datetime", "timestamp", "date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col).sort_index()
                    break
            df = df[df.index >= "2022-01-01"]

            if len(df) < 500:
                continue

            result = strat_func(df, symbol)
            if result and result["trades"] >= 5:
                results.append(result)
                all_results.append(result)

        if results:
            df_results = pd.DataFrame(results)
            profitable = df_results[df_results["pf"] > 1.0]
            logger.info(f"  Tested: {len(results)} symbols")
            logger.info(
                f"  Profitable: {len(profitable)} ({len(profitable)/len(results)*100:.1f}%)"
            )
            logger.info(
                f"  Avg PF: {df_results[df_results['pf'] < 999]['pf'].mean():.2f}"
            )
            logger.info(f"  Avg Return: {df_results['ret'].mean():+.1f}%")

            # Top 5
            top5 = df_results.nlargest(5, "pf")
            logger.info(f"\n  Top 5 {strat_name}:")
            for _, r in top5.iterrows():
                logger.info(
                    f"    {r['symbol']:<14} PF={r['pf']:.2f} Ret={r['ret']:+.1f}% Trades={r['trades']}"
                )

    # Overall summary
    logger.info("\n" + "=" * 70)
    logger.info("OVERALL EVENT STRATEGY SUMMARY")
    logger.info("=" * 70)

    if all_results:
        df_all = pd.DataFrame(all_results)

        # Group by strategy
        for strategy in df_all["strategy"].unique():
            subset = df_all[df_all["strategy"] == strategy]
            profitable = (subset["pf"] > 1.0).sum()
            avg_pf = subset[subset["pf"] < 999]["pf"].mean()
            logger.info(
                f"{strategy:<10}: {profitable}/{len(subset)} profitable ({profitable/len(subset)*100:.1f}%), "
                f"Avg PF={avg_pf:.2f}"
            )

        # Best overall
        logger.info("\nTop 15 Event-Based Opportunities:")
        top = df_all.nlargest(15, "pf")
        for _, r in top.iterrows():
            logger.info(
                f"  {r['symbol']:<14} {r['strategy']:<10} PF={r['pf']:.2f} "
                f"Ret={r['ret']:+.1f}% Trades={r['trades']}"
            )

        # Save results
        df_all.to_csv(DATA_ROOT / "event_strategy_results.csv", index=False)
        logger.info(f"\nResults saved to: {DATA_ROOT / 'event_strategy_results.csv'}")

    return all_results


if __name__ == "__main__":
    results = main()
