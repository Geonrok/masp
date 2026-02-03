#!/usr/bin/env python3
"""
Cross-Exchange Arbitrage Strategy Backtester
- Spot-Futures basis arbitrage (funding rate arbitrage)
- Simulated cross-exchange price differences
- Statistical arbitrage between correlated pairs

Note: Real arbitrage requires:
1. Multiple exchange accounts
2. Low latency connections
3. Significant capital for small spreads
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

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

    def load_futures(self, symbol: str, tf: str = "4h") -> pd.DataFrame:
        filepath = self.root / f"binance_futures_{tf}" / f"{symbol}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df.set_index("datetime").sort_index()
        return pd.DataFrame()

    def load_spot(self, symbol: str, tf: str = "4h") -> pd.DataFrame:
        filepath = self.root / f"binance_spot_{tf}" / f"{symbol}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df.set_index("datetime").sort_index()
        return pd.DataFrame()

    def load_funding_rate(self, symbol: str) -> pd.DataFrame:
        filepath = self.root / "binance_funding_rate" / f"{symbol}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df.set_index("datetime").sort_index()
        return pd.DataFrame()


class BasisArbitrage:
    """
    Spot-Futures Basis Arbitrage (Funding Rate Arbitrage)
    - Long spot + Short futures when funding > threshold
    - Collect funding payments while delta neutral
    """

    def __init__(self, funding_threshold: float = 0.0005):
        self.funding_threshold = funding_threshold  # 0.05% per 8h = ~55% APY

    def backtest(
        self,
        spot_df: pd.DataFrame,
        futures_df: pd.DataFrame,
        funding_df: pd.DataFrame,
        symbol: str,
    ) -> Dict:
        """Run basis arbitrage backtest"""

        # Align data
        common_idx = spot_df.index.intersection(futures_df.index)
        if len(common_idx) < 500:
            return None

        spot = spot_df.loc[common_idx, "close"]
        futures = futures_df.loc[common_idx, "close"]

        # Calculate basis
        basis = (futures - spot) / spot * 100  # Percentage

        # Resample funding to match price data
        if not funding_df.empty:
            funding = (
                funding_df["funding_rate"].reindex(common_idx, method="ffill").fillna(0)
            )
        else:
            # Simulate funding from basis
            funding = basis / 100 / 3  # Rough approximation

        capital = 10000
        init = capital
        position = None
        trades = []
        funding_collected = 0
        equity = [capital]

        for i in range(50, len(common_idx)):
            current_funding = funding.iloc[i] if i < len(funding) else 0
            current_basis = basis.iloc[i]
            spot_price = spot.iloc[i]
            futures_price = futures.iloc[i]

            # Collect funding if in position
            if position:
                # Funding is paid every 8 hours (3x per day, ~0.5 per 4h bar)
                if i % 2 == 0:  # Every 2 bars = 8 hours
                    funding_payment = position["size"] * current_funding
                    funding_collected += funding_payment
                    capital += funding_payment

                # Exit condition: funding turns negative or basis narrows significantly
                should_exit = (
                    current_funding < 0 or current_basis < position["entry_basis"] * 0.3
                )

                if should_exit:
                    # Close position
                    spot_pnl = (spot_price / position["spot_entry"] - 1) * position[
                        "size"
                    ]
                    futures_pnl = (
                        position["futures_entry"] / futures_price - 1
                    ) * position["size"]
                    total_pnl = spot_pnl + futures_pnl + position["funding_collected"]

                    trades.append(
                        {
                            "pnl": total_pnl,
                            "funding": position["funding_collected"],
                            "duration": i - position["entry_idx"],
                        }
                    )
                    capital += spot_pnl + futures_pnl  # Funding already added
                    position = None

            # Entry condition: high positive funding
            if (
                not position
                and current_funding > self.funding_threshold
                and current_basis > 0.1
            ):
                size = capital * 0.5  # 50% allocation (delta neutral)
                position = {
                    "spot_entry": spot_price,
                    "futures_entry": futures_price,
                    "entry_basis": current_basis,
                    "size": size,
                    "entry_idx": i,
                    "funding_collected": 0,
                }

            equity.append(capital)

        # Close final position
        if position:
            spot_pnl = (spot.iloc[-1] / position["spot_entry"] - 1) * position["size"]
            futures_pnl = (position["futures_entry"] / futures.iloc[-1] - 1) * position[
                "size"
            ]
            trades.append(
                {
                    "pnl": spot_pnl + futures_pnl + position["funding_collected"],
                    "funding": position["funding_collected"],
                    "duration": len(common_idx) - position["entry_idx"],
                }
            )
            capital += spot_pnl + futures_pnl

        if len(trades) < 3:
            return None

        pnls = [t["pnl"] for t in trades]
        total_funding = sum(t["funding"] for t in trades)
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
            "strategy": "basis_arb",
            "pf": min(pf, 999),
            "ret": (capital / init - 1) * 100,
            "wr": wins / len(trades) * 100,
            "mdd": mdd,
            "trades": len(trades),
            "funding_collected": total_funding,
            "avg_duration": np.mean([t["duration"] for t in trades]),
        }


class PairArbitrage:
    """
    Statistical Arbitrage between correlated pairs
    - Find cointegrated pairs
    - Trade mean reversion of spread
    """

    def __init__(self, zscore_entry: float = 2.0, zscore_exit: float = 0.5):
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit

    def find_cointegrated_pairs(
        self, price_dict: Dict[str, pd.Series], min_correlation: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """Find potentially cointegrated pairs"""
        pairs = []
        symbols = list(price_dict.keys())

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1 :]:
                # Calculate correlation
                combined = pd.concat(
                    [price_dict[sym1], price_dict[sym2]], axis=1
                ).dropna()
                if len(combined) < 500:
                    continue

                corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
                if corr > min_correlation:
                    pairs.append((sym1, sym2, corr))

        return sorted(pairs, key=lambda x: -x[2])

    def backtest_pair(
        self, df1: pd.DataFrame, df2: pd.DataFrame, sym1: str, sym2: str
    ) -> Dict:
        """Backtest pair trading strategy"""

        # Align data
        common_idx = df1.index.intersection(df2.index)
        if len(common_idx) < 500:
            return None

        price1 = df1.loc[common_idx, "close"]
        price2 = df2.loc[common_idx, "close"]

        # Calculate spread (ratio)
        spread = np.log(price1 / price2)
        spread_ma = spread.rolling(50).mean()
        spread_std = spread.rolling(50).std()
        zscore = (spread - spread_ma) / spread_std

        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]

        for i in range(100, len(common_idx)):
            z = zscore.iloc[i]
            p1 = price1.iloc[i]
            p2 = price2.iloc[i]

            if position:
                # Exit conditions
                should_exit = False
                if position["direction"] == "long_spread":
                    should_exit = z < self.zscore_exit or z > 0
                else:
                    should_exit = z > -self.zscore_exit or z < 0

                if should_exit:
                    # Calculate P&L
                    if position["direction"] == "long_spread":
                        pnl1 = (p1 / position["p1"] - 1) * position["size"]
                        pnl2 = (position["p2"] / p2 - 1) * position["size"]
                    else:
                        pnl1 = (position["p1"] / p1 - 1) * position["size"]
                        pnl2 = (p2 / position["p2"] - 1) * position["size"]

                    total_pnl = pnl1 + pnl2 - position["size"] * 0.004  # Trading costs
                    trades.append(
                        {
                            "pnl": total_pnl,
                            "zscore_entry": position["z_entry"],
                            "zscore_exit": z,
                        }
                    )
                    capital += total_pnl
                    position = None

            # Entry conditions
            if not position:
                if z > self.zscore_entry:  # Short spread: short sym1, long sym2
                    position = {
                        "direction": "short_spread",
                        "p1": p1,
                        "p2": p2,
                        "size": capital * 0.3,
                        "z_entry": z,
                    }
                elif z < -self.zscore_entry:  # Long spread: long sym1, short sym2
                    position = {
                        "direction": "long_spread",
                        "p1": p1,
                        "p2": p2,
                        "size": capital * 0.3,
                        "z_entry": z,
                    }

            equity.append(capital)

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
            "pair": f"{sym1}/{sym2}",
            "strategy": "pair_arb",
            "pf": min(pf, 999),
            "ret": (capital / init - 1) * 100,
            "wr": wins / len(trades) * 100,
            "mdd": mdd,
            "trades": len(trades),
        }


def main():
    loader = DataLoader()

    logger.info("=" * 70)
    logger.info("ARBITRAGE STRATEGY BACKTESTER")
    logger.info("=" * 70)
    logger.info("Strategies: Basis Arbitrage, Pair Trading")
    logger.info("=" * 70)

    # 1. Basis Arbitrage (Funding Rate Arb)
    logger.info("\n[1] BASIS ARBITRAGE (Spot-Futures Funding)")
    logger.info("-" * 50)

    basis_arb = BasisArbitrage(funding_threshold=0.0003)
    basis_results = []

    funding_dir = DATA_ROOT / "binance_funding_rate"
    symbols = [f.stem for f in funding_dir.glob("*.csv") if f.stem.endswith("USDT")]

    for symbol in symbols:
        spot = loader.load_spot(symbol, "4h")
        futures = loader.load_futures(symbol, "4h")
        funding = loader.load_funding_rate(symbol)

        if spot.empty or futures.empty:
            continue

        spot = spot[spot.index >= "2022-01-01"]
        futures = futures[futures.index >= "2022-01-01"]

        result = basis_arb.backtest(spot, futures, funding, symbol)
        if result:
            basis_results.append(result)
            logger.info(
                f"  {symbol:<14} PF={result['pf']:.2f} Ret={result['ret']:+.1f}% "
                f"Funding=${result['funding_collected']:.0f}"
            )

    if basis_results:
        df_basis = pd.DataFrame(basis_results)
        profitable = len(df_basis[df_basis["pf"] > 1.0])
        logger.info(
            f"\n  Summary: {profitable}/{len(df_basis)} profitable "
            f"({profitable/len(df_basis)*100:.1f}%)"
        )
        logger.info(f"  Avg PF: {df_basis[df_basis['pf'] < 999]['pf'].mean():.2f}")
        logger.info(f"  Avg Return: {df_basis['ret'].mean():+.1f}%")

    # 2. Pair Trading (Statistical Arbitrage)
    logger.info("\n[2] PAIR TRADING (Statistical Arbitrage)")
    logger.info("-" * 50)

    pair_arb = PairArbitrage(zscore_entry=2.0, zscore_exit=0.5)

    # Load price data for major coins
    major_symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "AVAXUSDT",
        "DOGEUSDT",
        "LINKUSDT",
        "MATICUSDT",
    ]

    price_dict = {}
    df_dict = {}
    for symbol in major_symbols:
        df = loader.load_futures(symbol, "4h")
        if not df.empty:
            df = df[df.index >= "2022-01-01"]
            if len(df) > 500:
                price_dict[symbol] = df["close"]
                df_dict[symbol] = df

    # Find cointegrated pairs
    pairs = pair_arb.find_cointegrated_pairs(price_dict, min_correlation=0.7)
    logger.info(f"  Found {len(pairs)} correlated pairs")

    pair_results = []
    for sym1, sym2, corr in pairs[:15]:  # Top 15 pairs
        result = pair_arb.backtest_pair(df_dict[sym1], df_dict[sym2], sym1, sym2)
        if result:
            result["correlation"] = corr
            pair_results.append(result)
            logger.info(
                f"  {result['pair']:<20} Corr={corr:.2f} PF={result['pf']:.2f} "
                f"Ret={result['ret']:+.1f}%"
            )

    if pair_results:
        df_pairs = pd.DataFrame(pair_results)
        profitable = len(df_pairs[df_pairs["pf"] > 1.0])
        logger.info(f"\n  Summary: {profitable}/{len(df_pairs)} profitable")
        logger.info(f"  Avg PF: {df_pairs[df_pairs['pf'] < 999]['pf'].mean():.2f}")

    # Combined results
    logger.info("\n" + "=" * 70)
    logger.info("OVERALL ARBITRAGE SUMMARY")
    logger.info("=" * 70)

    all_results = basis_results + pair_results
    if all_results:
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(DATA_ROOT / "arbitrage_results.csv", index=False)

        logger.info(f"Total strategies tested: {len(all_results)}")
        logger.info(f"Profitable: {len(df_all[df_all['pf'] > 1.0])}")
        logger.info(f"\nTop 10 Arbitrage Opportunities:")

        top = df_all.nlargest(10, "pf")
        for _, r in top.iterrows():
            name = r.get("symbol", r.get("pair", "Unknown"))
            logger.info(
                f"  {name:<20} {r['strategy']:<12} PF={r['pf']:.2f} Ret={r['ret']:+.1f}%"
            )

    return all_results


if __name__ == "__main__":
    results = main()
