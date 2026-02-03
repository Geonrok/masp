#!/usr/bin/env python3
"""
Alternative Data Collector

Collects data from multiple FREE sources:
1. CoinGecko - Market data, trading volume
2. DefiLlama - TVL data (market health proxy)
3. Blockchain.com - Basic on-chain metrics
4. Alternative.me - Fear & Greed (update)

No API key required for basic usage.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("E:/data/crypto_ohlcv")


class FreeDataCollector:
    """Collect data from free APIs."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }
        )

    def _request(self, url: str, params: dict = None) -> dict:
        """Make request with error handling."""
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
        finally:
            time.sleep(1)  # Rate limiting

    # =========================================================================
    # CoinGecko - Market Data
    # =========================================================================

    def get_coingecko_market(self, coin_id: str = "bitcoin") -> dict:
        """Get current market data from CoinGecko."""
        logger.info(f"Fetching CoinGecko market data for {coin_id}...")

        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false",
        }

        data = self._request(url, params)
        if not data:
            return {}

        market_data = data.get("market_data", {})

        return {
            "datetime": datetime.now(),
            "coin": coin_id,
            "price_usd": market_data.get("current_price", {}).get("usd"),
            "market_cap_usd": market_data.get("market_cap", {}).get("usd"),
            "total_volume_usd": market_data.get("total_volume", {}).get("usd"),
            "price_change_24h_pct": market_data.get("price_change_percentage_24h"),
            "price_change_7d_pct": market_data.get("price_change_percentage_7d"),
            "price_change_30d_pct": market_data.get("price_change_percentage_30d"),
            "ath_usd": market_data.get("ath", {}).get("usd"),
            "ath_change_pct": market_data.get("ath_change_percentage", {}).get("usd"),
            "circulating_supply": market_data.get("circulating_supply"),
            "total_supply": market_data.get("total_supply"),
        }

    def get_coingecko_history(
        self,
        coin_id: str = "bitcoin",
        days: int = 365,
    ) -> pd.DataFrame:
        """Get historical market data from CoinGecko."""
        logger.info(f"Fetching CoinGecko history for {coin_id} ({days} days)...")

        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily",
        }

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        # Parse prices
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        market_caps = data.get("market_caps", [])

        records = []
        for i, (ts, price) in enumerate(prices):
            record = {
                "datetime": pd.to_datetime(ts, unit="ms"),
                "price": price,
            }
            if i < len(volumes):
                record["volume"] = volumes[i][1]
            if i < len(market_caps):
                record["market_cap"] = market_caps[i][1]
            records.append(record)

        df = pd.DataFrame(records)
        df["coin"] = coin_id

        logger.info(f"  Collected {len(df)} records")
        return df

    # =========================================================================
    # DefiLlama - TVL Data
    # =========================================================================

    def get_defillama_tvl_history(self) -> pd.DataFrame:
        """Get total DeFi TVL history."""
        logger.info("Fetching DefiLlama TVL history...")

        url = "https://api.llama.fi/v2/historicalChainTvl"

        data = self._request(url)
        if not data:
            return pd.DataFrame()

        records = []
        for item in data:
            records.append(
                {
                    "datetime": pd.to_datetime(item.get("date"), unit="s"),
                    "tvl_usd": item.get("tvl"),
                }
            )

        df = pd.DataFrame(records)
        logger.info(f"  Collected {len(df)} TVL records")

        return df

    def get_defillama_chain_tvl(self, chain: str = "Ethereum") -> pd.DataFrame:
        """Get TVL history for specific chain."""
        logger.info(f"Fetching DefiLlama TVL for {chain}...")

        url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}"

        data = self._request(url)
        if not data:
            return pd.DataFrame()

        records = []
        for item in data:
            records.append(
                {
                    "datetime": pd.to_datetime(item.get("date"), unit="s"),
                    "tvl_usd": item.get("tvl"),
                }
            )

        df = pd.DataFrame(records)
        df["chain"] = chain

        logger.info(f"  Collected {len(df)} records")
        return df

    # =========================================================================
    # Blockchain.com - On-chain Metrics
    # =========================================================================

    def get_blockchain_metrics(self) -> dict:
        """Get basic Bitcoin on-chain metrics from Blockchain.com."""
        logger.info("Fetching Blockchain.com metrics...")

        metrics = {}

        # Hash rate
        url = "https://api.blockchain.info/charts/hash-rate?timespan=1year&format=json"
        data = self._request(url)
        if data and "values" in data:
            records = [
                {"datetime": pd.to_datetime(v["x"], unit="s"), "hash_rate": v["y"]}
                for v in data["values"]
            ]
            metrics["hash_rate"] = pd.DataFrame(records)
            logger.info(f"  Hash rate: {len(records)} records")

        time.sleep(2)

        # Transaction count
        url = "https://api.blockchain.info/charts/n-transactions?timespan=1year&format=json"
        data = self._request(url)
        if data and "values" in data:
            records = [
                {"datetime": pd.to_datetime(v["x"], unit="s"), "tx_count": v["y"]}
                for v in data["values"]
            ]
            metrics["tx_count"] = pd.DataFrame(records)
            logger.info(f"  TX count: {len(records)} records")

        time.sleep(2)

        # Active addresses (unique addresses)
        url = "https://api.blockchain.info/charts/n-unique-addresses?timespan=1year&format=json"
        data = self._request(url)
        if data and "values" in data:
            records = [
                {
                    "datetime": pd.to_datetime(v["x"], unit="s"),
                    "active_addresses": v["y"],
                }
                for v in data["values"]
            ]
            metrics["active_addresses"] = pd.DataFrame(records)
            logger.info(f"  Active addresses: {len(records)} records")

        return metrics

    # =========================================================================
    # Alternative.me - Fear & Greed Index
    # =========================================================================

    def get_fear_greed_history(self, limit: int = 0) -> pd.DataFrame:
        """
        Get Fear & Greed Index history.
        limit=0 means all available data.
        """
        logger.info("Fetching Fear & Greed Index...")

        url = "https://api.alternative.me/fng/"
        params = {"limit": limit, "format": "json"}

        data = self._request(url, params)
        if not data or "data" not in data:
            return pd.DataFrame()

        records = []
        for item in data["data"]:
            records.append(
                {
                    "datetime": pd.to_datetime(int(item["timestamp"]), unit="s"),
                    "value": int(item["value"]),
                    "classification": item["value_classification"],
                }
            )

        df = pd.DataFrame(records)
        df = df.sort_values("datetime").reset_index(drop=True)

        logger.info(
            f"  Collected {len(df)} F&G records ({df['datetime'].min()} to {df['datetime'].max()})"
        )

        return df


def main():
    parser = argparse.ArgumentParser(description="Alternative Data Collector")
    parser.add_argument("--all", action="store_true", help="Collect all data")
    parser.add_argument(
        "--coingecko", action="store_true", help="Collect CoinGecko data"
    )
    parser.add_argument(
        "--defillama", action="store_true", help="Collect DefiLlama TVL"
    )
    parser.add_argument(
        "--blockchain", action="store_true", help="Collect Blockchain.com metrics"
    )
    parser.add_argument(
        "--fear-greed", action="store_true", help="Update Fear & Greed Index"
    )

    args = parser.parse_args()

    if args.all:
        args.coingecko = True
        args.defillama = True
        args.blockchain = True
        args.fear_greed = True

    collector = FreeDataCollector()

    logger.info("=" * 60)
    logger.info("ALTERNATIVE DATA COLLECTOR")
    logger.info("=" * 60)

    # CoinGecko
    if args.coingecko:
        logger.info("\n[COINGECKO]")

        for coin in ["bitcoin", "ethereum"]:
            # Current market data
            market = collector.get_coingecko_market(coin)
            if market:
                df = pd.DataFrame([market])
                output_dir = OUTPUT_DIR / "coingecko"
                output_dir.mkdir(parents=True, exist_ok=True)

                # Append to daily file
                daily_path = output_dir / f"{coin}_market_daily.csv"
                if daily_path.exists():
                    existing = pd.read_csv(daily_path)
                    combined = pd.concat([existing, df], ignore_index=True)
                else:
                    combined = df
                combined.to_csv(daily_path, index=False)
                logger.info(f"  Saved: {daily_path}")

            time.sleep(3)

            # Historical data
            df_hist = collector.get_coingecko_history(coin, days=365)
            if not df_hist.empty:
                path = output_dir / f"{coin}_history_1y.csv"
                df_hist.to_csv(path, index=False)
                logger.info(f"  Saved: {path}")

            time.sleep(3)

    # DefiLlama
    if args.defillama:
        logger.info("\n[DEFILLAMA]")

        output_dir = OUTPUT_DIR / "defillama"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Total TVL
        df_tvl = collector.get_defillama_tvl_history()
        if not df_tvl.empty:
            path = output_dir / "total_tvl_history.csv"
            df_tvl.to_csv(path, index=False)
            logger.info(f"  Saved: {path}")

        time.sleep(2)

        # Chain-specific TVL
        for chain in ["Ethereum", "BSC", "Solana", "Arbitrum"]:
            df_chain = collector.get_defillama_chain_tvl(chain)
            if not df_chain.empty:
                path = output_dir / f"{chain.lower()}_tvl_history.csv"
                df_chain.to_csv(path, index=False)
                logger.info(f"  Saved: {path}")
            time.sleep(2)

    # Blockchain.com
    if args.blockchain:
        logger.info("\n[BLOCKCHAIN.COM]")

        output_dir = OUTPUT_DIR / "onchain"
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = collector.get_blockchain_metrics()

        for name, df in metrics.items():
            if not df.empty:
                path = output_dir / f"btc_{name}.csv"
                df.to_csv(path, index=False)
                logger.info(f"  Saved: {path}")

    # Fear & Greed
    if args.fear_greed:
        logger.info("\n[FEAR & GREED]")

        df_fg = collector.get_fear_greed_history(limit=0)
        if not df_fg.empty:
            path = OUTPUT_DIR / "FEAR_GREED_INDEX_updated.csv"
            df_fg.to_csv(path, index=False)
            logger.info(f"  Saved: {path}")

            # Compare with existing
            existing_path = OUTPUT_DIR / "FEAR_GREED_INDEX.csv"
            if existing_path.exists():
                existing = pd.read_csv(existing_path)
                logger.info(f"  Existing: {len(existing)} records")
                logger.info(f"  Updated: {len(df_fg)} records")
                logger.info(f"  New records: {len(df_fg) - len(existing)}")

    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
