#!/usr/bin/env python3
"""
Coinglass Data Collector

Collects:
1. Open Interest (aggregated across exchanges)
2. Liquidation data
3. Funding rates (aggregated)
4. Long/Short ratio

Note: Coinglass has rate limits. Use responsibly.

Usage:
    python collect_coinglass.py --all
    python collect_coinglass.py --oi
    python collect_coinglass.py --liquidation
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

OUTPUT_DIR = Path("E:/data/crypto_ohlcv/coinglass")

# Coinglass API endpoints (free tier)
BASE_URL = "https://open-api.coinglass.com/public/v2"

# Note: Free tier has limited endpoints
# For full access, you need API key from https://www.coinglass.com/pricing


class CoinglassCollector:
    """Coinglass data collector using public API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.session = requests.Session()

        if api_key:
            self.session.headers.update({
                "coinglassSecret": api_key,
            })

        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        })

    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with rate limiting."""
        url = f"{BASE_URL}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get("code") != "0":
                logger.warning(f"API error: {data.get('msg', 'Unknown error')}")
                return None

            return data.get("data")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None

        finally:
            time.sleep(1)  # Rate limiting

    def get_open_interest(self, symbol: str = "BTC") -> pd.DataFrame:
        """
        Get aggregated Open Interest across all exchanges.

        Returns current OI snapshot (not historical).
        For historical data, need to collect over time.
        """
        logger.info(f"Fetching OI for {symbol}...")

        data = self._request("open_interest", {"symbol": symbol})

        if not data:
            return pd.DataFrame()

        records = []
        for exchange_data in data:
            records.append({
                "exchange": exchange_data.get("exchangeName"),
                "open_interest": float(exchange_data.get("openInterest", 0)),
                "open_interest_usd": float(exchange_data.get("openInterestAmount", 0)),
                "h24_change_pct": float(exchange_data.get("h24Change", 0)),
            })

        df = pd.DataFrame(records)
        df["datetime"] = datetime.now()
        df["symbol"] = symbol

        # Calculate total
        total_oi = df["open_interest"].sum()
        total_oi_usd = df["open_interest_usd"].sum()

        logger.info(f"  Total OI: {total_oi:.2f} {symbol} (${total_oi_usd/1e9:.2f}B)")

        return df

    def get_liquidation_history(self, symbol: str = "BTC", time_type: str = "h24") -> pd.DataFrame:
        """
        Get liquidation data.

        time_type: h1, h4, h12, h24
        """
        logger.info(f"Fetching liquidations for {symbol} ({time_type})...")

        data = self._request("liquidation_history", {
            "symbol": symbol,
            "timeType": time_type,
        })

        if not data:
            return pd.DataFrame()

        records = []
        for item in data:
            records.append({
                "datetime": pd.to_datetime(item.get("createTime"), unit="ms"),
                "long_liquidation_usd": float(item.get("longLiquidationUsd", 0)),
                "short_liquidation_usd": float(item.get("shortLiquidationUsd", 0)),
                "total_liquidation_usd": float(item.get("liquidationUsd", 0)),
            })

        df = pd.DataFrame(records)
        df["symbol"] = symbol

        if not df.empty:
            total_liq = df["total_liquidation_usd"].sum()
            logger.info(f"  Total liquidations ({time_type}): ${total_liq/1e6:.2f}M")

        return df

    def get_funding_rate(self, symbol: str = "BTC") -> pd.DataFrame:
        """Get current funding rates across exchanges."""
        logger.info(f"Fetching funding rates for {symbol}...")

        data = self._request("funding", {"symbol": symbol})

        if not data:
            return pd.DataFrame()

        records = []
        for exchange_data in data:
            rate = exchange_data.get("rate")
            if rate is not None:
                records.append({
                    "exchange": exchange_data.get("exchangeName"),
                    "funding_rate": float(rate),
                    "next_funding_time": pd.to_datetime(
                        exchange_data.get("nextFundingTime"), unit="ms"
                    ) if exchange_data.get("nextFundingTime") else None,
                })

        df = pd.DataFrame(records)
        df["datetime"] = datetime.now()
        df["symbol"] = symbol

        if not df.empty:
            avg_rate = df["funding_rate"].mean()
            logger.info(f"  Avg funding rate: {avg_rate*100:.4f}%")

        return df

    def get_long_short_ratio(self, symbol: str = "BTC") -> pd.DataFrame:
        """Get long/short ratio across exchanges."""
        logger.info(f"Fetching L/S ratio for {symbol}...")

        data = self._request("long_short", {"symbol": symbol})

        if not data:
            return pd.DataFrame()

        records = []
        for exchange_data in data:
            records.append({
                "exchange": exchange_data.get("exchangeName"),
                "long_ratio": float(exchange_data.get("longRate", 0)),
                "short_ratio": float(exchange_data.get("shortRate", 0)),
                "long_short_ratio": float(exchange_data.get("longShortRatio", 0)),
            })

        df = pd.DataFrame(records)
        df["datetime"] = datetime.now()
        df["symbol"] = symbol

        return df


def collect_and_save(collector: CoinglassCollector, symbol: str = "BTC"):
    """Collect all data and save to files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now().strftime("%Y-%m-%d")

    # 1. Open Interest
    df_oi = collector.get_open_interest(symbol)
    if not df_oi.empty:
        # Save snapshot
        path = OUTPUT_DIR / f"oi_{symbol}_{timestamp}.csv"
        df_oi.to_csv(path, index=False)
        logger.info(f"Saved: {path}")

        # Append to daily file
        daily_path = OUTPUT_DIR / f"oi_{symbol}_daily.csv"
        if daily_path.exists():
            existing = pd.read_csv(daily_path)
            combined = pd.concat([existing, df_oi], ignore_index=True)
        else:
            combined = df_oi
        combined.to_csv(daily_path, index=False)

    time.sleep(2)

    # 2. Liquidation
    df_liq = collector.get_liquidation_history(symbol, "h24")
    if not df_liq.empty:
        path = OUTPUT_DIR / f"liquidation_{symbol}_{timestamp}.csv"
        df_liq.to_csv(path, index=False)
        logger.info(f"Saved: {path}")

        daily_path = OUTPUT_DIR / f"liquidation_{symbol}_daily.csv"
        if daily_path.exists():
            existing = pd.read_csv(daily_path)
            combined = pd.concat([existing, df_liq], ignore_index=True)
        else:
            combined = df_liq
        combined.to_csv(daily_path, index=False)

    time.sleep(2)

    # 3. Funding Rate
    df_fr = collector.get_funding_rate(symbol)
    if not df_fr.empty:
        path = OUTPUT_DIR / f"funding_{symbol}_{timestamp}.csv"
        df_fr.to_csv(path, index=False)
        logger.info(f"Saved: {path}")

    time.sleep(2)

    # 4. Long/Short Ratio
    df_ls = collector.get_long_short_ratio(symbol)
    if not df_ls.empty:
        path = OUTPUT_DIR / f"longshort_{symbol}_{timestamp}.csv"
        df_ls.to_csv(path, index=False)
        logger.info(f"Saved: {path}")


def create_summary(symbol: str = "BTC"):
    """Create summary from collected data."""
    logger.info("\n" + "=" * 60)
    logger.info(f"SUMMARY FOR {symbol}")
    logger.info("=" * 60)

    # Read latest OI
    oi_path = OUTPUT_DIR / f"oi_{symbol}_daily.csv"
    if oi_path.exists():
        df = pd.read_csv(oi_path)
        total_oi = df.groupby("datetime")["open_interest_usd"].sum().iloc[-1]
        logger.info(f"Total OI: ${total_oi/1e9:.2f}B")

    # Read latest liquidation
    liq_path = OUTPUT_DIR / f"liquidation_{symbol}_daily.csv"
    if liq_path.exists():
        df = pd.read_csv(liq_path)
        if not df.empty:
            latest = df.iloc[-1]
            logger.info(f"24h Long Liq: ${latest['long_liquidation_usd']/1e6:.2f}M")
            logger.info(f"24h Short Liq: ${latest['short_liquidation_usd']/1e6:.2f}M")


def main():
    parser = argparse.ArgumentParser(description="Coinglass Data Collector")
    parser.add_argument("--all", action="store_true", help="Collect all data")
    parser.add_argument("--oi", action="store_true", help="Collect Open Interest")
    parser.add_argument("--liquidation", action="store_true", help="Collect liquidations")
    parser.add_argument("--funding", action="store_true", help="Collect funding rates")
    parser.add_argument("--longshort", action="store_true", help="Collect L/S ratio")
    parser.add_argument("--symbols", type=str, default="BTC,ETH", help="Symbols")
    parser.add_argument("--api-key", type=str, default=None, help="Coinglass API key")

    args = parser.parse_args()

    if args.all:
        args.oi = True
        args.liquidation = True
        args.funding = True
        args.longshort = True

    symbols = args.symbols.split(",")

    logger.info("=" * 60)
    logger.info("COINGLASS DATA COLLECTOR")
    logger.info("=" * 60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"API Key: {'Set' if args.api_key else 'Not set (using free tier)'}")

    collector = CoinglassCollector(api_key=args.api_key)

    for symbol in symbols:
        logger.info(f"\n--- Collecting {symbol} ---")
        collect_and_save(collector, symbol)
        create_summary(symbol)

    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Data saved to: {OUTPUT_DIR}")
    logger.info("\nNote: Run this script daily to build historical data.")
    logger.info("Consider setting up a scheduled task (cron/Task Scheduler).")


if __name__ == "__main__":
    main()
