#!/usr/bin/env python3
"""
Binance Futures Data Collector

Collects:
1. Funding Rate (8-hour intervals, 2020-present)
2. Open Interest History (daily, 2020-present)
3. Long/Short Ratio (daily, 2020-present)
4. Taker Buy/Sell Volume (daily, 2020-present)

Usage:
    python collect_binance_data.py --all
    python collect_binance_data.py --funding-rate
    python collect_binance_data.py --open-interest
"""

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("E:/data/crypto_ohlcv")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

BASE_URL = "https://fapi.binance.com"


def collect_funding_rate(symbol: str, start_date: str = "2020-01-01") -> pd.DataFrame:
    """Collect historical funding rate."""
    url = f"{BASE_URL}/fapi/v1/fundingRate"

    all_data = []
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp.now().timestamp() * 1000)

    logger.info(f"Collecting funding rate for {symbol} from {start_date}...")

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "startTime": start_ts,
            "limit": 1000,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break

        if not data:
            break

        all_data.extend(data)
        start_ts = data[-1]["fundingTime"] + 1

        if len(all_data) % 5000 == 0:
            logger.info(f"  Collected {len(all_data)} funding rate records...")

        time.sleep(0.2)  # Rate limit

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)
    df = df[["datetime", "fundingRate", "fundingTime"]].drop_duplicates()
    df = df.sort_values("datetime").reset_index(drop=True)

    logger.info(
        f"  Total: {len(df)} records ({df['datetime'].min()} to {df['datetime'].max()})"
    )

    return df


def collect_open_interest(
    symbol: str, period: str = "1d", start_date: str = "2020-01-01"
) -> pd.DataFrame:
    """Collect historical Open Interest (max 500 records, no startTime support)."""
    url = f"{BASE_URL}/futures/data/openInterestHist"

    logger.info(f"Collecting OI for {symbol} ({period})...")

    params = {
        "symbol": symbol,
        "period": period,
        "limit": 500,  # Maximum allowed
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
    df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
    df = df[["datetime", "sumOpenInterest", "sumOpenInterestValue"]].drop_duplicates()
    df = df.sort_values("datetime").reset_index(drop=True)

    logger.info(
        f"  Total: {len(df)} records ({df['datetime'].min()} to {df['datetime'].max()})"
    )

    return df


def collect_long_short_ratio(
    symbol: str, period: str = "1d", start_date: str = "2020-01-01"
) -> pd.DataFrame:
    """Collect top trader long/short ratio (max 500 records)."""
    url = f"{BASE_URL}/futures/data/topLongShortAccountRatio"

    logger.info(f"Collecting L/S ratio for {symbol} ({period})...")

    params = {
        "symbol": symbol,
        "period": period,
        "limit": 500,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["longShortRatio"] = df["longShortRatio"].astype(float)
    df["longAccount"] = df["longAccount"].astype(float)
    df["shortAccount"] = df["shortAccount"].astype(float)
    df = df[
        ["datetime", "longShortRatio", "longAccount", "shortAccount"]
    ].drop_duplicates()
    df = df.sort_values("datetime").reset_index(drop=True)

    logger.info(
        f"  Total: {len(df)} records ({df['datetime'].min()} to {df['datetime'].max()})"
    )

    return df


def collect_taker_buy_sell(
    symbol: str, period: str = "1d", start_date: str = "2020-01-01"
) -> pd.DataFrame:
    """Collect taker buy/sell volume ratio (max 500 records)."""
    url = f"{BASE_URL}/futures/data/takerlongshortRatio"

    logger.info(f"Collecting taker volume for {symbol} ({period})...")

    params = {
        "symbol": symbol,
        "period": period,
        "limit": 500,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f"Error: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["buySellRatio"] = df["buySellRatio"].astype(float)
    df["buyVol"] = df["buyVol"].astype(float)
    df["sellVol"] = df["sellVol"].astype(float)
    df = df[["datetime", "buySellRatio", "buyVol", "sellVol"]].drop_duplicates()
    df = df.sort_values("datetime").reset_index(drop=True)

    logger.info(
        f"  Total: {len(df)} records ({df['datetime'].min()} to {df['datetime'].max()})"
    )

    return df


def save_data(df: pd.DataFrame, folder: str, filename: str):
    """Save DataFrame to CSV."""
    output_path = OUTPUT_DIR / folder
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / filename
    df.to_csv(file_path, index=False)
    logger.info(f"Saved: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect Binance Futures Data")
    parser.add_argument("--all", action="store_true", help="Collect all data types")
    parser.add_argument(
        "--funding-rate", action="store_true", help="Collect funding rate"
    )
    parser.add_argument(
        "--open-interest", action="store_true", help="Collect open interest"
    )
    parser.add_argument(
        "--long-short", action="store_true", help="Collect long/short ratio"
    )
    parser.add_argument(
        "--taker-volume", action="store_true", help="Collect taker buy/sell"
    )
    parser.add_argument(
        "--symbols", type=str, default="BTCUSDT,ETHUSDT", help="Symbols to collect"
    )
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date")

    args = parser.parse_args()

    if args.all:
        args.funding_rate = True
        args.open_interest = True
        args.long_short = True
        args.taker_volume = True

    symbols = args.symbols.split(",")

    logger.info("=" * 60)
    logger.info("BINANCE DATA COLLECTOR")
    logger.info("=" * 60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Start date: {args.start}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # Funding Rate
    if args.funding_rate:
        logger.info("\n[1/4] FUNDING RATE")
        for symbol in symbols:
            df = collect_funding_rate(symbol, args.start)
            if not df.empty:
                save_data(df, "binance_funding_rate", f"{symbol}_funding_full.csv")

    # Open Interest
    if args.open_interest:
        logger.info("\n[2/4] OPEN INTEREST")
        for symbol in symbols:
            df = collect_open_interest(symbol, "1d", args.start)
            if not df.empty:
                save_data(df, "binance_open_interest", f"{symbol}_oi_full.csv")

    # Long/Short Ratio
    if args.long_short:
        logger.info("\n[3/4] LONG/SHORT RATIO")
        for symbol in symbols:
            df = collect_long_short_ratio(symbol, "1d", args.start)
            if not df.empty:
                save_data(df, "binance_long_short_ratio", f"{symbol}_lsratio_full.csv")

    # Taker Buy/Sell
    if args.taker_volume:
        logger.info("\n[4/4] TAKER BUY/SELL VOLUME")
        for symbol in symbols:
            df = collect_taker_buy_sell(symbol, "1d", args.start)
            if not df.empty:
                save_data(df, "binance_taker_volume", f"{symbol}_taker.csv")

    logger.info("\n" + "=" * 60)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
