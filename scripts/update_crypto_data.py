"""
Crypto Data Updater

Downloads latest OHLCV data from:
- Upbit (KRW pairs)
- Binance Spot (USDT pairs)

Updates existing CSV files in E:/data/crypto_ohlcv/
"""

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_ROOT = Path("E:/data/crypto_ohlcv")


def update_upbit_data(
    symbols: Optional[List[str]] = None, days_back: int = 200
) -> Dict[str, int]:
    """
    Update Upbit OHLCV data.

    Args:
        symbols: List of symbols (without KRW- prefix). If None, updates all existing.
        days_back: Number of days to fetch

    Returns:
        Dict of symbol -> rows added
    """
    try:
        import pyupbit
    except ImportError:
        logger.error("pyupbit not installed. Run: pip install pyupbit")
        return {}

    upbit_dir = DATA_ROOT / "upbit_1d"
    results = {}

    if symbols is None:
        # Get all existing symbols
        symbols = [f.stem for f in upbit_dir.glob("*.csv")]

    logger.info(f"[Upbit] Updating {len(symbols)} symbols...")

    for i, symbol in enumerate(symbols):
        try:
            ticker = f"KRW-{symbol}"

            # Fetch data
            df = pyupbit.get_ohlcv(ticker, interval="day", count=days_back)

            if df is None or df.empty:
                logger.warning(f"[Upbit] No data for {symbol}")
                continue

            # Rename columns to match existing format
            df = df.reset_index()
            df.columns = ["date", "open", "high", "low", "close", "volume", "value"]
            df = df[["date", "open", "high", "low", "close", "volume"]]

            # Load existing data
            csv_path = upbit_dir / f"{symbol}.csv"
            if csv_path.exists():
                existing = pd.read_csv(csv_path, parse_dates=["date"])

                # Get last date in existing data
                last_date = existing["date"].max()

                # Filter new data
                df["date"] = pd.to_datetime(df["date"])
                new_data = df[df["date"] > last_date]

                if not new_data.empty:
                    # Append new data
                    combined = pd.concat([existing, new_data], ignore_index=True)
                    combined = combined.sort_values("date").drop_duplicates(
                        subset=["date"]
                    )
                    combined.to_csv(csv_path, index=False)
                    results[symbol] = len(new_data)
                    logger.info(
                        f"[Upbit] {symbol}: +{len(new_data)} rows (total: {len(combined)})"
                    )
                else:
                    results[symbol] = 0
            else:
                # Save new file
                df.to_csv(csv_path, index=False)
                results[symbol] = len(df)
                logger.info(f"[Upbit] {symbol}: Created with {len(df)} rows")

            # Rate limiting
            if (i + 1) % 10 == 0:
                time.sleep(1)

        except Exception as e:
            logger.error(f"[Upbit] Error updating {symbol}: {e}")
            continue

    return results


def update_binance_data(
    symbols: Optional[List[str]] = None, days_back: int = 200, market_type: str = "spot"
) -> Dict[str, int]:
    """
    Update Binance OHLCV data.

    Args:
        symbols: List of symbols (e.g., ["BTC", "ETH"]). If None, updates all existing.
        days_back: Number of days to fetch
        market_type: "spot" or "futures"

    Returns:
        Dict of symbol -> rows added
    """
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed. Run: pip install ccxt")
        return {}

    binance_dir = DATA_ROOT / f"binance_{market_type}_1d"
    results = {}

    if symbols is None:
        # Get all existing symbols
        symbols = [f.stem for f in binance_dir.glob("*.csv")]

    logger.info(f"[Binance {market_type}] Updating {len(symbols)} symbols...")

    # Initialize exchange
    if market_type == "futures":
        exchange = ccxt.binanceusdm({"enableRateLimit": True})
    else:
        exchange = ccxt.binance({"enableRateLimit": True})

    since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)

    for i, symbol in enumerate(symbols):
        try:
            # Construct pair symbol
            if symbol.endswith("USDT"):
                pair = symbol
            else:
                pair = f"{symbol}USDT" if not symbol.endswith("USDT") else symbol

            # Fetch OHLCV
            ohlcv = exchange.fetch_ohlcv(pair, timeframe="1d", since=since, limit=1000)

            if not ohlcv:
                logger.warning(f"[Binance] No data for {symbol}")
                continue

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime(
                "%Y-%m-%d"
            )

            # Load existing data
            csv_path = binance_dir / f"{symbol}.csv"
            if csv_path.exists():
                existing = pd.read_csv(csv_path)

                # Get last date in existing data
                last_date = existing["timestamp"].max()

                # Filter new data
                new_data = df[df["timestamp"] > last_date]

                if not new_data.empty:
                    # Append new data
                    combined = pd.concat([existing, new_data], ignore_index=True)
                    combined = combined.sort_values("timestamp").drop_duplicates(
                        subset=["timestamp"]
                    )
                    combined.to_csv(csv_path, index=False)
                    results[symbol] = len(new_data)
                    logger.info(
                        f"[Binance] {symbol}: +{len(new_data)} rows (total: {len(combined)})"
                    )
                else:
                    results[symbol] = 0
            else:
                # Save new file
                df.to_csv(csv_path, index=False)
                results[symbol] = len(df)
                logger.info(f"[Binance] {symbol}: Created with {len(df)} rows")

            # Rate limiting
            if (i + 1) % 5 == 0:
                time.sleep(0.5)

        except Exception as e:
            logger.error(f"[Binance] Error updating {symbol}: {e}")
            continue

    return results


def get_data_summary() -> Dict[str, dict]:
    """Get summary of data in each folder."""
    summary = {}

    for folder in DATA_ROOT.iterdir():
        if folder.is_dir() and not folder.name.startswith("."):
            csv_files = list(folder.glob("*.csv"))
            if csv_files:
                # Sample one file to get date range
                sample = pd.read_csv(csv_files[0])
                date_col = "date" if "date" in sample.columns else "timestamp"

                if date_col in sample.columns:
                    min_date = sample[date_col].min()
                    max_date = sample[date_col].max()
                else:
                    min_date = max_date = "unknown"

                summary[folder.name] = {
                    "files": len(csv_files),
                    "min_date": min_date,
                    "max_date": max_date,
                }

    return summary


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Update crypto OHLCV data")
    parser.add_argument(
        "--exchange", choices=["upbit", "binance", "all"], default="all"
    )
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to update")
    parser.add_argument("--days", type=int, default=60, help="Days to fetch")
    parser.add_argument("--summary", action="store_true", help="Show data summary only")

    args = parser.parse_args()

    if args.summary:
        print("\n=== Data Summary ===")
        summary = get_data_summary()
        for folder, info in sorted(summary.items()):
            print(f"\n{folder}:")
            print(f"  Files: {info['files']}")
            print(f"  Date range: {info['min_date']} ~ {info['max_date']}")
        return

    print(f"\n=== Updating Crypto Data (last {args.days} days) ===\n")

    total_updated = 0

    if args.exchange in ["upbit", "all"]:
        results = update_upbit_data(symbols=args.symbols, days_back=args.days)
        updated = sum(v for v in results.values() if v > 0)
        total_updated += updated
        print(
            f"\n[Upbit] Updated {updated} rows across {len([v for v in results.values() if v > 0])} symbols"
        )

    if args.exchange in ["binance", "all"]:
        results = update_binance_data(
            symbols=args.symbols, days_back=args.days, market_type="spot"
        )
        updated = sum(v for v in results.values() if v > 0)
        total_updated += updated
        print(
            f"\n[Binance Spot] Updated {updated} rows across {len([v for v in results.values() if v > 0])} symbols"
        )

    print(f"\n=== Total: {total_updated} rows updated ===")

    # Show final summary
    print("\n=== Final Data Summary ===")
    summary = get_data_summary()
    for folder in ["upbit_1d", "binance_spot_1d"]:
        if folder in summary:
            info = summary[folder]
            print(
                f"{folder}: {info['files']} files, {info['min_date']} ~ {info['max_date']}"
            )


if __name__ == "__main__":
    main()
