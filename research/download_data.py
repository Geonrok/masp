#!/usr/bin/env python3
"""
Download free data for strategy testing:
1. Social sentiment: alternative.me Fear & Greed (full history, refresh)
2. Binance trade data: aggregate trades from data.binance.vision
3. Deribit options OHLCV from CryptoDataDownload
"""

import os
import io
import zipfile
import time
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Use project HTTP wrapper if available, else requests
try:
    from libs.adapters.resilient_client import ResilientClient

    session = ResilientClient()
except Exception:
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 CryptoResearch/1.0"})

DATA_ROOT = Path("E:/data")
SOCIAL_DIR = DATA_ROOT / "social_sentiment"
TRADES_DIR = DATA_ROOT / "binance_trades"
OPTIONS_DIR = DATA_ROOT / "crypto_options"

for d in [SOCIAL_DIR, TRADES_DIR, OPTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def download_fear_greed_full():
    """Download full Fear & Greed Index history from alternative.me"""
    print("=" * 60)
    print("1. Fear & Greed Index (Full History)")
    print("=" * 60)

    url = "https://api.alternative.me/fng/?limit=0&format=json"
    try:
        resp = session.get(url, timeout=30)
        data = resp.json()
        records = data.get("data", [])
        print(f"  Downloaded {len(records)} days")

        rows = []
        for r in records:
            rows.append(
                {
                    "datetime": pd.to_datetime(int(r["timestamp"]), unit="s"),
                    "fear_greed_value": int(r["value"]),
                    "classification": r["value_classification"],
                }
            )

        df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
        path = SOCIAL_DIR / "fear_greed_full.csv"
        df.to_csv(path, index=False)
        print(f"  Saved to {path}")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()


def download_binance_aggr_trades(symbol="BTCUSDT", months=24):
    """Download monthly aggregate trade data from data.binance.vision"""
    print(f"\n{'=' * 60}")
    print(f"2. Binance Aggregate Trades: {symbol}")
    print("=" * 60)

    base_url = "https://data.binance.vision/data/futures/um/monthly/aggTrades"
    symbol_dir = TRADES_DIR / symbol
    symbol_dir.mkdir(exist_ok=True)

    now = datetime.now()
    all_dfs = []

    for m in range(months, 0, -1):
        dt = now - timedelta(days=m * 30)
        year = dt.year
        month = dt.month
        filename = f"{symbol}-aggTrades-{year}-{month:02d}.zip"
        url = f"{base_url}/{symbol}/{filename}"
        csv_path = symbol_dir / f"{year}-{month:02d}.csv"

        if csv_path.exists():
            print(f"  {year}-{month:02d}: already exists, skipping")
            try:
                df = pd.read_csv(csv_path, nrows=1)
                all_dfs.append(csv_path)
            except:
                pass
            continue

        try:
            print(f"  {year}-{month:02d}: downloading...", end=" ", flush=True)
            resp = session.get(url, timeout=60)
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}")
                continue

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(
                        f,
                        header=None,
                        names=[
                            "agg_trade_id",
                            "price",
                            "quantity",
                            "first_trade_id",
                            "last_trade_id",
                            "timestamp",
                            "is_buyer_maker",
                            "is_best_match",
                        ],
                    )
                    df.to_csv(csv_path, index=False)
                    print(f"{len(df):,} trades")
                    all_dfs.append(csv_path)

            time.sleep(0.5)
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n  Total months downloaded: {len(all_dfs)}")
    return all_dfs


def build_trade_features(symbol="BTCUSDT"):
    """Build hourly features from aggregate trade data."""
    print(f"\n{'=' * 60}")
    print(f"3. Building Trade Features: {symbol}")
    print("=" * 60)

    symbol_dir = TRADES_DIR / symbol
    if not symbol_dir.exists():
        print("  No trade data found")
        return pd.DataFrame()

    all_chunks = []
    for f in sorted(symbol_dir.glob("*.csv")):
        try:
            df = pd.read_csv(f)
            if "timestamp" in df.columns:
                # Handle both ms and us timestamps
                ts = df["timestamp"]
                if ts.iloc[0] > 1e15:  # microseconds
                    df["datetime"] = pd.to_datetime(ts, unit="us")
                else:
                    df["datetime"] = pd.to_datetime(ts, unit="ms")
                all_chunks.append(df)
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")

    if not all_chunks:
        print("  No valid data")
        return pd.DataFrame()

    print(f"  Loaded {len(all_chunks)} monthly files")
    trades = pd.concat(all_chunks, ignore_index=True)
    trades = trades.sort_values("datetime").reset_index(drop=True)
    print(f"  Total trades: {len(trades):,}")
    print(f"  Date range: {trades['datetime'].min()} to {trades['datetime'].max()}")

    # Resample to hourly
    trades.set_index("datetime", inplace=True)
    trades["price"] = trades["price"].astype(float)
    trades["quantity"] = trades["quantity"].astype(float)
    trades["usd_volume"] = trades["price"] * trades["quantity"]
    trades["is_buyer_maker"] = trades["is_buyer_maker"].astype(bool)

    hourly = pd.DataFrame()
    hourly["close"] = trades["price"].resample("1h").last()
    hourly["open"] = trades["price"].resample("1h").first()
    hourly["high"] = trades["price"].resample("1h").max()
    hourly["low"] = trades["price"].resample("1h").min()
    hourly["volume"] = trades["quantity"].resample("1h").sum()
    hourly["usd_volume"] = trades["usd_volume"].resample("1h").sum()
    hourly["trade_count"] = trades["price"].resample("1h").count()

    # Buy/sell volume
    buy_trades = trades[~trades["is_buyer_maker"]]
    sell_trades = trades[trades["is_buyer_maker"]]
    hourly["buy_volume"] = buy_trades["quantity"].resample("1h").sum()
    hourly["sell_volume"] = sell_trades["quantity"].resample("1h").sum()
    hourly["buy_volume"] = hourly["buy_volume"].fillna(0)
    hourly["sell_volume"] = hourly["sell_volume"].fillna(0)

    # Derived features
    hourly["buy_sell_ratio"] = hourly["buy_volume"] / (hourly["sell_volume"] + 1e-10)
    hourly["buy_pct"] = hourly["buy_volume"] / (
        hourly["buy_volume"] + hourly["sell_volume"] + 1e-10
    )
    hourly["avg_trade_size"] = hourly["usd_volume"] / (hourly["trade_count"] + 1e-10)

    # Large trade detection (trades > mean + 2*std)
    trades["is_large"] = (
        trades["usd_volume"]
        > trades["usd_volume"].rolling(1000, min_periods=100).mean()
        + 2 * trades["usd_volume"].rolling(1000, min_periods=100).std()
    )
    hourly["large_trade_count"] = trades["is_large"].resample("1h").sum()
    hourly["large_trade_pct"] = hourly["large_trade_count"] / (
        hourly["trade_count"] + 1e-10
    )

    hourly = hourly.dropna(subset=["close"]).reset_index()
    path = SOCIAL_DIR / f"{symbol}_trade_features_1h.csv"
    hourly.to_csv(path, index=False)
    print(f"  Saved hourly features: {len(hourly)} rows to {path}")
    print(f"  Columns: {list(hourly.columns)}")

    return hourly


def download_options_data():
    """Download Deribit BTC options data from CryptoDataDownload."""
    print(f"\n{'=' * 60}")
    print("4. Deribit Options Data")
    print("=" * 60)

    # CryptoDataDownload Deribit DVOL (volatility index)
    urls = {
        "deribit_btc_dvol": "https://www.cryptodatadownload.com/cdd/Deribit_BTCDVOL_1h.csv",
        "deribit_eth_dvol": "https://www.cryptodatadownload.com/cdd/Deribit_ETHDVOL_1h.csv",
    }

    for name, url in urls.items():
        path = OPTIONS_DIR / f"{name}.csv"
        if path.exists():
            print(f"  {name}: already exists")
            continue

        try:
            print(f"  {name}: downloading...", end=" ", flush=True)
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                # Skip the first line (header comment)
                lines = resp.text.split("\n")
                # Find the actual header
                start = 0
                for i, line in enumerate(lines):
                    if "date" in line.lower() or "unix" in line.lower():
                        start = i
                        break
                content = "\n".join(lines[start:])
                with open(path, "w") as f:
                    f.write(content)
                df = pd.read_csv(path)
                print(f"{len(df)} rows")
            else:
                print(f"HTTP {resp.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")


def main():
    print("=" * 70)
    print("DATA DOWNLOAD FOR STRATEGY TESTING")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}\n")

    # 1. Fear & Greed (refresh)
    fgi = download_fear_greed_full()

    # 2. Binance aggregate trades (BTC, ETH - last 24 months)
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        download_binance_aggr_trades(symbol, months=24)

    # 3. Build trade features
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        build_trade_features(symbol)

    # 4. Options data
    download_options_data()

    print(f"\n{'=' * 70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'=' * 70}")
    print(f"Finished: {datetime.now().isoformat()}")

    # Summary
    print("\nData available for testing:")
    for d in [SOCIAL_DIR, TRADES_DIR, OPTIONS_DIR]:
        files = list(d.rglob("*.csv"))
        total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"  {d}: {len(files)} files, {total_size:.0f} MB")


if __name__ == "__main__":
    main()
