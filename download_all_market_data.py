"""
Unified Market Data Downloader (KOSPI & KOSDAQ)
===============================================
Downloads OHLCV data for all listed stocks in KOSPI and KOSDAQ.
Skips already downloaded files to save time.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from pykrx import stock
from tqdm import tqdm

# Configuration
BASE_DIR = Path("E:/투자/data/kr_stock")
KOSPI_DIR = BASE_DIR / "kospi_ohlcv"
KOSDAQ_DIR = BASE_DIR / "kosdaq_ohlcv"

KOSPI_DIR.mkdir(parents=True, exist_ok=True)
KOSDAQ_DIR.mkdir(parents=True, exist_ok=True)

END_DATE = datetime.now().strftime("%Y%m%d")
START_DATE = (datetime.now() - timedelta(days=365 * 10)).strftime("%Y%m%d")


def get_market_tickers(market):
    try:
        tickers = stock.get_market_ticker_list(END_DATE, market=market)
        print(f"Found {len(tickers)} {market} tickers")
        return tickers
    except Exception as e:
        print(f"Error fetching {market} tickers: {e}")
        return []


def download_ticker(ticker, market, save_dir):
    try:
        # Check if already exists
        file_path = save_dir / f"{ticker}.csv"
        if file_path.exists():
            return "skipped"

        df = stock.get_market_ohlcv(START_DATE, END_DATE, ticker)

        if df.empty:
            return "empty"

        if len(df) < 100:
            return "too_short"

        df = df.reset_index()

        # Standardize columns
        # pykrx returns: 날짜, 시가, 고가, 저가, 종가, 거래량, [거래대금, 등락률...]
        # We need: Date, Open, High, Low, Close, Volume

        # 1. Rename Korean columns to English
        col_map = {
            "날짜": "Date",
            "시가": "Open",
            "고가": "High",
            "저가": "Low",
            "종가": "Close",
            "거래량": "Volume",
        }
        df = df.rename(columns=col_map)

        # 2. Check if English columns exist (pykrx sometimes returns English)
        required = ["Date", "Open", "High", "Low", "Close", "Volume"]

        # If 'Date' is not in columns but index was reset, the first column is date
        if "Date" not in df.columns:
            df.columns.values[0] = "Date"

        # Select only required
        if all(c in df.columns for c in required):
            df = df[required]
        else:
            # Fallback by index
            df = df.iloc[:, :6]
            df.columns = required

        # Save
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        return "downloaded"

    except Exception as e:
        # print(f"Error {ticker}: {e}")
        return "error"


def run_download(market):
    print(f"\nStarting {market} download...")
    tickers = get_market_tickers(market)
    save_dir = KOSPI_DIR if market == "KOSPI" else KOSDAQ_DIR

    stats = {"downloaded": 0, "skipped": 0, "empty": 0, "too_short": 0, "error": 0}

    pbar = tqdm(tickers)
    for ticker in pbar:
        status = download_ticker(ticker, market, save_dir)
        stats[status] += 1
        pbar.set_description(
            f"{market} | DL: {stats['downloaded']} | Skip: {stats['skipped']}"
        )

        if status == "downloaded":
            time.sleep(0.1)  # Rate limit

    print(f"\n{market} Complete. Stats: {stats}")


if __name__ == "__main__":
    print(f"Downloading data from {START_DATE} to {END_DATE}")
    run_download("KOSPI")
    run_download("KOSDAQ")
