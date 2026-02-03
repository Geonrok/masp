"""
Download Additional Data for Extended Backtesting
- KOSDAQ OHLCV
- Investor trading data (Foreign, Institution, Individual)
- Fundamental data (PER, PBR, etc.)
- Sector indices
"""

import sys

sys.path.insert(0, "E:/투자/Multi-Asset Strategy Platform")

import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from pykrx import stock

# Directories
DATA_DIR = Path("E:/투자/data/kr_stock")
KOSDAQ_DIR = DATA_DIR / "kosdaq_ohlcv"
INVESTOR_DIR = DATA_DIR / "investor_trading"
FUNDAMENTAL_DIR = DATA_DIR / "fundamentals"
SECTOR_DIR = DATA_DIR / "sector_indices"

for d in [KOSDAQ_DIR, INVESTOR_DIR, FUNDAMENTAL_DIR, SECTOR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Date range
END_DATE = datetime.now().strftime("%Y%m%d")
START_DATE = (datetime.now() - timedelta(days=365 * 10)).strftime("%Y%m%d")

print("=" * 70)
print("DOWNLOADING ADDITIONAL DATA")
print("=" * 70)
print(f"Date range: {START_DATE} ~ {END_DATE}")
print()

# ============================================================================
# 1. KOSDAQ OHLCV DATA
# ============================================================================
print("=" * 70)
print("1. KOSDAQ OHLCV DATA")
print("=" * 70)

try:
    kosdaq_tickers = stock.get_market_ticker_list(END_DATE, market="KOSDAQ")
    print(f"Total KOSDAQ tickers: {len(kosdaq_tickers)}")

    downloaded = 0
    errors = 0

    for i, ticker in enumerate(kosdaq_tickers):
        try:
            df = stock.get_market_ohlcv(START_DATE, END_DATE, ticker)
            if len(df) > 100:
                df = df.reset_index()
                df.columns = [
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "Value",
                    "Change",
                ]
                df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
                df.to_csv(
                    KOSDAQ_DIR / f"{ticker}.csv", index=False, encoding="utf-8-sig"
                )
                downloaded += 1

            if (i + 1) % 100 == 0:
                print(
                    f"  Progress: {i+1}/{len(kosdaq_tickers)} - Downloaded: {downloaded}"
                )
                time.sleep(1)

        except Exception:
            errors += 1
            continue

    print(f"KOSDAQ downloaded: {downloaded} stocks")
    print(f"Errors: {errors}")

except Exception as e:
    print(f"KOSDAQ download failed: {e}")

# ============================================================================
# 2. INVESTOR TRADING DATA (KOSPI)
# ============================================================================
print()
print("=" * 70)
print("2. INVESTOR TRADING DATA (KOSPI)")
print("=" * 70)

try:
    kospi_tickers = stock.get_market_ticker_list(END_DATE, market="KOSPI")
    print(f"Getting investor data for {len(kospi_tickers)} KOSPI stocks")

    # Sample tickers for testing (top 100 by market cap would be ideal)
    # For now, download for all tickers
    downloaded = 0

    for i, ticker in enumerate(kospi_tickers):
        try:
            # Get investor trading data
            df = stock.get_market_trading_value_by_date(START_DATE, END_DATE, ticker)
            if len(df) > 100:
                df = df.reset_index()
                # Columns: 날짜, 기관합계, 기타법인, 개인, 외국인합계, 전체
                df.to_csv(
                    INVESTOR_DIR / f"{ticker}_investor.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                downloaded += 1

            if (i + 1) % 50 == 0:
                print(
                    f"  Progress: {i+1}/{len(kospi_tickers)} - Downloaded: {downloaded}"
                )
                time.sleep(1)

        except Exception:
            continue

    print(f"Investor data downloaded: {downloaded} stocks")

except Exception as e:
    print(f"Investor data download failed: {e}")

# ============================================================================
# 3. FUNDAMENTAL DATA (KOSPI)
# ============================================================================
print()
print("=" * 70)
print("3. FUNDAMENTAL DATA (KOSPI)")
print("=" * 70)

try:
    # Get fundamental data for recent dates (monthly snapshots)
    dates_to_check = (
        pd.date_range(start=START_DATE, end=END_DATE, freq="M")
        .strftime("%Y%m%d")
        .tolist()
    )

    all_fundamentals = []

    for date in dates_to_check[-36:]:  # Last 3 years monthly
        try:
            df = stock.get_market_fundamental(date, market="KOSPI")
            if len(df) > 0:
                df = df.reset_index()
                df["Date"] = date
                all_fundamentals.append(df)
            time.sleep(0.5)
        except:
            continue

    if all_fundamentals:
        fundamental_df = pd.concat(all_fundamentals, ignore_index=True)
        fundamental_df.to_csv(
            FUNDAMENTAL_DIR / "kospi_fundamentals.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Fundamental data saved: {len(fundamental_df)} rows")
    else:
        print("No fundamental data retrieved")

except Exception as e:
    print(f"Fundamental data download failed: {e}")

# ============================================================================
# 4. SECTOR INDEX DATA
# ============================================================================
print()
print("=" * 70)
print("4. SECTOR INDEX DATA")
print("=" * 70)

try:
    # Get sector indices
    sector_list = [
        ("1001", "KOSPI"),
        ("1002", "KOSPI_Large"),
        ("1003", "KOSPI_Mid"),
        ("1004", "KOSPI_Small"),
        ("2001", "KOSDAQ"),
        ("2002", "KOSDAQ_Large"),
        ("2003", "KOSDAQ_Mid"),
        ("2004", "KOSDAQ_Small"),
    ]

    for code, name in sector_list:
        try:
            df = stock.get_index_ohlcv(START_DATE, END_DATE, code)
            if len(df) > 100:
                df = df.reset_index()
                df.to_csv(SECTOR_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
                print(f"  Saved: {name}")
        except Exception as e:
            print(f"  Failed: {name} - {e}")

    print("Sector indices downloaded")

except Exception as e:
    print(f"Sector data download failed: {e}")

print()
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)
print(f"KOSDAQ: {KOSDAQ_DIR}")
print(f"Investor: {INVESTOR_DIR}")
print(f"Fundamental: {FUNDAMENTAL_DIR}")
print(f"Sector: {SECTOR_DIR}")
