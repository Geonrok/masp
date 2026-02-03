"""
Round 14 Data Prep: Fundamental Data Download
=============================================
Downloads daily fundamental metrics (PER, PBR, DIV, BPS, etc.) for all KOSPI/KOSDAQ stocks.
Source: KRX (via pykrx)
"""

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from pykrx import stock
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("E:/투자/data/kr_stock")
FUND_DIR = DATA_DIR / "fundamental_data"
FUND_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "20160101"
END_DATE = datetime.now().strftime("%Y%m%d")


def get_all_tickers():
    kospi = stock.get_market_ticker_list(END_DATE, market="KOSPI")
    kosdaq = stock.get_market_ticker_list(END_DATE, market="KOSDAQ")
    return list(set(kospi + kosdaq))


def download_fundamental(ticker):
    save_path = FUND_DIR / f"{ticker}.csv"
    if save_path.exists():
        return "skipped"

    try:
        # get_market_fundamental_by_date returns daily fundamentals for a range
        # Columns: BPS, PER, PBR, EPS, DIV, DPS
        df = stock.get_market_fundamental_by_date(START_DATE, END_DATE, ticker)

        if df.empty:
            return "empty"

        # Rename for consistency
        # KRX cols might vary, usually: ['BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS']
        # pykrx returns English cols mostly now

        df.index.name = "Date"
        df.to_csv(save_path)
        time.sleep(0.1)
        return "downloaded"
    except Exception:
        # print(f"Error {ticker}: {e}")
        return "error"


def main():
    print("=" * 80)
    print("Downloading Fundamental Data (PER, PBR, EPS...)")
    print("=" * 80)

    tickers = get_all_tickers()
    print(f"Target Tickers: {len(tickers)}")

    # Check existing
    existing = {f.stem for f in FUND_DIR.glob("*.csv")}
    targets = [t for t in tickers if t not in existing]
    print(f"To Download: {len(targets)}")

    # Using ThreadPool for I/O bound task, but gentle on KRX
    # KRX blocks aggressive scraping. Sequential might be safer or low threads.

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(download_fundamental, t): t for t in targets}

        for f in tqdm(as_completed(futures), total=len(targets)):
            pass


if __name__ == "__main__":
    main()
