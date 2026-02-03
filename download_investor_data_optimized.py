"""
Round 13 Data Prep: Investor Data (Optimized)
=============================================
Downloads investor data only for top liquid stocks and viable candidates.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from pykrx import stock
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("E:/투자/data/kr_stock")
SUGEUP_DIR = DATA_DIR / "investor_data"
SUGEUP_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "20160101"
END_DATE = datetime.now().strftime("%Y%m%d")


def get_target_tickers():
    """Get Top 500 by Market Cap + Round 11 Winners"""
    print("Fetching target tickers...")

    # 1. Market Cap Top
    kospi = stock.get_market_cap_by_ticker(END_DATE, market="KOSPI")
    kosdaq = stock.get_market_cap_by_ticker(END_DATE, market="KOSDAQ")

    top_kospi = kospi.sort_values("시가총액", ascending=False).head(300).index.tolist()
    top_kosdaq = (
        kosdaq.sort_values("시가총액", ascending=False).head(200).index.tolist()
    )

    # 2. Round 11 Winners (If exists)
    r11_path = Path("E:/투자/data/round11_results")
    r11_tickers = []
    for f in r11_path.glob("*viable*.csv"):
        try:
            df = pd.read_csv(f)
            # Ensure ticker column is string and zero-padded
            if "ticker" in df.columns:
                r11_tickers.extend(df["ticker"].astype(str).str.zfill(6).tolist())
        except:
            pass

    targets = list(set(top_kospi + top_kosdaq + r11_tickers))
    print(f"Target Tickers: {len(targets)}")
    return targets


def download_one(ticker):
    save_path = SUGEUP_DIR / f"{ticker}.csv"
    if save_path.exists():
        return "skipped"

    try:
        df = stock.get_market_net_purchases_of_equities_by_ticker(
            START_DATE, END_DATE, ticker
        )
        if df.empty:
            return "empty"

        rename_map = {
            "연기금등": "Pension",
            "연기금": "Pension",
            "외국인": "Foreigner",
            "기관합계": "Institution",
            "개인": "Individual",
            "사모": "PrivateEquity",
            "금융투자": "FinancialInv",
            "보험": "Insurance",
            "투신": "Trust",
            "은행": "Bank",
            "기타금융": "EtcFinance",
            "기타법인": "EtcCorp",
            "전체": "Total",
        }
        df = df.rename(columns=rename_map)
        df.index.name = "Date"
        df.to_csv(save_path)
        return "downloaded"
    except:
        return "error"


def main():
    targets = get_target_tickers()

    print(f"Downloading Investor Data for {len(targets)} stocks...")

    # Sequential download (Parallel often blocked by KRX)
    # But let's try ThreadPool with low workers

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(download_one, t): t for t in targets}

        for f in tqdm(as_completed(futures), total=len(targets)):
            pass


if __name__ == "__main__":
    main()
