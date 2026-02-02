"""
Round 13 Data Prep: Investor Trading Activities (Sugeup)
========================================================
Downloads daily Net Buying Amount for Foreigners and Institutions.
Focuses on the full universe to find "Smart Money" flows.

Data Source: KRX (via pykrx)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pykrx import stock
from tqdm import tqdm
import time
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("E:/투자/data/kr_stock")
SUGEUP_DIR = DATA_DIR / "investor_data"
SUGEUP_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "20160101"
END_DATE = datetime.now().strftime("%Y%m%d")

def download_investor_data():
    print("="*80)
    print("Downloading Investor (Foreigner/Institution) Data")
    print("="*80)
    
    # Get all tickers
    kospi = stock.get_market_ticker_list(END_DATE, market="KOSPI")
    kosdaq = stock.get_market_ticker_list(END_DATE, market="KOSDAQ")
    all_tickers = kospi + kosdaq
    
    print(f"Total Tickers: {len(all_tickers)}")
    
    # Check existing
    existing = {f.stem for f in SUGEUP_DIR.glob("*.csv")}
    print(f"Already downloaded: {len(existing)}")
    
    # Download loop
    for i, ticker in enumerate(tqdm(all_tickers)):
        if ticker in existing:
            continue
            
        try:
            # Fetch Investor Data (Net Purchase)
            # Returns: Date, Individual, Foreigner, Institution, Etc...
            df = stock.get_market_net_purchases_of_equities_by_ticker(START_DATE, END_DATE, ticker, detail=True)
            
            if df.empty: continue
            
            # We need: Date, Foreigner, Institution, Individual, Pension(연기금)
            # Column names might be in Korean
            # Typical cols: ['개인', '외국인', '기관합계', '금융투자', '보험', '투신', '기타금융', '은행', '연기금등', '사모', '기타법인', '전체']
            
            # Rename for safety
            rename_map = {
                '연기금등': 'Pension',
                '연기금': 'Pension',
                '외국인': 'Foreigner',
                '기관합계': 'Institution',
                '개인': 'Individual',
                '사모': 'PrivateEquity'
            }
            df = df.rename(columns=rename_map)
            
            # Select key columns if they exist
            cols_to_keep = ['Foreigner', 'Institution', 'Individual', 'Pension', 'PrivateEquity']
            available_cols = [c for c in cols_to_keep if c in df.columns]
            
            if not available_cols:
                # Retry with basic API if detail failed
                df = stock.get_market_net_purchases_of_equities_by_ticker(START_DATE, END_DATE, ticker)
                df = df.rename(columns=rename_map)
                available_cols = [c for c in cols_to_keep if c in df.columns]
            
            if available_cols:
                final_df = df[available_cols]
                final_df.index.name = 'Date'
                final_df.to_csv(SUGEUP_DIR / f"{ticker}.csv")
            
            time.sleep(0.5) # Politeness delay
            
        except Exception as e:
            # print(f"Error {ticker}: {e}")
            time.sleep(1)
            continue

if __name__ == "__main__":
    download_investor_data()
