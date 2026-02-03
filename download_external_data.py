# -*- coding: utf-8 -*-
"""
외부 데이터 다운로드
- S&P500 (SPY)
- NASDAQ (QQQ)
- VIX
- USD/KRW
"""

import numpy as np
import pandas as pd

try:
    import yfinance as yf

    print("yfinance 설치됨")
except ImportError:
    print("yfinance 설치 필요: pip install yfinance")
    import subprocess

    subprocess.run(["pip", "install", "yfinance", "-q"])
    import yfinance as yf

DATA_DIR = "E:/투자/data/kosdaq_futures/"


def download_data():
    """외부 데이터 다운로드"""
    print("=" * 60)
    print("외부 데이터 다운로드")
    print("=" * 60)

    # 기간 설정 (KOSDAQ ETF와 동일)
    start_date = "2015-01-01"
    end_date = "2026-02-01"

    tickers = {
        "SPY": "spy_daily.parquet",  # S&P500 ETF
        "QQQ": "qqq_daily.parquet",  # NASDAQ ETF
        "^VIX": "vix_daily.parquet",  # VIX
        "USDKRW=X": "usdkrw_daily.parquet",  # USD/KRW
    }

    for ticker, filename in tickers.items():
        print(f"\n다운로드: {ticker}...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(data) > 0:
                # 컬럼 정리
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                data.to_parquet(DATA_DIR + filename)
                print(
                    f"  저장: {filename}, {len(data)} rows, {data.index.min()} ~ {data.index.max()}"
                )
            else:
                print(f"  데이터 없음: {ticker}")
        except Exception as e:
            print(f"  오류: {e}")

    print("\n완료!")


if __name__ == "__main__":
    download_data()
