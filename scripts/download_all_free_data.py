"""
무료 데이터 전체 다운로드 스크립트

다운로드 항목:
1. Binance 선물 - Funding Rate
2. Binance 선물 - Open Interest
3. Binance 선물 - Long/Short Ratio
4. Binance - 다중 시간프레임 OHLCV (4h, 1h)
5. Fear & Greed Index
6. 거시경제 데이터 (DXY, VIX 등)
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

# ccxt 사용
try:
    import ccxt
except ImportError:
    print("ccxt 설치 필요: pip install ccxt")
    sys.exit(1)

DATA_ROOT = Path("E:/data/crypto_ohlcv")
DATA_ROOT.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    """로그 출력"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ============================================================
# 1. Binance Funding Rate
# ============================================================
def download_binance_funding_rate(symbols: List[str] = None):
    """Binance 선물 Funding Rate 다운로드"""
    log("=" * 60)
    log("Binance Funding Rate 다운로드 시작")
    log("=" * 60)

    output_dir = DATA_ROOT / "binance_funding_rate"
    output_dir.mkdir(parents=True, exist_ok=True)

    exchange = ccxt.binance(
        {"enableRateLimit": True, "options": {"defaultType": "future"}}
    )

    if symbols is None:
        # 주요 심볼만
        symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "DOGE/USDT",
            "SOL/USDT",
            "DOT/USDT",
            "MATIC/USDT",
            "LTC/USDT",
            "AVAX/USDT",
            "LINK/USDT",
            "ATOM/USDT",
            "UNI/USDT",
            "ETC/USDT",
            "XLM/USDT",
            "ALGO/USDT",
            "NEAR/USDT",
            "FTM/USDT",
            "SAND/USDT",
        ]

    for symbol in symbols:
        try:
            log(f"  {symbol} funding rate...")

            # Binance API로 직접 호출 (ccxt는 funding rate 히스토리 제한적)
            symbol_id = symbol.replace("/", "")
            url = "https://fapi.binance.com/fapi/v1/fundingRate"

            all_data = []
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int(
                (datetime.now() - timedelta(days=365 * 3)).timestamp() * 1000
            )

            while True:
                params = {
                    "symbol": symbol_id,
                    "startTime": start_time,
                    "endTime": end_time,
                    "limit": 1000,
                }

                response = requests.get(url, params=params)
                if response.status_code != 200:
                    break

                data = response.json()
                if not data:
                    break

                all_data.extend(data)

                # 다음 페이지
                start_time = data[-1]["fundingTime"] + 1
                if start_time >= end_time:
                    break

                time.sleep(0.1)  # Rate limit

            if all_data:
                df = pd.DataFrame(all_data)
                df["datetime"] = pd.to_datetime(df["fundingTime"], unit="ms")
                df["fundingRate"] = df["fundingRate"].astype(float)
                df = df[["datetime", "fundingRate", "fundingTime"]]
                df = df.sort_values("datetime").drop_duplicates()

                filename = output_dir / f"{symbol_id}_funding.csv"
                df.to_csv(filename, index=False)
                log(f"    저장: {len(df)}개 레코드")

            time.sleep(0.2)

        except Exception as e:
            log(f"    에러: {e}")
            continue

    log(f"Funding Rate 다운로드 완료: {output_dir}")
    return output_dir


# ============================================================
# 2. Binance Open Interest
# ============================================================
def download_binance_open_interest(symbols: List[str] = None):
    """Binance 선물 Open Interest 다운로드"""
    log("=" * 60)
    log("Binance Open Interest 다운로드 시작")
    log("=" * 60)

    output_dir = DATA_ROOT / "binance_open_interest"
    output_dir.mkdir(parents=True, exist_ok=True)

    if symbols is None:
        symbols = [
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "ADAUSDT",
            "DOGEUSDT",
            "SOLUSDT",
            "DOTUSDT",
            "MATICUSDT",
            "LTCUSDT",
            "AVAXUSDT",
            "LINKUSDT",
            "ATOMUSDT",
            "UNIUSDT",
            "ETCUSDT",
        ]

    for symbol in symbols:
        try:
            log(f"  {symbol} open interest...")

            url = "https://fapi.binance.com/futures/data/openInterestHist"

            all_data = []
            end_time = int(datetime.now().timestamp() * 1000)

            # 최대 30일씩 가져오기 (API 제한)
            for _ in range(36):  # 약 3년
                params = {
                    "symbol": symbol,
                    "period": "1d",
                    "limit": 30,
                    "endTime": end_time,
                }

                response = requests.get(url, params=params)
                if response.status_code != 200:
                    break

                data = response.json()
                if not data:
                    break

                all_data.extend(data)
                end_time = data[0]["timestamp"] - 1

                time.sleep(0.2)

            if all_data:
                df = pd.DataFrame(all_data)
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
                df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)
                df = df[["datetime", "sumOpenInterest", "sumOpenInterestValue"]]
                df = df.sort_values("datetime").drop_duplicates()

                filename = output_dir / f"{symbol}_oi.csv"
                df.to_csv(filename, index=False)
                log(f"    저장: {len(df)}개 레코드")

        except Exception as e:
            log(f"    에러: {e}")
            continue

    log(f"Open Interest 다운로드 완료: {output_dir}")
    return output_dir


# ============================================================
# 3. Binance Long/Short Ratio
# ============================================================
def download_binance_long_short_ratio(symbols: List[str] = None):
    """Binance 선물 Long/Short Ratio 다운로드"""
    log("=" * 60)
    log("Binance Long/Short Ratio 다운로드 시작")
    log("=" * 60)

    output_dir = DATA_ROOT / "binance_long_short_ratio"
    output_dir.mkdir(parents=True, exist_ok=True)

    if symbols is None:
        symbols = [
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "XRPUSDT",
            "ADAUSDT",
            "DOGEUSDT",
            "SOLUSDT",
            "DOTUSDT",
            "MATICUSDT",
            "LTCUSDT",
        ]

    for symbol in symbols:
        try:
            log(f"  {symbol} long/short ratio...")

            # Global Long/Short Account Ratio
            url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"

            all_data = []
            end_time = int(datetime.now().timestamp() * 1000)

            for _ in range(36):
                params = {
                    "symbol": symbol,
                    "period": "1d",
                    "limit": 30,
                    "endTime": end_time,
                }

                response = requests.get(url, params=params)
                if response.status_code != 200:
                    break

                data = response.json()
                if not data:
                    break

                all_data.extend(data)
                end_time = data[0]["timestamp"] - 1

                time.sleep(0.2)

            if all_data:
                df = pd.DataFrame(all_data)
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["longShortRatio"] = df["longShortRatio"].astype(float)
                df["longAccount"] = df["longAccount"].astype(float)
                df["shortAccount"] = df["shortAccount"].astype(float)
                df = df[["datetime", "longShortRatio", "longAccount", "shortAccount"]]
                df = df.sort_values("datetime").drop_duplicates()

                filename = output_dir / f"{symbol}_lsratio.csv"
                df.to_csv(filename, index=False)
                log(f"    저장: {len(df)}개 레코드")

        except Exception as e:
            log(f"    에러: {e}")
            continue

    log(f"Long/Short Ratio 다운로드 완료: {output_dir}")
    return output_dir


# ============================================================
# 4. Binance 다중 시간프레임 OHLCV
# ============================================================
def download_binance_multi_timeframe(
    symbols: List[str] = None, timeframes: List[str] = None
):
    """Binance 다중 시간프레임 OHLCV 다운로드"""
    log("=" * 60)
    log("Binance 다중 시간프레임 OHLCV 다운로드 시작")
    log("=" * 60)

    if timeframes is None:
        timeframes = ["4h", "1h"]

    if symbols is None:
        symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "DOGE/USDT",
            "SOL/USDT",
            "DOT/USDT",
            "MATIC/USDT",
            "LTC/USDT",
            "AVAX/USDT",
            "LINK/USDT",
            "ATOM/USDT",
            "UNI/USDT",
            "ETC/USDT",
            "XLM/USDT",
            "ALGO/USDT",
            "NEAR/USDT",
            "FTM/USDT",
            "SAND/USDT",
        ]

    exchange = ccxt.binance(
        {"enableRateLimit": True, "options": {"defaultType": "future"}}
    )

    for tf in timeframes:
        output_dir = DATA_ROOT / f"binance_futures_{tf}"
        output_dir.mkdir(parents=True, exist_ok=True)

        log(f"\n시간프레임: {tf}")

        for symbol in symbols:
            try:
                log(f"  {symbol}...")

                all_data = []
                since = exchange.parse8601("2020-01-01T00:00:00Z")

                while True:
                    ohlcv = exchange.fetch_ohlcv(symbol, tf, since=since, limit=1000)
                    if not ohlcv:
                        break

                    all_data.extend(ohlcv)
                    since = ohlcv[-1][0] + 1

                    if len(ohlcv) < 1000:
                        break

                    time.sleep(0.1)

                if all_data:
                    df = pd.DataFrame(
                        all_data,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df = df[["datetime", "open", "high", "low", "close", "volume"]]
                    df = df.drop_duplicates("datetime").sort_values("datetime")

                    symbol_clean = symbol.replace("/", "")
                    filename = output_dir / f"{symbol_clean}.csv"
                    df.to_csv(filename, index=False)
                    log(f"    저장: {len(df)}개 캔들")

            except Exception as e:
                log(f"    에러: {e}")
                continue

    log(f"\n다중 시간프레임 다운로드 완료")


# ============================================================
# 5. Fear & Greed Index
# ============================================================
def download_fear_greed_index():
    """Fear & Greed Index 다운로드 (Alternative.me)"""
    log("=" * 60)
    log("Fear & Greed Index 다운로드 시작")
    log("=" * 60)

    output_dir = DATA_ROOT / "sentiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            data = response.json()

            if "data" in data:
                df = pd.DataFrame(data["data"])
                df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
                df["value"] = df["value"].astype(int)
                df = df[["datetime", "value", "value_classification"]]
                df.columns = ["datetime", "fear_greed_value", "classification"]
                df = df.sort_values("datetime")

                filename = output_dir / "fear_greed_index.csv"
                df.to_csv(filename, index=False)
                log(f"저장: {len(df)}개 레코드 -> {filename}")

    except Exception as e:
        log(f"에러: {e}")

    return output_dir


# ============================================================
# 6. 거시경제 데이터 (Yahoo Finance 대체)
# ============================================================
def download_macro_data():
    """거시경제 데이터 다운로드 (yfinance 사용)"""
    log("=" * 60)
    log("거시경제 데이터 다운로드 시작")
    log("=" * 60)

    try:
        import yfinance as yf
    except ImportError:
        log("yfinance 설치 필요: pip install yfinance")
        return None

    output_dir = DATA_ROOT / "macro"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 다운로드할 티커
    tickers = {
        "DX-Y.NYB": "DXY",  # 달러 인덱스
        "^VIX": "VIX",  # 변동성 지수
        "^GSPC": "SP500",  # S&P 500
        "^IXIC": "NASDAQ",  # 나스닥
        "^TNX": "US10Y",  # 미국 10년물 금리
        "GC=F": "GOLD",  # 금
        "CL=F": "OIL",  # 원유 (WTI)
    }

    for ticker, name in tickers.items():
        try:
            log(f"  {name} ({ticker})...")

            data = yf.download(ticker, start="2017-01-01", progress=False)

            if len(data) > 0:
                data = data.reset_index()
                data.columns = [
                    "datetime",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                ]
                data = data[["datetime", "open", "high", "low", "close", "volume"]]

                filename = output_dir / f"{name}.csv"
                data.to_csv(filename, index=False)
                log(f"    저장: {len(data)}개 레코드")

        except Exception as e:
            log(f"    에러: {e}")
            continue

    log(f"거시경제 데이터 다운로드 완료: {output_dir}")
    return output_dir


# ============================================================
# 7. Binance 현물 추가 시간프레임
# ============================================================
def download_binance_spot_multi_timeframe(
    symbols: List[str] = None, timeframes: List[str] = None
):
    """Binance 현물 다중 시간프레임 OHLCV 다운로드"""
    log("=" * 60)
    log("Binance 현물 다중 시간프레임 OHLCV 다운로드 시작")
    log("=" * 60)

    if timeframes is None:
        timeframes = ["4h", "1h"]

    if symbols is None:
        symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "DOGE/USDT",
            "SOL/USDT",
            "DOT/USDT",
            "MATIC/USDT",
            "LTC/USDT",
        ]

    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
        }
    )

    for tf in timeframes:
        output_dir = DATA_ROOT / f"binance_spot_{tf}"
        output_dir.mkdir(parents=True, exist_ok=True)

        log(f"\n시간프레임: {tf}")

        for symbol in symbols:
            try:
                log(f"  {symbol}...")

                all_data = []
                since = exchange.parse8601("2020-01-01T00:00:00Z")

                while True:
                    ohlcv = exchange.fetch_ohlcv(symbol, tf, since=since, limit=1000)
                    if not ohlcv:
                        break

                    all_data.extend(ohlcv)
                    since = ohlcv[-1][0] + 1

                    if len(ohlcv) < 1000:
                        break

                    time.sleep(0.1)

                if all_data:
                    df = pd.DataFrame(
                        all_data,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df = df[["datetime", "open", "high", "low", "close", "volume"]]
                    df = df.drop_duplicates("datetime").sort_values("datetime")

                    symbol_clean = symbol.replace("/", "")
                    filename = output_dir / f"{symbol_clean}.csv"
                    df.to_csv(filename, index=False)
                    log(f"    저장: {len(df)}개 캔들")

            except Exception as e:
                log(f"    에러: {e}")
                continue


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("무료 데이터 전체 다운로드")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    start = datetime.now()

    # 1. Funding Rate
    download_binance_funding_rate()

    # 2. Open Interest
    download_binance_open_interest()

    # 3. Long/Short Ratio
    download_binance_long_short_ratio()

    # 4. 선물 다중 시간프레임
    download_binance_multi_timeframe()

    # 5. Fear & Greed
    download_fear_greed_index()

    # 6. 거시경제
    download_macro_data()

    # 7. 현물 다중 시간프레임
    download_binance_spot_multi_timeframe()

    elapsed = datetime.now() - start

    print("\n" + "=" * 70)
    print("다운로드 완료!")
    print(f"소요 시간: {elapsed}")
    print("=" * 70)

    # 저장된 폴더 목록
    print("\n저장된 데이터 폴더:")
    for folder in sorted(DATA_ROOT.iterdir()):
        if folder.is_dir():
            file_count = len(list(folder.glob("*.csv")))
            print(f"  {folder.name}: {file_count}개 파일")


if __name__ == "__main__":
    main()
