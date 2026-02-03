#!/usr/bin/env python3
"""
Data Collection Plan for Strategy Development

Phase 1: Free Data Sources
Phase 2: Paid Data Sources (if needed)

Target: Profit Factor > 1.3, Win Rate > 40%
"""

# =============================================================================
# PHASE 1: FREE DATA (Priority)
# =============================================================================

FREE_DATA_SOURCES = {
    # ---------------------------------------------------------------------
    # 1. Binance Futures API (Most Important)
    # ---------------------------------------------------------------------
    "binance_funding_rate": {
        "description": "Funding rate every 8 hours",
        "endpoint": "GET /fapi/v1/fundingRate",
        "params": {"symbol": "BTCUSDT", "limit": 1000},
        "history": "2019-09 onwards",
        "frequency": "8 hours",
        "priority": "HIGH",
        "status": "PARTIAL",  # We have 2023+ only
        "action": "Collect 2020-2022 data",
    },
    "binance_open_interest_hist": {
        "description": "Historical Open Interest",
        "endpoint": "GET /futures/data/openInterestHist",
        "params": {"symbol": "BTCUSDT", "period": "1d", "limit": 500},
        "history": "2020-01 onwards",
        "frequency": "5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d",
        "priority": "HIGH",
        "status": "PARTIAL",  # Only recent 1 month
        "action": "Collect full history (1d interval)",
    },
    "binance_long_short_ratio": {
        "description": "Top trader long/short ratio",
        "endpoint": "GET /futures/data/topLongShortAccountRatio",
        "params": {"symbol": "BTCUSDT", "period": "1d", "limit": 500},
        "history": "2020-01 onwards",
        "frequency": "5m to 1d",
        "priority": "HIGH",
        "status": "PARTIAL",
        "action": "Collect full history",
    },
    "binance_taker_volume": {
        "description": "Taker buy/sell volume",
        "endpoint": "GET /futures/data/takerlongshortRatio",
        "params": {"symbol": "BTCUSDT", "period": "1d", "limit": 500},
        "history": "2020-01 onwards",
        "frequency": "5m to 1d",
        "priority": "MEDIUM",
        "status": "NOT_COLLECTED",
        "action": "Collect full history",
    },
    # ---------------------------------------------------------------------
    # 2. Coinglass (Free Tier)
    # ---------------------------------------------------------------------
    "coinglass_liquidation": {
        "description": "Liquidation data by exchange",
        "source": "https://www.coinglass.com/",
        "method": "Web scraping or API",
        "history": "2020 onwards",
        "frequency": "Daily",
        "priority": "HIGH",
        "status": "NOT_COLLECTED",
        "action": "Scrape or use API",
    },
    "coinglass_oi_aggregated": {
        "description": "Aggregated OI across exchanges",
        "source": "https://www.coinglass.com/",
        "priority": "MEDIUM",
        "status": "NOT_COLLECTED",
    },
    # ---------------------------------------------------------------------
    # 3. Google Trends
    # ---------------------------------------------------------------------
    "google_trends_bitcoin": {
        "description": "Search interest for 'Bitcoin'",
        "source": "pytrends library",
        "history": "2004 onwards",
        "frequency": "Weekly (daily for <90 days)",
        "priority": "MEDIUM",
        "status": "NOT_COLLECTED",
        "action": "Use pytrends to collect",
        "code_example": """
from pytrends.request import TrendReq
pytrends = TrendReq()
pytrends.build_payload(["bitcoin"], timeframe="2020-01-01 2025-12-31")
df = pytrends.interest_over_time()
        """,
    },
    # ---------------------------------------------------------------------
    # 4. Alternative.me Fear & Greed (Already have)
    # ---------------------------------------------------------------------
    "fear_greed_index": {
        "description": "Crypto Fear & Greed Index",
        "source": "https://api.alternative.me/fng/",
        "history": "2018-02 onwards",
        "frequency": "Daily",
        "priority": "HIGH",
        "status": "COLLECTED",
        "action": "Keep updated",
    },
}

# =============================================================================
# PHASE 2: PAID DATA (If Phase 1 insufficient)
# =============================================================================

PAID_DATA_SOURCES = {
    # ---------------------------------------------------------------------
    # 1. Glassnode (Best for on-chain)
    # ---------------------------------------------------------------------
    "glassnode": {
        "pricing": "$29/month (Advanced), $799/month (Professional)",
        "key_metrics": [
            "Exchange Netflow",
            "MVRV Ratio",
            "SOPR (Spent Output Profit Ratio)",
            "Active Addresses",
            "Realized Price",
            "NUPL (Net Unrealized Profit/Loss)",
        ],
        "priority": "HIGH if Phase 1 fails",
        "recommendation": "Start with Advanced tier",
    },
    # ---------------------------------------------------------------------
    # 2. CryptoQuant (Alternative to Glassnode)
    # ---------------------------------------------------------------------
    "cryptoquant": {
        "pricing": "$29/month (Basic), $99/month (Pro)",
        "key_metrics": [
            "Exchange Reserve",
            "Exchange Inflow/Outflow",
            "Miner Outflow",
            "Fund Flow Ratio",
        ],
        "priority": "MEDIUM",
    },
    # ---------------------------------------------------------------------
    # 3. Santiment (Social + On-chain)
    # ---------------------------------------------------------------------
    "santiment": {
        "pricing": "$49/month (Pro)",
        "key_metrics": [
            "Social Volume",
            "Weighted Sentiment",
            "Development Activity",
            "Whale Transaction Count",
        ],
        "priority": "LOW",
    },
}

# =============================================================================
# DATA COLLECTION SCRIPT TEMPLATE
# =============================================================================

COLLECTION_SCRIPT = '''
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def collect_binance_funding_rate(symbol="BTCUSDT", start_date="2020-01-01"):
    """Collect historical funding rate from Binance."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"

    all_data = []
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp.now().timestamp() * 1000)

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "startTime": start_ts,
            "limit": 1000,
        }

        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        all_data.extend(data)
        start_ts = data[-1]["fundingTime"] + 1

        print(f"Collected {len(all_data)} records...")
        time.sleep(0.5)  # Rate limit

    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)

    return df[["datetime", "fundingRate", "fundingTime"]]


def collect_binance_oi_history(symbol="BTCUSDT", period="1d", start_date="2020-01-01"):
    """Collect historical Open Interest from Binance."""
    url = "https://fapi.binance.com/futures/data/openInterestHist"

    all_data = []
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp.now().timestamp() * 1000)

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": start_ts,
            "limit": 500,
        }

        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        all_data.extend(data)
        start_ts = data[-1]["timestamp"] + 1

        print(f"Collected {len(all_data)} OI records...")
        time.sleep(0.5)

    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


def collect_binance_long_short_ratio(symbol="BTCUSDT", period="1d", start_date="2020-01-01"):
    """Collect top trader long/short ratio."""
    url = "https://fapi.binance.com/futures/data/topLongShortAccountRatio"

    all_data = []
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp.now().timestamp() * 1000)

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": start_ts,
            "limit": 500,
        }

        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        all_data.extend(data)
        start_ts = data[-1]["timestamp"] + 1

        print(f"Collected {len(all_data)} L/S ratio records...")
        time.sleep(0.5)

    df = pd.DataFrame(all_data)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


if __name__ == "__main__":
    import os

    OUTPUT_DIR = "E:/data/crypto_ohlcv"

    # 1. Funding Rate
    print("Collecting Funding Rate...")
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        df = collect_binance_funding_rate(symbol, "2020-01-01")
        output_path = f"{OUTPUT_DIR}/binance_funding_rate/{symbol}_funding_full.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}: {len(df)} records")

    # 2. Open Interest
    print("\\nCollecting Open Interest...")
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        df = collect_binance_oi_history(symbol, "1d", "2020-01-01")
        output_path = f"{OUTPUT_DIR}/binance_open_interest/{symbol}_oi_full.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}: {len(df)} records")

    # 3. Long/Short Ratio
    print("\\nCollecting Long/Short Ratio...")
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        df = collect_binance_long_short_ratio(symbol, "1d", "2020-01-01")
        output_path = f"{OUTPUT_DIR}/binance_long_short_ratio/{symbol}_lsratio_full.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}: {len(df)} records")
'''

# =============================================================================
# EXPECTED IMPACT ON STRATEGY
# =============================================================================

EXPECTED_IMPACT = {
    "funding_rate_full": {
        "current_pf": 1.05,
        "expected_pf_improvement": "+0.1 to +0.2",
        "reason": "Extreme funding (>0.1%) = contrarian signal",
    },
    "open_interest_change": {
        "expected_pf_improvement": "+0.1 to +0.15",
        "reason": "OI divergence from price = reversal signal",
    },
    "liquidation_data": {
        "expected_pf_improvement": "+0.15 to +0.25",
        "reason": "Large liquidations = capitulation = entry opportunity",
    },
    "exchange_netflow": {
        "expected_pf_improvement": "+0.1 to +0.2",
        "reason": "Inflow spike = sell pressure incoming",
    },
    "combined_all": {
        "target_pf": "1.3 to 1.5",
        "target_win_rate": "40-45%",
        "confidence": "Medium",
    },
}

if __name__ == "__main__":
    print("=" * 60)
    print("DATA COLLECTION PLAN")
    print("=" * 60)

    print("\n[PHASE 1: FREE DATA]")
    for name, info in FREE_DATA_SOURCES.items():
        status = info.get("status", "UNKNOWN")
        priority = info.get("priority", "UNKNOWN")
        print(f"  {name}: {status} (Priority: {priority})")

    print("\n[PHASE 2: PAID DATA]")
    for name, info in PAID_DATA_SOURCES.items():
        pricing = info.get("pricing", "Unknown")
        print(f"  {name}: {pricing}")

    print("\n[EXPECTED IMPACT]")
    for name, impact in EXPECTED_IMPACT.items():
        improvement = impact.get(
            "expected_pf_improvement", impact.get("target_pf", "N/A")
        )
        print(f"  {name}: PF {improvement}")
