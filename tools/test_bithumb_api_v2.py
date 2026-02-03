"""
Bithumb API v2 Connection Test
- Test JWT authentication with real API
"""

import logging
import os

from dotenv import load_dotenv

from libs.adapters.bithumb_api_v2 import BithumbAPIV2

load_dotenv()
api_key = os.getenv("BITHUMB_API_KEY")
secret_key = os.getenv("BITHUMB_SECRET_KEY")
debug_enabled = os.getenv("BITHUMB_JWT_DEBUG") == "1"
force_empty_query_hash = os.getenv("BITHUMB_JWT_INCLUDE_EMPTY_QUERY_HASH") == "1"

logging.basicConfig(level=logging.DEBUG if debug_enabled else logging.INFO)

print("API Key loaded:", "Yes" if api_key else "No")
print("Secret Key loaded:", "Yes" if secret_key else "No")
print("JWT debug logging:", "On" if debug_enabled else "Off")
print("Force empty query_hash:", "On" if force_empty_query_hash else "Off")

if not api_key or not secret_key:
    print("ERROR: API keys not found in .env")
    exit(1)

client = BithumbAPIV2(api_key, secret_key)

print()
print("=== Test 1: GET /v1/accounts (Balance) ===")
try:
    accounts = client.get_accounts()
    print(f"SUCCESS: {type(accounts).__name__}")
    if isinstance(accounts, list):
        print(f"  Accounts count: {len(accounts)}")
        for acc in accounts[:3]:
            currency = acc.get("currency", "?")
            balance = acc.get("balance", "?")
            print(f"  - {currency}: {balance}")
    elif isinstance(accounts, dict):
        print(f"  Response keys: {list(accounts.keys())}")
except Exception as e:
    print(f"FAILED: {e}")

print()
print("=== Test 2: GET /v1/ticker (Price) ===")
try:
    ticker = client.get_ticker(["KRW-BTC"])
    print(f"SUCCESS: {type(ticker).__name__}")
    if isinstance(ticker, list) and ticker:
        price = ticker[0].get("trade_price", "?")
        print(f"  BTC price: {price}")
    elif isinstance(ticker, dict):
        print(f"  Response keys: {list(ticker.keys())}")
except Exception as e:
    print(f"FAILED: {e}")

print()
print("=== Done ===")
