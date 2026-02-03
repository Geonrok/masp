"""새 API 키 테스트"""

import os

from dotenv import load_dotenv

# 강제 reload
load_dotenv(override=True)

api_key = os.getenv("BITHUMB_API_KEY")
secret_key = os.getenv("BITHUMB_SECRET_KEY")

print(f"API_KEY: {api_key[:8]}... (len={len(api_key)})")
print(f"SECRET: {secret_key[:8]}... (len={len(secret_key)})")

from libs.adapters.bithumb_api_v2 import BithumbAPIV2

client = BithumbAPIV2(api_key, secret_key)

print()
print("=== GET /v1/accounts ===")
try:
    result = client.get_accounts()
    print(f"SUCCESS: {type(result).__name__}")
    if isinstance(result, list):
        for acc in result[:5]:
            currency = acc.get("currency", "?")
            balance = acc.get("balance", "?")
            print(f"  {currency}: {balance}")
except Exception as e:
    print(f"FAILED: {e}")

print()
print("=== GET /v1/ticker ===")
try:
    result = client.get_ticker(["KRW-BTC"])
    print(f'SUCCESS: BTC = {result[0].get("trade_price")} KRW')
except Exception as e:
    print(f"FAILED: {e}")
