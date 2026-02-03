import os
from dotenv import load_dotenv

load_dotenv(override=False)

print("=== A6-1: API 키 존재 여부 (값 출력 금지) ===")
keys = ["BITHUMB_API_KEY", "BITHUMB_SECRET_KEY", "UPBIT_ACCESS_KEY", "UPBIT_SECRET_KEY"]
for k in keys:
    v = os.getenv(k)
    if not v:
        status = "MISSING"
    elif len(v) < 20:
        status = f"TOO_SHORT ({len(v)} chars)"
    else:
        status = f"SET ({len(v)} chars)"
    print(f"{k}: {status}")
