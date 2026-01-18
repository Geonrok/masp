print("=== A3: Public API 연결성 테스트 (get_quote 사용) ===")
import time
from libs.adapters.factory import AdapterFactory

tests = [("upbit_spot", "BTC/KRW"), ("bithumb_spot", "BTC/KRW")]

for ex, sym in tests:
    try:
        md = AdapterFactory.create_market_data(ex)
        quote = md.get_quote(sym)
        if quote and getattr(quote, "last", None):
            print(f"{ex} get_quote({sym}): PASS")
        else:
            print(f"{ex} get_quote({sym}): FAIL (None 또는 last 없음)")
    except Exception as e:
        print(f"{ex} get_quote({sym}): FAIL ({type(e).__name__}: {str(e)[:80]})")
    time.sleep(0.1)
