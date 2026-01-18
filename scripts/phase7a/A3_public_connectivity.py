print("=== A3: Public API 연결성 테스트 (키 불필요) ===")

from libs.adapters.factory import AdapterFactory

tests = [("upbit_spot", "BTC/KRW"), ("bithumb_spot", "BTC/KRW")]

for ex, sym in tests:
    try:
        md = AdapterFactory.create_market_data(ex)
        ticker = md.get_ticker(sym)
        if ticker:
            print(f"{ex} get_ticker({sym}): PASS")
        else:
            print(f"{ex} get_ticker({sym}): FAIL (None 반환)")
    except Exception as e:
        print(f"{ex} get_ticker({sym}): FAIL ({type(e).__name__}: {str(e)[:80]})")
