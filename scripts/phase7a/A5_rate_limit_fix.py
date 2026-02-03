import time

print("=== A5: Rate Limit 테스트 (get_quote 사용) ===")

from libs.adapters.factory import AdapterFactory

tests = [("upbit_spot", "BTC/KRW"), ("bithumb_spot", "BTC/KRW")]
N = 20
SLEEP = 0.12

for ex, sym in tests:
    md = AdapterFactory.create_market_data(ex)
    errors = 0
    rate_limit_hit = False
    t0 = time.time()

    for i in range(N):
        try:
            md.get_quote(sym)
        except Exception as e:
            errors += 1
            msg = str(e).lower()
            if "429" in msg or "too many" in msg or "rate" in msg:
                print(f"{ex} Rate Limit: FAIL (429 감지)")
                rate_limit_hit = True
                break
        time.sleep(SLEEP)

    dt = time.time() - t0
    if not rate_limit_hit:
        if errors == 0:
            print(f"{ex} Rate Limit: PASS (N={N}, elapsed={dt:.2f}s)")
        else:
            print(f"{ex} Rate Limit: WARN (errors={errors})")
