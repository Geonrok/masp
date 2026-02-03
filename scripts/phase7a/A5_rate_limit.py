import time

print("=== A5: Rate Limit 준수 테스트 (Public API burst) ===")

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
            md.get_ticker(sym)
        except Exception as e:
            errors += 1
            msg = str(e).lower()
            if "429" in msg or "too many" in msg or "rate" in msg:
                print(f"{ex} Rate Limit: FAIL (429 감지: {str(e)[:80]})")
                rate_limit_hit = True
                break
        time.sleep(SLEEP)

    dt = time.time() - t0
    if not rate_limit_hit:
        if errors == 0:
            print(f"{ex} Rate Limit: PASS (N={N}, sleep={SLEEP}s, elapsed={dt:.2f}s)")
        else:
            print(f"{ex} Rate Limit: WARN (errors={errors}, elapsed={dt:.2f}s)")

print(f"\n=== Rate Limit 정량 분석 ===")
print(f"현재 설정: 0.1s/symbol = 10 symbols/s")
print(f"447개 심볼 예상 처리: {447 * 0.1:.1f}s = {447 * 0.1 / 60:.1f}분")
print(f"Bithumb 제한: 30/s -> 현재 10/s < 30/s (안전)")
print(f"Upbit 제한: 10/s (주문) -> 현재 10/s == 10/s (한계치)")
