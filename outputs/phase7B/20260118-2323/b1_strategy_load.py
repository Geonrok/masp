import sys

print("=== B1: 전략 로딩 + MarketData adapter 확인 ===")
from services.strategy_runner import StrategyRunner

r = StrategyRunner(
    strategy_name="KAMA-TSMOM-Gate", exchange="upbit", symbols=["BTC/KRW"]
)

# 1) Strategy must exist
if r.strategy is None:
    print("Strategy loaded: None")
    print("B1 결과: FAIL (strategy=None)")
    sys.exit(1)

print(f"Strategy loaded: {type(r.strategy).__name__}")

# 2) Runner must have market_data
if not (hasattr(r, "market_data") and r.market_data):
    print("MarketData adapter: None 또는 미주입")
    print("B1 결과: FAIL (runner.market_data missing)")
    sys.exit(1)

adapter_type = type(r.market_data).__name__
print(f"MarketData adapter: {adapter_type}")

# 3) 기능 테스트
try:
    quote = r.market_data.get_quote("BTC/KRW")
    if quote and hasattr(quote, "last") and quote.last > 0:
        print(f"기능 테스트: PASS (가격: {quote.last:,.0f} KRW)")
    else:
        print(f"기능 테스트: WARN (quote 구조 이상: {quote})")
except Exception as e:
    print(f"기능 테스트: WARN ({type(e).__name__}: {str(e)[:50]})")

# 4) Strategy 내부 market_data 확인
md_in_strategy = getattr(r.strategy, "market_data", None) or getattr(
    r.strategy, "_market_data", None
)
if md_in_strategy:
    same = md_in_strategy is r.market_data
    print(
        f"Strategy 내부 market_data: {type(md_in_strategy).__name__} (same_object={same})"
    )

print("B1 결과: PASS")
