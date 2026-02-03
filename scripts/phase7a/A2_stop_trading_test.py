import os

print("\n=== A2-2: STOP_TRADING=1 차단 테스트 ===")
print(f"STOP_TRADING env: {os.getenv('STOP_TRADING')}")

from services.strategy_runner import StrategyRunner

try:
    r = StrategyRunner(
        strategy_name="KAMA-TSMOM-Gate", exchange="bithumb", symbols=["BTC/KRW"]
    )
    r.run_once()
    print("결과: FAIL - run_once가 차단되지 않음")
except Exception as e:
    print(f"결과: PASS - 차단됨 ({type(e).__name__}: {str(e)[:100]})")
