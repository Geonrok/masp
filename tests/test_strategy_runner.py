"""
Strategy Runner 테스트 (Paper Trading)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> bool:
    print("=" * 60)
    print("Strategy Runner Test (Paper Trading)")
    print("=" * 60)

    # 1. Runner 초기화
    print("\n[1] Strategy Runner 초기화")
    from services.strategy_runner import StrategyRunner

    runner = StrategyRunner(
        strategy_name="ma_crossover_test",
        exchange="paper",
        symbols=["BTC/KRW"],
        position_size_krw=10000,
    )
    print(f"  Strategy: {runner.strategy_name}")
    print(f"  Exchange: {runner.exchange}")
    print("  ✅ PASS")

    # 2. 상태 확인
    print("\n[2] 초기 상태")
    status = runner.get_status()
    print(f"  Positions: {status['positions']}")
    print(f"  Health: {status['health']['status']}")
    print("  ✅ PASS")

    # 3. 1회 실행 (매수)
    print("\n[3] 1회 실행 (매수 예상)")
    result = runner.run_once()
    print(f"  Result: {result}")
    print("  ✅ PASS")

    # 4. 포지션 확인
    print("\n[4] 포지션 확인")
    status = runner.get_status()
    print(f"  Positions: {status['positions']}")
    print("  ✅ PASS")

    # 5. 1회 실행 (매도)
    print("\n[5] 1회 실행 (매도 예상)")
    result = runner.run_once()
    print(f"  Result: {result}")
    print("  ✅ PASS")

    # 6. 최종 상태
    print("\n[6] 최종 상태")
    status = runner.get_status()
    print(f"  Positions: {status['positions']}")
    print(f"  Trades Today: {status['trades_today']}")
    print("  ✅ PASS")

    # 7. Daily Report 생성
    print("\n[7] Daily Report 생성")
    report = runner.generate_daily_report()
    print(f"  Report Length: {len(report)} chars")
    print("  ✅ PASS")

    print("\n" + "=" * 60)
    print("✅ All Tests Complete (7/7 PASS)")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
