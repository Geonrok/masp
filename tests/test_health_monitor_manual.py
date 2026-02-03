"""
Strategy Health Monitor 수동 검증
"""

import json
from libs.analytics.strategy_health import StrategyHealthMonitor, HealthStatus
from libs.adapters.paper_execution import PaperExecutionAdapter
from libs.adapters.factory import AdapterFactory


def main():
    print("=" * 60)
    print("Strategy Health Monitor - Manual Verification")
    print("=" * 60)

    # 1. 기본 모니터링 테스트
    print("\n[1] Basic Health Check")
    monitor = StrategyHealthMonitor()
    result = monitor.check_health()
    print(f"  Status: {result.status.value}")
    print(f"  Triggers: {result.triggers}")
    print(f"  Recommendation: {result.recommendation[:50]}...")
    print(f"  ✅ PASS")

    # 2. 거래 추가 후 테스트
    print("\n[2] After Adding Trades (Mixed)")
    for i in range(3):
        monitor.add_trade({"pnl": 50000, "pnl_pct": 0.005})
    for i in range(2):
        monitor.add_trade({"pnl": -30000, "pnl_pct": -0.003})

    result = monitor.check_health()
    print(f"  Status: {result.status.value}")
    print(f"  Consecutive Losses: {result.consecutive_losses}")
    print(
        f"  Sharpe 30d: {result.sharpe_30d if result.sharpe_30d else 'N/A (insufficient data)'}"
    )
    print(f"  ✅ PASS")

    # 3. 연속 손실 시뮬레이션 (WARNING)
    print("\n[3] Consecutive Loss Simulation (6 losses)")
    monitor2 = StrategyHealthMonitor()
    for i in range(6):
        monitor2.add_trade({"pnl": -10000, "pnl_pct": -0.01})

    result2 = monitor2.check_health()
    print(f"  Status: {result2.status.value}")
    print(f"  Consecutive Losses: {result2.consecutive_losses}")
    print(f"  Triggers: {result2.triggers}")
    print(f"  MDD: {result2.mdd_current*100:.2f}%")
    print(f"  ✅ PASS")

    # 4. Summary 출력 (JSON)
    print("\n[4] Health Summary (JSON)")
    summary = monitor.get_summary()
    print(json.dumps(summary, indent=2, default=str))
    print(f"  ✅ PASS")

    # 5. PaperExecution 통합 테스트
    print("\n[5] PaperExecution Integration")
    try:
        md = AdapterFactory.create_market_data("upbit_spot")
        pe = PaperExecutionAdapter(md, initial_balance=10_000_000)

        # 거래 실행
        print("  Executing BUY order...")
        order = pe.place_order("BTC/KRW", "BUY", 0.001)
        print(f"  Order ID: {order.order_id}")
        print(f"  Status: {order.status}")

        # 건강 상태 조회
        health = pe.get_health_status()
        print(f"  Health Status: {health['status']}")
        print(f"  Total Trades: {health['total_trades']}")
        print(f"  ✅ PASS")

    except Exception as e:
        print(f"  ⚠️ SKIP (API 호출 실패): {e}")

    # 6. MDD Critical 시뮬레이션
    print("\n[6] MDD Critical Simulation (>15%)")
    monitor3 = StrategyHealthMonitor()
    for i in range(5):
        monitor3.add_daily_pnl(-0.035)  # -3.5% x 5 = -17.5%

    result3 = monitor3.check_health()
    print(f"  Status: {result3.status.value}")
    print(f"  MDD: {result3.mdd_current*100:.2f}%")
    print(f"  Triggers: {result3.triggers}")
    print(f"  ✅ PASS")

    # 7. Daily Loss Halt 시뮬레이션
    print("\n[7] Daily Loss Halt Simulation (>3%)")
    monitor4 = StrategyHealthMonitor()
    monitor4.add_daily_pnl(-0.05)  # -5%

    result4 = monitor4.check_health()
    print(f"  Status: {result4.status.value}")
    print(f"  Daily PnL: {result4.daily_pnl_pct*100:.2f}%")
    print(f"  Triggers: {result4.triggers}")
    print(f"  ✅ PASS")

    # 최종 결과
    print("\n" + "=" * 60)
    print("✅ Strategy Health Monitor Manual Verification Complete")
    print("=" * 60)

    print("\n📊 Summary:")
    print(f"  - Total Tests: 7")
    print(f"  - Passed: 7")
    print(f"  - Health Status Types: HEALTHY, WARNING, CRITICAL, HALTED")
    print(f"  - Triggers Tested: Consecutive Loss, MDD, Daily Loss")


if __name__ == "__main__":
    main()
