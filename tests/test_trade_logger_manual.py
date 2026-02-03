"""
Trade Logger + Daily Report 수동 검증
- 전용 테스트 디렉토리 사용 (누적 영향 최소화)
- 실패 시 즉시 exit code 반영
"""

import os
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> bool:
    print("=" * 60)
    print("Trade Logger + Daily Report - Manual Test")
    print("=" * 60)

    # 전용 테스트 디렉토리
    test_root = Path("logs/_manual_test")
    trades_dir = test_root / "trades"
    reports_dir = test_root / "reports"

    # 1. Trade Logger 초기화
    print("\n[1] Trade Logger 초기화")
    from libs.adapters.trade_logger import TradeLogger

    trade_logger = TradeLogger(log_dir=str(trades_dir))
    print(f"  Log Dir: {trade_logger.log_dir}")
    print("  ✅ PASS")

    # 2. 거래 기록 테스트
    print("\n[2] 거래 기록 테스트")
    run_tag = datetime.now().strftime("%H%M%S")
    test_trades = [
        {
            "exchange": "upbit",
            "order_id": f"test_{run_tag}_001",
            "symbol": "BTC/KRW",
            "side": "BUY",
            "quantity": 0.001,
            "price": 133000000,
            "fee": 665,
            "pnl": 0,
            "status": "FILLED",
        },
        {
            "exchange": "upbit",
            "order_id": f"test_{run_tag}_002",
            "symbol": "ETH/KRW",
            "side": "buy",
            "quantity": 0.1,
            "price": 4500000,
            "fee": 225,
            "pnl": 0,
            "status": "FILLED",
        },
        {
            "exchange": "bithumb",
            "order_id": f"test_{run_tag}_003",
            "symbol": "BTC/KRW",
            "side": "SELL",
            "quantity": 0.001,
            "price": 133500000,
            "fee": 668,
            "pnl": 500,
            "status": "FILLED",
        },
    ]

    for trade in test_trades:
        result = trade_logger.log_trade(trade)
        print(f"  {trade['symbol']} {trade['side']}: {'✅' if result else '❌'}")
        if not result:
            print("  ❌ FAIL: log_trade returned False")
            return False
    print("  ✅ PASS")

    # 3. 거래 조회 테스트
    print("\n[3] 거래 조회 테스트")
    trades = trade_logger.get_trades(date.today())
    print(f"  Today's Trades (in test dir): {len(trades)}")
    if len(trades) < 3:
        print("  ❌ FAIL: expected at least 3 trades")
        return False
    print("  ✅ PASS")

    # 4. 일일 요약 테스트
    print("\n[4] 일일 요약 테스트")
    summary = trade_logger.get_daily_summary()
    print(f"  Total Trades: {summary['total_trades']}")
    print(f"  Buy/Sell: {summary['buy_count']}/{summary['sell_count']}")
    print(f"  Volume: {summary['total_volume']:,.0f} KRW")
    print(f"  Fees: {summary['total_fee']:,.0f} KRW")
    print(f"  PnL: {summary['total_pnl']:+,.0f} KRW")

    # side 정규화 검증 (소문자 "buy" → "BUY")
    if summary["buy_count"] < 2:
        print("  ❌ FAIL: side normalization failed")
        return False
    print("  ✅ PASS")

    # 5. Daily Report 생성
    print("\n[5] Daily Report 생성")
    from libs.analytics.daily_report import DailyReportGenerator
    from libs.analytics.strategy_health import StrategyHealthMonitor

    health_monitor = StrategyHealthMonitor()
    reporter = DailyReportGenerator(
        trade_logger, health_monitor, report_dir=str(reports_dir)
    )

    report = reporter.generate()
    expected_path = reports_dir / f"daily_{date.today().strftime('%Y-%m-%d')}.md"
    print(f"  Report saved to: {expected_path}")
    print(f"  Report length: {len(report)} chars")
    if not expected_path.exists():
        print("  ❌ FAIL: report file not created")
        return False
    print("  ✅ PASS")

    # 6. 콘솔 요약 출력
    print("\n[6] 콘솔 요약")
    reporter.print_summary()

    print("=" * 60)
    print("✅ All Tests Complete (6/6 PASS)")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
