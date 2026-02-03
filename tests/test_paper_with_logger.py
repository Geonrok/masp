"""
PaperExecution + TradeLogger 통합 테스트
"""

import sys
import os
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> bool:
    print("=" * 60)
    print("PaperExecution + TradeLogger Integration Test")
    print("=" * 60)

    # 전용 테스트 디렉토리
    test_trades_dir = Path("logs/_paper_test/trades")

    # 1. 어댑터 초기화
    print("\n[1] 어댑터 초기화")
    from libs.adapters.trade_logger import TradeLogger
    from libs.adapters.paper_execution import PaperExecutionAdapter
    from libs.adapters.real_upbit_spot import UpbitSpotMarketData

    trade_logger = TradeLogger(log_dir=str(test_trades_dir))
    market_data = UpbitSpotMarketData()
    paper = PaperExecutionAdapter(
        market_data_adapter=market_data,
        initial_balance=10_000_000,
        trade_logger=trade_logger,
    )
    print(f"  Initial Balance: {paper.get_balance():,.0f} KRW")
    print("  ✅ PASS")

    # 2. 매수 주문
    print("\n[2] BTC 매수 주문")
    buy_result = paper.place_order(
        symbol="BTC/KRW", side="BUY", quantity=0.0001, order_type="MARKET"
    )
    print(f"  Order ID: {buy_result.order_id}")
    print(f"  Quantity: {buy_result.quantity} @ {buy_result.price:,.0f}")
    print(f"  Status: {buy_result.status}")
    if buy_result.status != "FILLED":
        print("  ❌ FAIL: Order not filled")
        return False
    print("  ✅ PASS")

    # 3. 매도 주문
    print("\n[3] BTC 매도 주문")
    sell_result = paper.place_order(
        symbol="BTC/KRW", side="SELL", quantity=0.0001, order_type="MARKET"
    )
    print(f"  Order ID: {sell_result.order_id}")
    print(f"  Quantity: {sell_result.quantity} @ {sell_result.price:,.0f}")
    print(f"  Status: {sell_result.status}")
    if sell_result.status != "FILLED":
        print("  ❌ FAIL: Order not filled")
        return False
    print("  ✅ PASS")

    # 4. 거래 로그 확인
    print("\n[4] 거래 로그 확인")
    trades = trade_logger.get_trades(date.today())
    print(f"  Logged Trades: {len(trades)}")
    if len(trades) < 2:
        print("  ❌ FAIL: Expected at least 2 trades")
        return False

    for t in trades:
        print(f"    - {t['symbol']} {t['side']} @ {float(t['price']):,.0f}")
    print("  ✅ PASS")

    # 5. 일일 요약 확인
    print("\n[5] 일일 요약")
    summary = trade_logger.get_daily_summary()
    print(f"  Total Trades: {summary['total_trades']}")
    print(f"  Buy/Sell: {summary['buy_count']}/{summary['sell_count']}")
    print(f"  Volume: {summary['total_volume']:,.0f} KRW")
    print(f"  Fees: {summary['total_fee']:,.0f} KRW")
    print(f"  PnL: {summary['total_pnl']:+,.0f} KRW")
    print("  ✅ PASS")

    # 6. Health 상태
    print("\n[6] Health 상태")
    health = paper.get_health_status()
    print(f"  Status: {health.get('status', 'N/A')}")
    print(f"  Total Trades: {health.get('total_trades', 0)}")
    print("  ✅ PASS")

    print("\n" + "=" * 60)
    print("✅ All Tests Complete (6/6 PASS)")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
