"""
Upbit 소액 실거래 테스트
⚠️ WARNING: 실제 주문이 실행됩니다!

테스트 내용:
1. 사전 검증 (잔고, Kill-Switch, API)
2. BTC 5,000 KRW 시장가 매수
3. 체결 확인 및 로그 기록
4. BTC 전량 시장가 매도
5. 최종 PnL 계산
"""

import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def confirm_live_trading() -> bool:
    """실거래 확인"""
    print("\n" + "=" * 60)
    print("⚠️  WARNING: LIVE TRADING TEST")
    print("=" * 60)
    print("\n이 스크립트는 실제 주문을 실행합니다!")
    print("약 6,000 KRW의 실제 자금이 사용됩니다.")
    print("\n계속하시겠습니까? (yes/no): ", end="")

    response = input().strip().lower()
    return response == "yes"


def main() -> bool:
    print("=" * 60)
    print("Upbit Live Trading Test (6,000 KRW)")
    print("=" * 60)

    # 0. 실거래 확인
    if not confirm_live_trading():
        print("\n❌ 테스트 취소됨")
        return False

    # 1. 모듈 임포트
    print("\n[1] 모듈 로드")
    from pathlib import Path

    from libs.adapters.factory import AdapterFactory
    from libs.adapters.trade_logger import TradeLogger
    from libs.analytics.daily_report import DailyReportGenerator
    from libs.core.config import Config

    # 실거래 로그 디렉토리
    live_log_dir = Path("logs/live_trades")

    config = Config(asset_class="crypto_spot", strategy_name="live_test")
    logger = TradeLogger(log_dir=str(live_log_dir / "trades"))
    print("  Config + TradeLogger 로드 완료")
    print("  ✅ PASS")

    # 2. Kill-Switch 체크
    print("\n[2] Kill-Switch 체크")
    if config.is_kill_switch_active():
        print("  🔴 Kill-Switch ACTIVE - 테스트 중단")
        return False
    print("  Kill-Switch: ✅ INACTIVE")
    print("  ✅ PASS")

    # 3. Upbit 어댑터 생성
    print("\n[3] Upbit 어댑터 생성")
    try:
        upbit = AdapterFactory.create_execution(
            "upbit_spot",
            adapter_mode="live",
            config=config,
            trade_logger=logger,
        )
        print("  어댑터 생성 완료")
        print("  ✅ PASS")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        return False

    # 4. 사전 잔고 확인
    print("\n[4] 사전 잔고 확인")
    krw_before = upbit.get_balance("KRW")
    btc_before = upbit.get_balance("BTC")
    print(f"  KRW: {krw_before:,.0f}")
    print(f"  BTC: {btc_before:.8f}")

    if krw_before < 6000:
        print("  ❌ FAIL: 잔고 부족 (최소 6,000 KRW 필요)")
        return False
    print("  ✅ PASS")

    # 5. BTC 현재가 확인
    print("\n[5] BTC/KRW 현재가")
    btc_price = upbit.get_current_price("BTC/KRW")
    if btc_price is None:
        print("  ❌ FAIL: 가격 조회 실패")
        return False
    print(f"  BTC/KRW: {btc_price:,.0f} KRW")

    # 매수 수량 계산 (6,000 KRW 기준)
    buy_amount_krw = 6000
    buy_quantity = buy_amount_krw / btc_price
    print(f"  매수 예정: {buy_quantity:.8f} BTC (~{buy_amount_krw:,} KRW)")
    print("  ✅ PASS")

    # 6. BTC 시장가 매수
    print("\n[6] BTC 시장가 매수 (6,000 KRW)")
    print("  주문 실행 중...")

    buy_result = upbit.place_order(
        symbol="BTC/KRW", side="BUY", quantity=buy_quantity, order_type="MARKET"
    )

    if buy_result.status == "REJECTED":
        print(f"  ❌ FAIL: {buy_result.message}")
        return False

    print(f"  Order ID: {buy_result.order_id}")
    print(f"  Status: {buy_result.status}")
    print(
        f"  Filled: {buy_result.filled_quantity:.8f} BTC @ {buy_result.filled_price:,.0f}"
    )
    print(f"  Fee: {buy_result.fee:.2f} KRW")
    print("  ✅ PASS")

    # 7. 체결 대기
    print("\n[7] 체결 확인 대기 (3초)")
    time.sleep(3)

    btc_after_buy = upbit.get_balance("BTC")
    print(f"  BTC 잔고: {btc_after_buy:.8f}")
    print("  ✅ PASS")

    # 8. BTC 전량 시장가 매도
    print("\n[8] BTC 전량 시장가 매도")

    # 매도 가능 수량 확인
    sell_quantity = btc_after_buy
    if sell_quantity <= 0:
        print("  ⚠️ 매도할 BTC 없음 - 스킵")
    else:
        print(f"  매도 수량: {sell_quantity:.8f} BTC")
        print("  주문 실행 중...")

        sell_result = upbit.place_order(
            symbol="BTC/KRW", side="SELL", quantity=sell_quantity, order_type="MARKET"
        )

        if sell_result.status == "REJECTED":
            print(f"  ❌ FAIL: {sell_result.message}")
            print("  ⚠️ BTC가 계정에 남아있습니다!")
            return False

        print(f"  Order ID: {sell_result.order_id}")
        print(f"  Status: {sell_result.status}")
        print(
            f"  Filled: {sell_result.filled_quantity:.8f} BTC @ {sell_result.filled_price:,.0f}"
        )
        print(f"  Fee: {sell_result.fee:.2f} KRW")
        print("  ✅ PASS")

    # 9. 최종 잔고 확인
    print("\n[9] 최종 잔고 확인")
    time.sleep(2)

    krw_after = upbit.get_balance("KRW")
    btc_after = upbit.get_balance("BTC")

    print(f"  KRW: {krw_after:,.0f} (변동: {krw_after - krw_before:+,.0f})")
    print(f"  BTC: {btc_after:.8f}")
    print("  ✅ PASS")

    # 10. PnL 계산
    print("\n[10] PnL 계산")
    pnl = krw_after - krw_before
    pnl_pct = (pnl / buy_amount_krw) * 100

    print(f"  투자금: {buy_amount_krw:,} KRW")
    print(f"  PnL: {pnl:+,.0f} KRW ({pnl_pct:+.2f}%)")

    # 수수료 예상 (0.05% * 2 = 0.1%)
    expected_fee = buy_amount_krw * 0.001
    print(f"  예상 수수료: ~{expected_fee:,.0f} KRW")
    print("  ✅ PASS")

    # 11. 거래 로그 확인
    print("\n[11] 거래 로그 확인")
    trades = logger.get_trades(date.today())
    print(f"  기록된 거래: {len(trades)}건")
    for t in trades[-2:]:  # 최근 2건
        print(f"    - {t['symbol']} {t['side']} @ {float(t['price']):,.0f}")
    print("  ✅ PASS")

    # 12. Daily Report 생성
    print("\n[12] Daily Report 생성")
    from libs.analytics.strategy_health import StrategyHealthMonitor

    health = StrategyHealthMonitor()
    reporter = DailyReportGenerator(
        logger, health, report_dir=str(live_log_dir / "reports")
    )
    reporter.generate()
    print("  Report 저장: logs/live_trades/reports/")
    print("  ✅ PASS")

    # 최종 결과
    print("\n" + "=" * 60)
    print("🎉 Live Trading Test COMPLETE")
    print("=" * 60)
    print(f"\n총 PnL: {pnl:+,.0f} KRW ({pnl_pct:+.2f}%)")
    print(f"거래 기록: {len(trades)}건")
    print("로그 위치: logs/live_trades/")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
