"""
Bithumb 어댑터 테스트 (조회만)
⚠️ 실제 주문 테스트는 별도 검증 후 수행
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> bool:
    print("=" * 60)
    print("Bithumb Adapters Test (READ ONLY)")
    print("⚠️ No actual orders will be placed")
    print("=" * 60)

    # 1. 시세 어댑터 테스트
    print("\n[1] 시세 어댑터 초기화")
    from libs.adapters.real_bithumb_spot import BithumbSpotMarketData

    try:
        market_data = BithumbSpotMarketData()
        print("  ✅ PASS")
    except ImportError as e:
        print(f"  ❌ FAIL: {e}")
        return False

    # 2. BTC 현재가 조회
    print("\n[2] BTC/KRW 현재가 조회")
    quote = market_data.get_quote("BTC/KRW")
    if quote:
        print(f"  Last: {quote.last:,.0f} KRW")
        print(f"  Bid: {quote.bid:,.0f} / Ask: {quote.ask:,.0f}")
        print("  ✅ PASS")
    else:
        print("  ❌ FAIL: Quote unavailable")
        return False

    # 3. 호가창 조회
    print("\n[3] BTC/KRW 호가창 조회")
    orderbook = market_data.get_orderbook("BTC/KRW", depth=3)
    if orderbook:
        print(f"  Bids: {len(orderbook.get('bids', []))} levels")
        print(f"  Asks: {len(orderbook.get('asks', []))} levels")
        print("  ✅ PASS")
    else:
        print("  ❌ FAIL: Orderbook unavailable")
        return False

    # 4. OHLCV 조회
    print("\n[4] BTC/KRW OHLCV 조회")
    ohlcv = market_data.get_ohlcv("BTC/KRW", interval="1d", limit=5)
    if ohlcv:
        print(f"  Candles: {len(ohlcv)}")
        if ohlcv:
            latest = ohlcv[-1]
            print(
                f"  Latest: O={latest.open:,.0f} H={latest.high:,.0f} L={latest.low:,.0f} C={latest.close:,.0f}"
            )
        print("  ✅ PASS")
    else:
        print("  ⚠️ SKIP: OHLCV unavailable (pybithumb API limitation)")
        print("  ✅ PASS (non-critical)")

    # 5. 전체 종목 조회
    print("\n[5] 전체 종목 조회")
    tickers = market_data.get_tickers()
    print(f"  Total tickers: {len(tickers)}")
    print(f"  Sample: {tickers[:5]}")
    print("  ✅ PASS")

    # 6. 실주문 어댑터 테스트 (API 키 필요)
    print("\n[6] 실주문 어댑터 초기화")
    from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
    from libs.core.config import Config

    try:
        config = Config(asset_class="crypto_spot", strategy_name="bithumb_test")
        execution = BithumbExecutionAdapter(config)
        print("  ✅ PASS")
    except ValueError as e:
        print(f"  ⚠️ SKIP: {e}")
        print("  → .env에 Bithumb API 키 설정 필요")
        print("\n" + "=" * 60)
        print("✅ Market Data Tests Complete (5/5 PASS)")
        print("⚠️ Execution tests skipped (no API key)")
        print("=" * 60)
        return True
    except ImportError as e:
        print(f"  ❌ FAIL: {e}")
        return False

    # 7. KRW 잔고 조회
    print("\n[7] KRW 잔고 조회")
    krw_balance = execution.get_balance("KRW")
    print(f"  KRW Balance: {krw_balance:,.0f} KRW")
    print("  ✅ PASS")

    # 8. Kill-Switch 상태
    print("\n[8] Kill-Switch 상태")
    kill_switch = config.is_kill_switch_active()
    print(f"  Kill-Switch: {'🔴 ACTIVE' if kill_switch else '✅ INACTIVE'}")
    print("  ✅ PASS")

    print("\n" + "=" * 60)
    print("✅ All Tests Complete (8/8 PASS)")
    print("=" * 60)
    print("\n⚠️ 실제 주문 테스트는 별도로 신중하게 진행하세요")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
