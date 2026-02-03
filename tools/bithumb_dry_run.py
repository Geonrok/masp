"""
Bithumb Live Dry Run Script
- 10,000원 소액 시장가 매수/매도 테스트
"""

from dotenv import load_dotenv
import os
import time

load_dotenv(override=True)

from libs.adapters.bithumb_api_v2 import BithumbAPIV2


def main():
    api_key = os.getenv("BITHUMB_API_KEY")
    secret_key = os.getenv("BITHUMB_SECRET_KEY")

    client = BithumbAPIV2(api_key, secret_key)

    # 1. 현재 잔고 확인
    print("=== Step 1: 현재 잔고 확인 ===")
    accounts = client.get_accounts()
    for acc in accounts:
        if acc.get("currency") in ["KRW", "BTC"]:
            print(f"  {acc['currency']}: {acc['balance']}")

    # 2. 현재가 확인
    print("\n=== Step 2: BTC 현재가 ===")
    ticker = client.get_ticker(["KRW-BTC"])
    btc_price = float(ticker[0]["trade_price"])
    print(f"  BTC: {btc_price:,.0f} KRW")

    # 3. 시장가 매수 (10,000원)
    print("\n=== Step 3: BTC 시장가 매수 (10,000원) ===")
    try:
        result = client.post_order(
            market="KRW-BTC",
            side="bid",
            ord_type="price",  # 시장가 매수 (KRW 금액 지정)
            price="10000",
        )
        print(f"  주문 결과: {result}")
        order_id = result.get("uuid")
        print(f"  주문 ID: {order_id}")
    except Exception as e:
        print(f"  ❌ 매수 실패: {e}")
        return

    # 4. 체결 대기
    print("\n=== Step 4: 체결 대기 (3초) ===")
    time.sleep(3)

    # 5. 주문 상태 확인
    print("\n=== Step 5: 주문 상태 확인 ===")
    try:
        order_status = client.get_order(order_id)
        print(f"  상태: {order_status.get('state')}")
        print(f"  체결량: {order_status.get('executed_volume')}")
    except Exception as e:
        print(f"  ⚠️ 조회 실패: {e}")

    # 6. 잔고 확인 (매수 후)
    print("\n=== Step 6: 매수 후 잔고 ===")
    accounts = client.get_accounts()
    btc_balance = 0
    for acc in accounts:
        if acc.get("currency") in ["KRW", "BTC"]:
            print(f"  {acc['currency']}: {acc['balance']}")
            if acc.get("currency") == "BTC":
                btc_balance = float(acc["balance"])

    # 7. 시장가 매도 (전량)
    print(f"\n=== Step 7: BTC 시장가 매도 ({btc_balance:.8f} BTC) ===")
    if btc_balance > 0.00001:  # 최소 주문량 확인
        try:
            result = client.post_order(
                market="KRW-BTC",
                side="ask",
                ord_type="market",  # 시장가 매도 (수량 지정)
                volume=f"{btc_balance:.8f}",
            )
            print(f"  주문 결과: {result}")
            sell_order_id = result.get("uuid")
            print(f"  주문 ID: {sell_order_id}")
        except Exception as e:
            print(f"  ❌ 매도 실패: {e}")
    else:
        print(f"  ⚠️ 잔고 부족 (최소 0.00001 BTC 필요)")

    # 8. 최종 잔고 확인
    print("\n=== Step 8: 최종 잔고 ===")
    time.sleep(3)
    accounts = client.get_accounts()
    for acc in accounts:
        if acc.get("currency") in ["KRW", "BTC"]:
            print(f"  {acc['currency']}: {acc['balance']}")

    print("\n=== Dry Run 완료 ===")


if __name__ == "__main__":
    main()
