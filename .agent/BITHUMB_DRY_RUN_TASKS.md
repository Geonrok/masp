# Bithumb Live Dry Run 작업 지시서

## 📋 작업 개요

```yaml
프로젝트: MASP (Multi-Asset Strategy Platform)
작업: Bithumb API 2.0 소액 Dry Run (10,000원)
날짜: 2026-01-15 16:30 KST
상태: API 연결 성공 - 실거래 테스트 준비 완료
```

---

## ✅ 사전 검증 완료

### API 연결 성공
```
✅ GET /v1/accounts: SUCCESS
   KRW: 42,959원
   BTC: 0.00001417
   ETH: 0.0090172

✅ GET /v1/ticker: SUCCESS
   BTC = 141,639,000 KRW

✅ 테스트: 157 passed, 5 skipped
```

---

## 🎯 작업 목표

**10,000원 소액으로 시장가 매수/매도 테스트**

1. BTC 시장가 매수 (10,000원)
2. 주문 결과 확인 (order_id, 체결량)
3. BTC 시장가 매도 (전량)
4. 최종 잔고 확인

---

## 📁 작업 대상 파일

### 1. 신규: `tools/bithumb_dry_run.py`
```python
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
    api_key = os.getenv('BITHUMB_API_KEY')
    secret_key = os.getenv('BITHUMB_SECRET_KEY')
    
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
            price="10000"
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
                btc_balance = float(acc['balance'])
    
    # 7. 시장가 매도 (전량)
    print(f"\n=== Step 7: BTC 시장가 매도 ({btc_balance:.8f} BTC) ===")
    if btc_balance > 0.00001:  # 최소 주문량 확인
        try:
            result = client.post_order(
                market="KRW-BTC",
                side="ask",
                ord_type="market",  # 시장가 매도 (수량 지정)
                volume=f"{btc_balance:.8f}"
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
```

---

## 🧪 실행 방법

```bash
# Dry Run 실행
python tools/bithumb_dry_run.py
```

---

## ✅ 성공 기준

| 항목 | 기준 |
|------|------|
| 시장가 매수 | order_id 반환 |
| 체결 확인 | state = "done" 또는 "wait" |
| 시장가 매도 | order_id 반환 |
| 잔고 변화 | KRW 감소 → 복구 (수수료 제외) |

---

## ⚠️ 주의사항

1. **실제 거래**: 이 스크립트는 실제 KRW를 사용합니다
2. **소액 테스트**: 10,000원으로 제한
3. **수수료**: 매수/매도 시 0.25% 수수료 발생
4. **최소 주문량**: BTC 최소 0.0001 BTC
