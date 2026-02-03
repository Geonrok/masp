print("=== A4b: Private API 인증 (직접 라이브러리, 읽기 전용) ===")
print("주의: place_order 호출 절대 금지\n")

import os

from dotenv import load_dotenv

load_dotenv(override=False)


def mask(s):
    return f"SET(len={len(s)})" if s else "MISSING"


# Upbit 인증
ak = os.getenv("UPBIT_ACCESS_KEY", "")
sk = os.getenv("UPBIT_SECRET_KEY", "")
if ak and sk:
    try:
        import pyupbit

        upbit = pyupbit.Upbit(ak, sk)
        balances = upbit.get_balances()
        if balances is not None:
            print("UPBIT 인증: PASS (private endpoint OK)")
        else:
            print("UPBIT 인증: WARN (balances=None)")
    except Exception as e:
        msg = str(e)[:100]
        print(f"UPBIT 인증: FAIL ({type(e).__name__}: {msg})")
else:
    print(f"UPBIT 인증: SKIP (keys: ak={mask(ak)}, sk={mask(sk)})")

# Bithumb 인증
bk = os.getenv("BITHUMB_API_KEY", "")
bs = os.getenv("BITHUMB_SECRET_KEY", "")
if bk and bs:
    try:
        import pybithumb

        bithumb = pybithumb.Bithumb(bk, bs)
        balance = bithumb.get_balance("BTC")
        if balance is not None:
            print("BITHUMB 인증: PASS (private endpoint OK)")
        else:
            print("BITHUMB 인증: WARN (balance=None)")
    except Exception as e:
        msg = str(e)[:100]
        print(f"BITHUMB 인증: FAIL ({type(e).__name__}: {msg})")
else:
    print(f"BITHUMB 인증: SKIP (keys: bk={mask(bk)}, bs={mask(bs)})")
