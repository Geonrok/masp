print("=== A4: Private API 인증 테스트 (get_balance만 호출) ===")
print("주의: place_order 호출 절대 금지\n")

from libs.adapters.factory import AdapterFactory
from libs.core.config import Config

cfg = Config(asset_class="crypto_spot", strategy_name="kama_tsmom_gate")

targets = [
    ("upbit_spot", "UPBIT"),
    ("bithumb_spot", "BITHUMB"),
]

for ex, label in targets:
    try:
        exa = AdapterFactory.create_execution(ex, adapter_mode="live", config=cfg)
        bal = exa.get_balance("KRW")
        if bal is not None:
            print(f"{label} get_balance: PASS (인증 성공)")
        else:
            print(f"{label} get_balance: WARN (None 반환, 잔고 0 가능)")
    except Exception as e:
        msg = str(e)[:120]
        if "401" in msg or "unauthorized" in msg.lower() or "signature" in msg.lower():
            print(f"{label} get_balance: FAIL (인증 오류: {msg})")
        else:
            print(f"{label} get_balance: FAIL ({type(e).__name__}: {msg})")
