print("=== A4a: 가드레일 검증 (Live 모드 차단 = 정상) ===")
from libs.adapters.factory import AdapterFactory
from libs.core.config import Config

cfg = Config(asset_class="crypto_spot", strategy_name="kama_tsmom_gate")

for ex in ["upbit_spot", "bithumb_spot"]:
    try:
        AdapterFactory.create_execution(ex, adapter_mode="live", config=cfg)
        print(f"{ex} guardrail: FAIL (UNEXPECTED: live adapter created)")
    except RuntimeError as e:
        msg = str(e).lower()
        if "live trading disabled" in msg:
            print(f"{ex} guardrail: PASS (차단됨 - 정상)")
        else:
            print(f"{ex} guardrail: FAIL (unexpected error: {msg[:80]})")
    except Exception as e:
        print(f"{ex} guardrail: FAIL ({type(e).__name__}: {str(e)[:80]})")
