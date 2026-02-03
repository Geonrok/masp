import os

from libs.core.config import Config

print("=== A2-1: Kill-Switch 상태 확인 ===")
try:
    c = Config(asset_class="crypto_spot", strategy_name="kama_tsmom_gate")
    is_active = c.is_kill_switch_active()
    print(f"Config 로드: PASS")
    print(f"Kill-Switch Active: {is_active}")
    print(f"STOP_TRADING env: {os.getenv('STOP_TRADING', 'Not Set')}")
except Exception as e:
    print(f"Config 로드: FAIL - {e}")
