"""
Kill-Switch validation for live trading tests (v2.2 Final).
"""
from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, ROOT_DIR)

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from libs.core.config import Config
from _live_test_utils import live_test_enabled

if pytest and not live_test_enabled():
    pytest.skip("Live trading tests disabled (env guard not satisfied)", allow_module_level=True)


def test_kill_switch() -> bool:
    """Fail fast if kill-switch is active."""
    print("=" * 60)
    print("Kill-Switch Check")
    print("=" * 60)

    if os.getenv("STOP_TRADING") == "1":
        print("❌ STOP_TRADING is set - halt")
        return False

    config = Config(asset_class="crypto_spot", strategy_name="live_test")
    if config.is_kill_switch_active():
        print("❌ Kill-Switch ACTIVE - halt")
        return False

    print("✅ Kill-Switch INACTIVE")
    return True


if __name__ == "__main__":
    success = test_kill_switch()
    sys.exit(0 if success else 1)
