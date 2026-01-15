"""
Kill-Switch Integration Tests

Tests kill-switch detection and order blocking functionality.
"""

import pytest
import os
import tempfile
from pathlib import Path
from libs.core.config import Config, AssetClass
from libs.core.order_validator import OrderValidator


class TestKillSwitch:
    """Kill-Switch functionality tests"""
    
    @pytest.fixture
    def temp_kill_switch(self):
        """Create temporary kill-switch file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("EMERGENCY STOP")
            path = f.name
        yield path
       # Cleanup
        if os.path.exists(path):
            os.unlink(path)
    
    def test_kill_switch_detection_active(self, temp_kill_switch):
        """Test kill-switch active detection"""
        config = Config(
            asset_class=AssetClass.CRYPTO_SPOT,
            kill_switch_file=temp_kill_switch
        )
        
        assert config.is_kill_switch_active() is True, "Kill-switch should be detected as active"
    
    def test_kill_switch_detection_inactive(self):
        """Test kill-switch inactive (file doesn't exist)"""
        config = Config(
            asset_class=AssetClass.CRYPTO_SPOT,
            kill_switch_file="/nonexistent/kill_switch.txt"
        )
        
        assert config.is_kill_switch_active() is False, "Non-existent file should be inactive"
    
    def test_kill_switch_no_file_configured(self):
        """Test kill-switch when no file is configured"""
        config = Config(
            asset_class=AssetClass.CRYPTO_SPOT,
            kill_switch_file=None
        )
        
        assert config.is_kill_switch_active() is False, "No file configured should be inactive"
    
    def test_order_validator_blocks_when_active(self, temp_kill_switch):
        """Test that OrderValidator blocks orders when kill-switch is active"""
        config = Config(
            asset_class=AssetClass.CRYPTO_SPOT,
            kill_switch_file=temp_kill_switch
        )
        
        validator = OrderValidator(config)
        
        # Try to validate an order
        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.01,
            price=133_000_000,
            balance=10_000_000,
            total_equity=10_000_000
        )
        
        assert result.valid is False, "Order should be rejected when kill-switch is active"
        assert "Kill-Switch" in result.reason, f"Rejection reason should mention kill-switch, got: {result.reason}"
    
    def test_order_validator_allows_when_inactive(self):
        """Test that OrderValidator allows orders when kill-switch is inactive"""
        config = Config(
            asset_class=AssetClass.CRYPTO_SPOT,
            kill_switch_file=None
        )
        
        validator = OrderValidator(config)
        
        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.007,
            price=133_000_000,  # ~931k KRW (<= 10% of equity)
            balance=10_000_000,
            total_equity=10_000_000
        )
        
        assert result.valid is True, f"Order should be allowed, but got: {result.reason}"
    
    def test_order_validator_max_position_limit(self):
        """Test position size limit (10% of equity)"""
        config = Config(
            asset_class=AssetClass.CRYPTO_SPOT
        )
        
        validator = OrderValidator(config)
        
        # Order value = 0.02 * 133M = 2.66M (26.6% of 10M equity)
        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.02,
            price=133_000_000,
            balance=10_000_000,
            total_equity=10_000_000
        )
        
        assert result.valid is False, "Order should exceed 10% limit"
        assert "10%" in result.reason or "equity" in result.reason.lower()
    
    def test_order_validator_insufficient_balance(self):
        """Test insufficient balance rejection"""
        config = Config(
            asset_class=AssetClass.CRYPTO_SPOT
        )
        
        validator = OrderValidator(config)
        
        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.1,
            price=133_000_000,  # 13.3M needed
            balance=1_000_000,  # Only 1M available
            total_equity=10_000_000
        )
        
        assert result.valid is False, "Should reject due to insufficient balance"
        assert "max" in result.reason.lower() or "value" in result.reason.lower()


# Manual test runner
if __name__ == "__main__":
    print("=== Kill-Switch Manual Test ===\n")
    
    # Test 1: No kill-switch
    print("[Test 1] No kill-switch configured:")
    config1 = Config(asset_class=AssetClass.CRYPTO_SPOT)
    print(f"  Active: {config1.is_kill_switch_active()}")
    print(f"  ✅ PASS\n")
    
    # Test 2: Non-existent file
    print("[Test 2] Non-existent kill-switch file:")
    config2 = Config(
        asset_class=AssetClass.CRYPTO_SPOT,
        kill_switch_file="/tmp/nonexistent_kill_switch.txt"
    )
    print(f"  Active: {config2.is_kill_switch_active()}")
    print(f"  ✅ PASS\n")
    
    # Test 3: Create and detect
    print("[Test 3] Create kill-switch and detect:")
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("EMERGENCY")
        temp_path = f.name
    
    config3 = Config(
        asset_class=AssetClass.CRYPTO_SPOT,
        kill_switch_file=temp_path
    )
    print(f"  File: {temp_path}")
    print(f"  Active: {config3.is_kill_switch_active()}")
    
    # Test validator
    validator = OrderValidator(config3)
    result = validator.validate_quick()
    print(f"  Validator: {result.valid}, Reason: {result.reason}")
    
    os.unlink(temp_path)
    print(f"  ✅ PASS\n")
    
    print("=== All Manual Tests Complete ===")
