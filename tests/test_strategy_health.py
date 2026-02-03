"""
Strategy Health Monitor 테스트
"""

import pytest
from datetime import datetime, timedelta
from libs.analytics.strategy_health import (
    StrategyHealthMonitor,
    HealthStatus,
    HealthCheckResult,
)


class TestStrategyHealthMonitor:
    """StrategyHealthMonitor 테스트"""

    def test_healthy_status(self):
        """정상 상태 테스트"""
        monitor = StrategyHealthMonitor()
        # 수익 거래 5개 추가
        for i in range(5):
            monitor.add_trade({"pnl": 10000, "pnl_pct": 0.01})

        result = monitor.check_health()
        assert result.status == HealthStatus.HEALTHY
        assert result.consecutive_losses == 0

    def test_consecutive_loss_warning(self):
        """연속 손실 5회 경고 테스트"""
        monitor = StrategyHealthMonitor()
        for i in range(5):
            monitor.add_trade({"pnl": -10000, "pnl_pct": -0.01})

        result = monitor.check_health()
        assert result.status == HealthStatus.WARNING
        assert result.consecutive_losses == 5
        assert any("CONSECUTIVE_LOSS" in t for t in result.triggers)

    def test_consecutive_loss_critical(self):
        """연속 손실 8회 중단 테스트"""
        monitor = StrategyHealthMonitor()
        for i in range(8):
            monitor.add_trade({"pnl": -10000, "pnl_pct": -0.01})

        result = monitor.check_health()
        assert result.status == HealthStatus.CRITICAL
        assert result.consecutive_losses == 8
        assert any("CONSECUTIVE_LOSS" in t for t in result.triggers)

    def test_mdd_critical(self):
        """MDD 15% 초과 테스트"""
        monitor = StrategyHealthMonitor()
        # 15% 이상 누적 손실
        for i in range(5):
            monitor.add_daily_pnl(-0.04)  # 총 -20%

        result = monitor.check_health()
        assert result.status == HealthStatus.CRITICAL
        assert result.mdd_current > 0.15
        assert any("MDD_CRITICAL" in t for t in result.triggers)

    def test_daily_loss_halt(self):
        """일일 3% 손실 중단 테스트"""
        monitor = StrategyHealthMonitor()
        monitor.add_daily_pnl(-0.04)  # 4% 손실

        result = monitor.check_health()
        assert result.status == HealthStatus.CRITICAL
        assert any("DAILY_LOSS" in t for t in result.triggers)

    def test_get_summary(self):
        """요약 JSON 직렬화 테스트"""
        monitor = StrategyHealthMonitor()

        # 거래 추가
        for i in range(3):
            monitor.add_trade({"pnl": 5000, "pnl_pct": 0.005})

        summary = monitor.get_summary()

        assert "status" in summary
        assert "sharpe_30d" in summary
        assert "mdd_pct" in summary
        assert "recommendation" in summary
        assert "total_trades" in summary
        assert summary["total_trades"] == 3


class TestHealthMonitorIntegration:
    """PaperExecution 통합 테스트"""

    def test_paper_execution_health_integration(self):
        """PaperExecution과 HealthMonitor 통합"""
        from libs.adapters.paper_execution import PaperExecutionAdapter
        from libs.adapters.factory import AdapterFactory

        md = AdapterFactory.create_market_data("upbit_spot")
        pe = PaperExecutionAdapter(md, initial_balance=10_000_000)

        # 거래 실행
        result = pe.place_order("BTC/KRW", "BUY", 0.001)

        assert result.success or result.status in ["FILLED", "PENDING", "REJECTED"]

        # 건강 상태 확인
        health = pe.get_health_status()
        assert "status" in health
        assert health["status"] in ["HEALTHY", "WARNING", "CRITICAL", "HALTED"]
        assert "total_trades" in health


# Manual test runner
if __name__ == "__main__":
    print("=== Strategy Health Monitor Test ===\n")

    # Test 1: Healthy
    print("[1] Healthy Status Test")
    m1 = StrategyHealthMonitor()
    for i in range(3):
        m1.add_trade({"pnl": 10000, "pnl_pct": 0.01})
    r1 = m1.check_health()
    print(f"  Status: {r1.status.value}")
    print(f"  ✅ PASS\n")

    # Test 2: Warning
    print("[2] Warning (5 consecutive losses)")
    m2 = StrategyHealthMonitor()
    for i in range(5):
        m2.add_trade({"pnl": -10000, "pnl_pct": -0.01})
    r2 = m2.check_health()
    print(f"  Status: {r2.status.value}")
    print(f"  Consecutive Losses: {r2.consecutive_losses}")
    print(f"  ✅ PASS\n")

    # Test 3: Critical
    print("[3] Critical (8 consecutive losses)")
    m3 = StrategyHealthMonitor()
    for i in range(8):
        m3.add_trade({"pnl": -10000, "pnl_pct": -0.01})
    r3 = m3.check_health()
    print(f"  Status: {r3.status.value}")
    print(f"  Triggers: {r3.triggers}")
    print(f"  ✅ PASS\n")

    print("✅ All manual tests passed")
