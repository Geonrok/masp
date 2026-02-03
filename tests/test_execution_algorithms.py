"""
Tests for execution algorithms.
"""

import pytest
from datetime import datetime, timedelta

from libs.execution.algorithms import (
    TWAPAlgorithm,
    VWAPAlgorithm,
    POVAlgorithm,
    ExecutionSide,
    ExecutionStatus,
    ExecutionSlice,
    ExecutionPlan,
    create_execution_plan,
)


class TestTWAPAlgorithm:
    """Tests for TWAP algorithm."""

    def test_create_plan(self):
        """Test TWAP plan creation."""
        algo = TWAPAlgorithm()
        plan = algo.create_plan(
            symbol="BTC",
            side=ExecutionSide.BUY,
            total_quantity=100,
            duration_minutes=60,
            num_slices=10,
        )

        assert plan.algorithm == "twap"
        assert plan.symbol == "BTC"
        assert plan.side == ExecutionSide.BUY
        assert plan.total_quantity == 100
        assert len(plan.slices) == 10

    def test_equal_slices(self):
        """Test that TWAP creates equal slices."""
        algo = TWAPAlgorithm()
        plan = algo.create_plan(
            symbol="BTC",
            side=ExecutionSide.BUY,
            total_quantity=100,
            num_slices=10,
        )

        # All slices should be approximately equal
        quantities = [s.target_quantity for s in plan.slices]
        assert all(abs(q - 10) < 0.01 for q in quantities)

    def test_time_intervals(self):
        """Test that slices are evenly spaced."""
        start = datetime(2024, 1, 1, 9, 0, 0)
        algo = TWAPAlgorithm()
        plan = algo.create_plan(
            symbol="BTC",
            side=ExecutionSide.BUY,
            total_quantity=100,
            duration_minutes=60,
            num_slices=6,
            start_time=start,
        )

        # Check intervals (60 min / 6 slices = 10 min)
        for i, s in enumerate(plan.slices):
            expected_time = start + timedelta(minutes=10 * i)
            assert s.scheduled_time == expected_time


class TestVWAPAlgorithm:
    """Tests for VWAP algorithm."""

    def test_create_plan(self):
        """Test VWAP plan creation."""
        algo = VWAPAlgorithm()
        plan = algo.create_plan(
            symbol="ETH",
            side=ExecutionSide.SELL,
            total_quantity=50,
            duration_minutes=30,
            num_slices=6,
        )

        assert plan.algorithm == "vwap"
        assert len(plan.slices) == 6
        assert sum(s.target_quantity for s in plan.slices) == pytest.approx(
            50, rel=0.01
        )

    def test_volume_profile_weighting(self):
        """Test that slices are weighted by volume profile."""
        # Custom profile with clear differences
        profile = [0.5, 0.3, 0.2]  # 50%, 30%, 20%

        algo = VWAPAlgorithm()
        plan = algo.create_plan(
            symbol="ETH",
            side=ExecutionSide.BUY,
            total_quantity=100,
            num_slices=3,
            volume_profile=profile,
        )

        # First slice should be largest
        quantities = [s.target_quantity for s in plan.slices]
        assert quantities[0] > quantities[1] > quantities[2]

    def test_total_matches(self):
        """Test that total quantity matches plan."""
        algo = VWAPAlgorithm()
        plan = algo.create_plan(
            symbol="BTC",
            side=ExecutionSide.BUY,
            total_quantity=100,
            num_slices=10,
        )

        total = sum(s.target_quantity for s in plan.slices)
        assert total == pytest.approx(100, rel=0.01)


class TestPOVAlgorithm:
    """Tests for POV algorithm."""

    def test_create_plan(self):
        """Test POV plan creation."""
        algo = POVAlgorithm(participation_rate=0.1)
        plan = algo.create_plan(
            symbol="BTC",
            side=ExecutionSide.BUY,
            total_quantity=100,
            duration_minutes=60,
        )

        assert plan.algorithm == "pov"
        assert plan.metadata["participation_rate"] == 0.1

    def test_calculate_next_slice(self):
        """Test dynamic slice calculation."""
        algo = POVAlgorithm(participation_rate=0.1)

        # 10% of 1000 volume = 100
        slice_size = algo.calculate_next_slice(
            remaining_quantity=500,
            market_volume=1000,
            elapsed_time=30,
            total_time=60,
        )

        assert slice_size > 0
        assert slice_size <= 500  # Can't exceed remaining

    def test_urgency_factor(self):
        """Test that urgency increases slice size over time."""
        algo = POVAlgorithm(participation_rate=0.1, urgency=1.0)

        early_slice = algo.calculate_next_slice(
            remaining_quantity=500,
            market_volume=1000,
            elapsed_time=10,
            total_time=60,
        )

        late_slice = algo.calculate_next_slice(
            remaining_quantity=500,
            market_volume=1000,
            elapsed_time=50,
            total_time=60,
        )

        # Late slice should be larger due to urgency
        assert late_slice > early_slice


class TestExecutionSlice:
    """Tests for ExecutionSlice."""

    def test_mark_executed(self):
        """Test marking slice as executed."""
        slice = ExecutionSlice(
            slice_id=0,
            scheduled_time=datetime.now(),
            target_quantity=100,
            target_percentage=10,
        )

        slice.mark_executed(50, 1000)
        assert slice.executed_quantity == 50
        assert slice.executed_price == 1000
        assert slice.status == ExecutionStatus.IN_PROGRESS

        slice.mark_executed(50, 1100)
        assert slice.executed_quantity == 100
        assert slice.executed_price == 1050  # Weighted average
        assert slice.status == ExecutionStatus.COMPLETED

    def test_remaining_quantity(self):
        """Test remaining quantity calculation."""
        slice = ExecutionSlice(
            slice_id=0,
            scheduled_time=datetime.now(),
            target_quantity=100,
            target_percentage=10,
        )

        assert slice.remaining_quantity == 100

        slice.mark_executed(60, 1000)
        assert slice.remaining_quantity == 40


class TestExecutionPlan:
    """Tests for ExecutionPlan."""

    def test_completion_tracking(self):
        """Test plan completion tracking."""
        slices = [
            ExecutionSlice(
                slice_id=i,
                scheduled_time=datetime.now() + timedelta(minutes=i),
                target_quantity=10,
                target_percentage=10,
            )
            for i in range(10)
        ]

        plan = ExecutionPlan(
            plan_id="test",
            symbol="BTC",
            side=ExecutionSide.BUY,
            total_quantity=100,
            slices=slices,
        )

        assert plan.completion_percentage == 0

        # Execute half
        for i in range(5):
            slices[i].mark_executed(10, 1000)

        assert plan.executed_quantity == 50
        assert plan.completion_percentage == 50

    def test_average_price(self):
        """Test volume weighted average price."""
        slices = [
            ExecutionSlice(
                slice_id=0,
                scheduled_time=datetime.now(),
                target_quantity=50,
                target_percentage=50,
            ),
            ExecutionSlice(
                slice_id=1,
                scheduled_time=datetime.now(),
                target_quantity=50,
                target_percentage=50,
            ),
        ]

        plan = ExecutionPlan(
            plan_id="test",
            symbol="BTC",
            side=ExecutionSide.BUY,
            total_quantity=100,
            slices=slices,
        )

        slices[0].mark_executed(50, 1000)
        slices[1].mark_executed(50, 1100)

        # VWAP = (50*1000 + 50*1100) / 100 = 1050
        assert plan.average_price == 1050


class TestFactoryFunction:
    """Tests for create_execution_plan factory."""

    def test_create_twap(self):
        """Test creating TWAP plan via factory."""
        plan = create_execution_plan(
            algorithm="twap",
            symbol="BTC",
            side="buy",
            total_quantity=100,
        )

        assert plan.algorithm == "twap"

    def test_create_vwap(self):
        """Test creating VWAP plan via factory."""
        plan = create_execution_plan(
            algorithm="vwap",
            symbol="ETH",
            side="sell",
            total_quantity=50,
        )

        assert plan.algorithm == "vwap"

    def test_create_pov(self):
        """Test creating POV plan via factory."""
        plan = create_execution_plan(
            algorithm="pov",
            symbol="BTC",
            side="buy",
            total_quantity=100,
            participation_rate=0.15,
        )

        assert plan.algorithm == "pov"

    def test_invalid_algorithm(self):
        """Test error on invalid algorithm."""
        with pytest.raises(ValueError):
            create_execution_plan(
                algorithm="invalid",
                symbol="BTC",
                side="buy",
                total_quantity=100,
            )
