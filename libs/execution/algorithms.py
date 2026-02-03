"""
Execution Algorithms

Advanced order execution algorithms designed to minimize market impact
and achieve optimal execution prices.

Algorithms:
- VWAP: Volume Weighted Average Price
- TWAP: Time Weighted Average Price
- POV: Percentage of Volume
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class ExecutionSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class ExecutionStatus(Enum):
    """Execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ExecutionSlice:
    """
    A single slice of an execution plan.

    Represents a portion of the total order to be executed
    at a specific time.
    """

    slice_id: int
    scheduled_time: datetime
    target_quantity: float
    target_percentage: float  # Percentage of total order
    executed_quantity: float = 0.0
    executed_price: float = 0.0
    status: ExecutionStatus = ExecutionStatus.PENDING
    actual_time: Optional[datetime] = None

    @property
    def remaining_quantity(self) -> float:
        """Remaining quantity to execute."""
        return max(0, self.target_quantity - self.executed_quantity)

    @property
    def is_complete(self) -> bool:
        """Check if slice is fully executed."""
        return self.status == ExecutionStatus.COMPLETED

    def mark_executed(
        self,
        quantity: float,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Mark quantity as executed."""
        self.executed_quantity += quantity
        if self.executed_quantity > 0:
            # Weighted average price
            total_value = (
                self.executed_price * (self.executed_quantity - quantity)
            ) + (price * quantity)
            self.executed_price = total_value / self.executed_quantity

        self.actual_time = timestamp or datetime.now()

        if self.executed_quantity >= self.target_quantity:
            self.status = ExecutionStatus.COMPLETED
        else:
            self.status = ExecutionStatus.IN_PROGRESS


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for an order.

    Contains all slices and tracks overall progress.
    """

    plan_id: str
    symbol: str
    side: ExecutionSide
    total_quantity: float
    slices: List[ExecutionSlice] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    algorithm: str = "unknown"
    status: ExecutionStatus = ExecutionStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def executed_quantity(self) -> float:
        """Total executed quantity across all slices."""
        return sum(s.executed_quantity for s in self.slices)

    @property
    def remaining_quantity(self) -> float:
        """Remaining quantity to execute."""
        return max(0, self.total_quantity - self.executed_quantity)

    @property
    def completion_percentage(self) -> float:
        """Percentage of order completed."""
        if self.total_quantity == 0:
            return 100.0
        return (self.executed_quantity / self.total_quantity) * 100

    @property
    def average_price(self) -> float:
        """Volume weighted average execution price."""
        total_value = sum(s.executed_price * s.executed_quantity for s in self.slices)
        total_qty = self.executed_quantity
        return total_value / total_qty if total_qty > 0 else 0.0

    @property
    def active_slices(self) -> List[ExecutionSlice]:
        """Get slices that are ready to execute."""
        now = datetime.now()
        return [
            s
            for s in self.slices
            if s.status in [ExecutionStatus.PENDING, ExecutionStatus.IN_PROGRESS]
            and s.scheduled_time <= now
        ]

    def get_next_slice(self) -> Optional[ExecutionSlice]:
        """Get the next slice to execute."""
        now = datetime.now()
        for s in self.slices:
            if s.status == ExecutionStatus.PENDING and s.scheduled_time <= now:
                return s
        return None

    def update_status(self) -> None:
        """Update overall plan status based on slices."""
        if all(s.status == ExecutionStatus.COMPLETED for s in self.slices):
            self.status = ExecutionStatus.COMPLETED
            self.end_time = datetime.now()
        elif any(s.status == ExecutionStatus.FAILED for s in self.slices):
            self.status = ExecutionStatus.FAILED
        elif any(s.status == ExecutionStatus.IN_PROGRESS for s in self.slices):
            self.status = ExecutionStatus.IN_PROGRESS
        elif any(s.status == ExecutionStatus.CANCELLED for s in self.slices):
            self.status = ExecutionStatus.CANCELLED


class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms."""

    algorithm_name: str = "base"

    def __init__(
        self,
        participation_rate: float = 0.1,
        min_slice_size: float = 0.0,
        max_slice_size: Optional[float] = None,
        urgency: float = 0.5,  # 0 = patient, 1 = aggressive
    ):
        """
        Initialize execution algorithm.

        Args:
            participation_rate: Target participation rate (for POV)
            min_slice_size: Minimum slice size
            max_slice_size: Maximum slice size (None = no limit)
            urgency: Urgency level affecting execution speed
        """
        self.participation_rate = participation_rate
        self.min_slice_size = min_slice_size
        self.max_slice_size = max_slice_size
        self.urgency = max(0, min(1, urgency))

    @abstractmethod
    def create_plan(
        self,
        symbol: str,
        side: ExecutionSide,
        total_quantity: float,
        duration_minutes: int = 60,
        num_slices: int = 10,
        start_time: Optional[datetime] = None,
        volume_profile: Optional[List[float]] = None,
    ) -> ExecutionPlan:
        """
        Create an execution plan.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            total_quantity: Total quantity to execute
            duration_minutes: Total execution duration
            num_slices: Number of execution slices
            start_time: Plan start time (default: now)
            volume_profile: Historical volume profile (for VWAP)

        Returns:
            ExecutionPlan
        """
        pass

    def _generate_plan_id(self) -> str:
        """Generate unique plan ID."""
        import uuid

        return f"{self.algorithm_name.upper()}-{uuid.uuid4().hex[:8]}"

    def _apply_size_limits(self, size: float) -> float:
        """Apply min/max size limits to a slice."""
        size = max(size, self.min_slice_size)
        if self.max_slice_size:
            size = min(size, self.max_slice_size)
        return size


class TWAPAlgorithm(ExecutionAlgorithm):
    """
    Time Weighted Average Price Algorithm.

    Splits order into equal-sized slices executed at regular intervals.
    Simple and predictable, good for liquid markets with stable volume.
    """

    algorithm_name = "twap"

    def create_plan(
        self,
        symbol: str,
        side: ExecutionSide,
        total_quantity: float,
        duration_minutes: int = 60,
        num_slices: int = 10,
        start_time: Optional[datetime] = None,
        volume_profile: Optional[List[float]] = None,
    ) -> ExecutionPlan:
        """
        Create TWAP execution plan.

        Divides order into equal slices at equal time intervals.
        """
        start = start_time or datetime.now()
        interval = timedelta(minutes=duration_minutes / num_slices)

        # Equal quantity per slice
        base_quantity = total_quantity / num_slices

        slices = []
        for i in range(num_slices):
            slice_qty = self._apply_size_limits(base_quantity)

            slices.append(
                ExecutionSlice(
                    slice_id=i,
                    scheduled_time=start + (interval * i),
                    target_quantity=slice_qty,
                    target_percentage=100 / num_slices,
                )
            )

        # Adjust last slice to ensure total matches
        total_planned = sum(s.target_quantity for s in slices)
        if slices and total_planned != total_quantity:
            slices[-1].target_quantity += total_quantity - total_planned

        plan = ExecutionPlan(
            plan_id=self._generate_plan_id(),
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            slices=slices,
            start_time=start,
            algorithm=self.algorithm_name,
            metadata={
                "duration_minutes": duration_minutes,
                "num_slices": num_slices,
                "interval_seconds": interval.total_seconds(),
            },
        )

        logger.info(
            f"[TWAP] Created plan {plan.plan_id}: "
            f"{total_quantity} {symbol} over {duration_minutes}min in {num_slices} slices"
        )

        return plan


class VWAPAlgorithm(ExecutionAlgorithm):
    """
    Volume Weighted Average Price Algorithm.

    Sizes slices based on historical volume profile to minimize market impact.
    Executes more during high volume periods, less during low volume.
    """

    algorithm_name = "vwap"

    def __init__(
        self,
        participation_rate: float = 0.1,
        min_slice_size: float = 0.0,
        max_slice_size: Optional[float] = None,
        urgency: float = 0.5,
        default_volume_profile: Optional[List[float]] = None,
    ):
        """
        Initialize VWAP algorithm.

        Args:
            default_volume_profile: Default intraday volume profile (normalized)
        """
        super().__init__(participation_rate, min_slice_size, max_slice_size, urgency)

        # Default Korean market volume profile (hourly from 9AM to 3:30PM)
        self.default_volume_profile = default_volume_profile or [
            0.15,  # 9:00-10:00 - High opening volume
            0.12,  # 10:00-11:00
            0.10,  # 11:00-12:00
            0.08,  # 12:00-13:00 - Lunch lull
            0.10,  # 13:00-14:00
            0.12,  # 14:00-15:00
            0.18,  # 15:00-15:30 - High closing volume
        ]

    def create_plan(
        self,
        symbol: str,
        side: ExecutionSide,
        total_quantity: float,
        duration_minutes: int = 60,
        num_slices: int = 10,
        start_time: Optional[datetime] = None,
        volume_profile: Optional[List[float]] = None,
    ) -> ExecutionPlan:
        """
        Create VWAP execution plan.

        Sizes slices proportional to expected volume.
        """
        start = start_time or datetime.now()
        interval = timedelta(minutes=duration_minutes / num_slices)

        # Use provided profile or default
        profile = volume_profile or self.default_volume_profile

        # Normalize profile to sum to 1
        profile_sum = sum(profile)
        if profile_sum > 0:
            normalized_profile = [v / profile_sum for v in profile]
        else:
            normalized_profile = [1 / len(profile)] * len(profile)

        # Map slices to profile
        slices = []
        for i in range(num_slices):
            # Map slice to profile bucket
            profile_idx = int(i / num_slices * len(normalized_profile))
            profile_idx = min(profile_idx, len(normalized_profile) - 1)

            volume_weight = normalized_profile[profile_idx]
            slice_qty = (
                total_quantity * volume_weight * (num_slices / len(normalized_profile))
            )
            slice_qty = self._apply_size_limits(slice_qty)

            slices.append(
                ExecutionSlice(
                    slice_id=i,
                    scheduled_time=start + (interval * i),
                    target_quantity=slice_qty,
                    target_percentage=volume_weight * 100,
                )
            )

        # Normalize to ensure total matches
        total_planned = sum(s.target_quantity for s in slices)
        if total_planned > 0:
            scale = total_quantity / total_planned
            for s in slices:
                s.target_quantity *= scale

        plan = ExecutionPlan(
            plan_id=self._generate_plan_id(),
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            slices=slices,
            start_time=start,
            algorithm=self.algorithm_name,
            metadata={
                "duration_minutes": duration_minutes,
                "num_slices": num_slices,
                "volume_profile": normalized_profile,
            },
        )

        logger.info(
            f"[VWAP] Created plan {plan.plan_id}: "
            f"{total_quantity} {symbol} over {duration_minutes}min"
        )

        return plan


class POVAlgorithm(ExecutionAlgorithm):
    """
    Percentage of Volume Algorithm.

    Executes at a fixed percentage of market volume.
    Adapts to real-time market conditions.
    """

    algorithm_name = "pov"

    def __init__(
        self,
        participation_rate: float = 0.1,  # 10% of volume
        min_slice_size: float = 0.0,
        max_slice_size: Optional[float] = None,
        urgency: float = 0.5,
        check_interval_seconds: int = 60,
    ):
        """
        Initialize POV algorithm.

        Args:
            participation_rate: Target percentage of market volume
            check_interval_seconds: Interval to check volume and adjust
        """
        super().__init__(participation_rate, min_slice_size, max_slice_size, urgency)
        self.check_interval_seconds = check_interval_seconds

    def create_plan(
        self,
        symbol: str,
        side: ExecutionSide,
        total_quantity: float,
        duration_minutes: int = 60,
        num_slices: int = 10,
        start_time: Optional[datetime] = None,
        volume_profile: Optional[List[float]] = None,
    ) -> ExecutionPlan:
        """
        Create POV execution plan.

        Creates initial plan with estimated slices.
        Actual execution should adapt based on real-time volume.
        """
        start = start_time or datetime.now()

        # Calculate number of check intervals
        num_intervals = max(1, duration_minutes * 60 // self.check_interval_seconds)
        interval = timedelta(seconds=self.check_interval_seconds)

        # Estimate slice size based on participation rate
        # This is a rough estimate; actual execution adapts to volume
        estimated_slice = total_quantity / num_intervals
        estimated_slice = self._apply_size_limits(estimated_slice)

        slices = []
        remaining = total_quantity
        for i in range(num_intervals):
            if remaining <= 0:
                break

            slice_qty = min(estimated_slice, remaining)

            slices.append(
                ExecutionSlice(
                    slice_id=i,
                    scheduled_time=start + (interval * i),
                    target_quantity=slice_qty,
                    target_percentage=self.participation_rate * 100,
                )
            )

            remaining -= slice_qty

        plan = ExecutionPlan(
            plan_id=self._generate_plan_id(),
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            slices=slices,
            start_time=start,
            algorithm=self.algorithm_name,
            metadata={
                "participation_rate": self.participation_rate,
                "check_interval_seconds": self.check_interval_seconds,
                "duration_minutes": duration_minutes,
            },
        )

        logger.info(
            f"[POV] Created plan {plan.plan_id}: "
            f"{total_quantity} {symbol} at {self.participation_rate*100}% participation"
        )

        return plan

    def calculate_next_slice(
        self,
        remaining_quantity: float,
        market_volume: float,
        elapsed_time: float,
        total_time: float,
    ) -> float:
        """
        Calculate next slice size based on real-time volume.

        Args:
            remaining_quantity: Quantity left to execute
            market_volume: Recent market volume
            elapsed_time: Time elapsed
            total_time: Total execution time

        Returns:
            Recommended slice size
        """
        # Target slice based on participation
        target_slice = market_volume * self.participation_rate

        # Adjust for remaining time (increase if behind schedule)
        time_ratio = elapsed_time / total_time if total_time > 0 else 1
        urgency_factor = 1 + (self.urgency * time_ratio)

        adjusted_slice = target_slice * urgency_factor

        # Don't exceed remaining quantity
        return min(self._apply_size_limits(adjusted_slice), remaining_quantity)


def create_execution_plan(
    algorithm: str,
    symbol: str,
    side: str,
    total_quantity: float,
    duration_minutes: int = 60,
    num_slices: int = 10,
    **kwargs,
) -> ExecutionPlan:
    """
    Factory function to create execution plans.

    Args:
        algorithm: Algorithm name (twap, vwap, pov)
        symbol: Trading symbol
        side: "buy" or "sell"
        total_quantity: Total quantity to execute
        duration_minutes: Execution duration
        num_slices: Number of slices
        **kwargs: Algorithm-specific parameters

    Returns:
        ExecutionPlan
    """
    exec_side = ExecutionSide(side.lower())

    algorithms = {
        "twap": TWAPAlgorithm,
        "vwap": VWAPAlgorithm,
        "pov": POVAlgorithm,
    }

    algo_class = algorithms.get(algorithm.lower())
    if not algo_class:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Available: {list(algorithms.keys())}"
        )

    algo = algo_class(**kwargs)
    return algo.create_plan(
        symbol=symbol,
        side=exec_side,
        total_quantity=total_quantity,
        duration_minutes=duration_minutes,
        num_slices=num_slices,
    )
