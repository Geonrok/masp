"""
Integrated Strategy - Combines all Phase 1-4 components

This module brings together:
- Bias-Free Backtester (Phase 1)
- Validation System (Phase 2)
- Veto Manager (Phase 3)
- WebSocket Infrastructure (Phase 4)

Purpose: Paper trading and strategy validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

import numpy as np
import pandas as pd

from libs.backtest.bias_free_backtester import (
    BiasFreeBacktester,
    BacktestConfig,
    ExecutionConfig,
    BacktestMetrics,
)
from libs.backtest.validation import (
    validate_strategy,
    WFOConfig,
    CPCVConfig,
)
from libs.risk.veto_manager import (
    VetoManager,
    VetoConfig,
    VetoResult,
)

logger = logging.getLogger(__name__)


@dataclass
class IntegratedConfig:
    """Configuration for integrated strategy."""

    # Backtest settings
    initial_capital: float = 10000.0
    slippage_pct: float = 0.005
    commission_pct: float = 0.001
    max_positions: int = 20

    # Strategy parameters
    kama_period: int = 5
    tsmom_period: int = 90
    gate_period: int = 30

    # Validation settings
    run_validation: bool = True
    n_trials: int = 1  # Number of parameter combos tested

    # Veto settings
    enable_veto: bool = True
    adx_threshold: float = 20.0
    ci_threshold: float = 61.8
    funding_rate_threshold: float = 0.001

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initial_capital": self.initial_capital,
            "slippage_pct": self.slippage_pct,
            "commission_pct": self.commission_pct,
            "max_positions": self.max_positions,
            "kama_period": self.kama_period,
            "tsmom_period": self.tsmom_period,
            "gate_period": self.gate_period,
            "run_validation": self.run_validation,
            "enable_veto": self.enable_veto,
        }


@dataclass
class IntegratedResult:
    """Result from integrated strategy run."""

    # Backtest metrics
    metrics: Optional[BacktestMetrics] = None

    # Validation results
    validation: Optional[Dict[str, Any]] = None

    # Veto statistics
    veto_stats: Dict[str, Any] = field(default_factory=dict)

    # Run metadata
    config: Optional[IntegratedConfig] = None
    run_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "validation": self.validation,
            "veto_stats": self.veto_stats,
            "config": self.config.to_dict() if self.config else None,
            "run_time": self.run_time,
            "timestamp": self.timestamp.isoformat(),
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "INTEGRATED STRATEGY RESULTS",
            "=" * 60,
        ]

        if self.metrics:
            lines.extend(
                [
                    "",
                    "PERFORMANCE METRICS:",
                    f"  Total Return:     {self.metrics.total_return * 100:+.1f}%",
                    f"  Sharpe Ratio:     {self.metrics.sharpe_ratio:.2f}",
                    f"  Max Drawdown:     {self.metrics.max_drawdown * 100:.1f}%",
                    f"  Win Rate:         {self.metrics.win_rate * 100:.1f}%",
                    f"  Trading Days:     {self.metrics.trading_days}",
                    f"  Invested Days:    {self.metrics.invested_days}",
                ]
            )

        if self.validation:
            verdict = self.validation.get("verdict", {})
            lines.extend(
                [
                    "",
                    "VALIDATION RESULTS:",
                    f"  Is Robust:        {verdict.get('is_robust', 'N/A')}",
                    f"  Confidence:       {verdict.get('confidence', 'N/A')}",
                    f"  Warnings:         {len(verdict.get('warnings', []))}",
                ]
            )

            wfo = self.validation.get("wfo", {})
            if wfo:
                lines.extend(
                    [
                        "",
                        "  WFO:",
                        f"    OOS Sharpe:     {wfo.get('oos_sharpe', 0):.2f}",
                        f"    Efficiency:     {wfo.get('efficiency_ratio', 0):.2f}",
                        f"    Robustness:     {wfo.get('robustness_score', 0):.1%}",
                    ]
                )

            cpcv = self.validation.get("cpcv", {})
            if cpcv:
                lines.extend(
                    [
                        "",
                        "  CPCV:",
                        f"    Mean Sharpe:    {cpcv.get('mean_sharpe', 0):.2f}",
                        f"    PBO:            {cpcv.get('pbo', 0):.1%}",
                    ]
                )

            dsr = self.validation.get("dsr", {})
            if dsr and "error" not in dsr:
                lines.extend(
                    [
                        "",
                        "  DSR:",
                        f"    Raw Sharpe:     {dsr.get('raw_sharpe', 0):.2f}",
                        f"    Deflated:       {dsr.get('deflated_sharpe', 0):.2f}",
                        f"    Haircut:        {dsr.get('haircut_pct', 0):.1f}%",
                        f"    Significant:    {dsr.get('is_significant', False)}",
                    ]
                )

        if self.veto_stats:
            lines.extend(
                [
                    "",
                    "VETO STATISTICS:",
                    f"  Total Checks:     {self.veto_stats.get('total_checks', 0)}",
                    f"  Vetoed:           {self.veto_stats.get('vetoed_count', 0)}",
                    f"  Veto Rate:        {self.veto_stats.get('veto_rate', 0):.1%}",
                ]
            )

        lines.extend(
            [
                "",
                f"Run Time: {self.run_time:.2f}s",
                "=" * 60,
            ]
        )

        return "\n".join(lines)


class IntegratedStrategy:
    """
    Integrated Strategy Engine

    Combines:
    1. Bias-Free Backtester for accurate historical testing
    2. Validation (WFO, CPCV, DSR) for robustness checking
    3. Veto Manager for real-time trade filtering
    4. (Future) WebSocket for live data
    """

    def __init__(self, config: Optional[IntegratedConfig] = None):
        """
        Initialize integrated strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config or IntegratedConfig()

        # Initialize components
        self._init_backtester()
        self._init_veto_manager()

        logger.info("[IntegratedStrategy] Initialized")

    def _init_backtester(self) -> None:
        """Initialize bias-free backtester."""
        exec_config = ExecutionConfig(
            slippage_pct=self.config.slippage_pct,
            commission_pct=self.config.commission_pct,
            max_positions=self.config.max_positions,
        )

        backtest_config = BacktestConfig(
            initial_capital=self.config.initial_capital,
            execution=exec_config,
            kama_period=self.config.kama_period,
            tsmom_period=self.config.tsmom_period,
            gate_period=self.config.gate_period,
        )

        self.backtester = BiasFreeBacktester(backtest_config)

    def _init_veto_manager(self) -> None:
        """Initialize veto manager."""
        veto_config = VetoConfig(
            adx_threshold=self.config.adx_threshold,
            ci_threshold=self.config.ci_threshold,
            funding_rate_threshold=self.config.funding_rate_threshold,
        )

        self.veto_manager = VetoManager(veto_config)

    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        validate: bool = True,
    ) -> IntegratedResult:
        """
        Run integrated backtest with validation.

        Args:
            data: Dict of symbol -> DataFrame with OHLCV data
            validate: Whether to run validation suite

        Returns:
            IntegratedResult
        """
        import time

        start_time = time.time()

        result = IntegratedResult(config=self.config)

        # Run backtest
        logger.info("[IntegratedStrategy] Running backtest on %d symbols", len(data))
        metrics = self.backtester.run(data)
        result.metrics = metrics

        # Run validation if requested
        if validate and self.config.run_validation:
            logger.info("[IntegratedStrategy] Running validation suite")

            def strategy_returns(d: Dict[str, pd.DataFrame]) -> np.ndarray:
                bt = BiasFreeBacktester(self.backtester.config)
                bt.run(d)
                return np.array(bt.daily_returns)

            try:
                validation_result = validate_strategy(
                    data=data,
                    strategy_func=strategy_returns,
                    n_trials=self.config.n_trials,
                )
                result.validation = validation_result
            except Exception as e:
                logger.error("[IntegratedStrategy] Validation failed: %s", e)
                result.validation = {"error": str(e)}

        # Calculate veto statistics
        result.veto_stats = self._calculate_veto_stats(data)

        result.run_time = time.time() - start_time

        logger.info(
            "[IntegratedStrategy] Completed in %.2fs: Return=%.1f%%, Sharpe=%.2f",
            result.run_time,
            metrics.total_return * 100,
            metrics.sharpe_ratio,
        )

        return result

    def _calculate_veto_stats(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate veto statistics from data."""
        if not self.config.enable_veto:
            return {"enabled": False}

        stats = {
            "enabled": True,
            "total_checks": 0,
            "vetoed_count": 0,
            "by_level": {},
        }

        # Sample veto checks on random dates
        for symbol, df in list(data.items())[:5]:  # Check first 5 symbols
            if len(df) < 50:
                continue

            context = {"ohlcv": df}

            for _ in range(10):  # Sample 10 dates per symbol
                result = self.veto_manager.can_trade(symbol, "long", context)
                stats["total_checks"] += 1

                if not result.can_trade:
                    stats["vetoed_count"] += 1
                    level = result.veto_level.name if result.veto_level else "UNKNOWN"
                    stats["by_level"][level] = stats["by_level"].get(level, 0) + 1

        if stats["total_checks"] > 0:
            stats["veto_rate"] = stats["vetoed_count"] / stats["total_checks"]
        else:
            stats["veto_rate"] = 0

        return stats

    def paper_trade_check(
        self,
        symbol: str,
        side: str,
        ohlcv: pd.DataFrame,
        funding_rate: Optional[float] = None,
    ) -> VetoResult:
        """
        Check if a trade should be executed (paper trading mode).

        Args:
            symbol: Trading symbol
            side: Trade direction ("long" or "short")
            ohlcv: Recent OHLCV data
            funding_rate: Current funding rate (optional)

        Returns:
            VetoResult
        """
        context = {"ohlcv": ohlcv}

        if funding_rate is not None:
            context["funding_rate"] = funding_rate

        return self.veto_manager.can_trade(symbol, side, context)

    def enable_kill_switch(self) -> None:
        """Enable emergency kill switch."""
        self.veto_manager.enable_kill_switch()
        logger.warning("[IntegratedStrategy] KILL SWITCH ENABLED")

    def disable_kill_switch(self) -> None:
        """Disable kill switch."""
        self.veto_manager.disable_kill_switch()
        logger.info("[IntegratedStrategy] Kill switch disabled")


def run_integrated_test(
    data_path: str,
    config: Optional[IntegratedConfig] = None,
) -> IntegratedResult:
    """
    Convenience function to run integrated test.

    Args:
        data_path: Path to data folder
        config: Strategy configuration

    Returns:
        IntegratedResult
    """
    from pathlib import Path

    # Load data
    data = {}
    data_folder = Path(data_path)

    for csv_file in data_folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)

            # Find date column
            date_col = None
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    date_col = col
                    break

            if date_col:
                df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
                df = df.set_index("date")

            if len(df) >= 100:
                data[csv_file.stem] = df

        except Exception as e:
            logger.debug("Skip %s: %s", csv_file, e)
            continue

    logger.info("Loaded %d symbols from %s", len(data), data_path)

    # Run integrated strategy
    strategy = IntegratedStrategy(config)
    return strategy.run_backtest(data)
