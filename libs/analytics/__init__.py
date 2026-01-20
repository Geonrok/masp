"""
Analytics Module

Provides comprehensive performance analytics and metrics for trading strategies.
"""

from libs.analytics.performance import (
    PerformanceMetrics,
    calculate_sharpe,
    calculate_sortino,
    calculate_max_drawdown,
    calculate_calmar,
    calculate_performance_metrics,
)
from libs.analytics.advanced_metrics import (
    RiskMetrics,
    DrawdownMetrics,
    AdvancedPerformanceMetrics,
    calculate_omega_ratio,
    calculate_var,
    calculate_cvar,
    calculate_information_ratio,
    calculate_treynor_ratio,
    calculate_tail_ratio,
    calculate_gain_to_pain,
    calculate_skewness,
    calculate_kurtosis,
    analyze_drawdowns,
    calculate_expectancy,
    calculate_payoff_ratio,
    calculate_advanced_metrics,
    generate_performance_report,
)

__all__ = [
    # Basic performance
    "PerformanceMetrics",
    "calculate_sharpe",
    "calculate_sortino",
    "calculate_max_drawdown",
    "calculate_calmar",
    "calculate_performance_metrics",
    # Advanced metrics
    "RiskMetrics",
    "DrawdownMetrics",
    "AdvancedPerformanceMetrics",
    "calculate_omega_ratio",
    "calculate_var",
    "calculate_cvar",
    "calculate_information_ratio",
    "calculate_treynor_ratio",
    "calculate_tail_ratio",
    "calculate_gain_to_pain",
    "calculate_skewness",
    "calculate_kurtosis",
    "analyze_drawdowns",
    "calculate_expectancy",
    "calculate_payoff_ratio",
    "calculate_advanced_metrics",
    "generate_performance_report",
]
