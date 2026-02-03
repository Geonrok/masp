"""
Performance metrics for backtesting.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd


def calculate_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.03,
    trading_days: int = 252,
) -> Dict:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Daily returns series
        benchmark_returns: Benchmark daily returns
        risk_free_rate: Annual risk-free rate
        trading_days: Trading days per year

    Returns:
        Dict of performance metrics
    """
    returns = returns.dropna()

    if len(returns) < 2:
        return _empty_metrics()

    # Cumulative returns
    cum_returns = (1 + returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1

    # Annualized metrics
    n_days = len(returns)
    n_years = n_days / trading_days

    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Risk metrics
    daily_std = returns.std()
    annual_std = daily_std * np.sqrt(trading_days)

    # Daily risk-free rate
    rf_daily = (1 + risk_free_rate) ** (1 / trading_days) - 1

    # Sharpe Ratio
    excess_returns = returns - rf_daily
    if daily_std > 0:
        sharpe = np.sqrt(trading_days) * excess_returns.mean() / daily_std
    else:
        sharpe = 0

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(trading_days)
        sortino = (cagr - risk_free_rate) / downside_std if downside_std > 0 else 0
    else:
        sortino = 0

    # Maximum Drawdown
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float("inf")

    # Average win/loss
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0

    # Expectancy
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # Best/Worst day
    best_day = returns.max()
    worst_day = returns.min()

    # Skewness and Kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    # Alpha and Beta (if benchmark provided)
    alpha = 0
    beta = 0
    information_ratio = 0
    tracking_error = 0

    if benchmark_returns is not None and len(benchmark_returns) > 10:
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) > 10:
            strategy_ret = aligned.iloc[:, 0]
            bench_ret = aligned.iloc[:, 1]

            # Beta
            cov = np.cov(strategy_ret, bench_ret)
            if cov[1, 1] > 0:
                beta = cov[0, 1] / cov[1, 1]
                bench_cagr = (1 + bench_ret).prod() ** (
                    trading_days / len(bench_ret)
                ) - 1
                alpha = cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate))

            # Tracking error and information ratio
            active_returns = strategy_ret - bench_ret
            tracking_error = active_returns.std() * np.sqrt(trading_days)
            if tracking_error > 0:
                information_ratio = (
                    active_returns.mean() * trading_days / tracking_error
                )

    metrics = {
        # Returns
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": annual_std,
        # Risk-adjusted
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        # Drawdown
        "max_drawdown": max_drawdown,
        # Trading
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_day": best_day,
        "worst_day": worst_day,
        # Distribution
        "skewness": skewness,
        "kurtosis": kurtosis,
        # Benchmark comparison
        "alpha": alpha,
        "beta": beta,
        "information_ratio": information_ratio,
        "tracking_error": tracking_error,
        # Meta
        "n_trading_days": n_days,
        "n_years": n_years,
    }

    return metrics


def _empty_metrics() -> Dict:
    """Return empty metrics dict."""
    return {
        "total_return": 0,
        "cagr": 0,
        "annual_volatility": 0,
        "sharpe_ratio": 0,
        "sortino_ratio": 0,
        "calmar_ratio": 0,
        "max_drawdown": 0,
        "win_rate": 0,
        "profit_factor": 0,
        "expectancy": 0,
        "avg_win": 0,
        "avg_loss": 0,
        "best_day": 0,
        "worst_day": 0,
        "skewness": 0,
        "kurtosis": 0,
        "alpha": 0,
        "beta": 0,
        "information_ratio": 0,
        "tracking_error": 0,
        "n_trading_days": 0,
        "n_years": 0,
    }


def format_metrics(metrics: Dict) -> str:
    """Format metrics for display."""
    lines = [
        "=" * 50,
        "PERFORMANCE SUMMARY",
        "=" * 50,
        "",
        "Returns:",
        f"  Total Return:     {metrics['total_return']:>10.2%}",
        f"  CAGR:             {metrics['cagr']:>10.2%}",
        f"  Annual Volatility:{metrics['annual_volatility']:>10.2%}",
        "",
        "Risk-Adjusted:",
        f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}",
        f"  Sortino Ratio:    {metrics['sortino_ratio']:>10.2f}",
        f"  Calmar Ratio:     {metrics['calmar_ratio']:>10.2f}",
        "",
        "Drawdown:",
        f"  Max Drawdown:     {metrics['max_drawdown']:>10.2%}",
        "",
        "Trading Statistics:",
        f"  Win Rate:         {metrics['win_rate']:>10.2%}",
        f"  Profit Factor:    {metrics['profit_factor']:>10.2f}",
        f"  Expectancy:       {metrics['expectancy']:>10.4f}",
        f"  Avg Win:          {metrics['avg_win']:>10.4f}",
        f"  Avg Loss:         {metrics['avg_loss']:>10.4f}",
        "",
        f"  Best Day:         {metrics['best_day']:>10.2%}",
        f"  Worst Day:        {metrics['worst_day']:>10.2%}",
        "",
        "Benchmark Comparison:",
        f"  Alpha:            {metrics['alpha']:>10.2%}",
        f"  Beta:             {metrics['beta']:>10.2f}",
        f"  Information Ratio:{metrics['information_ratio']:>10.2f}",
        "",
        f"Trading Days: {metrics['n_trading_days']}, Years: {metrics['n_years']:.2f}",
        "=" * 50,
    ]
    return "\n".join(lines)
