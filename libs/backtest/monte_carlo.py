"""
Monte Carlo Simulation Module for MASP

Provides statistical analysis of trading strategy performance:
- Return distribution simulation
- Confidence intervals
- Maximum drawdown estimation
- Probability of ruin calculation
- Risk of loss assessment
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    n_simulations: int = 10000
    n_periods: int = 252  # Trading days in a year
    confidence_levels: list[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])
    seed: Optional[int] = None


@dataclass
class DrawdownDistribution:
    """Distribution of maximum drawdowns."""

    mean: float
    median: float
    std: float
    percentile_90: float
    percentile_95: float
    percentile_99: float
    max_observed: float
    probability_exceed_10pct: float
    probability_exceed_20pct: float
    probability_exceed_30pct: float


@dataclass
class ReturnDistribution:
    """Distribution of simulated returns."""

    mean: float
    median: float
    std: float
    skewness: float
    kurtosis: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    min_return: float
    max_return: float


@dataclass
class RuinProbability:
    """Probability of ruin (capital loss) analysis."""

    prob_lose_10pct: float
    prob_lose_25pct: float
    prob_lose_50pct: float
    prob_lose_75pct: float
    prob_total_ruin: float  # >90% loss
    expected_min_capital: float
    worst_case_capital: float  # 1st percentile


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric."""

    level: float
    lower: float
    upper: float
    point_estimate: float


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo simulation results."""

    config: MonteCarloConfig
    return_distribution: ReturnDistribution
    drawdown_distribution: DrawdownDistribution
    ruin_probability: RuinProbability
    final_capital_distribution: np.ndarray
    equity_curves: np.ndarray  # Shape: (n_simulations, n_periods)
    confidence_intervals: dict[str, list[ConfidenceInterval]]
    sharpe_distribution: list[float]
    summary: dict


class MonteCarloSimulator:
    """
    Monte Carlo Simulator for Trading Strategies.

    Generates statistical estimates of strategy performance by
    resampling historical returns or generating synthetic returns.
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        """
        Initialize Monte Carlo Simulator.

        Args:
            config: Simulation configuration (optional)
        """
        self.config = config or MonteCarloConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)
        logger.info(
            f"[MonteCarloSimulator] Initialized: "
            f"{self.config.n_simulations} simulations, "
            f"{self.config.n_periods} periods"
        )

    def simulate_from_returns(
        self,
        historical_returns: np.ndarray,
        initial_capital: float = 1000000,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation using bootstrap resampling of historical returns.

        This method randomly resamples (with replacement) from historical returns
        to generate multiple possible future equity curves.

        Args:
            historical_returns: Array of historical period returns (e.g., daily)
            initial_capital: Starting capital

        Returns:
            MonteCarloResult with simulation statistics
        """
        if len(historical_returns) < 30:
            logger.warning(
                "[MonteCarloSimulator] Small sample size may yield unreliable results"
            )

        n_sims = self.config.n_simulations
        n_periods = self.config.n_periods

        # Generate simulated return paths via bootstrap
        simulated_returns = np.random.choice(
            historical_returns,
            size=(n_sims, n_periods),
            replace=True,
        )

        # Calculate equity curves
        equity_curves = self._calculate_equity_curves(
            simulated_returns, initial_capital
        )

        # Calculate final capitals
        final_capitals = equity_curves[:, -1]

        # Calculate max drawdowns for each simulation
        max_drawdowns = self._calculate_max_drawdowns(equity_curves)

        # Calculate period returns for each simulation
        period_returns = (final_capitals / initial_capital) - 1

        # Build results
        return self._build_results(
            equity_curves=equity_curves,
            final_capitals=final_capitals,
            max_drawdowns=max_drawdowns,
            period_returns=period_returns,
            initial_capital=initial_capital,
            simulated_returns=simulated_returns,
        )

    def simulate_parametric(
        self,
        mean_return: float,
        std_return: float,
        initial_capital: float = 1000000,
        distribution: str = "normal",
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation using parametric distribution.

        Generates synthetic returns from a specified distribution
        with given mean and standard deviation.

        Args:
            mean_return: Expected period return (e.g., 0.001 for 0.1%)
            std_return: Standard deviation of returns
            initial_capital: Starting capital
            distribution: "normal" or "t" (student's t with fat tails)

        Returns:
            MonteCarloResult with simulation statistics
        """
        n_sims = self.config.n_simulations
        n_periods = self.config.n_periods

        # Generate simulated returns
        if distribution == "normal":
            simulated_returns = np.random.normal(
                mean_return, std_return, size=(n_sims, n_periods)
            )
        elif distribution == "t":
            # Student's t with 5 degrees of freedom for fat tails
            simulated_returns = stats.t.rvs(
                df=5, loc=mean_return, scale=std_return, size=(n_sims, n_periods)
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Calculate equity curves
        equity_curves = self._calculate_equity_curves(
            simulated_returns, initial_capital
        )

        # Calculate final capitals
        final_capitals = equity_curves[:, -1]

        # Calculate max drawdowns
        max_drawdowns = self._calculate_max_drawdowns(equity_curves)

        # Calculate period returns
        period_returns = (final_capitals / initial_capital) - 1

        return self._build_results(
            equity_curves=equity_curves,
            final_capitals=final_capitals,
            max_drawdowns=max_drawdowns,
            period_returns=period_returns,
            initial_capital=initial_capital,
            simulated_returns=simulated_returns,
        )

    def simulate_from_trades(
        self,
        trade_returns: np.ndarray,
        trades_per_period: float = 1.0,
        initial_capital: float = 1000000,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation resampling individual trade returns.

        Useful when strategy has discrete trades rather than continuous exposure.

        Args:
            trade_returns: Array of individual trade returns (percentage)
            trades_per_period: Average number of trades per period
            initial_capital: Starting capital

        Returns:
            MonteCarloResult with simulation statistics
        """
        n_sims = self.config.n_simulations
        n_periods = self.config.n_periods

        # Generate number of trades per period (Poisson distribution)
        n_trades_per_period = np.random.poisson(
            trades_per_period, size=(n_sims, n_periods)
        )

        # Generate simulated period returns
        simulated_returns = np.zeros((n_sims, n_periods))
        for i in range(n_sims):
            for j in range(n_periods):
                n_trades = n_trades_per_period[i, j]
                if n_trades > 0:
                    trades = np.random.choice(
                        trade_returns, size=n_trades, replace=True
                    )
                    # Compound trades within period
                    period_return = np.prod(1 + trades) - 1
                    simulated_returns[i, j] = period_return

        # Calculate equity curves
        equity_curves = self._calculate_equity_curves(
            simulated_returns, initial_capital
        )

        final_capitals = equity_curves[:, -1]
        max_drawdowns = self._calculate_max_drawdowns(equity_curves)
        period_returns = (final_capitals / initial_capital) - 1

        return self._build_results(
            equity_curves=equity_curves,
            final_capitals=final_capitals,
            max_drawdowns=max_drawdowns,
            period_returns=period_returns,
            initial_capital=initial_capital,
            simulated_returns=simulated_returns,
        )

    def _calculate_equity_curves(
        self,
        returns: np.ndarray,
        initial_capital: float,
    ) -> np.ndarray:
        """Calculate equity curves from return matrix."""
        # returns shape: (n_simulations, n_periods)
        cumulative_returns = np.cumprod(1 + returns, axis=1)
        equity_curves = initial_capital * cumulative_returns
        return equity_curves

    def _calculate_max_drawdowns(self, equity_curves: np.ndarray) -> np.ndarray:
        """Calculate maximum drawdown for each simulation."""
        n_sims = equity_curves.shape[0]
        max_drawdowns = np.zeros(n_sims)

        for i in range(n_sims):
            curve = equity_curves[i]
            peak = np.maximum.accumulate(curve)
            drawdown = (peak - curve) / peak
            max_drawdowns[i] = np.max(drawdown)

        return max_drawdowns

    def _build_results(
        self,
        equity_curves: np.ndarray,
        final_capitals: np.ndarray,
        max_drawdowns: np.ndarray,
        period_returns: np.ndarray,
        initial_capital: float,
        simulated_returns: np.ndarray,
    ) -> MonteCarloResult:
        """Build complete Monte Carlo results."""
        # Return distribution
        return_distribution = ReturnDistribution(
            mean=float(np.mean(period_returns)),
            median=float(np.median(period_returns)),
            std=float(np.std(period_returns)),
            skewness=float(stats.skew(period_returns)),
            kurtosis=float(stats.kurtosis(period_returns)),
            percentile_5=float(np.percentile(period_returns, 5)),
            percentile_25=float(np.percentile(period_returns, 25)),
            percentile_75=float(np.percentile(period_returns, 75)),
            percentile_95=float(np.percentile(period_returns, 95)),
            min_return=float(np.min(period_returns)),
            max_return=float(np.max(period_returns)),
        )

        # Drawdown distribution
        drawdown_distribution = DrawdownDistribution(
            mean=float(np.mean(max_drawdowns)),
            median=float(np.median(max_drawdowns)),
            std=float(np.std(max_drawdowns)),
            percentile_90=float(np.percentile(max_drawdowns, 90)),
            percentile_95=float(np.percentile(max_drawdowns, 95)),
            percentile_99=float(np.percentile(max_drawdowns, 99)),
            max_observed=float(np.max(max_drawdowns)),
            probability_exceed_10pct=float(np.mean(max_drawdowns > 0.10)),
            probability_exceed_20pct=float(np.mean(max_drawdowns > 0.20)),
            probability_exceed_30pct=float(np.mean(max_drawdowns > 0.30)),
        )

        # Ruin probability
        capital_ratios = final_capitals / initial_capital
        ruin_probability = RuinProbability(
            prob_lose_10pct=float(np.mean(capital_ratios < 0.90)),
            prob_lose_25pct=float(np.mean(capital_ratios < 0.75)),
            prob_lose_50pct=float(np.mean(capital_ratios < 0.50)),
            prob_lose_75pct=float(np.mean(capital_ratios < 0.25)),
            prob_total_ruin=float(np.mean(capital_ratios < 0.10)),
            expected_min_capital=float(np.mean(np.min(equity_curves, axis=1))),
            worst_case_capital=float(np.percentile(final_capitals, 1)),
        )

        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            period_returns, final_capitals, max_drawdowns
        )

        # Sharpe distribution
        sharpe_distribution = self._calculate_sharpe_distribution(simulated_returns)

        # Summary
        summary = {
            "n_simulations": self.config.n_simulations,
            "n_periods": self.config.n_periods,
            "initial_capital": initial_capital,
            "expected_final_capital": float(np.mean(final_capitals)),
            "median_final_capital": float(np.median(final_capitals)),
            "expected_return": return_distribution.mean,
            "median_return": return_distribution.median,
            "return_std": return_distribution.std,
            "expected_max_drawdown": drawdown_distribution.mean,
            "probability_profit": float(np.mean(period_returns > 0)),
            "probability_loss_10pct": ruin_probability.prob_lose_10pct,
            "worst_case_drawdown_99pct": drawdown_distribution.percentile_99,
        }

        return MonteCarloResult(
            config=self.config,
            return_distribution=return_distribution,
            drawdown_distribution=drawdown_distribution,
            ruin_probability=ruin_probability,
            final_capital_distribution=final_capitals,
            equity_curves=equity_curves,
            confidence_intervals=confidence_intervals,
            sharpe_distribution=sharpe_distribution,
            summary=summary,
        )

    def _calculate_confidence_intervals(
        self,
        period_returns: np.ndarray,
        final_capitals: np.ndarray,
        max_drawdowns: np.ndarray,
    ) -> dict[str, list[ConfidenceInterval]]:
        """Calculate confidence intervals for key metrics."""
        intervals = {
            "final_return": [],
            "final_capital": [],
            "max_drawdown": [],
        }

        for level in self.config.confidence_levels:
            alpha = 1 - level
            lower_pct = alpha / 2 * 100
            upper_pct = (1 - alpha / 2) * 100

            # Return CI
            intervals["final_return"].append(
                ConfidenceInterval(
                    level=level,
                    lower=float(np.percentile(period_returns, lower_pct)),
                    upper=float(np.percentile(period_returns, upper_pct)),
                    point_estimate=float(np.mean(period_returns)),
                )
            )

            # Capital CI
            intervals["final_capital"].append(
                ConfidenceInterval(
                    level=level,
                    lower=float(np.percentile(final_capitals, lower_pct)),
                    upper=float(np.percentile(final_capitals, upper_pct)),
                    point_estimate=float(np.mean(final_capitals)),
                )
            )

            # Drawdown CI (note: higher is worse)
            intervals["max_drawdown"].append(
                ConfidenceInterval(
                    level=level,
                    lower=float(np.percentile(max_drawdowns, lower_pct)),
                    upper=float(np.percentile(max_drawdowns, upper_pct)),
                    point_estimate=float(np.mean(max_drawdowns)),
                )
            )

        return intervals

    def _calculate_sharpe_distribution(
        self,
        simulated_returns: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> list[float]:
        """Calculate distribution of Sharpe ratios across simulations."""
        # Calculate annualized Sharpe for each simulation
        sharpe_ratios = []
        annualization_factor = np.sqrt(252)  # Assuming daily returns

        for i in range(simulated_returns.shape[0]):
            returns = simulated_returns[i]
            mean_return = np.mean(returns) - risk_free_rate / 252
            std_return = np.std(returns)
            if std_return > 0:
                sharpe = (mean_return / std_return) * annualization_factor
            else:
                sharpe = 0.0
            sharpe_ratios.append(float(sharpe))

        return sharpe_ratios


def format_monte_carlo_report(result: MonteCarloResult) -> str:
    """
    Format Monte Carlo results as human-readable report.

    Args:
        result: MonteCarloResult to format

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "MONTE CARLO SIMULATION REPORT",
        "=" * 60,
        f"Simulations: {result.config.n_simulations:,}",
        f"Periods: {result.config.n_periods}",
        "",
        "--- RETURN DISTRIBUTION ---",
        f"  Expected Return:   {result.return_distribution.mean:.2%}",
        f"  Median Return:     {result.return_distribution.median:.2%}",
        f"  Std Deviation:     {result.return_distribution.std:.2%}",
        f"  Skewness:          {result.return_distribution.skewness:.2f}",
        f"  5th Percentile:    {result.return_distribution.percentile_5:.2%}",
        f"  95th Percentile:   {result.return_distribution.percentile_95:.2%}",
        "",
        "--- DRAWDOWN RISK ---",
        f"  Expected Max DD:   {result.drawdown_distribution.mean:.1%}",
        f"  Median Max DD:     {result.drawdown_distribution.median:.1%}",
        f"  95th Pctl Max DD:  {result.drawdown_distribution.percentile_95:.1%}",
        f"  99th Pctl Max DD:  {result.drawdown_distribution.percentile_99:.1%}",
        f"  P(DD > 10%):       {result.drawdown_distribution.probability_exceed_10pct:.1%}",
        f"  P(DD > 20%):       {result.drawdown_distribution.probability_exceed_20pct:.1%}",
        f"  P(DD > 30%):       {result.drawdown_distribution.probability_exceed_30pct:.1%}",
        "",
        "--- RUIN PROBABILITY ---",
        f"  P(Lose 10%):       {result.ruin_probability.prob_lose_10pct:.1%}",
        f"  P(Lose 25%):       {result.ruin_probability.prob_lose_25pct:.1%}",
        f"  P(Lose 50%):       {result.ruin_probability.prob_lose_50pct:.1%}",
        f"  P(Total Ruin):     {result.ruin_probability.prob_total_ruin:.1%}",
        f"  Worst Case (1%):   {result.ruin_probability.worst_case_capital:,.0f}",
        "",
        "--- CONFIDENCE INTERVALS ---",
    ]

    for ci in result.confidence_intervals["final_return"]:
        lines.append(f"  {ci.level:.0%} CI Return: [{ci.lower:.2%}, {ci.upper:.2%}]")

    lines.extend(
        [
            "",
            "--- SHARPE RATIO DISTRIBUTION ---",
            f"  Mean Sharpe:       {np.mean(result.sharpe_distribution):.2f}",
            f"  Median Sharpe:     {np.median(result.sharpe_distribution):.2f}",
            f"  5th Pctl Sharpe:   {np.percentile(result.sharpe_distribution, 5):.2f}",
            f"  95th Pctl Sharpe:  {np.percentile(result.sharpe_distribution, 95):.2f}",
            "",
            "--- SUMMARY ---",
            f"  Probability of Profit:      {result.summary['probability_profit']:.1%}",
            f"  Expected Final Capital:     {result.summary['expected_final_capital']:,.0f}",
            f"  Median Final Capital:       {result.summary['median_final_capital']:,.0f}",
            "=" * 60,
        ]
    )

    return "\n".join(lines)
