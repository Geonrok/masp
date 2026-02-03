"""
Portfolio Optimization

Multi-strategy portfolio optimization with various methods for
allocating capital across multiple trading strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""

    MEAN_VARIANCE = "mean_variance"  # Markowitz
    RISK_PARITY = "risk_parity"
    MAX_SHARPE = "max_sharpe"
    MIN_VARIANCE = "min_variance"
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOLATILITY = "inverse_volatility"
    MAX_DIVERSIFICATION = "max_diversification"
    HRP = "hrp"  # Hierarchical Risk Parity


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization."""

    min_weight: float = 0.0  # Minimum weight per strategy
    max_weight: float = 1.0  # Maximum weight per strategy
    target_return: Optional[float] = None  # Target return for mean-variance
    max_volatility: Optional[float] = None  # Maximum portfolio volatility
    strategy_groups: Optional[Dict[str, List[int]]] = None  # Groups for sector limits
    group_limits: Optional[Dict[str, Tuple[float, float]]] = (
        None  # (min, max) per group
    )
    allow_short: bool = False  # Allow negative weights
    sum_to_one: bool = True  # Weights must sum to 1


@dataclass
class StrategyReturns:
    """Historical returns for a strategy."""

    name: str
    returns: np.ndarray  # Array of periodic returns
    benchmark_returns: Optional[np.ndarray] = None

    @property
    def mean_return(self) -> float:
        """Annualized mean return."""
        return float(np.mean(self.returns) * 252)

    @property
    def volatility(self) -> float:
        """Annualized volatility."""
        return float(np.std(self.returns) * np.sqrt(252))

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio (assuming 0 risk-free rate)."""
        if self.volatility == 0:
            return 0.0
        return self.mean_return / self.volatility


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""

    weights: np.ndarray
    strategy_names: List[str]
    method: OptimizationMethod
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    effective_n: float  # Effective number of bets
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_allocation(self) -> Dict[str, float]:
        """Get weight allocation as dictionary."""
        return {
            name: float(weight)
            for name, weight in zip(self.strategy_names, self.weights)
        }

    def __str__(self) -> str:
        """String representation."""
        lines = [
            f"Portfolio Optimization Result ({self.method.value})",
            f"  Expected Return: {self.expected_return*100:.2f}%",
            f"  Expected Volatility: {self.expected_volatility*100:.2f}%",
            f"  Sharpe Ratio: {self.sharpe_ratio:.3f}",
            f"  Diversification Ratio: {self.diversification_ratio:.3f}",
            f"  Effective N: {self.effective_n:.2f}",
            "  Weights:",
        ]
        for name, weight in sorted(
            zip(self.strategy_names, self.weights),
            key=lambda x: x[1],
            reverse=True,
        ):
            if weight > 0.001:
                lines.append(f"    {name}: {weight*100:.2f}%")
        return "\n".join(lines)


class PortfolioOptimizer:
    """
    Multi-strategy portfolio optimizer.

    Supports various optimization methods for allocating capital
    across multiple trading strategies.
    """

    def __init__(
        self,
        strategies: List[StrategyReturns],
        risk_free_rate: float = 0.02,  # 2% annual
    ):
        """
        Initialize portfolio optimizer.

        Args:
            strategies: List of strategy return series
            risk_free_rate: Annual risk-free rate
        """
        self.strategies = strategies
        self.risk_free_rate = risk_free_rate
        self.n_strategies = len(strategies)

        # Calculate covariance matrix
        self._calculate_statistics()

        logger.info(
            f"[PortfolioOptimizer] Initialized with {self.n_strategies} strategies"
        )

    def _calculate_statistics(self) -> None:
        """Calculate return statistics and covariance matrix."""
        # Build returns matrix
        min_len = min(len(s.returns) for s in self.strategies)
        self.returns_matrix = np.column_stack(
            [s.returns[-min_len:] for s in self.strategies]
        )

        # Mean returns (annualized)
        self.mean_returns = np.mean(self.returns_matrix, axis=0) * 252

        # Covariance matrix (annualized)
        self.cov_matrix = np.cov(self.returns_matrix, rowvar=False) * 252

        # Handle single strategy case (cov returns scalar)
        if self.n_strategies == 1:
            self.cov_matrix = np.array([[self.cov_matrix]])
            self.mean_returns = np.array([self.mean_returns])

        # Correlation matrix
        std_devs = np.sqrt(np.diag(self.cov_matrix))
        # Avoid division by zero
        std_devs = np.where(std_devs == 0, 1e-10, std_devs)
        self.corr_matrix = self.cov_matrix / np.outer(std_devs, std_devs)

        # Ensure correlation matrix diagonal is exactly 1.0 (numerical stability)
        np.fill_diagonal(self.corr_matrix, 1.0)

        # Individual volatilities
        self.volatilities = std_devs

        logger.debug(
            f"[PortfolioOptimizer] Stats calculated: "
            f"mean_ret={self.mean_returns}, vol={self.volatilities}"
        )

    def optimize(
        self,
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio weights.

        Args:
            method: Optimization method
            constraints: Optimization constraints

        Returns:
            OptimizationResult
        """
        if constraints is None:
            constraints = OptimizationConstraints()

        logger.info(f"[PortfolioOptimizer] Running {method.value} optimization")

        if method == OptimizationMethod.MEAN_VARIANCE:
            weights = self._mean_variance(constraints)
        elif method == OptimizationMethod.RISK_PARITY:
            weights = self._risk_parity(constraints)
        elif method == OptimizationMethod.MAX_SHARPE:
            weights = self._max_sharpe(constraints)
        elif method == OptimizationMethod.MIN_VARIANCE:
            weights = self._min_variance(constraints)
        elif method == OptimizationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight()
        elif method == OptimizationMethod.INVERSE_VOLATILITY:
            weights = self._inverse_volatility(constraints)
        elif method == OptimizationMethod.MAX_DIVERSIFICATION:
            weights = self._max_diversification(constraints)
        elif method == OptimizationMethod.HRP:
            weights = self._hierarchical_risk_parity()
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Calculate portfolio metrics
        result = self._build_result(weights, method)

        logger.info(
            f"[PortfolioOptimizer] Optimization complete: "
            f"return={result.expected_return*100:.2f}%, "
            f"vol={result.expected_volatility*100:.2f}%, "
            f"sharpe={result.sharpe_ratio:.3f}"
        )

        return result

    def _build_result(
        self,
        weights: np.ndarray,
        method: OptimizationMethod,
    ) -> OptimizationResult:
        """Build optimization result with metrics."""
        # Ensure weights is 1D array
        weights = np.atleast_1d(weights).flatten()

        # Portfolio return and volatility
        port_return = np.dot(weights, self.mean_returns.flatten())
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        # Ensure scalars (in case of single strategy)
        port_return = float(np.asarray(port_return).item())
        port_vol = float(np.asarray(port_vol).item())

        # Sharpe ratio
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0.0

        # Diversification ratio
        weighted_vol = float(np.dot(weights, self.volatilities.flatten()))
        div_ratio = weighted_vol / port_vol if port_vol > 0 else 1.0

        # Effective N (Herfindahl index inverse)
        effective_n = 1.0 / float(np.sum(weights**2)) if np.any(weights > 0) else 0.0

        return OptimizationResult(
            weights=weights,
            strategy_names=[s.name for s in self.strategies],
            method=method,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=float(sharpe),
            diversification_ratio=float(div_ratio),
            effective_n=float(effective_n),
            metadata={
                "mean_returns": self.mean_returns.flatten().tolist(),
                "volatilities": self.volatilities.flatten().tolist(),
                "correlation_matrix": self.corr_matrix.tolist(),
            },
        )

    def _equal_weight(self) -> np.ndarray:
        """Equal weight allocation."""
        return np.ones(self.n_strategies) / self.n_strategies

    def _inverse_volatility(
        self,
        constraints: OptimizationConstraints,
    ) -> np.ndarray:
        """Inverse volatility weighting."""
        inv_vol = 1.0 / self.volatilities
        weights = inv_vol / inv_vol.sum()

        # Apply constraints
        weights = self._apply_weight_constraints(weights, constraints)

        return weights

    def _min_variance(
        self,
        constraints: OptimizationConstraints,
    ) -> np.ndarray:
        """Minimum variance portfolio."""

        def portfolio_variance(w: np.ndarray) -> float:
            return float(np.dot(w.T, np.dot(self.cov_matrix, w)))

        # Initial guess
        x0 = np.ones(self.n_strategies) / self.n_strategies

        # Bounds
        bounds = self._get_bounds(constraints)

        # Constraints
        cons = self._get_scipy_constraints(constraints)

        result = optimize.minimize(
            portfolio_variance,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        return result.x if result.success else self._equal_weight()

    def _max_sharpe(
        self,
        constraints: OptimizationConstraints,
    ) -> np.ndarray:
        """Maximum Sharpe ratio portfolio."""

        def neg_sharpe(w: np.ndarray) -> float:
            port_return = np.dot(w, self.mean_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
            if port_vol < 1e-10:
                return 1e10
            return -(port_return - self.risk_free_rate) / port_vol

        x0 = np.ones(self.n_strategies) / self.n_strategies
        bounds = self._get_bounds(constraints)
        cons = self._get_scipy_constraints(constraints)

        result = optimize.minimize(
            neg_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        return result.x if result.success else self._equal_weight()

    def _mean_variance(
        self,
        constraints: OptimizationConstraints,
    ) -> np.ndarray:
        """Mean-variance optimization (Markowitz)."""
        if constraints.target_return is not None:
            # Optimize for target return
            return self._mean_variance_target_return(constraints)
        else:
            # Optimize for max Sharpe
            return self._max_sharpe(constraints)

    def _mean_variance_target_return(
        self,
        constraints: OptimizationConstraints,
    ) -> np.ndarray:
        """Mean-variance with target return constraint."""

        def portfolio_variance(w: np.ndarray) -> float:
            return float(np.dot(w.T, np.dot(self.cov_matrix, w)))

        x0 = np.ones(self.n_strategies) / self.n_strategies
        bounds = self._get_bounds(constraints)

        # Add target return constraint
        cons = self._get_scipy_constraints(constraints)
        cons.append(
            {
                "type": "eq",
                "fun": lambda w: np.dot(w, self.mean_returns)
                - constraints.target_return,
            }
        )

        result = optimize.minimize(
            portfolio_variance,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        return result.x if result.success else self._equal_weight()

    def _risk_parity(
        self,
        constraints: OptimizationConstraints,
    ) -> np.ndarray:
        """
        Risk parity optimization.

        Equalizes risk contribution from each strategy.
        """

        def risk_budget_objective(w: np.ndarray) -> float:
            # Portfolio volatility
            port_vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))

            if port_vol < 1e-10:
                return 1e10

            # Marginal risk contribution
            mrc = np.dot(self.cov_matrix, w) / port_vol

            # Risk contribution
            rc = w * mrc

            # Target: equal risk contribution
            target_rc = port_vol / self.n_strategies

            # Minimize squared deviation from target
            return float(np.sum((rc - target_rc) ** 2))

        x0 = np.ones(self.n_strategies) / self.n_strategies
        bounds = self._get_bounds(constraints)
        cons = self._get_scipy_constraints(constraints)

        result = optimize.minimize(
            risk_budget_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        return result.x if result.success else self._inverse_volatility(constraints)

    def _max_diversification(
        self,
        constraints: OptimizationConstraints,
    ) -> np.ndarray:
        """
        Maximum diversification portfolio.

        Maximizes the diversification ratio: sum(w*vol) / portfolio_vol
        """

        def neg_div_ratio(w: np.ndarray) -> float:
            port_vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
            if port_vol < 1e-10:
                return 1e10
            weighted_vol = np.dot(w, self.volatilities)
            return -weighted_vol / port_vol

        x0 = np.ones(self.n_strategies) / self.n_strategies
        bounds = self._get_bounds(constraints)
        cons = self._get_scipy_constraints(constraints)

        result = optimize.minimize(
            neg_div_ratio,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        return result.x if result.success else self._equal_weight()

    def _hierarchical_risk_parity(self) -> np.ndarray:
        """
        Hierarchical Risk Parity (HRP).

        Uses hierarchical clustering to determine allocation,
        more robust to estimation error than mean-variance.
        """
        # Step 1: Tree clustering
        dist = self._correlation_distance()
        link = hierarchy.linkage(squareform(dist), method="single")

        # Step 2: Quasi-diagonalization
        sorted_idx = self._get_quasi_diag(link)

        # Step 3: Recursive bisection
        weights = self._recursive_bisection(self.cov_matrix, sorted_idx)

        return weights

    def _correlation_distance(self) -> np.ndarray:
        """Calculate correlation distance matrix."""
        dist = np.sqrt(0.5 * (1 - self.corr_matrix))
        # Ensure diagonal is exactly 0 (required by squareform)
        np.fill_diagonal(dist, 0.0)
        return dist

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """Sort indices based on hierarchical clustering."""
        link = link.astype(int)
        n = link.shape[0] + 1

        def recurse(cluster_id: int) -> List[int]:
            if cluster_id < n:
                return [cluster_id]
            left = int(link[cluster_id - n, 0])
            right = int(link[cluster_id - n, 1])
            return recurse(left) + recurse(right)

        return recurse(2 * n - 2)

    def _recursive_bisection(
        self,
        cov: np.ndarray,
        sorted_idx: List[int],
    ) -> np.ndarray:
        """Recursively allocate weights using inverse variance."""
        weights = np.ones(len(sorted_idx))

        def _allocate(
            items: List[int],
            w: np.ndarray,
        ) -> None:
            if len(items) <= 1:
                return

            # Split in half
            mid = len(items) // 2
            left = items[:mid]
            right = items[mid:]

            # Calculate cluster variances
            def cluster_var(idx_list: List[int]) -> float:
                sub_cov = cov[np.ix_(idx_list, idx_list)]
                inv_diag = 1.0 / np.diag(sub_cov)
                cluster_w = inv_diag / inv_diag.sum()
                return float(np.dot(cluster_w, np.dot(sub_cov, cluster_w)))

            var_left = cluster_var(left)
            var_right = cluster_var(right)

            # Allocate based on inverse variance
            alpha = 1 - var_left / (var_left + var_right)

            w[left] *= alpha
            w[right] *= 1 - alpha

            # Recurse
            _allocate(left, w)
            _allocate(right, w)

        _allocate(sorted_idx, weights)

        # Map back to original order
        result = np.zeros(self.n_strategies)
        for i, idx in enumerate(sorted_idx):
            result[idx] = weights[i]

        return result

    def _get_bounds(
        self,
        constraints: OptimizationConstraints,
    ) -> List[Tuple[float, float]]:
        """Get scipy bounds from constraints."""
        if constraints.allow_short:
            return [(-1.0, 1.0)] * self.n_strategies
        return [(constraints.min_weight, constraints.max_weight)] * self.n_strategies

    def _get_scipy_constraints(
        self,
        constraints: OptimizationConstraints,
    ) -> List[Dict[str, Any]]:
        """Get scipy constraints from constraints."""
        cons = []

        if constraints.sum_to_one:
            cons.append(
                {
                    "type": "eq",
                    "fun": lambda w: np.sum(w) - 1.0,
                }
            )

        if constraints.max_volatility is not None:
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda w: constraints.max_volatility
                    - np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w))),
                }
            )

        # Group constraints
        if constraints.strategy_groups and constraints.group_limits:
            for group_name, indices in constraints.strategy_groups.items():
                if group_name in constraints.group_limits:
                    min_limit, max_limit = constraints.group_limits[group_name]
                    cons.append(
                        {
                            "type": "ineq",
                            "fun": lambda w, idx=indices: np.sum(w[idx]) - min_limit,
                        }
                    )
                    cons.append(
                        {
                            "type": "ineq",
                            "fun": lambda w, idx=indices, mx=max_limit: mx
                            - np.sum(w[idx]),
                        }
                    )

        return cons

    def _apply_weight_constraints(
        self,
        weights: np.ndarray,
        constraints: OptimizationConstraints,
    ) -> np.ndarray:
        """Apply min/max constraints and renormalize."""
        weights = np.clip(weights, constraints.min_weight, constraints.max_weight)
        if constraints.sum_to_one and np.sum(weights) > 0:
            weights /= np.sum(weights)
        return weights

    def get_efficient_frontier(
        self,
        n_points: int = 50,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> List[OptimizationResult]:
        """
        Calculate efficient frontier.

        Args:
            n_points: Number of points on frontier
            constraints: Optimization constraints

        Returns:
            List of OptimizationResult on efficient frontier
        """
        if constraints is None:
            constraints = OptimizationConstraints()

        # Find min and max returns
        min_var_result = self.optimize(OptimizationMethod.MIN_VARIANCE, constraints)
        max_sharpe_result = self.optimize(OptimizationMethod.MAX_SHARPE, constraints)

        min_ret = min_var_result.expected_return
        max_ret = max(self.mean_returns)

        # Generate target returns
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []
        for target in target_returns:
            c = OptimizationConstraints(
                min_weight=constraints.min_weight,
                max_weight=constraints.max_weight,
                target_return=target,
            )
            try:
                result = self.optimize(OptimizationMethod.MEAN_VARIANCE, c)
                frontier.append(result)
            except Exception:
                continue

        return frontier

    def get_risk_contributions(
        self,
        weights: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate risk contribution of each strategy.

        Args:
            weights: Portfolio weights

        Returns:
            Dictionary of strategy name to risk contribution
        """
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        if port_vol < 1e-10:
            return {s.name: 0.0 for s in self.strategies}

        # Marginal risk contribution
        mrc = np.dot(self.cov_matrix, weights) / port_vol

        # Risk contribution (as percentage)
        rc = weights * mrc / port_vol * 100

        return {s.name: float(r) for s, r in zip(self.strategies, rc)}


# Convenience functions
def mean_variance_optimize(
    returns: np.ndarray,
    target_return: Optional[float] = None,
) -> np.ndarray:
    """
    Quick mean-variance optimization.

    Args:
        returns: NxM matrix of returns (N periods, M assets)
        target_return: Target portfolio return (optional)

    Returns:
        Optimal weights
    """
    strategies = [
        StrategyReturns(name=f"Asset_{i}", returns=returns[:, i])
        for i in range(returns.shape[1])
    ]
    opt = PortfolioOptimizer(strategies)
    constraints = OptimizationConstraints(target_return=target_return)
    result = opt.optimize(OptimizationMethod.MEAN_VARIANCE, constraints)
    return result.weights


def risk_parity_optimize(returns: np.ndarray) -> np.ndarray:
    """
    Quick risk parity optimization.

    Args:
        returns: NxM matrix of returns (N periods, M assets)

    Returns:
        Optimal weights
    """
    strategies = [
        StrategyReturns(name=f"Asset_{i}", returns=returns[:, i])
        for i in range(returns.shape[1])
    ]
    opt = PortfolioOptimizer(strategies)
    result = opt.optimize(OptimizationMethod.RISK_PARITY)
    return result.weights


def max_sharpe_optimize(returns: np.ndarray) -> np.ndarray:
    """
    Quick maximum Sharpe optimization.

    Args:
        returns: NxM matrix of returns (N periods, M assets)

    Returns:
        Optimal weights
    """
    strategies = [
        StrategyReturns(name=f"Asset_{i}", returns=returns[:, i])
        for i in range(returns.shape[1])
    ]
    opt = PortfolioOptimizer(strategies)
    result = opt.optimize(OptimizationMethod.MAX_SHARPE)
    return result.weights


def min_variance_optimize(returns: np.ndarray) -> np.ndarray:
    """
    Quick minimum variance optimization.

    Args:
        returns: NxM matrix of returns (N periods, M assets)

    Returns:
        Optimal weights
    """
    strategies = [
        StrategyReturns(name=f"Asset_{i}", returns=returns[:, i])
        for i in range(returns.shape[1])
    ]
    opt = PortfolioOptimizer(strategies)
    result = opt.optimize(OptimizationMethod.MIN_VARIANCE)
    return result.weights
