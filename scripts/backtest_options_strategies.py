#!/usr/bin/env python3
"""
Cryptocurrency Options Strategy Backtester
- Uses Black-Scholes model for synthetic option pricing
- Tests various options strategies: Covered Call, Protective Put, Straddle, etc.
- Integrates with Multi-Factor signals for entry timing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
from enum import Enum

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

DATA_ROOT = Path("E:/data/crypto_ohlcv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Black-Scholes Option Pricing
# ============================================================================


def black_scholes_call(S, K, T, r, sigma):
    """Calculate call option price using Black-Scholes"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """Calculate put option price using Black-Scholes"""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def calc_historical_volatility(prices: pd.Series, window: int = 30) -> pd.Series:
    """Calculate annualized historical volatility"""
    returns = np.log(prices / prices.shift(1))
    # Annualize: 4h bars = 6 per day * 365 days
    return returns.rolling(window).std() * np.sqrt(6 * 365)


def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def calc_sma(s, p):
    return s.rolling(p).mean()


def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


# ============================================================================
# Data Loader
# ============================================================================


class DataLoader:
    def __init__(self):
        self.root = DATA_ROOT
        self._cache = {}

    def load_ohlcv(self, symbol, tf="4h"):
        key = f"{symbol}_{tf}"
        if key in self._cache:
            return self._cache[key].copy()
        folder = {"4h": "binance_futures_4h", "1d": "binance_futures_1d"}.get(tf)
        for fn in [f"{symbol}.csv", f"{symbol.replace('USDT', '')}.csv"]:
            fp = self.root / folder / fn
            if fp.exists():
                df = pd.read_csv(fp)
                for col in ["datetime", "timestamp"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col)
                        break
                if all(
                    c in df.columns for c in ["open", "high", "low", "close", "volume"]
                ):
                    df = df[["open", "high", "low", "close", "volume"]].sort_index()
                    self._cache[key] = df
                    return df.copy()
        return pd.DataFrame()

    def load_fear_greed(self):
        for fn in ["FEAR_GREED_INDEX_updated.csv", "FEAR_GREED_INDEX.csv"]:
            fp = self.root / fn
            if fp.exists():
                df = pd.read_csv(fp)
                for col in ["datetime", "timestamp"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col)
                        break
                if "close" in df.columns:
                    return df[["close"]].rename(columns={"close": "fear_greed"})
        return pd.DataFrame()


# ============================================================================
# Options Strategies
# ============================================================================


class OptionStrategy(Enum):
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    PROTECTIVE_PUT = "protective_put"
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    IRON_CONDOR = "iron_condor"


@dataclass
class OptionPosition:
    strategy: OptionStrategy
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp
    spot_entry: float
    strike_call: Optional[float]
    strike_put: Optional[float]
    premium_paid: float
    premium_received: float
    contracts: int
    underlying_position: float  # +1 for long, -1 for short, 0 for none


@dataclass
class StrategyResult:
    strategy: str
    symbol: str
    total_trades: int
    winning_trades: int
    total_pnl: float
    total_premium_collected: float
    total_premium_paid: float
    win_rate: float
    avg_trade_pnl: float
    max_drawdown: float
    sharpe_ratio: float


class OptionsBacktester:
    def __init__(self, risk_free_rate: float = 0.05, expiry_days: int = 7):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
            expiry_days: Days to expiration for options (default 7 = weekly)
        """
        self.r = risk_free_rate
        self.expiry_bars = expiry_days * 6  # 6 bars per day for 4h
        self.commission_rate = 0.001  # 0.1% per trade

    def run_covered_call(self, df: pd.DataFrame, signals: pd.Series) -> StrategyResult:
        """
        Covered Call: Long underlying + Short OTM call
        - Enter when signal > 0 (bullish but limited upside expected)
        - Strike: 5% OTM
        """
        vol = calc_historical_volatility(df["close"], 30)
        trades = []
        equity = [10000]
        capital = 10000
        position = None

        for i in range(50, len(df) - self.expiry_bars):
            price = df["close"].iloc[i]
            signal = signals.iloc[i] if i < len(signals) else 0
            sigma = vol.iloc[i] if not pd.isna(vol.iloc[i]) else 0.8

            # Check for expiry
            if position and i >= position["expiry_idx"]:
                expiry_price = df["close"].iloc[position["expiry_idx"]]
                strike = position["strike"]
                premium = position["premium"]

                # P&L calculation
                underlying_pnl = (expiry_price - position["entry"]) * position[
                    "contracts"
                ]

                # Call assignment if ITM
                if expiry_price > strike:
                    # Called away at strike
                    call_pnl = -(expiry_price - strike) * position["contracts"]
                else:
                    # Keep premium
                    call_pnl = 0

                total_pnl = underlying_pnl + call_pnl + premium
                trades.append(
                    {
                        "pnl": total_pnl,
                        "premium": premium,
                        "assigned": expiry_price > strike,
                    }
                )
                capital += total_pnl
                position = None

            # Enter new position
            if not position and signal > 1.5 and sigma > 0:
                strike = price * 1.05  # 5% OTM
                T = self.expiry_bars / (6 * 365)  # Time to expiry in years
                premium = black_scholes_call(price, strike, T, self.r, sigma)

                contracts = (capital * 0.3) / price  # 30% position size
                position = {
                    "entry": price,
                    "strike": strike,
                    "premium": premium * contracts,
                    "contracts": contracts,
                    "expiry_idx": i + self.expiry_bars,
                }
                capital -= price * contracts * self.commission_rate

            equity.append(capital)

        return self._calc_result("Covered Call", "BTCUSDT", trades, equity)

    def run_cash_secured_put(
        self, df: pd.DataFrame, signals: pd.Series
    ) -> StrategyResult:
        """
        Cash-Secured Put: Sell OTM puts with cash collateral
        - Enter when signal > 0 (want to buy dip)
        - Strike: 5% OTM (below current price)
        """
        vol = calc_historical_volatility(df["close"], 30)
        trades = []
        equity = [10000]
        capital = 10000
        position = None

        for i in range(50, len(df) - self.expiry_bars):
            price = df["close"].iloc[i]
            signal = signals.iloc[i] if i < len(signals) else 0
            sigma = vol.iloc[i] if not pd.isna(vol.iloc[i]) else 0.8

            if position and i >= position["expiry_idx"]:
                expiry_price = df["close"].iloc[position["expiry_idx"]]
                strike = position["strike"]
                premium = position["premium"]

                if expiry_price < strike:
                    # Assigned - must buy at strike
                    assignment_loss = (strike - expiry_price) * position["contracts"]
                    total_pnl = premium - assignment_loss
                else:
                    # Keep full premium
                    total_pnl = premium

                trades.append(
                    {
                        "pnl": total_pnl,
                        "premium": premium,
                        "assigned": expiry_price < strike,
                    }
                )
                capital += total_pnl
                position = None

            if not position and signal > 1.0 and sigma > 0:
                strike = price * 0.95  # 5% OTM put
                T = self.expiry_bars / (6 * 365)
                premium = black_scholes_put(price, strike, T, self.r, sigma)

                # Position size based on cash needed for assignment
                contracts = (capital * 0.3) / strike
                position = {
                    "strike": strike,
                    "premium": premium * contracts,
                    "contracts": contracts,
                    "expiry_idx": i + self.expiry_bars,
                }

            equity.append(capital)

        return self._calc_result("Cash-Secured Put", "BTCUSDT", trades, equity)

    def run_protective_put(
        self, df: pd.DataFrame, signals: pd.Series
    ) -> StrategyResult:
        """
        Protective Put: Long underlying + Long ATM put (insurance)
        - Enter when signal > 2 (strong bullish but want protection)
        """
        vol = calc_historical_volatility(df["close"], 30)
        trades = []
        equity = [10000]
        capital = 10000
        position = None

        for i in range(50, len(df) - self.expiry_bars):
            price = df["close"].iloc[i]
            signal = signals.iloc[i] if i < len(signals) else 0
            sigma = vol.iloc[i] if not pd.isna(vol.iloc[i]) else 0.8

            if position and i >= position["expiry_idx"]:
                expiry_price = df["close"].iloc[position["expiry_idx"]]
                strike = position["strike"]
                premium_paid = position["premium"]

                underlying_pnl = (expiry_price - position["entry"]) * position[
                    "contracts"
                ]

                # Put payoff
                if expiry_price < strike:
                    put_payoff = (strike - expiry_price) * position["contracts"]
                else:
                    put_payoff = 0

                total_pnl = underlying_pnl + put_payoff - premium_paid
                trades.append(
                    {
                        "pnl": total_pnl,
                        "premium": -premium_paid,
                        "protected": expiry_price < strike,
                    }
                )
                capital += total_pnl
                position = None

            if not position and signal > 2.0 and sigma > 0:
                strike = price  # ATM put
                T = self.expiry_bars / (6 * 365)
                premium = black_scholes_put(price, strike, T, self.r, sigma)

                contracts = (capital * 0.3) / price
                position = {
                    "entry": price,
                    "strike": strike,
                    "premium": premium * contracts,
                    "contracts": contracts,
                    "expiry_idx": i + self.expiry_bars,
                }
                capital -= (price + premium) * contracts * self.commission_rate

            equity.append(capital)

        return self._calc_result("Protective Put", "BTCUSDT", trades, equity)

    def run_long_straddle(self, df: pd.DataFrame, signals: pd.Series) -> StrategyResult:
        """
        Long Straddle: Long ATM call + Long ATM put
        - Profit from large moves in either direction
        - Enter when volatility is low but expected to increase
        """
        vol = calc_historical_volatility(df["close"], 30)
        vol_ma = vol.rolling(60).mean()
        trades = []
        equity = [10000]
        capital = 10000
        position = None

        for i in range(100, len(df) - self.expiry_bars):
            price = df["close"].iloc[i]
            sigma = vol.iloc[i] if not pd.isna(vol.iloc[i]) else 0.8
            sigma_ma = vol_ma.iloc[i] if not pd.isna(vol_ma.iloc[i]) else sigma

            if position and i >= position["expiry_idx"]:
                expiry_price = df["close"].iloc[position["expiry_idx"]]
                strike = position["strike"]
                premium_paid = position["premium"]

                call_payoff = max(expiry_price - strike, 0) * position["contracts"]
                put_payoff = max(strike - expiry_price, 0) * position["contracts"]

                total_pnl = call_payoff + put_payoff - premium_paid
                trades.append(
                    {
                        "pnl": total_pnl,
                        "premium": -premium_paid,
                        "move_pct": abs(expiry_price - strike) / strike * 100,
                    }
                )
                capital += total_pnl
                position = None

            # Enter when vol is below average (expecting vol expansion)
            if not position and sigma < sigma_ma * 0.9 and sigma > 0:
                strike = price  # ATM
                T = self.expiry_bars / (6 * 365)
                call_premium = black_scholes_call(price, strike, T, self.r, sigma)
                put_premium = black_scholes_put(price, strike, T, self.r, sigma)
                total_premium = call_premium + put_premium

                contracts = (capital * 0.2) / total_premium
                position = {
                    "strike": strike,
                    "premium": total_premium * contracts,
                    "contracts": contracts,
                    "expiry_idx": i + self.expiry_bars,
                }

            equity.append(capital)

        return self._calc_result("Long Straddle", "BTCUSDT", trades, equity)

    def run_short_straddle(
        self, df: pd.DataFrame, signals: pd.Series
    ) -> StrategyResult:
        """
        Short Straddle: Short ATM call + Short ATM put
        - Profit from low volatility / sideways market
        - Enter when volatility is high and expected to decrease
        """
        vol = calc_historical_volatility(df["close"], 30)
        vol_ma = vol.rolling(60).mean()
        trades = []
        equity = [10000]
        capital = 10000
        position = None

        for i in range(100, len(df) - self.expiry_bars):
            price = df["close"].iloc[i]
            sigma = vol.iloc[i] if not pd.isna(vol.iloc[i]) else 0.8
            sigma_ma = vol_ma.iloc[i] if not pd.isna(vol_ma.iloc[i]) else sigma

            if position and i >= position["expiry_idx"]:
                expiry_price = df["close"].iloc[position["expiry_idx"]]
                strike = position["strike"]
                premium_received = position["premium"]

                call_loss = max(expiry_price - strike, 0) * position["contracts"]
                put_loss = max(strike - expiry_price, 0) * position["contracts"]

                total_pnl = premium_received - call_loss - put_loss
                trades.append(
                    {
                        "pnl": total_pnl,
                        "premium": premium_received,
                        "move_pct": abs(expiry_price - strike) / strike * 100,
                    }
                )
                capital += total_pnl
                position = None

            # Enter when vol is above average (expecting vol contraction)
            if not position and sigma > sigma_ma * 1.2 and sigma > 0:
                strike = price
                T = self.expiry_bars / (6 * 365)
                call_premium = black_scholes_call(price, strike, T, self.r, sigma)
                put_premium = black_scholes_put(price, strike, T, self.r, sigma)
                total_premium = call_premium + put_premium

                contracts = (capital * 0.15) / price  # Smaller size due to risk
                position = {
                    "strike": strike,
                    "premium": total_premium * contracts,
                    "contracts": contracts,
                    "expiry_idx": i + self.expiry_bars,
                }

            equity.append(capital)

        return self._calc_result("Short Straddle", "BTCUSDT", trades, equity)

    def run_iron_condor(self, df: pd.DataFrame, signals: pd.Series) -> StrategyResult:
        """
        Iron Condor: Sell OTM put spread + Sell OTM call spread
        - Profit from range-bound market
        - Limited risk, limited reward
        """
        vol = calc_historical_volatility(df["close"], 30)
        vol_ma = vol.rolling(60).mean()
        trades = []
        equity = [10000]
        capital = 10000
        position = None

        for i in range(100, len(df) - self.expiry_bars):
            price = df["close"].iloc[i]
            sigma = vol.iloc[i] if not pd.isna(vol.iloc[i]) else 0.8
            sigma_ma = vol_ma.iloc[i] if not pd.isna(vol_ma.iloc[i]) else sigma

            if position and i >= position["expiry_idx"]:
                expiry_price = df["close"].iloc[position["expiry_idx"]]

                # Calculate P&L for each leg
                put_short_strike = position["put_short"]
                put_long_strike = position["put_long"]
                call_short_strike = position["call_short"]
                call_long_strike = position["call_long"]
                premium = position["premium"]
                contracts = position["contracts"]

                # Put spread P&L
                put_short_loss = max(put_short_strike - expiry_price, 0)
                put_long_gain = max(put_long_strike - expiry_price, 0)
                put_spread_pnl = (put_long_gain - put_short_loss) * contracts

                # Call spread P&L
                call_short_loss = max(expiry_price - call_short_strike, 0)
                call_long_gain = max(expiry_price - call_long_strike, 0)
                call_spread_pnl = (call_long_gain - call_short_loss) * contracts

                total_pnl = premium + put_spread_pnl + call_spread_pnl
                trades.append(
                    {
                        "pnl": total_pnl,
                        "premium": premium,
                        "in_range": put_short_strike < expiry_price < call_short_strike,
                    }
                )
                capital += total_pnl
                position = None

            # Enter when vol is moderate to high
            if not position and sigma > sigma_ma * 1.1 and sigma > 0:
                T = self.expiry_bars / (6 * 365)

                # Define strikes (10% wings)
                put_short = price * 0.93
                put_long = price * 0.88
                call_short = price * 1.07
                call_long = price * 1.12

                # Calculate premiums
                put_short_prem = black_scholes_put(price, put_short, T, self.r, sigma)
                put_long_prem = black_scholes_put(price, put_long, T, self.r, sigma)
                call_short_prem = black_scholes_call(
                    price, call_short, T, self.r, sigma
                )
                call_long_prem = black_scholes_call(price, call_long, T, self.r, sigma)

                net_credit = (
                    put_short_prem - put_long_prem + call_short_prem - call_long_prem
                )

                if net_credit > 0:
                    contracts = (capital * 0.2) / (
                        price * 0.05
                    )  # Max loss = width of spread
                    position = {
                        "put_short": put_short,
                        "put_long": put_long,
                        "call_short": call_short,
                        "call_long": call_long,
                        "premium": net_credit * contracts,
                        "contracts": contracts,
                        "expiry_idx": i + self.expiry_bars,
                    }

            equity.append(capital)

        return self._calc_result("Iron Condor", "BTCUSDT", trades, equity)

    def _calc_result(
        self, strategy: str, symbol: str, trades: List[dict], equity: List[float]
    ) -> StrategyResult:
        if not trades:
            return StrategyResult(
                strategy=strategy,
                symbol=symbol,
                total_trades=0,
                winning_trades=0,
                total_pnl=0,
                total_premium_collected=0,
                total_premium_paid=0,
                win_rate=0,
                avg_trade_pnl=0,
                max_drawdown=0,
                sharpe_ratio=0,
            )

        pnls = [t["pnl"] for t in trades]
        premiums = [t.get("premium", 0) for t in trades]

        equity_s = pd.Series(equity)
        drawdown = (equity_s - equity_s.expanding().max()) / equity_s.expanding().max()

        returns = pd.Series(equity).pct_change().dropna()
        sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252 * 6)

        return StrategyResult(
            strategy=strategy,
            symbol=symbol,
            total_trades=len(trades),
            winning_trades=sum(1 for p in pnls if p > 0),
            total_pnl=sum(pnls),
            total_premium_collected=sum(p for p in premiums if p > 0),
            total_premium_paid=abs(sum(p for p in premiums if p < 0)),
            win_rate=sum(1 for p in pnls if p > 0) / len(pnls) * 100,
            avg_trade_pnl=np.mean(pnls),
            max_drawdown=drawdown.min() * 100,
            sharpe_ratio=sharpe,
        )


def generate_signals(df: pd.DataFrame, fear_greed: pd.DataFrame) -> pd.Series:
    """Generate multi-factor signals for timing options entries"""
    scores = pd.Series(0.0, index=df.index)

    # Technical
    ema20 = calc_ema(df["close"], 20)
    ema50 = calc_ema(df["close"], 50)
    rsi = calc_rsi(df["close"], 14)

    tech = np.where(
        (df["close"] > ema20) & (ema20 > ema50),
        1.5,
        np.where((df["close"] < ema20) & (ema20 < ema50), -1.5, 0),
    )

    rsi_score = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
    scores += tech + rsi_score

    # Fear & Greed
    if not fear_greed.empty:
        fg = fear_greed["fear_greed"].reindex(df.index, method="ffill")
        fg_score = np.where(fg < 25, 1.5, np.where(fg > 75, -1.5, 0))
        scores += pd.Series(fg_score, index=df.index).fillna(0)

    return scores


def main():
    loader = DataLoader()

    logger.info("=" * 70)
    logger.info("CRYPTOCURRENCY OPTIONS STRATEGY BACKTESTER")
    logger.info("=" * 70)
    logger.info("Using Black-Scholes synthetic pricing")
    logger.info("Weekly options (7-day expiry)")
    logger.info("=" * 70)

    # Load data
    symbols = ["BTCUSDT", "ETHUSDT"]
    fear_greed = loader.load_fear_greed()

    all_results = []

    for symbol in symbols:
        df = loader.load_ohlcv(symbol, "4h")
        if df.empty:
            continue

        df = df[df.index >= "2022-01-01"]
        if len(df) < 500:
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"Testing {symbol}")
        logger.info(f"{'='*70}")
        logger.info(
            f"Period: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}"
        )
        logger.info(f"Data points: {len(df)}")

        signals = generate_signals(df, fear_greed)
        backtester = OptionsBacktester(expiry_days=7)

        # Run all strategies
        strategies = [
            ("Covered Call", backtester.run_covered_call),
            ("Cash-Secured Put", backtester.run_cash_secured_put),
            ("Protective Put", backtester.run_protective_put),
            ("Long Straddle", backtester.run_long_straddle),
            ("Short Straddle", backtester.run_short_straddle),
            ("Iron Condor", backtester.run_iron_condor),
        ]

        logger.info(
            f"\n{'Strategy':<20} {'Trades':>8} {'Win%':>8} {'Total PnL':>12} {'Avg PnL':>10} {'MDD':>10} {'Sharpe':>8}"
        )
        logger.info("-" * 80)

        for name, func in strategies:
            result = func(df, signals)
            result.symbol = symbol
            all_results.append(result)

            logger.info(
                f"{result.strategy:<20} {result.total_trades:>8} {result.win_rate:>7.1f}% "
                f"${result.total_pnl:>11.0f} ${result.avg_trade_pnl:>9.0f} "
                f"{result.max_drawdown:>9.1f}% {result.sharpe_ratio:>7.2f}"
            )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("OVERALL SUMMARY")
    logger.info("=" * 70)

    # Group by strategy
    strategy_summary = {}
    for r in all_results:
        if r.strategy not in strategy_summary:
            strategy_summary[r.strategy] = []
        strategy_summary[r.strategy].append(r)

    logger.info(
        f"\n{'Strategy':<20} {'Avg Win%':>10} {'Avg PnL':>12} {'Avg MDD':>10} {'Avg Sharpe':>10}"
    )
    logger.info("-" * 65)

    best_strategy = None
    best_sharpe = -999

    for strategy, results in strategy_summary.items():
        avg_wr = np.mean([r.win_rate for r in results])
        avg_pnl = np.mean([r.total_pnl for r in results])
        avg_mdd = np.mean([r.max_drawdown for r in results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])

        logger.info(
            f"{strategy:<20} {avg_wr:>9.1f}% ${avg_pnl:>11.0f} {avg_mdd:>9.1f}% {avg_sharpe:>9.2f}"
        )

        if avg_sharpe > best_sharpe:
            best_sharpe = avg_sharpe
            best_strategy = strategy

    logger.info(
        f"\nBest Strategy by Sharpe Ratio: {best_strategy} (Sharpe: {best_sharpe:.2f})"
    )

    # Recommendations
    logger.info("\n" + "=" * 70)
    logger.info("STRATEGY RECOMMENDATIONS")
    logger.info("=" * 70)
    logger.info("""
    1. COVERED CALL: Best when moderately bullish
       - Collect premium while holding spot
       - Caps upside but provides income

    2. CASH-SECURED PUT: Best when wanting to buy dips
       - Collect premium while waiting for entry
       - Get assigned at lower price

    3. PROTECTIVE PUT: Best when holding but worried
       - Insurance against crash
       - Costs premium but limits downside

    4. LONG STRADDLE: Best before major events
       - Profit from big moves either direction
       - Needs volatility expansion

    5. SHORT STRADDLE: Best in consolidation
       - Profit from sideways action
       - High risk if market moves sharply

    6. IRON CONDOR: Best in range-bound market
       - Limited risk/reward
       - Profit if price stays in range
    """)

    return all_results


if __name__ == "__main__":
    results = main()
