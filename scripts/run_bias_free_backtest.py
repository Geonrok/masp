"""
Run Bias-Free Backtest with Real Data

This script:
1. Loads real OHLCV data from E:/data/crypto_ohlcv/
2. Runs the bias-free backtester (Phase 1)
3. Optionally runs validation (Phase 2)
4. Prints comprehensive results

Usage:
    python scripts/run_bias_free_backtest.py --exchange upbit
    python scripts/run_bias_free_backtest.py --exchange binance --validate
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from libs.strategy.integrated_strategy import (
    IntegratedConfig,
    IntegratedStrategy,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_ROOT = Path("E:/data/crypto_ohlcv")


def load_exchange_data(
    exchange: str, timeframe: str = "1d", min_days: int = 100
) -> dict:
    """
    Load OHLCV data from exchange folder.

    Args:
        exchange: Exchange name (upbit, binance_spot, binance_futures)
        timeframe: Timeframe folder suffix (1d, 4h, 1h)
        min_days: Minimum number of days required

    Returns:
        Dict of symbol -> DataFrame
    """
    folder = DATA_ROOT / f"{exchange}_{timeframe}"

    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")

    data = {}
    csv_files = list(folder.glob("*.csv"))

    logger.info(f"Loading data from {folder} ({len(csv_files)} files)...")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Identify date column
            date_col = None
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    date_col = col
                    break

            if date_col is None:
                continue

            # Parse dates and set index
            df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
            df = df.set_index("date")

            # Ensure required columns
            required = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required):
                continue

            df = df[required]

            # Filter by minimum days
            if len(df) >= min_days:
                symbol = csv_file.stem
                data[symbol] = df

        except Exception as e:
            logger.debug(f"Skip {csv_file.name}: {e}")
            continue

    logger.info(f"Loaded {len(data)} symbols with {min_days}+ days")
    return data


def run_backtest(
    exchange: str = "upbit",
    validate: bool = False,
    min_days: int = 100,
    initial_capital: float = 10000.0,
) -> None:
    """
    Run bias-free backtest.

    Args:
        exchange: Exchange to test (upbit, binance_spot, binance_futures)
        validate: Whether to run WFO/CPCV/DSR validation
        min_days: Minimum days required per symbol
        initial_capital: Starting capital
    """
    print("\n" + "=" * 70)
    print("BIAS-FREE BACKTEST")
    print(f"Exchange: {exchange}")
    print(f"Validate: {validate}")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print("=" * 70 + "\n")

    # Load data
    data = load_exchange_data(exchange, timeframe="1d", min_days=min_days)

    if not data:
        print("ERROR: No data loaded!")
        return

    # Get date range
    all_dates = []
    for df in data.values():
        all_dates.extend(df.index.tolist())

    min_date = min(all_dates)
    max_date = max(all_dates)

    print(
        f"Data Range: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}"
    )
    print(f"Symbols: {len(data)}")
    print()

    # Configure strategy
    config = IntegratedConfig(
        initial_capital=initial_capital,
        slippage_pct=0.005,  # 0.5% slippage
        commission_pct=0.001,  # 0.1% commission
        max_positions=20,
        kama_period=5,
        tsmom_period=90,
        gate_period=30,
        run_validation=validate,
        enable_veto=True,
    )

    # Run backtest
    strategy = IntegratedStrategy(config)
    result = strategy.run_backtest(data, validate=validate)

    # Print results
    print(result.summary())

    # Additional analysis
    if result.metrics:
        m = result.metrics

        print("\n" + "=" * 70)
        print("COMPARISON WITH ORIGINAL (BIASED) RESULTS")
        print("=" * 70)
        print()
        print("Original Biased Results (INVALID):")
        print("  - Upbit: +133% ~ +176%")
        print("  - Binance: Similar inflated returns")
        print()
        print("Bias-Free Results (VALID):")
        print(f"  - Total Return: {m.total_return * 100:+.1f}%")
        print(f"  - Sharpe Ratio: {m.sharpe_ratio:.2f}")
        print(f"  - Max Drawdown: {m.max_drawdown * 100:.1f}%")
        print()

        if m.total_return < 0:
            print("NOTE: Negative returns confirm the Look-Ahead Bias hypothesis.")
            print(
                "      The original +133% was due to Day T signal -> Day T execution."
            )
            print(
                "      Correct methodology (Day T signal -> Day T+1 execution) shows reality."
            )
        elif m.total_return < 0.1:  # Less than 10%
            print(
                "NOTE: Much lower returns than original confirms bias was significant."
            )

        print()

    return result


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Bias-Free Backtest")
    parser.add_argument(
        "--exchange",
        choices=["upbit", "binance_spot", "binance_futures"],
        default="upbit",
        help="Exchange to test",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run WFO/CPCV/DSR validation"
    )
    parser.add_argument(
        "--min-days", type=int, default=100, help="Minimum days required per symbol"
    )
    parser.add_argument(
        "--capital", type=float, default=10000.0, help="Initial capital"
    )

    args = parser.parse_args()

    run_backtest(
        exchange=args.exchange,
        validate=args.validate,
        min_days=args.min_days,
        initial_capital=args.capital,
    )


if __name__ == "__main__":
    main()
