#!/usr/bin/env python
"""
MASP Daemon Mode Runner.
Starts the DailyScheduler in run_forever() mode.

Usage:
    python scripts/run_daemon.py --strategy kama_tsmom_gate --exchange paper
    python scripts/run_daemon.py --strategy atlas_futures_p04 --exchange binance_futures
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(override=False)

from services.scheduler import DailyScheduler
from services.strategy_runner import StrategyRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MASP Daemon Mode - Run strategy scheduler in background",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Paper trading with KAMA-TSMOM-Gate (default)
    python scripts/run_daemon.py

    # Paper trading with ATLAS-Futures
    python scripts/run_daemon.py --strategy atlas_futures_p04

    # Live trading (requires env vars)
    python scripts/run_daemon.py --strategy kama_tsmom_gate --exchange upbit

    # Binance Futures with ATLAS
    python scripts/run_daemon.py --strategy atlas_futures_p04 --exchange binance_futures
        """,
    )
    parser.add_argument(
        "--strategy",
        default="kama_tsmom_gate",
        help="Strategy ID (default: kama_tsmom_gate)",
    )
    parser.add_argument(
        "--exchange",
        default="paper",
        choices=[
            "paper",
            "upbit",
            "bithumb",
            "binance_spot",
            "binance_futures",
            "ebest",
        ],
        help="Exchange to use (default: paper)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Trading symbols (default: auto-detect from strategy/exchange)",
    )
    parser.add_argument(
        "--size-krw",
        type=float,
        default=10000,
        help="Position size in KRW (default: 10000)",
    )
    parser.add_argument(
        "--size-usdt",
        type=float,
        default=10,
        help="Position size in USDT for Binance (default: 10)",
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=1,
        help="Leverage for futures (default: 1)",
    )
    parser.add_argument(
        "--config",
        default="config/schedule_config.yaml",
        help="Schedule config path (default: config/schedule_config.yaml)",
    )
    parser.add_argument(
        "--no-validate-keys",
        action="store_true",
        help="Skip API key validation at startup",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit if API key validation fails",
    )
    return parser.parse_args()


def get_default_symbols(strategy: str, exchange: str) -> list[str]:
    """Get default symbols based on strategy and exchange."""
    if strategy == "atlas_futures_p04":
        return ["BTCUSDT", "ETHUSDT", "LINKUSDT"]
    if exchange in ("binance_spot", "binance_futures"):
        return ["BTCUSDT"]
    return ["BTC/KRW"]


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Determine symbols
    symbols = args.symbols or get_default_symbols(args.strategy, args.exchange)

    # Determine position size
    is_usdt = args.exchange in ("binance_spot", "binance_futures")
    position_size_krw = 0 if is_usdt else args.size_krw
    position_size_usdt = args.size_usdt if is_usdt else 0

    logger.info("=" * 60)
    logger.info("MASP Daemon Mode Starting")
    logger.info("=" * 60)
    logger.info("Strategy: %s", args.strategy)
    logger.info("Exchange: %s", args.exchange)
    logger.info("Symbols: %s", symbols)
    logger.info("Position Size: %s %s",
                position_size_usdt if is_usdt else position_size_krw,
                "USDT" if is_usdt else "KRW")
    logger.info("Leverage: %dx", args.leverage)
    logger.info("Config: %s", args.config)
    logger.info("Live Trading: %s", os.getenv("MASP_ENABLE_LIVE_TRADING", "0"))
    logger.info("=" * 60)

    try:
        # Create StrategyRunner
        runner = StrategyRunner(
            strategy_name=args.strategy,
            exchange=args.exchange,
            symbols=symbols,
            position_size_krw=position_size_krw,
            position_size_usdt=position_size_usdt,
            leverage=args.leverage,
        )

        # Create DailyScheduler
        scheduler = DailyScheduler(runner, config_path=args.config)

        logger.info("Scheduler trigger: %s", scheduler.trigger)
        logger.info("Scheduler jitter: %d seconds", scheduler.jitter)
        logger.info("Starting scheduler loop (Ctrl+C to stop)...")

        # Run forever
        asyncio.run(
            scheduler.run_forever(
                validate_keys=not args.no_validate_keys,
                strict=args.strict,
            )
        )

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        return 0
    except Exception as exc:
        logger.error("Daemon error: %s", exc, exc_info=True)
        return 1

    logger.info("Daemon stopped gracefully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
