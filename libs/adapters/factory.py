"""
Adapter Factory for creating market data and execution adapters

Provides a centralized way to create adapters for different exchanges.
Phase 2C - Extended with Upbit/Bithumb execution, Paper trading
Optimized with class caching for faster repeated instantiation.
"""

import logging
import os
import warnings
from typing import Literal, Type

from libs.adapters.base import ExecutionAdapter, MarketDataAdapter

logger = logging.getLogger(__name__)

# ============================================================================
# Adapter Class Cache - Prevents repeated imports
# ============================================================================
_market_data_classes: dict[str, Type[MarketDataAdapter]] = {}
_execution_classes: dict[str, Type[ExecutionAdapter]] = {}


def clear_adapter_cache() -> None:
    """
    Clear the adapter class cache.

    Use this in tests to ensure fresh adapter loading.
    """
    _market_data_classes.clear()
    _execution_classes.clear()


def get_cached_adapters() -> dict:
    """
    Get info about currently cached adapter classes.

    Returns:
        dict with 'market_data' and 'execution' lists of cached adapter names.
    """
    return {
        "market_data": list(_market_data_classes.keys()),
        "execution": list(_execution_classes.keys()),
    }


AdapterType = Literal[
    "upbit_spot",
    "bithumb_spot",
    "binance_spot",
    "binance_futures",
    "ebest_spot",
    "ebest_kospi",
    "ebest_kosdaq",
    "kiwoom_spot",
    "mock",
    "paper",
]

# Constants for exchange name sets
EBEST_EXCHANGES = {"ebest", "ebest_spot", "ebest_kospi", "ebest_kosdaq"}
KIWOOM_EXCHANGES = {"kiwoom", "kiwoom_spot"}


class AdapterFactory:
    """
    Factory for creating exchange adapters.

    Market Data:
        - upbit_spot: Upbit 현물 시세
        - bithumb_spot: Bithumb 현물 시세
        - binance_spot: Binance 현물 시세
        - binance_futures: Binance 선물 시세
        - mock: Mock 시세

    Execution:
        - paper: 모의 거래 (Paper Trading)
        - upbit_spot: Upbit 실거래
        - upbit: Upbit 실거래 (deprecated alias)
        - bithumb: Bithumb 실거래
        - binance_spot: Binance 현물 실거래
        - binance_futures: Binance 선물 실거래
        - mock: Mock 실행

    Usage:
        # 시세 조회
        market_data = AdapterFactory.create_market_data("upbit_spot")

        # 모의 거래
        paper = AdapterFactory.create_execution("paper")

        # 실거래 (Config 필요)
        upbit = AdapterFactory.create_execution("upbit_spot", adapter_mode="live")
    """

    @staticmethod
    def _get_market_data_class(exchange_name: str) -> Type[MarketDataAdapter]:
        """Get MarketData adapter class with caching."""
        # Normalize name
        if exchange_name == "upbit":
            exchange_name = "upbit_spot"

        if exchange_name not in _market_data_classes:
            if exchange_name == "upbit_spot":
                from libs.adapters.real_upbit_spot import UpbitSpotMarketData

                _market_data_classes[exchange_name] = UpbitSpotMarketData
            elif exchange_name == "bithumb_spot":
                from libs.adapters.real_bithumb_spot import BithumbSpotMarketData

                _market_data_classes[exchange_name] = BithumbSpotMarketData
            elif exchange_name == "binance_spot":
                from libs.adapters.real_binance_spot import BinanceSpotMarketData

                _market_data_classes[exchange_name] = BinanceSpotMarketData
            elif exchange_name == "binance_futures":
                from libs.adapters.real_binance_futures import BinanceFuturesMarketData

                _market_data_classes[exchange_name] = BinanceFuturesMarketData
            elif exchange_name == "mock":
                from libs.adapters.mock import MockMarketDataAdapter

                _market_data_classes[exchange_name] = MockMarketDataAdapter
            elif exchange_name in EBEST_EXCHANGES:
                from libs.adapters.real_ebest_spot import EbestSpotMarketData

                _market_data_classes[exchange_name] = EbestSpotMarketData
            elif exchange_name in KIWOOM_EXCHANGES:
                from libs.adapters.real_kiwoom_spot import KiwoomSpotMarketData

                _market_data_classes[exchange_name] = KiwoomSpotMarketData
            else:
                raise ValueError(f"Unknown market data adapter type: {exchange_name}")

        return _market_data_classes[exchange_name]

    @staticmethod
    def create_market_data(exchange_name: str, **kwargs) -> MarketDataAdapter:
        """
        Create a MarketData adapter (uses cached class loading).

        Args:
            exchange_name: "upbit_spot" | "bithumb_spot" | "binance_futures" | "mock"
            **kwargs: Adapter-specific configuration

        Returns:
            MarketDataAdapter instance

        Raises:
            ValueError: If exchange_name is unknown
        """
        logger.info(f"[FACTORY] Creating MarketData adapter: {exchange_name}")
        adapter_class = AdapterFactory._get_market_data_class(exchange_name)
        return adapter_class(**kwargs)

    @staticmethod
    def create_execution(
        exchange_name: str,
        adapter_mode: str = "paper",
        config=None,
        trade_logger=None,
        **kwargs,
    ) -> ExecutionAdapter:
        """
        Create an Execution adapter.

        Args:
            exchange_name: "upbit_spot" (recommended), "upbit" (deprecated)
            adapter_mode: "paper" for simulation or "live" for execution
            **kwargs: Adapter-specific configuration

        Returns:
            ExecutionAdapter instance

        Raises:
            ValueError: If exchange_name is unknown
        """
        logger.info(f"[FACTORY] Creating Execution adapter: {exchange_name}")

        if config is None:
            config = kwargs.pop("config", None)
        else:
            kwargs.pop("config", None)
        if trade_logger is None:
            trade_logger = kwargs.pop("trade_logger", None)
        else:
            kwargs.pop("trade_logger", None)

        if exchange_name == "upbit":
            warnings.warn(
                "'upbit' is deprecated, use 'upbit_spot' instead. "
                "This alias will be removed in the next release.",
                DeprecationWarning,
                stacklevel=2,
            )
            exchange_name = "upbit_spot"

        if exchange_name == "upbit_spot":
            if adapter_mode in {"live", "execution"}:
                if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
                    raise RuntimeError(
                        "[Factory] Live trading disabled. "
                        "Set MASP_ENABLE_LIVE_TRADING=1 or use adapter_mode='paper'"
                    )
                from libs.adapters.real_upbit_spot import UpbitSpotExecution

                return UpbitSpotExecution(
                    access_key=kwargs.get("access_key"),
                    secret_key=kwargs.get("secret_key"),
                )

            from libs.adapters.paper_execution import PaperExecutionAdapter

            market_data = AdapterFactory.create_market_data("upbit_spot")
            return PaperExecutionAdapter(
                market_data_adapter=market_data,
                initial_balance=kwargs.pop("initial_balance", 1_000_000),
                config=config,
                trade_logger=trade_logger,
                **kwargs,
            )

        if exchange_name != "mock" and exchange_name != "paper":
            logger.warning(
                "[FACTORY] ⚠️ Real trading adapter requested. "
                "Ensure Kill-Switch is configured and Config.is_kill_switch_active() is called."
            )

        if exchange_name == "paper":
            from libs.adapters.paper_execution import PaperExecutionAdapter

            market_data = AdapterFactory.create_market_data("upbit_spot")
            return PaperExecutionAdapter(
                market_data_adapter=market_data,
                initial_balance=kwargs.pop("initial_balance", 1_000_000),
                config=config,
                trade_logger=trade_logger,
                **kwargs,
            )

        if exchange_name in {"bithumb", "bithumb_spot"}:
            if adapter_mode in {"live", "execution"}:
                if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
                    raise RuntimeError(
                        "[Factory] Bithumb live trading disabled. "
                        "Set MASP_ENABLE_LIVE_TRADING=1 or use adapter_mode='paper'"
                    )
                from libs.adapters.real_bithumb_execution import BithumbExecutionAdapter
                from libs.core.config import Config as ConfigClass

                if config is None:
                    config = ConfigClass()
                adapter = BithumbExecutionAdapter(config, **kwargs)
                if trade_logger:
                    adapter.set_trade_logger(trade_logger)
                return adapter

            # Paper mode for bithumb
            from libs.adapters.paper_execution import PaperExecutionAdapter

            market_data = AdapterFactory.create_market_data("bithumb_spot")
            return PaperExecutionAdapter(
                market_data_adapter=market_data,
                initial_balance=kwargs.pop("initial_balance", 1_000_000),
                config=config,
                trade_logger=trade_logger,
                **kwargs,
            )

        if exchange_name == "binance_spot":
            if adapter_mode in {"live", "execution"}:
                if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
                    raise RuntimeError(
                        "[Factory] Binance Spot live trading disabled. "
                        "Set MASP_ENABLE_LIVE_TRADING=1 or use adapter_mode='paper'"
                    )
                from libs.adapters.real_binance_spot import BinanceSpotExecution

                return BinanceSpotExecution(**kwargs)

            # Paper mode for binance_spot
            from libs.adapters.paper_execution import PaperExecutionAdapter

            market_data = AdapterFactory.create_market_data("binance_spot")
            return PaperExecutionAdapter(
                market_data_adapter=market_data,
                initial_balance=kwargs.pop("initial_balance", 10_000),  # USDT
                config=config,
                trade_logger=trade_logger,
                **kwargs,
            )

        if exchange_name == "binance_futures":
            if adapter_mode in {"live", "execution"}:
                if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
                    raise RuntimeError(
                        "[Factory] Binance Futures live trading disabled. "
                        "Set MASP_ENABLE_LIVE_TRADING=1 or use adapter_mode='paper'"
                    )
                from libs.adapters.real_binance_futures import BinanceFuturesExecution

                return BinanceFuturesExecution(**kwargs)

            # Paper mode for binance_futures
            from libs.adapters.paper_execution import PaperExecutionAdapter

            market_data = AdapterFactory.create_market_data("binance_futures")
            return PaperExecutionAdapter(
                market_data_adapter=market_data,
                initial_balance=kwargs.pop("initial_balance", 10_000),  # USDT
                config=config,
                trade_logger=trade_logger,
                **kwargs,
            )

        if exchange_name == "mock":
            from libs.adapters.mock import MockExecutionAdapter

            return MockExecutionAdapter(**kwargs)

        if exchange_name in EBEST_EXCHANGES:
            # Extract credentials from config or kwargs
            ebest_app_key = kwargs.get("app_key")
            ebest_app_secret = kwargs.get("app_secret")
            ebest_account_no = kwargs.get("account_no")
            ebest_account_pwd = kwargs.get("account_pwd")

            if config is not None:
                if (
                    ebest_app_key is None
                    and hasattr(config, "ebest_app_key")
                    and config.ebest_app_key
                ):
                    ebest_app_key = config.ebest_app_key.get_secret_value()
                if (
                    ebest_app_secret is None
                    and hasattr(config, "ebest_app_secret")
                    and config.ebest_app_secret
                ):
                    ebest_app_secret = config.ebest_app_secret.get_secret_value()
                if (
                    ebest_account_no is None
                    and hasattr(config, "ebest_account_no")
                    and config.ebest_account_no
                ):
                    ebest_account_no = config.ebest_account_no.get_secret_value()
                if (
                    ebest_account_pwd is None
                    and hasattr(config, "ebest_account_pwd")
                    and config.ebest_account_pwd
                ):
                    ebest_account_pwd = config.ebest_account_pwd.get_secret_value()

            if adapter_mode in {"live", "execution"}:
                if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
                    raise RuntimeError(
                        "[Factory] eBest live trading disabled. "
                        "Set MASP_ENABLE_LIVE_TRADING=1 or use adapter_mode='paper'"
                    )
                # Validate credentials for live mode
                if not ebest_app_key or not ebest_app_secret:
                    raise ValueError(
                        "[Factory] eBest live trading requires EBEST_APP_KEY and EBEST_APP_SECRET. "
                        "Set environment variables or provide via config."
                    )
                from libs.adapters.real_ebest_execution import EbestSpotExecution

                adapter = EbestSpotExecution(
                    app_key=ebest_app_key,
                    app_secret=ebest_app_secret,
                    account_no=ebest_account_no,
                    account_pwd=ebest_account_pwd,
                )
                if trade_logger:
                    adapter.set_trade_logger(trade_logger)
                return adapter

            # Paper mode for eBest
            from libs.adapters.paper_execution import PaperExecutionAdapter

            market_data = AdapterFactory.create_market_data(
                "ebest_spot",
                app_key=ebest_app_key,
                app_secret=ebest_app_secret,
            )
            return PaperExecutionAdapter(
                market_data_adapter=market_data,
                initial_balance=kwargs.pop("initial_balance", 10_000_000),  # 10M KRW
                config=config,
                trade_logger=trade_logger,
                **kwargs,
            )

        if exchange_name in KIWOOM_EXCHANGES:
            # Extract credentials from config or kwargs
            kiwoom_app_key = kwargs.get("app_key")
            kiwoom_app_secret = kwargs.get("app_secret")
            kiwoom_account_no = kwargs.get("account_no")

            if config is not None:
                if (
                    kiwoom_app_key is None
                    and hasattr(config, "kiwoom_app_key")
                    and config.kiwoom_app_key
                ):
                    kiwoom_app_key = config.kiwoom_app_key.get_secret_value()
                if (
                    kiwoom_app_secret is None
                    and hasattr(config, "kiwoom_app_secret")
                    and config.kiwoom_app_secret
                ):
                    kiwoom_app_secret = config.kiwoom_app_secret.get_secret_value()
                if (
                    kiwoom_account_no is None
                    and hasattr(config, "kiwoom_account_no")
                    and config.kiwoom_account_no
                ):
                    kiwoom_account_no = config.kiwoom_account_no.get_secret_value()

            if adapter_mode in {"live", "execution"}:
                if os.getenv("MASP_ENABLE_LIVE_TRADING") != "1":
                    raise RuntimeError(
                        "[Factory] Kiwoom live trading disabled. "
                        "Set MASP_ENABLE_LIVE_TRADING=1 or use adapter_mode='paper'"
                    )
                # Validate credentials for live mode
                if not kiwoom_app_key or not kiwoom_app_secret:
                    raise ValueError(
                        "[Factory] Kiwoom live trading requires KIWOOM_APP_KEY and KIWOOM_APP_SECRET. "
                        "Set environment variables or provide via config."
                    )
                from libs.adapters.real_kiwoom_spot import KiwoomSpotExecution

                adapter = KiwoomSpotExecution(
                    app_key=kiwoom_app_key,
                    app_secret=kiwoom_app_secret,
                    account_no=kiwoom_account_no,
                )
                if trade_logger:
                    adapter.set_trade_logger(trade_logger)
                return adapter

            # Paper mode for Kiwoom
            from libs.adapters.paper_execution import PaperExecutionAdapter

            market_data = AdapterFactory.create_market_data(
                "kiwoom_spot",
                app_key=kiwoom_app_key,
                app_secret=kiwoom_app_secret,
            )
            return PaperExecutionAdapter(
                market_data_adapter=market_data,
                initial_balance=kwargs.pop(
                    "initial_balance", 1_000_000
                ),  # 1M KRW (소액)
                config=config,
                trade_logger=trade_logger,
                **kwargs,
            )

        raise ValueError(f"Unknown exchange: {exchange_name}")

    @staticmethod
    def get_available_exchanges() -> dict:
        """
        사용 가능한 거래소 목록

        Returns:
            dict: {"market_data": [...], "execution": [...]}
        """
        return {
            "market_data": [
                "upbit_spot",
                "bithumb_spot",
                "binance_spot",
                "binance_futures",
                "ebest_spot",
                "ebest_kospi",
                "ebest_kosdaq",
                "kiwoom_spot",
                "mock",
            ],
            "execution": [
                "paper",
                "upbit_spot",
                "upbit",
                "bithumb",
                "binance_spot",
                "binance_futures",
                "ebest",
                "ebest_spot",
                "ebest_kospi",
                "ebest_kosdaq",
                "kiwoom",
                "kiwoom_spot",
                "mock",
            ],
        }

    @staticmethod
    def list_available() -> list[str]:
        """
        List all available adapter types (legacy).

        Returns:
            List of adapter type names
        """
        all_types = set()
        exchanges = AdapterFactory.get_available_exchanges()
        all_types.update(exchanges["market_data"])
        all_types.update(exchanges["execution"])
        return sorted(list(all_types))


def create_execution_adapter(
    exchange: str, live_mode: bool = False, **kwargs
) -> ExecutionAdapter:
    """
    Legacy helper for compatibility with older scripts/tests.

    Args:
        exchange: Exchange name (e.g., "bithumb", "upbit_spot", "paper")
        live_mode: True for live execution, False for paper
    """
    adapter_mode = "live" if live_mode else "paper"
    return AdapterFactory.create_execution(
        exchange, adapter_mode=adapter_mode, **kwargs
    )
