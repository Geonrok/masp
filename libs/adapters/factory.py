"""
Adapter Factory for creating market data and execution adapters

Provides a centralized way to create adapters for different exchanges.
Phase 2C - Extended with Upbit/Bithumb execution, Paper trading
"""

import logging
import os
import warnings
from typing import Literal, Optional
from libs.adapters.base import MarketDataAdapter, ExecutionAdapter

logger = logging.getLogger(__name__)


AdapterType = Literal["upbit_spot", "bithumb_spot", "binance_futures", "mock", "paper"]


class AdapterFactory:
    """
    Factory for creating exchange adapters.
    
    Market Data:
        - upbit_spot: Upbit 현물 시세
        - bithumb_spot: Bithumb 현물 시세
        - binance_futures: Binance 선물 시세
        - mock: Mock 시세
        
    Execution:
        - paper: 모의 거래 (Paper Trading)
        - upbit_spot: Upbit 실거래
        - upbit: Upbit 실거래 (deprecated alias)
        - bithumb: Bithumb 실거래
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
    def create_market_data(
        exchange_name: str,
        **kwargs
    ) -> MarketDataAdapter:
        """
        Create a MarketData adapter.

        Args:
            exchange_name: "upbit_spot" | "bithumb_spot" | "binance_futures" | "mock"
            **kwargs: Adapter-specific configuration

        Returns:
            MarketDataAdapter instance

        Raises:
            ValueError: If exchange_name is unknown
        """
        logger.info(f"[FACTORY] Creating MarketData adapter: {exchange_name}")

        if exchange_name in {"upbit", "upbit_spot"}:
            from libs.adapters.real_upbit_spot import UpbitSpotMarketData
            return UpbitSpotMarketData(**kwargs)

        if exchange_name == "bithumb_spot":
            from libs.adapters.real_bithumb_spot import BithumbSpotMarketData
            return BithumbSpotMarketData(**kwargs)

        if exchange_name == "binance_futures":
            from libs.adapters.real_binance_futures import BinanceFuturesMarketData
            return BinanceFuturesMarketData(**kwargs)

        if exchange_name == "mock":
            from libs.adapters.mock import MockMarketDataAdapter
            return MockMarketDataAdapter(**kwargs)

        raise ValueError(f"Unknown market data adapter type: {exchange_name}")

    @staticmethod
    def create_execution(
        exchange_name: str,
        adapter_mode: str = "paper",
        config=None,
        trade_logger=None,
        **kwargs
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

        if exchange_name == "binance_futures":
            from libs.adapters.real_binance_futures import BinanceFuturesExecution
            return BinanceFuturesExecution(**kwargs)

        if exchange_name == "mock":
            from libs.adapters.mock import MockExecutionAdapter
            return MockExecutionAdapter(**kwargs)

        raise ValueError(f"Unknown exchange: {exchange_name}")

    @staticmethod
    def get_available_exchanges() -> dict:
        """
        사용 가능한 거래소 목록
        
        Returns:
            dict: {"market_data": [...], "execution": [...]}
        """
        return {
            "market_data": ["upbit_spot", "bithumb_spot", "binance_futures", "mock"],
            "execution": ["paper", "upbit_spot", "upbit", "bithumb", "binance_futures", "mock"]
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


def create_execution_adapter(exchange: str, live_mode: bool = False, **kwargs) -> ExecutionAdapter:
    """
    Legacy helper for compatibility with older scripts/tests.

    Args:
        exchange: Exchange name (e.g., "bithumb", "upbit_spot", "paper")
        live_mode: True for live execution, False for paper
    """
    adapter_mode = "live" if live_mode else "paper"
    return AdapterFactory.create_execution(exchange, adapter_mode=adapter_mode, **kwargs)
