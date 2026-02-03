"""
Exchange Registry - 거래소 등록 및 관리
- 거래소 메타데이터 관리
- 거래소 상태 모니터링
- 거래소별 설정 관리
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


logger = logging.getLogger(__name__)


class ExchangeStatus(Enum):
    """거래소 상태."""

    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ExchangeType(Enum):
    """거래소 유형."""

    SPOT = "spot"
    FUTURES = "futures"
    MARGIN = "margin"


class ExchangeRegion(Enum):
    """거래소 지역."""

    KOREA = "kr"
    GLOBAL = "global"
    US = "us"
    ASIA = "asia"


@dataclass
class ExchangeInfo:
    """거래소 정보."""

    name: str
    display_name: str
    exchange_type: ExchangeType
    region: ExchangeRegion
    base_currency: str = "KRW"
    supported_symbols: List[str] = field(default_factory=list)
    api_rate_limit: int = 10  # requests per second
    requires_auth: bool = False
    features: Set[str] = field(default_factory=set)

    # Runtime state
    status: ExchangeStatus = ExchangeStatus.UNKNOWN
    last_check: Optional[datetime] = None
    latency_ms: float = 0.0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "exchange_type": self.exchange_type.value,
            "region": self.region.value,
            "base_currency": self.base_currency,
            "supported_symbols": self.supported_symbols,
            "api_rate_limit": self.api_rate_limit,
            "requires_auth": self.requires_auth,
            "features": list(self.features),
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "latency_ms": self.latency_ms,
            "error_count": self.error_count,
        }


@dataclass
class ExchangeConfig:
    """거래소 설정."""

    exchange_name: str
    enabled: bool = True
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)

    # Trading settings
    max_order_size: Optional[float] = None
    min_order_size: Optional[float] = None
    default_slippage: float = 0.001  # 0.1%

    @classmethod
    def from_env(cls, exchange_name: str) -> "ExchangeConfig":
        """Load config from environment variables."""
        prefix = exchange_name.upper().replace("-", "_")

        return cls(
            exchange_name=exchange_name,
            enabled=os.getenv(f"{prefix}_ENABLED", "1") == "1",
            api_key=os.getenv(f"{prefix}_API_KEY"),
            api_secret=os.getenv(f"{prefix}_API_SECRET"),
        )


class ExchangeRegistry:
    """거래소 레지스트리 - 모든 거래소를 관리하는 싱글톤."""

    _instance: Optional["ExchangeRegistry"] = None

    # Default exchange definitions
    DEFAULT_EXCHANGES: Dict[str, ExchangeInfo] = {
        "upbit_spot": ExchangeInfo(
            name="upbit_spot",
            display_name="Upbit",
            exchange_type=ExchangeType.SPOT,
            region=ExchangeRegion.KOREA,
            base_currency="KRW",
            supported_symbols=["BTC", "ETH", "XRP", "SOL", "ADA", "DOGE"],
            api_rate_limit=10,
            requires_auth=True,
            features={"market_order", "limit_order", "cancel_order", "balance"},
        ),
        "bithumb_spot": ExchangeInfo(
            name="bithumb_spot",
            display_name="Bithumb",
            exchange_type=ExchangeType.SPOT,
            region=ExchangeRegion.KOREA,
            base_currency="KRW",
            supported_symbols=["BTC", "ETH", "XRP", "SOL", "ADA", "DOGE"],
            api_rate_limit=15,
            requires_auth=True,
            features={"market_order", "limit_order", "cancel_order", "balance"},
        ),
        "binance_futures": ExchangeInfo(
            name="binance_futures",
            display_name="Binance Futures",
            exchange_type=ExchangeType.FUTURES,
            region=ExchangeRegion.GLOBAL,
            base_currency="USDT",
            supported_symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            api_rate_limit=20,
            requires_auth=True,
            features={
                "market_order",
                "limit_order",
                "cancel_order",
                "balance",
                "leverage",
                "hedge_mode",
            },
        ),
    }

    def __new__(cls) -> "ExchangeRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._exchanges: Dict[str, ExchangeInfo] = {}
        self._configs: Dict[str, ExchangeConfig] = {}
        self._adapters: Dict[str, Any] = {}  # cached adapters
        self._health_callbacks: List[Callable[[str, ExchangeStatus], None]] = []

        # Register default exchanges
        for name, info in self.DEFAULT_EXCHANGES.items():
            self._exchanges[name] = info

        self._initialized = True
        logger.info(
            "[ExchangeRegistry] Initialized with %d exchanges", len(self._exchanges)
        )

    def register(self, info: ExchangeInfo) -> None:
        """Register a new exchange.

        Args:
            info: ExchangeInfo instance
        """
        self._exchanges[info.name] = info
        logger.info("[ExchangeRegistry] Registered: %s", info.name)

    def unregister(self, name: str) -> bool:
        """Unregister an exchange.

        Args:
            name: Exchange name

        Returns:
            True if removed
        """
        if name in self._exchanges:
            del self._exchanges[name]
            if name in self._adapters:
                del self._adapters[name]
            logger.info("[ExchangeRegistry] Unregistered: %s", name)
            return True
        return False

    def get(self, name: str) -> Optional[ExchangeInfo]:
        """Get exchange info by name.

        Args:
            name: Exchange name

        Returns:
            ExchangeInfo or None
        """
        return self._exchanges.get(name)

    def get_all(self) -> Dict[str, ExchangeInfo]:
        """Get all registered exchanges.

        Returns:
            Dict of exchange name to ExchangeInfo
        """
        return self._exchanges.copy()

    def get_by_type(self, exchange_type: ExchangeType) -> List[ExchangeInfo]:
        """Get exchanges by type.

        Args:
            exchange_type: SPOT, FUTURES, or MARGIN

        Returns:
            List of matching ExchangeInfo
        """
        return [e for e in self._exchanges.values() if e.exchange_type == exchange_type]

    def get_by_region(self, region: ExchangeRegion) -> List[ExchangeInfo]:
        """Get exchanges by region.

        Args:
            region: KOREA, GLOBAL, etc.

        Returns:
            List of matching ExchangeInfo
        """
        return [e for e in self._exchanges.values() if e.region == region]

    def get_online(self) -> List[ExchangeInfo]:
        """Get all online exchanges.

        Returns:
            List of online ExchangeInfo
        """
        return [
            e for e in self._exchanges.values() if e.status == ExchangeStatus.ONLINE
        ]

    def set_config(self, config: ExchangeConfig) -> None:
        """Set exchange configuration.

        Args:
            config: ExchangeConfig instance
        """
        self._configs[config.exchange_name] = config

    def get_config(self, name: str) -> Optional[ExchangeConfig]:
        """Get exchange configuration.

        Args:
            name: Exchange name

        Returns:
            ExchangeConfig or None
        """
        if name not in self._configs:
            # Try loading from environment
            self._configs[name] = ExchangeConfig.from_env(name)
        return self._configs.get(name)

    def update_status(
        self,
        name: str,
        status: ExchangeStatus,
        latency_ms: float = 0.0,
        error: bool = False,
    ) -> None:
        """Update exchange status.

        Args:
            name: Exchange name
            status: New status
            latency_ms: Response latency in ms
            error: Whether this update is due to an error
        """
        if name not in self._exchanges:
            return

        exchange = self._exchanges[name]
        old_status = exchange.status

        exchange.status = status
        exchange.last_check = datetime.now()
        exchange.latency_ms = latency_ms

        if error:
            exchange.error_count += 1
        else:
            exchange.error_count = 0

        # Notify callbacks if status changed
        if old_status != status:
            logger.info(
                "[ExchangeRegistry] %s status: %s -> %s",
                name,
                old_status.value,
                status.value,
            )
            for callback in self._health_callbacks:
                try:
                    callback(name, status)
                except Exception as e:
                    logger.warning("Health callback error: %s", e)

    def add_health_callback(
        self, callback: Callable[[str, ExchangeStatus], None]
    ) -> None:
        """Add status change callback.

        Args:
            callback: Function(exchange_name, new_status)
        """
        self._health_callbacks.append(callback)

    def check_health(self, name: str) -> ExchangeStatus:
        """Check exchange health by attempting a simple API call.

        Args:
            name: Exchange name

        Returns:
            Current status
        """
        if name not in self._exchanges:
            return ExchangeStatus.UNKNOWN

        try:
            import time

            from libs.adapters.factory import AdapterFactory

            start = time.time()
            adapter = AdapterFactory.create_market_data(name)

            # Try to get a quote for a common symbol
            exchange = self._exchanges[name]
            if exchange.supported_symbols:
                test_symbol = exchange.supported_symbols[0]
                quote = adapter.get_quote(test_symbol)
            else:
                quote = True  # No symbols to test

            latency_ms = (time.time() - start) * 1000

            if quote:
                self.update_status(name, ExchangeStatus.ONLINE, latency_ms)
                return ExchangeStatus.ONLINE
            else:
                self.update_status(name, ExchangeStatus.DEGRADED, latency_ms)
                return ExchangeStatus.DEGRADED

        except Exception as e:
            logger.warning("[ExchangeRegistry] Health check failed for %s: %s", name, e)
            self.update_status(name, ExchangeStatus.OFFLINE, error=True)
            return ExchangeStatus.OFFLINE

    def check_all_health(self) -> Dict[str, ExchangeStatus]:
        """Check health of all exchanges.

        Returns:
            Dict of exchange name to status
        """
        results = {}
        for name in self._exchanges:
            results[name] = self.check_health(name)
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary.

        Returns:
            Summary dict with counts and status
        """
        exchanges = list(self._exchanges.values())

        return {
            "total": len(exchanges),
            "online": len([e for e in exchanges if e.status == ExchangeStatus.ONLINE]),
            "offline": len(
                [e for e in exchanges if e.status == ExchangeStatus.OFFLINE]
            ),
            "by_type": {
                "spot": len(
                    [e for e in exchanges if e.exchange_type == ExchangeType.SPOT]
                ),
                "futures": len(
                    [e for e in exchanges if e.exchange_type == ExchangeType.FUTURES]
                ),
            },
            "by_region": {
                "korea": len(
                    [e for e in exchanges if e.region == ExchangeRegion.KOREA]
                ),
                "global": len(
                    [e for e in exchanges if e.region == ExchangeRegion.GLOBAL]
                ),
            },
        }


# =============================================================================
# Global singleton instance
# =============================================================================


def get_registry() -> ExchangeRegistry:
    """Get the global ExchangeRegistry instance.

    Returns:
        ExchangeRegistry singleton
    """
    return ExchangeRegistry()
