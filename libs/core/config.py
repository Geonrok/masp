"""
Configuration schema and loader for Multi-Asset Strategy Platform.
Uses Pydantic for validation and versioning.
"""

from enum import Enum
from pathlib import Path
from typing import Optional
import json
import os

from pydantic import BaseModel, Field, SecretStr

# Phase 1: Load .env file if exists (for API keys)
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env from current directory or parent directories
except ImportError:
    pass  # python-dotenv not installed, skip


class AssetClass(str, Enum):
    """Supported asset classes."""
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_FUTURES = "crypto_futures"
    KR_STOCK_SPOT = "kr_stock_spot"
    KR_STOCK_FUTURES = "kr_stock_futures"


class ScheduleConfig(BaseModel):
    """Schedule configuration - supports interval or cron."""
    mode: str = Field(default="interval", description="'interval' or 'cron'")
    interval_seconds: int = Field(default=60, description="Interval in seconds (for interval mode)")
    cron_expression: Optional[str] = Field(default=None, description="Cron expression (for cron mode)")
    timezone: str = Field(default="Asia/Seoul", description="Timezone for scheduling (KST)")


class Config(BaseModel):
    """
    Main configuration schema for the strategy platform.
    Versioned for future migrations.
    """
    config_version: str = Field(default="0.1.0", description="Configuration schema version")
    asset_class: AssetClass = Field(..., description="Asset class this service handles")
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    enabled_strategies: list[str] = Field(
        default_factory=lambda: ["mock_strategy"],
        description="List of strategy IDs to enable"
    )
    storage_dsn: str = Field(
        default="storage/local.db",
        description="SQLite database path"
    )
    run_lock: bool = Field(default=True, description="Enable run lock to prevent concurrent execution")
    paper_mode: bool = Field(default=True, description="Enable paper trading (mock orders)")
    
    # Phase 1 준비: Adapter mode (default: mock)
    adapter_mode: str = Field(
        default="mock",
        description="Adapter mode: 'mock' (Phase 0, no API keys) or 'real' (Phase 1, requires API keys)"
    )
    
    # Phase 1 준비: Safety features
    dry_run: bool = Field(
        default=True,
        description="Dry-run mode: no actual orders executed (auto-enabled if adapter_mode=mock)"
    )
    kill_switch_file: Optional[str] = Field(
        default_factory=lambda: os.getenv("KILL_SWITCH_FILE"),
        description="Path to kill switch file - if exists, service will terminate immediately"
    )
    
    # Phase 1: API Keys (3중 방어: SecretStr + repr=False + exclude=True)
    upbit_access_key: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("UPBIT_ACCESS_KEY", "")) if os.getenv("UPBIT_ACCESS_KEY") else None,
        repr=False,
        exclude=True,
        description="Upbit API access key"
    )
    upbit_secret_key: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("UPBIT_SECRET_KEY", "")) if os.getenv("UPBIT_SECRET_KEY") else None,
        repr=False,
        exclude=True,
        description="Upbit API secret key"
    )
    binance_api_key: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("BINANCE_API_KEY", "")) if os.getenv("BINANCE_API_KEY") else None,
        repr=False,
        exclude=True,
        description="Binance API key"
    )
    binance_api_secret: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("BINANCE_API_SECRET", "")) if os.getenv("BINANCE_API_SECRET") else None,
        repr=False,
        exclude=True,
        description="Binance API secret"
    )
    
    # Bithumb API (Phase 2C)
    bithumb_api_key: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("BITHUMB_API_KEY", "")) if os.getenv("BITHUMB_API_KEY") else None,
        repr=False,
        exclude=True,
        description="Bithumb API key"
    )
    bithumb_secret_key: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("BITHUMB_SECRET_KEY", "")) if os.getenv("BITHUMB_SECRET_KEY") else None,
        repr=False,
        exclude=True,
        description="Bithumb secret key"
    )
    
    korea_broker_app_key: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("KOREA_BROKER_APP_KEY", "")) if os.getenv("KOREA_BROKER_APP_KEY") else None,
        repr=False,
        exclude=True,
        description="Korea broker app key"
    )
    korea_broker_app_secret: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("KOREA_BROKER_APP_SECRET", "")) if os.getenv("KOREA_BROKER_APP_SECRET") else None,
        repr=False,
        exclude=True,
        description="Korea broker app secret"
    )
    korea_broker_account_number: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("KOREA_BROKER_ACCOUNT_NUMBER", "")) if os.getenv("KOREA_BROKER_ACCOUNT_NUMBER") else None,
        repr=False,
        exclude=True,
        description="Korea broker account number"
    )

    # eBest (LS Securities) Open API
    ebest_app_key: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("EBEST_APP_KEY", "")) if os.getenv("EBEST_APP_KEY") else None,
        repr=False,
        exclude=True,
        description="eBest/LS Securities Open API app key"
    )
    ebest_app_secret: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("EBEST_APP_SECRET", "")) if os.getenv("EBEST_APP_SECRET") else None,
        repr=False,
        exclude=True,
        description="eBest/LS Securities Open API app secret"
    )
    ebest_account_no: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("EBEST_ACCOUNT_NO", "")) if os.getenv("EBEST_ACCOUNT_NO") else None,
        repr=False,
        exclude=True,
        description="eBest/LS Securities trading account number"
    )
    ebest_account_pwd: Optional[SecretStr] = Field(
        default_factory=lambda: SecretStr(os.getenv("EBEST_ACCOUNT_PWD", "")) if os.getenv("EBEST_ACCOUNT_PWD") else None,
        repr=False,
        exclude=True,
        description="eBest/LS Securities transaction password"
    )
    
    # Service metadata
    service_name: Optional[str] = Field(default=None, description="Service name override")
    build_version: str = Field(default="0.1.0", description="Build version")
    
    # Symbols to trade (mock for Phase 0)
    symbols: list[str] = Field(
        default_factory=list,
        description="List of symbols to process"
    )
    
    def validate_real_mode_requirements(self) -> None:
        """
        Phase 1 준비: real mode에서는 API 키 존재 체크
        Phase 0에서는 이 메서드를 호출하지 않음
        """
        if self.adapter_mode != "real":
            return
        
        # Phase 0 보호: real 모드는 아직 사용 금지
        raise RuntimeError(
            "[CONFIG] adapter_mode='real' is not allowed in Phase 0. "
            "Phase 0 is mock-only. Set adapter_mode='mock'."
        )
    
    @property
    def effective_service_name(self) -> str:
        """Get effective service name."""
        return self.service_name or f"{self.asset_class.value}_service"
    
    def __str__(self) -> str:
        """API 키 완전 마스킹"""
        return (
            f"Config(asset_class={self.asset_class.value}, "
            f"adapter_mode={self.adapter_mode}, "
            f"dry_run={self.dry_run}, "
            f"api_keys=<MASKED>)"
        )
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def is_kill_switch_active(self) -> bool:
        """
        Check if kill switch is active.
        
        Returns:
            True if kill switch file exists and is active
        """
        if not self.kill_switch_file:
            return False
        
        from pathlib import Path
        kill_switch_path = Path(self.kill_switch_file)
        return kill_switch_path.exists()



def get_default_symbols(asset_class: AssetClass) -> list[str]:
    """Get default symbols for each asset class."""
    defaults = {
        AssetClass.CRYPTO_SPOT: ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"],
        AssetClass.CRYPTO_FUTURES: ["BTC/USDT:PERP", "ETH/USDT:PERP", "SOL/USDT:PERP"],
        AssetClass.KR_STOCK_SPOT: ["005930", "000660", "035420", "051910", "006400"],  # Samsung, SK Hynix, NAVER, LG Chem, Samsung SDI
        AssetClass.KR_STOCK_FUTURES: ["101S6000", "101S3000", "101S9000"],  # KOSPI200 futures codes
    }
    return defaults.get(asset_class, [])


def load_config(
    asset_class: AssetClass,
    config_path: Optional[Path] = None,
    overrides: Optional[dict] = None
) -> Config:
    """
    Load configuration for a service.
    
    Priority:
    1. Explicit overrides
    2. Config file (if provided)
    3. Defaults
    """
    config_data = {
        "asset_class": asset_class.value,
        "symbols": get_default_symbols(asset_class),
    }
    
    # Load from file if provided
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            file_config = json.load(f)
            config_data.update(file_config)
    
    # Apply overrides
    if overrides:
        config_data.update(overrides)
    
    return Config(**config_data)


def create_default_config(asset_class: AssetClass) -> Config:
    """Create a default configuration for an asset class."""
    return Config(
        asset_class=asset_class,
        symbols=get_default_symbols(asset_class),
    )


