"""Dashboard data providers for real data integration."""
from services.dashboard.providers.portfolio_provider import get_portfolio_summary
from services.dashboard.providers.system_provider import get_system_resources, get_service_health
from services.dashboard.providers.order_provider import (
    get_execution_adapter,
    get_price_provider,
    get_balance_provider,
)

__all__ = [
    "get_portfolio_summary",
    "get_system_resources",
    "get_service_health",
    "get_execution_adapter",
    "get_price_provider",
    "get_balance_provider",
]
