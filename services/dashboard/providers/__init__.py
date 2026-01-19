"""Dashboard data providers for real data integration."""
from services.dashboard.providers.portfolio_provider import get_portfolio_summary
from services.dashboard.providers.system_provider import get_system_resources, get_service_health

__all__ = [
    "get_portfolio_summary",
    "get_system_resources",
    "get_service_health",
]
