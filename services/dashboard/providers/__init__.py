"""Dashboard data providers for real data integration."""

from services.dashboard.providers.alert_manager_provider import (
    check_system_anomalies,
    create_rule,
    delete_rule,
    get_alert_rules,
    get_alert_settings_callbacks,
    get_anomaly_thresholds,
    get_recent_alerts,
    save_thresholds,
    toggle_rule,
)
from services.dashboard.providers.alert_provider import get_alert_store
from services.dashboard.providers.backtest_provider import (
    get_backtest_data,
    get_backtest_list,
    get_backtest_provider,
    get_strategy_names,
)
from services.dashboard.providers.log_provider import get_log_provider
from services.dashboard.providers.multi_exchange_provider import (
    find_arbitrage_opportunities,
    get_best_exchange,
    get_exchange_list,
    get_exchange_status,
    get_price_comparison,
    get_registry_summary,
    perform_health_check,
)
from services.dashboard.providers.order_provider import (
    get_balance_provider,
    get_execution_adapter,
    get_price_provider,
)
from services.dashboard.providers.portfolio_provider import get_portfolio_summary
from services.dashboard.providers.positions_provider import get_positions_data
from services.dashboard.providers.risk_metrics_provider import get_risk_metrics_data
from services.dashboard.providers.scheduler_provider import get_scheduler_job_provider
from services.dashboard.providers.strategy_performance_provider import (
    get_strategy_performance_provider,
)
from services.dashboard.providers.system_provider import (
    get_service_health,
    get_system_resources,
)
from services.dashboard.providers.trade_history_provider import get_trade_history_client

__all__ = [
    # 1순위 providers
    "get_portfolio_summary",
    "get_system_resources",
    "get_service_health",
    "get_execution_adapter",
    "get_price_provider",
    "get_balance_provider",
    # 2순위 providers
    "get_trade_history_client",
    "get_log_provider",
    # 3순위 providers
    "get_alert_store",
    "get_scheduler_job_provider",
    "get_strategy_performance_provider",
    # 추가 providers
    "get_positions_data",
    "get_risk_metrics_data",
    # backtest providers
    "get_backtest_data",
    "get_backtest_list",
    "get_strategy_names",
    "get_backtest_provider",
    # alert manager providers
    "get_alert_rules",
    "get_anomaly_thresholds",
    "toggle_rule",
    "delete_rule",
    "create_rule",
    "save_thresholds",
    "get_recent_alerts",
    "check_system_anomalies",
    "get_alert_settings_callbacks",
    # multi-exchange providers
    "get_exchange_list",
    "get_exchange_status",
    "get_price_comparison",
    "find_arbitrage_opportunities",
    "get_best_exchange",
    "perform_health_check",
    "get_registry_summary",
]
