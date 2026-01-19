"""Dashboard data providers for real data integration."""
from services.dashboard.providers.portfolio_provider import get_portfolio_summary
from services.dashboard.providers.system_provider import get_system_resources, get_service_health
from services.dashboard.providers.order_provider import (
    get_execution_adapter,
    get_price_provider,
    get_balance_provider,
)
from services.dashboard.providers.trade_history_provider import get_trade_history_client
from services.dashboard.providers.log_provider import get_log_provider
from services.dashboard.providers.alert_provider import get_alert_store
from services.dashboard.providers.scheduler_provider import get_scheduler_job_provider
from services.dashboard.providers.strategy_performance_provider import get_strategy_performance_provider
from services.dashboard.providers.positions_provider import get_positions_data
from services.dashboard.providers.risk_metrics_provider import get_risk_metrics_data
from services.dashboard.providers.backtest_provider import (
    get_backtest_data,
    get_backtest_list,
    get_strategy_names,
    get_backtest_provider,
)
from services.dashboard.providers.alert_manager_provider import (
    get_alert_rules,
    get_anomaly_thresholds,
    toggle_rule,
    delete_rule,
    create_rule,
    save_thresholds,
    get_recent_alerts,
    check_system_anomalies,
    get_alert_settings_callbacks,
)
from services.dashboard.providers.multi_exchange_provider import (
    get_exchange_list,
    get_exchange_status,
    get_price_comparison,
    find_arbitrage_opportunities,
    get_best_exchange,
    perform_health_check,
    get_registry_summary,
)

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
    # backtest providers (Phase 7D-1)
    "get_backtest_data",
    "get_backtest_list",
    "get_strategy_names",
    "get_backtest_provider",
    # alert manager providers (Phase 7D-2)
    "get_alert_rules",
    "get_anomaly_thresholds",
    "toggle_rule",
    "delete_rule",
    "create_rule",
    "save_thresholds",
    "get_recent_alerts",
    "check_system_anomalies",
    "get_alert_settings_callbacks",
    # multi-exchange providers (Phase 7D-3)
    "get_exchange_list",
    "get_exchange_status",
    "get_price_comparison",
    "find_arbitrage_opportunities",
    "get_best_exchange",
    "perform_health_check",
    "get_registry_summary",
]
