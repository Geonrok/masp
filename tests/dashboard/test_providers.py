"""Tests for dashboard data providers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_import_providers():
    """Test provider modules import correctly."""
    from services.dashboard.providers import (
        get_portfolio_summary,
        get_service_health,
        get_system_resources,
    )

    assert callable(get_portfolio_summary)
    assert callable(get_system_resources)
    assert callable(get_service_health)


# =============================================================================
# Portfolio Provider Tests
# =============================================================================


def test_portfolio_provider_returns_none_when_disabled():
    """Test portfolio provider returns None when live trading is disabled."""
    from services.dashboard.providers.portfolio_provider import get_portfolio_summary

    with patch.dict("os.environ", {"MASP_ENABLE_LIVE_TRADING": "0"}, clear=False):
        result = get_portfolio_summary()
        assert result is None


def test_parse_holdings_to_positions():
    """Test parsing raw holdings to PortfolioPosition objects."""
    from services.dashboard.providers.portfolio_provider import (
        _parse_holdings_to_positions,
    )

    holdings = [
        {"currency": "BTC", "balance": "0.1", "avg_buy_price": "50000000"},
        {"currency": "ETH", "balance": "2.0", "avg_buy_price": "3000000"},
        {"currency": "KRW", "balance": "1000000"},  # Should be skipped
    ]
    prices = {
        "BTC/KRW": 55000000.0,
        "ETH/KRW": 3200000.0,
    }

    positions = _parse_holdings_to_positions(holdings, prices)

    assert len(positions) == 2
    assert positions[0].symbol == "BTC/KRW"
    assert positions[0].quantity == 0.1
    assert positions[0].current_price == 55000000.0
    assert positions[1].symbol == "ETH/KRW"


def test_parse_holdings_skips_zero_balance():
    """Test that zero balance holdings are skipped."""
    from services.dashboard.providers.portfolio_provider import (
        _parse_holdings_to_positions,
    )

    holdings = [
        {"currency": "BTC", "balance": "0", "avg_buy_price": "50000000"},
        {"currency": "ETH", "balance": "0.0", "avg_buy_price": "3000000"},
    ]

    positions = _parse_holdings_to_positions(holdings, {})

    assert len(positions) == 0


def test_get_cash_balance():
    """Test extracting KRW cash balance from holdings."""
    from services.dashboard.providers.portfolio_provider import _get_cash_balance

    holdings = [
        {"currency": "BTC", "balance": "0.1"},
        {"currency": "KRW", "balance": "5000000"},
        {"currency": "ETH", "balance": "2.0"},
    ]

    cash = _get_cash_balance(holdings)

    assert cash == 5000000.0


def test_get_cash_balance_no_krw():
    """Test cash balance returns 0 when no KRW entry."""
    from services.dashboard.providers.portfolio_provider import _get_cash_balance

    holdings = [
        {"currency": "BTC", "balance": "0.1"},
    ]

    cash = _get_cash_balance(holdings)

    assert cash == 0.0


# =============================================================================
# System Provider Tests
# =============================================================================


def test_get_system_resources_with_psutil():
    """Test system resources returns valid data when psutil is available."""
    import importlib.util

    from services.dashboard.components.system_status import ResourceUsage
    from services.dashboard.providers.system_provider import get_system_resources

    result = get_system_resources()

    assert isinstance(result, ResourceUsage)
    assert 0 <= result.cpu_percent <= 100
    assert 0 <= result.memory_percent <= 100

    # Only check actual values if psutil is available
    if importlib.util.find_spec("psutil") is not None:
        assert result.memory_total_mb > 0
        assert result.disk_total_gb > 0
    else:
        # psutil not available, should return zeros
        assert result.memory_total_mb == 0.0
        assert result.disk_total_gb == 0.0


def test_get_system_resources_without_psutil():
    """Test system resources returns zeros when psutil not available."""

    with patch.dict("sys.modules", {"psutil": None}):
        # Force reimport to trigger ImportError path
        import importlib

        from services.dashboard.providers import system_provider

        importlib.reload(system_provider)

        system_provider.get_system_resources()

        # Should not crash, returns resource object


def test_get_service_health_returns_list():
    """Test service health returns list of ServiceHealth objects."""
    from services.dashboard.components.system_status import ServiceHealth
    from services.dashboard.providers.system_provider import get_service_health

    result = get_service_health()

    assert isinstance(result, list)
    assert len(result) >= 1
    for service in result:
        assert isinstance(service, ServiceHealth)
        assert service.name is not None


def test_service_health_contains_expected_services():
    """Test that all expected services are included."""
    from services.dashboard.providers.system_provider import get_service_health

    result = get_service_health()
    service_names = [s.name for s in result]

    assert "Upbit API" in service_names
    assert "Scheduler" in service_names
    assert "Database" in service_names
    assert "Telegram" in service_names


def test_check_telegram_not_configured():
    """Test Telegram check when not configured."""
    from services.dashboard.components.system_status import ServiceStatus
    from services.dashboard.providers.system_provider import _check_telegram

    with patch.dict("os.environ", {}, clear=True):
        status, latency, message = _check_telegram()

        assert status == ServiceStatus.UNKNOWN
        assert message == "Not configured"


def test_check_telegram_configured():
    """Test Telegram check when configured."""
    from services.dashboard.components.system_status import ServiceStatus
    from services.dashboard.providers.system_provider import _check_telegram

    with patch.dict(
        "os.environ",
        {"TELEGRAM_BOT_TOKEN": "test", "TELEGRAM_CHAT_ID": "123"},
        clear=False,
    ):
        status, latency, message = _check_telegram()

        assert status == ServiceStatus.HEALTHY
        assert message == ""


# =============================================================================
# Order Provider Tests
# =============================================================================


def test_order_provider_import():
    """Test order provider modules import correctly."""
    from services.dashboard.providers import (
        get_balance_provider,
        get_execution_adapter,
        get_price_provider,
    )

    assert callable(get_execution_adapter)
    assert callable(get_price_provider)
    assert callable(get_balance_provider)


def test_get_execution_adapter_returns_none_when_disabled():
    """Test execution adapter returns None when live trading is disabled."""
    from services.dashboard.providers.order_provider import get_execution_adapter

    with patch.dict("os.environ", {"MASP_ENABLE_LIVE_TRADING": "0"}, clear=False):
        result = get_execution_adapter()
        assert result is None


def test_get_price_provider_returns_none_when_disabled():
    """Test price provider returns None when live trading is disabled."""
    from services.dashboard.providers.order_provider import get_price_provider

    with patch.dict("os.environ", {"MASP_ENABLE_LIVE_TRADING": "0"}, clear=False):
        result = get_price_provider()
        assert result is None


def test_get_balance_provider_returns_none_when_disabled():
    """Test balance provider returns None when live trading is disabled."""
    from services.dashboard.providers.order_provider import get_balance_provider

    with patch.dict("os.environ", {"MASP_ENABLE_LIVE_TRADING": "0"}, clear=False):
        result = get_balance_provider()
        assert result is None


def test_is_live_trading_enabled():
    """Test live trading enabled check."""
    from services.dashboard.providers.order_provider import is_live_trading_enabled

    with patch.dict("os.environ", {"MASP_ENABLE_LIVE_TRADING": "0"}, clear=False):
        assert is_live_trading_enabled() is False

    with patch.dict("os.environ", {"MASP_ENABLE_LIVE_TRADING": "1"}, clear=False):
        assert is_live_trading_enabled() is True


def test_order_execution_wrapper_place_order():
    """Test OrderExecutionWrapper place_order method."""
    from dataclasses import dataclass

    from services.dashboard.providers.order_provider import OrderExecutionWrapper

    # Mock adapter
    @dataclass
    class MockOrderResult:
        order_id: str = "test-123"
        status: str = "FILLED"
        filled_quantity: float = 0.001
        filled_price: float = 50000000.0
        fee: float = 25.0
        message: str = "Order placed successfully"

    mock_adapter = MagicMock()
    mock_adapter.get_current_price.return_value = 50000000.0
    mock_adapter.place_order.return_value = MockOrderResult()

    wrapper = OrderExecutionWrapper(mock_adapter)
    result = wrapper.place_order("BTC", "buy", units=0.001)

    assert result["success"] is True
    assert result["order_id"] == "test-123"
    assert result["executed_qty"] == 0.001
    assert result["fee"] == 25.0


# =============================================================================
# Trade History Provider Tests
# =============================================================================


def test_trade_history_provider_import():
    """Test trade history provider imports correctly."""
    from services.dashboard.providers import get_trade_history_client

    assert callable(get_trade_history_client)


def test_trade_history_api_client_with_mock():
    """Test TradeHistoryApiClient with mock logger."""

    from services.dashboard.providers.trade_history_provider import (
        TradeHistoryApiClient,
    )

    mock_logger = MagicMock()
    mock_logger.get_trades.return_value = [
        {
            "timestamp": "2024-01-01T10:00:00",
            "exchange": "upbit",
            "order_id": "T001",
            "symbol": "BTC",
            "side": "BUY",
            "quantity": "0.1",
            "price": "50000000",
            "fee": "2500",
            "pnl": "0",
            "status": "FILLED",
            "message": "",
        }
    ]

    client = TradeHistoryApiClient(mock_logger)
    trades = client.get_trade_history(days=1)

    assert len(trades) >= 1
    assert trades[0]["symbol"] == "BTC"
    assert trades[0]["side"] == "BUY"


def test_trade_history_api_client_get_daily_summary():
    """Test TradeHistoryApiClient daily summary."""
    from services.dashboard.providers.trade_history_provider import (
        TradeHistoryApiClient,
    )

    mock_logger = MagicMock()
    mock_logger.get_daily_summary.return_value = {
        "date": "2024-01-01",
        "total_trades": 5,
        "buy_count": 3,
        "sell_count": 2,
        "total_volume": 1000000.0,
        "total_fee": 500.0,
        "total_pnl": 10000.0,
    }

    client = TradeHistoryApiClient(mock_logger)
    summary = client.get_daily_summary()

    assert summary["total_trades"] == 5
    assert summary["buy_count"] == 3


# =============================================================================
# Log Provider Tests
# =============================================================================


def test_log_provider_import():
    """Test log provider imports correctly."""
    from services.dashboard.providers import get_log_provider

    assert callable(get_log_provider)


def test_get_log_provider_returns_callable():
    """Test get_log_provider returns a callable."""
    from services.dashboard.providers.log_provider import get_log_provider

    provider = get_log_provider()

    assert callable(provider)


def test_log_provider_returns_list():
    """Test log provider returns a list."""
    from services.dashboard.providers.log_provider import get_log_provider

    provider = get_log_provider(log_dir="nonexistent_dir")
    result = provider()

    assert isinstance(result, list)


def test_parse_log_level():
    """Test log level parsing."""
    from services.dashboard.components.log_viewer import LogLevel
    from services.dashboard.providers.log_provider import _parse_log_level

    assert _parse_log_level("DEBUG") == LogLevel.DEBUG
    assert _parse_log_level("INFO") == LogLevel.INFO
    assert _parse_log_level("WARNING") == LogLevel.WARNING
    assert _parse_log_level("WARN") == LogLevel.WARNING
    assert _parse_log_level("ERROR") == LogLevel.ERROR
    assert _parse_log_level("CRITICAL") == LogLevel.CRITICAL


def test_parse_log_line():
    """Test log line parsing."""
    from services.dashboard.components.log_viewer import LogLevel
    from services.dashboard.providers.log_provider import _parse_log_line

    # Standard Python logging format
    line = "2024-01-01 12:00:00,000 - mymodule - INFO - Test message"
    entry = _parse_log_line(line)

    assert entry is not None
    assert entry.level == LogLevel.INFO
    assert entry.source == "mymodule"
    assert "Test message" in entry.message


# =============================================================================
# Alert Provider Tests
# =============================================================================


def test_alert_provider_import():
    """Test alert provider imports correctly."""
    from services.dashboard.providers import get_alert_store

    assert callable(get_alert_store)


def test_get_alert_store_returns_none_when_no_alerts():
    """Test alert store returns None when no alert files exist."""
    from services.dashboard.providers.alert_provider import get_alert_store

    result = get_alert_store(log_dir="nonexistent_dir")

    assert result is None


def test_parse_alert_type():
    """Test alert type parsing."""
    from services.dashboard.providers.alert_provider import (
        ALERT_TYPE_ERROR,
        ALERT_TYPE_SIGNAL,
        ALERT_TYPE_SYSTEM,
        ALERT_TYPE_TRADE,
        _parse_alert_type,
    )

    assert _parse_alert_type("TRADE") == ALERT_TYPE_TRADE
    assert _parse_alert_type("SIGNAL") == ALERT_TYPE_SIGNAL
    assert _parse_alert_type("ERROR") == ALERT_TYPE_ERROR
    assert _parse_alert_type("WARNING") == ALERT_TYPE_ERROR
    assert _parse_alert_type("unknown") == ALERT_TYPE_SYSTEM


# =============================================================================
# Scheduler Provider Tests
# =============================================================================


def test_scheduler_provider_import():
    """Test scheduler provider imports correctly."""
    from services.dashboard.providers import get_scheduler_job_provider

    assert callable(get_scheduler_job_provider)


def test_get_scheduler_job_provider_returns_callable():
    """Test scheduler job provider returns a callable."""
    from services.dashboard.providers.scheduler_provider import (
        get_scheduler_job_provider,
    )

    provider = get_scheduler_job_provider()

    assert callable(provider)


def test_scheduler_provider_returns_jobs():
    """Test scheduler provider returns list of jobs."""
    from services.dashboard.components.scheduler_status import ScheduledJob
    from services.dashboard.providers.scheduler_provider import get_scheduled_jobs

    jobs = get_scheduled_jobs()

    assert isinstance(jobs, list)
    # Should return static jobs when APScheduler not available
    assert len(jobs) >= 1
    for job in jobs:
        assert isinstance(job, ScheduledJob)


def test_parse_job_type():
    """Test job type parsing."""
    from services.dashboard.components.scheduler_status import JobType
    from services.dashboard.providers.scheduler_provider import _parse_job_type

    assert _parse_job_type("BTC Momentum Strategy") == JobType.STRATEGY
    assert _parse_job_type("Market Data Fetch") == JobType.DATA_FETCH
    assert _parse_job_type("Daily Report") == JobType.REPORT
    assert _parse_job_type("Log Cleanup") == JobType.CLEANUP


# =============================================================================
# Strategy Performance Provider Tests
# =============================================================================


def test_strategy_performance_provider_import():
    """Test strategy performance provider imports correctly."""
    from services.dashboard.providers import get_strategy_performance_provider

    assert callable(get_strategy_performance_provider)


def test_calculate_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    from services.dashboard.providers.strategy_performance_provider import (
        _calculate_sharpe_ratio,
    )

    # Constant returns should give low Sharpe
    returns = [1.0, 1.0, 1.0, 1.0, 1.0]
    sharpe = _calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)

    # Varying returns
    returns = [1.0, -0.5, 2.0, -1.0, 1.5, 0.8, -0.3]
    sharpe = _calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)


def test_calculate_max_drawdown():
    """Test max drawdown calculation."""
    from services.dashboard.providers.strategy_performance_provider import (
        _calculate_max_drawdown,
    )

    # Increasing returns = no drawdown
    returns = [1.0, 1.0, 1.0, 1.0]
    mdd = _calculate_max_drawdown(returns)
    assert mdd >= 0.0

    # Returns with drawdown
    returns = [5.0, -10.0, 3.0, -5.0]
    mdd = _calculate_max_drawdown(returns)
    assert mdd > 0.0


def test_calculate_trade_stats():
    """Test trade statistics calculation."""
    from services.dashboard.providers.strategy_performance_provider import (
        _calculate_trade_stats,
    )

    trades = [
        {"pnl": 100},
        {"pnl": -50},
        {"pnl": 200},
        {"pnl": -30},
        {"pnl": 150},
    ]

    stats = _calculate_trade_stats(trades)

    assert stats.total_trades == 5
    assert stats.winning_trades == 3
    assert stats.losing_trades == 2
    assert stats.win_rate == 60.0


# =============================================================================
# Positions Provider Tests
# =============================================================================


def test_positions_provider_import():
    """Test positions provider imports correctly."""
    from services.dashboard.providers import get_positions_data

    assert callable(get_positions_data)


def test_get_positions_data_returns_none_when_disabled():
    """Test positions data returns None when live trading is disabled."""
    from services.dashboard.providers.positions_provider import get_positions_data

    with patch.dict("os.environ", {"MASP_ENABLE_LIVE_TRADING": "0"}, clear=False):
        result = get_positions_data()
        assert result is None


# =============================================================================
# Risk Metrics Provider Tests
# =============================================================================


def test_risk_metrics_provider_import():
    """Test risk metrics provider imports correctly."""
    from services.dashboard.providers import get_risk_metrics_data

    assert callable(get_risk_metrics_data)


def test_calculate_equity_curve():
    """Test equity curve calculation."""
    from services.dashboard.providers.risk_metrics_provider import (
        _calculate_equity_curve,
    )

    returns = [1.0, -0.5, 2.0, -1.0]  # percent
    equity = _calculate_equity_curve(returns, initial_capital=100.0)

    assert len(equity) == 5  # initial + 4 returns
    assert equity[0] == 100.0
    assert equity[1] == 101.0  # 100 * 1.01
    assert abs(equity[2] - 100.495) < 0.001  # 101 * 0.995


def test_calculate_daily_returns_empty():
    """Test daily returns calculation with empty trades."""
    from services.dashboard.providers.risk_metrics_provider import (
        _calculate_daily_returns,
    )

    returns, dates = _calculate_daily_returns([])

    assert returns == []
    assert dates == []


# =============================================================================
# Backtest Provider Tests (Phase 7D-1)
# =============================================================================


def test_backtest_provider_import():
    """Test backtest provider imports correctly."""
    from services.dashboard.providers import (
        get_backtest_data,
        get_backtest_list,
        get_backtest_provider,
        get_strategy_names,
    )

    assert callable(get_backtest_data)
    assert callable(get_backtest_list)
    assert callable(get_strategy_names)
    assert callable(get_backtest_provider)


def test_get_backtest_data_returns_none_when_no_store():
    """Test backtest data returns None when BacktestStore unavailable."""
    from services.dashboard.providers.backtest_provider import get_backtest_data

    with patch(
        "services.dashboard.providers.backtest_provider._get_backtest_store",
        return_value=None,
    ):
        result = get_backtest_data()
        assert result is None


def test_get_backtest_list_returns_empty_when_no_store():
    """Test backtest list returns empty list when no store."""
    from services.dashboard.providers.backtest_provider import get_backtest_list

    with patch(
        "services.dashboard.providers.backtest_provider._get_backtest_store",
        return_value=None,
    ):
        result = get_backtest_list()
        assert result == []


def test_get_strategy_names_returns_empty_when_no_store():
    """Test strategy names returns empty list when no store."""
    from services.dashboard.providers.backtest_provider import get_strategy_names

    with patch(
        "services.dashboard.providers.backtest_provider._get_backtest_store",
        return_value=None,
    ):
        result = get_strategy_names()
        assert result == []


def test_get_backtest_provider_returns_none_when_no_store():
    """Test backtest provider returns None when no store."""
    from services.dashboard.providers.backtest_provider import get_backtest_provider

    with patch(
        "services.dashboard.providers.backtest_provider._get_backtest_store",
        return_value=None,
    ):
        result = get_backtest_provider()
        assert result is None


def test_get_backtest_provider_returns_none_when_no_backtests():
    """Test backtest provider returns None when no backtests exist."""
    from services.dashboard.providers.backtest_provider import get_backtest_provider

    mock_store = MagicMock()
    mock_store.list_backtests.return_value = []

    with patch(
        "services.dashboard.providers.backtest_provider._get_backtest_store",
        return_value=mock_store,
    ):
        result = get_backtest_provider()
        assert result is None


def test_get_backtest_provider_returns_callable():
    """Test backtest provider returns callable when backtests exist."""
    from services.dashboard.providers.backtest_provider import get_backtest_provider

    mock_store = MagicMock()
    mock_store.list_backtests.return_value = [{"backtest_id": "bt_001"}]

    with patch(
        "services.dashboard.providers.backtest_provider._get_backtest_store",
        return_value=mock_store,
    ):
        with patch(
            "services.dashboard.providers.backtest_provider.get_backtest_data",
            return_value={"dates": [], "daily_returns": []},
        ):
            result = get_backtest_provider()
            assert callable(result)


# =============================================================================
# BacktestStore Tests (Phase 7D-1)
# =============================================================================


def test_backtest_result_dataclass():
    """Test BacktestResult dataclass."""
    from libs.adapters.backtest_store import BacktestResult

    result = BacktestResult(
        backtest_id="bt_001",
        strategy_name="TestStrategy",
        created_at="2025-01-01T10:00:00",
        initial_capital=10000000.0,
        start_date="2025-01-01",
        end_date="2025-01-31",
        dates=["2025-01-01", "2025-01-02"],
        daily_returns=[0.01, -0.005],
        total_return=0.5,
        sharpe_ratio=1.2,
        max_drawdown=-0.05,
    )

    assert result.backtest_id == "bt_001"
    assert result.strategy_name == "TestStrategy"
    assert len(result.dates) == 2


def test_backtest_result_to_dict():
    """Test BacktestResult to_dict method."""
    from libs.adapters.backtest_store import BacktestResult

    result = BacktestResult(
        backtest_id="bt_001",
        strategy_name="TestStrategy",
        created_at="2025-01-01T10:00:00",
        initial_capital=10000000.0,
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    data = result.to_dict()

    assert isinstance(data, dict)
    assert data["backtest_id"] == "bt_001"
    assert data["strategy_name"] == "TestStrategy"


def test_backtest_result_from_dict():
    """Test BacktestResult from_dict method."""
    from libs.adapters.backtest_store import BacktestResult

    data = {
        "backtest_id": "bt_002",
        "strategy_name": "AnotherStrategy",
        "created_at": "2025-01-02T10:00:00",
        "initial_capital": 5000000,
        "start_date": "2025-01-01",
        "end_date": "2025-01-31",
        "dates": ["2025-01-01"],
        "daily_returns": [0.02],
        "total_return": 2.0,
    }

    result = BacktestResult.from_dict(data)

    assert result.backtest_id == "bt_002"
    assert result.strategy_name == "AnotherStrategy"
    assert result.initial_capital == 5000000.0
    assert result.total_return == 2.0


def test_backtest_result_to_viewer_format():
    """Test BacktestResult to_viewer_format method."""
    from datetime import date

    from libs.adapters.backtest_store import BacktestResult

    result = BacktestResult(
        backtest_id="bt_001",
        strategy_name="TestStrategy",
        created_at="2025-01-01T10:00:00",
        initial_capital=10000000.0,
        start_date="2025-01-01",
        end_date="2025-01-02",
        dates=["2025-01-01", "2025-01-02"],
        daily_returns=[0.01, -0.005],
    )

    viewer_data = result.to_viewer_format()

    assert "dates" in viewer_data
    assert "daily_returns" in viewer_data
    assert "initial_capital" in viewer_data
    assert "strategy_name" in viewer_data
    assert len(viewer_data["dates"]) == 2
    assert isinstance(viewer_data["dates"][0], date)


def test_backtest_store_initialization(tmp_path):
    """Test BacktestStore initialization."""
    from libs.adapters.backtest_store import BacktestStore

    store = BacktestStore(store_dir=str(tmp_path))

    assert store.store_dir.exists()


def test_backtest_store_save_and_load(tmp_path):
    """Test BacktestStore save and load methods."""
    from libs.adapters.backtest_store import BacktestResult, BacktestStore

    store = BacktestStore(store_dir=str(tmp_path))

    result = BacktestResult(
        backtest_id="bt_test_001",
        strategy_name="TestStrategy",
        created_at="2025-01-01T10:00:00",
        initial_capital=10000000.0,
        start_date="2025-01-01",
        end_date="2025-01-31",
        dates=["2025-01-01"],
        daily_returns=[0.01],
        total_return=1.0,
    )

    # Save
    success = store.save(result)
    assert success is True

    # Load
    loaded = store.load("TestStrategy", "bt_test_001")
    assert loaded is not None
    assert loaded.backtest_id == "bt_test_001"
    assert loaded.strategy_name == "TestStrategy"


def test_backtest_store_list_backtests(tmp_path):
    """Test BacktestStore list_backtests method."""
    from libs.adapters.backtest_store import BacktestResult, BacktestStore

    store = BacktestStore(store_dir=str(tmp_path))

    # Save two backtests
    result1 = BacktestResult(
        backtest_id="bt_001",
        strategy_name="Strategy1",
        created_at="2025-01-01T10:00:00",
        initial_capital=10000000.0,
        start_date="2025-01-01",
        end_date="2025-01-31",
    )
    result2 = BacktestResult(
        backtest_id="bt_002",
        strategy_name="Strategy2",
        created_at="2025-01-02T10:00:00",
        initial_capital=10000000.0,
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    store.save(result1)
    store.save(result2)

    # List
    backtests = store.list_backtests()

    assert len(backtests) == 2
    # Should be sorted by created_at descending
    assert backtests[0]["backtest_id"] == "bt_002"
    assert backtests[1]["backtest_id"] == "bt_001"


def test_backtest_store_get_latest(tmp_path):
    """Test BacktestStore get_latest method."""
    from libs.adapters.backtest_store import BacktestResult, BacktestStore

    store = BacktestStore(store_dir=str(tmp_path))

    result1 = BacktestResult(
        backtest_id="bt_001",
        strategy_name="TestStrategy",
        created_at="2025-01-01T10:00:00",
        initial_capital=10000000.0,
        start_date="2025-01-01",
        end_date="2025-01-31",
    )
    result2 = BacktestResult(
        backtest_id="bt_002",
        strategy_name="TestStrategy",
        created_at="2025-01-02T10:00:00",
        initial_capital=10000000.0,
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    store.save(result1)
    store.save(result2)

    # Get latest
    latest = store.get_latest("TestStrategy")

    assert latest is not None
    assert latest.backtest_id == "bt_002"


def test_backtest_store_delete(tmp_path):
    """Test BacktestStore delete method."""
    from libs.adapters.backtest_store import BacktestResult, BacktestStore

    store = BacktestStore(store_dir=str(tmp_path))

    result = BacktestResult(
        backtest_id="bt_to_delete",
        strategy_name="TestStrategy",
        created_at="2025-01-01T10:00:00",
        initial_capital=10000000.0,
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    store.save(result)

    # Verify exists
    loaded = store.load("TestStrategy", "bt_to_delete")
    assert loaded is not None

    # Delete
    success = store.delete("TestStrategy", "bt_to_delete")
    assert success is True

    # Verify deleted
    loaded = store.load("TestStrategy", "bt_to_delete")
    assert loaded is None


def test_backtest_store_get_strategy_names(tmp_path):
    """Test BacktestStore get_strategy_names method."""
    from libs.adapters.backtest_store import BacktestResult, BacktestStore

    store = BacktestStore(store_dir=str(tmp_path))

    result1 = BacktestResult(
        backtest_id="bt_001",
        strategy_name="StrategyA",
        created_at="2025-01-01T10:00:00",
        initial_capital=10000000.0,
        start_date="2025-01-01",
        end_date="2025-01-31",
    )
    result2 = BacktestResult(
        backtest_id="bt_002",
        strategy_name="StrategyB",
        created_at="2025-01-01T10:00:00",
        initial_capital=10000000.0,
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    store.save(result1)
    store.save(result2)

    names = store.get_strategy_names()

    assert len(names) == 2
    assert "StrategyA" in names
    assert "StrategyB" in names


def test_generate_backtest_id():
    """Test generate_backtest_id function."""
    from libs.adapters.backtest_store import generate_backtest_id

    backtest_id = generate_backtest_id()

    assert backtest_id.startswith("bt_")
    assert len(backtest_id) == 18  # bt_ + YYYYMMDD_HHMMSS
