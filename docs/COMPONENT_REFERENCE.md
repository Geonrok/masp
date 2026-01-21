# MASP Component Reference

## Core Modules

### libs/core/

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `config.py` | Configuration management | `Config`, `get_config()` |
| `rbac.py` | Role-Based Access Control | `RBACManager`, `User`, `Permission`, `Role` |
| `audit_logger.py` | Audit logging | `AuditLogger`, `AuditEvent`, `AuditEventType` |
| `database.py` | Database connection | `get_db()`, `DatabaseManager` |
| `backup.py` | Database backup/restore | `backup_database()`, `restore_database()` |

### libs/adapters/

#### Market Data Adapters

| File | Exchange | Key Methods |
|------|----------|-------------|
| `real_upbit_spot.py` | Upbit (KRW) | `get_quote()`, `get_ohlcv()`, `get_all_symbols()` |
| `real_bithumb_spot.py` | Bithumb (KRW) | `get_quote()`, `get_ohlcv()`, `get_all_symbols()` |
| `real_binance_spot.py` | Binance Spot (USDT) | `get_quote()`, `get_ohlcv()`, `get_all_symbols()` |
| `real_binance_futures.py` | Binance Futures | `get_quote()`, `get_ohlcv()`, `get_funding_rate()` |
| `real_ebest_spot.py` | eBest (KOSPI/KOSDAQ) | `get_quote()`, `get_ohlcv()`, `get_all_symbols()` |
| `paper_market_data.py` | Paper Trading | Mock data for testing |

#### Execution Adapters

| File | Exchange | Key Methods |
|------|----------|-------------|
| `real_upbit_execution.py` | Upbit | `place_order()`, `cancel_order()`, `get_balance()` |
| `real_bithumb_execution.py` | Bithumb | `place_order()`, `cancel_order()`, `get_balance()` |
| `real_binance_execution.py` | Binance Spot | `place_order()`, `cancel_order()`, `get_balance()` |
| `real_binance_futures_execution.py` | Binance Futures | `place_order()`, `set_leverage()`, `get_position()` |
| `real_ebest_execution.py` | eBest | `place_order()`, `cancel_order()`, `get_balance()` |
| `paper_execution.py` | Paper Trading | Simulated execution |

#### Helper Modules

| File | Purpose | Key Components |
|------|---------|----------------|
| `execution_helpers.py` | Common utilities | `StandardOrderResult`, `retry_with_backoff`, `rate_limit`, `KillSwitchMixin` |
| `factory.py` | Adapter factory | `create_market_data()`, `create_execution()` |

### libs/strategies/

| File | Strategy | Description |
|------|----------|-------------|
| `base_strategy.py` | Base class | Abstract strategy interface |
| `kama_tsmom.py` | KAMA-TSMOM | Trend following with KAMA filter |
| `kama_tsmom_gate.py` | KAMA-TSMOM-Gate | Enhanced with volatility gating |
| `strategy_factory.py` | Factory | Strategy instantiation |

### libs/indicators/

| File | Indicators | Description |
|------|------------|-------------|
| `trend.py` | SMA, EMA, KAMA | Trend indicators |
| `momentum.py` | RSI, TSMOM | Momentum indicators |
| `volatility.py` | ATR, Bollinger | Volatility indicators |

### libs/positions/

| File | Purpose | Key Classes |
|------|---------|-------------|
| `position_tracker.py` | Track positions | `PositionTracker`, `Position` |
| `position_store.py` | Persistence | `PositionStore` |

### libs/risk/

| File | Purpose | Key Classes |
|------|---------|-------------|
| `position_validator.py` | Order validation | `PositionValidator` |
| `circuit_breaker.py` | Drawdown protection | `CircuitBreaker` |

### libs/monitoring/

| File | Purpose | Key Classes |
|------|---------|-------------|
| `system_monitor.py` | System metrics | `SystemMonitor`, `SystemMetrics` |
| `trading_metrics.py` | Trading metrics | `TradingMetricsAggregator` |
| `alert_manager.py` | Alert handling | `AlertManager`, `Alert` |

### libs/backtest/

| File | Purpose | Key Classes |
|------|---------|-------------|
| `engine.py` | Backtest engine | `BacktestEngine`, `BacktestResult` |
| `pipeline.py` | Automation | `BacktestPipeline`, `BacktestJob` |

---

## Service Layer

### services/api/

#### Main Application

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, routers, middleware |
| `config.py` | API configuration |

#### Routes (`services/api/routes/`)

| File | Endpoints | Authentication |
|------|-----------|----------------|
| `strategy.py` | `/api/v1/strategy/*` | Admin token |
| `positions.py` | `/api/v1/positions/*` | Admin token |
| `trades.py` | `/api/v1/trades/*` | Admin token |
| `health.py` | `/api/v1/health/*` | None (public) |
| `settings.py` | `/api/v1/settings/*` | Admin token |
| `config.py` | `/api/v1/config/*` | Admin token |
| `keys.py` | `/api/v1/keys/*` | Admin token |
| `monitoring.py` | `/api/v1/monitoring/*` | Admin token |
| `users.py` | `/api/v1/users/*` | RBAC |

#### Middleware (`services/api/middleware/`)

| File | Purpose |
|------|---------|
| `auth.py` | Authentication, RBAC integration |

#### WebSocket (`services/api/websocket/`)

| File | Purpose |
|------|---------|
| `stream.py` | Real-time streaming |

### services/dashboard/

| File | Purpose |
|------|---------|
| `main.py` | Streamlit dashboard |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/schedule_config.json` | Trading schedule configuration |
| `config/risk_config.json` | Risk parameters |
| `.env` | Environment variables |
| `pytest.ini` | Test configuration |
| `requirements.txt` | Python dependencies |

---

## Data Storage

| Directory | Contents |
|-----------|----------|
| `storage/` | SQLite databases, RBAC users |
| `storage/positions/` | Position history |
| `storage/backtest/` | Backtest results |
| `logs/` | Application logs |
| `logs/audit/` | Audit logs (JSON Lines) |

---

## Test Files

| Directory | Test Coverage |
|-----------|---------------|
| `tests/core/` | Config, RBAC, Audit Logger |
| `tests/adapters/` | All exchange adapters |
| `tests/strategies/` | Strategy implementations |
| `tests/risk/` | Risk management |
| `tests/positions/` | Position tracking |
| `tests/monitoring/` | Monitoring components |
| `tests/backtest/` | Backtest pipeline |
| `tests/api/` | API routes |

---

## Environment Variables Reference

### Exchange API Keys

```bash
# Upbit
UPBIT_ACCESS_KEY=your_access_key
UPBIT_SECRET_KEY=your_secret_key

# Bithumb
BITHUMB_API_KEY=your_api_key
BITHUMB_API_SECRET=your_api_secret

# Binance
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

# eBest (LS Securities)
EBEST_APP_KEY=your_app_key
EBEST_APP_SECRET=your_app_secret
EBEST_ACCOUNT_NO=your_account_number
```

### Security

```bash
# Admin authentication
MASP_ADMIN_TOKEN=your_secure_token

# API key encryption
API_KEY_HMAC_SECRET=your_hmac_secret

# RBAC storage
MASP_RBAC_STORAGE=storage/rbac_users.json
```

### API Server

```bash
# Server configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=0

# SSL/TLS
API_SSL_ENABLED=0
API_SSL_CERTFILE=/path/to/cert.pem
API_SSL_KEYFILE=/path/to/key.pem
API_SSL_KEYFILE_PASSWORD=optional_password
```

### Database

```bash
DATABASE_URL=sqlite:///storage/masp.db
```

---

## Quick Reference

### Starting the Platform

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export MASP_ADMIN_TOKEN=your_token
export UPBIT_ACCESS_KEY=...
# ... other keys

# 3. Start API server
python -m services.api.main

# 4. Start dashboard (optional)
streamlit run services/dashboard/main.py
```

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/core/test_rbac.py -v

# With coverage
pytest --cov=libs --cov-report=html
```

### Common Operations

```python
# Get config
from libs.core.config import get_config
config = get_config()

# Create adapter
from libs.adapters.factory import create_market_data, create_execution
md = create_market_data("upbit")
exec = create_execution("upbit")

# Track position
from libs.positions.position_tracker import PositionTracker
tracker = PositionTracker.get_instance()
tracker.open_position(symbol="BTC/KRW", ...)

# Audit log
from libs.core.audit_logger import get_audit_logger
logger = get_audit_logger()
logger.log_order_placed(...)

# Check permission (RBAC)
from libs.core.rbac import RBACManager, Permission
rbac = RBACManager.get_instance()
if rbac.check_permission(user_id, Permission.EXECUTE_TRADES):
    # Allow trade
```
