# MASP (Multi-Asset Strategy Platform) Architecture

## Overview

MASP is a modular, event-driven trading platform designed for automated execution of quantitative trading strategies across multiple exchanges and asset classes.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MASP Architecture                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Streamlit   │  │  REST API   │  │  WebSocket  │  │   CLI       │   │
│  │ Dashboard   │  │  (FastAPI)  │  │   Stream    │  │   Tools     │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         │                │                │                │           │
│         └────────────────┴────────────────┴────────────────┘           │
│                                    │                                    │
│                        ┌───────────┴───────────┐                       │
│                        │   Service Layer       │                       │
│                        │  ┌─────────────────┐  │                       │
│                        │  │ Strategy Runner │  │                       │
│                        │  │ Position Manager│  │                       │
│                        │  │ Order Executor  │  │                       │
│                        │  └─────────────────┘  │                       │
│                        └───────────┬───────────┘                       │
│                                    │                                    │
│   ┌────────────────────────────────┴────────────────────────────────┐  │
│   │                         Core Libraries                           │  │
│   │                                                                  │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │  │
│   │  │ Strategy │  │ Position │  │   Risk   │  │  Config  │        │  │
│   │  │  Engine  │  │ Tracker  │  │  Module  │  │  Manager │        │  │
│   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │  │
│   │                                                                  │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │  │
│   │  │  Audit   │  │   RBAC   │  │ Backtest │  │ Circuit  │        │  │
│   │  │  Logger  │  │  System  │  │ Pipeline │  │ Breaker  │        │  │
│   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │  │
│   └────────────────────────────────┬────────────────────────────────┘  │
│                                    │                                    │
│   ┌────────────────────────────────┴────────────────────────────────┐  │
│   │                         Adapter Layer                            │  │
│   │                                                                  │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │  │
│   │  │  Upbit   │  │ Bithumb  │  │ Binance  │  │  eBest   │        │  │
│   │  │ Adapter  │  │ Adapter  │  │ Adapter  │  │ Adapter  │        │  │
│   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │  │
│   │                                                                  │  │
│   │  ┌──────────┐  ┌──────────┐                                     │  │
│   │  │  Paper   │  │  Mock    │                                     │  │
│   │  │ Trading  │  │ Adapter  │                                     │  │
│   │  └──────────┘  └──────────┘                                     │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    ▼                                    │
│                        ┌─────────────────────┐                         │
│                        │  External Services  │                         │
│                        │  (Exchange APIs)    │                         │
│                        └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
Multi-Asset Strategy Platform/
├── libs/                    # Core library modules
│   ├── adapters/           # Exchange adapters
│   ├── backtest/           # Backtesting engine
│   ├── core/               # Core utilities (config, RBAC, audit)
│   ├── indicators/         # Technical indicators
│   ├── monitoring/         # System monitoring
│   ├── positions/          # Position tracking
│   ├── risk/               # Risk management
│   └── strategies/         # Strategy implementations
├── services/               # Service layer
│   ├── api/               # FastAPI REST & WebSocket API
│   └── dashboard/         # Streamlit dashboard
├── tests/                  # Test suites
├── storage/               # Data persistence
├── config/                # Configuration files
└── scripts/               # Utility scripts
```

## Core Components

### 1. Strategy Engine (`libs/strategies/`)

The strategy engine implements the trading logic using a modular approach.

**Key Files:**
- `base_strategy.py` - Abstract base class for strategies
- `kama_tsmom.py` - KAMA-TSMOM trend following strategy
- `strategy_factory.py` - Factory for strategy instantiation

**Strategy Interface:**
```python
class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Signal]:
        """Generate trading signals from market data."""
        pass

    @abstractmethod
    def calculate_position_size(self, signal: Signal, balance: float) -> float:
        """Calculate position size based on signal and available balance."""
        pass
```

### 2. Adapter Layer (`libs/adapters/`)

Adapters provide a unified interface to different exchanges.

**Supported Exchanges:**
| Exchange | Market Data | Execution | Status |
|----------|-------------|-----------|--------|
| Upbit | `real_upbit_spot.py` | `real_upbit_execution.py` | Production |
| Bithumb | `real_bithumb_spot.py` | `real_bithumb_execution.py` | Production |
| Binance Spot | `real_binance_spot.py` | `real_binance_execution.py` | Production |
| Binance Futures | `real_binance_futures.py` | `real_binance_futures_execution.py` | Production |
| eBest (KOSPI) | `real_ebest_spot.py` | `real_ebest_execution.py` | Development |
| Paper Trading | `paper_market_data.py` | `paper_execution.py` | Testing |

**Adapter Interface:**
```python
class MarketDataAdapter(ABC):
    @abstractmethod
    async def get_quote(self, symbol: str) -> MarketQuote:
        """Get current market quote."""
        pass

    @abstractmethod
    async def get_ohlcv(self, symbol: str, interval: str, limit: int) -> List[OHLCV]:
        """Get historical OHLCV data."""
        pass

class ExecutionAdapter(ABC):
    @abstractmethod
    async def place_order(self, symbol: str, side: str, quantity: float, ...) -> OrderResult:
        """Place an order."""
        pass

    @abstractmethod
    async def get_balance(self, currency: str) -> float:
        """Get available balance."""
        pass
```

### 3. Position Tracking (`libs/positions/`)

Tracks open positions across all exchanges.

**Key Classes:**
- `PositionTracker` - Singleton for tracking positions
- `Position` - Position dataclass with P&L calculation
- `PositionStore` - Persistence layer

**Position Structure:**
```python
@dataclass
class Position:
    symbol: str
    exchange: str
    side: str          # LONG or SHORT
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    opened_at: datetime
```

### 4. Risk Management (`libs/risk/`)

Implements risk controls and position limits.

**Components:**
- `PositionValidator` - Validates order sizes against limits
- `CircuitBreaker` - Automatic trading halt on drawdown
- `KillSwitch` - Emergency stop functionality

**Risk Parameters:**
```python
@dataclass
class RiskConfig:
    max_position_pct: float = 0.10      # Max 10% per position
    max_daily_drawdown_pct: float = 0.05 # Max 5% daily drawdown
    max_total_exposure_pct: float = 0.80 # Max 80% total exposure
    max_single_order_krw: float = 1_000_000
    max_daily_orders: int = 100
```

### 5. RBAC System (`libs/core/rbac.py`)

Role-Based Access Control for API security.

**Roles:**
| Role | Description |
|------|-------------|
| `viewer` | Read-only access to positions and trades |
| `analyst` | Viewer + backtest execution |
| `trader` | Analyst + trade execution |
| `manager` | Trader + config changes, kill switch |
| `admin` | Manager + user management |
| `super_admin` | Full system access |

**Permission Structure:**
```python
class Permission(Enum):
    READ_POSITIONS = "positions:read"
    READ_TRADES = "trades:read"
    WRITE_CONFIG = "config:write"
    EXECUTE_TRADES = "trades:execute"
    ADMIN_USERS = "users:admin"
    # ... more permissions
```

### 6. Audit Logging (`libs/core/audit_logger.py`)

Comprehensive logging for compliance and debugging.

**Event Types:**
- Trading: `order.placed`, `order.filled`, `order.cancelled`
- Security: `auth.success`, `auth.failure`, `permission.denied`
- System: `system.startup`, `killswitch.activated`, `circuit_breaker.triggered`
- Config: `config.changed`, `strategy.enabled`, `strategy.disabled`

**Log Format:** JSON Lines (`.jsonl`)

### 7. Monitoring (`libs/monitoring/`)

System and trading metrics collection.

**Components:**
- `SystemMonitor` - CPU, memory, disk monitoring
- `TradingMetricsAggregator` - Order success rates, latency
- `AlertManager` - Alert generation and notification

### 8. Backtest Pipeline (`libs/backtest/`)

Automated backtesting infrastructure.

**Components:**
- `BacktestEngine` - Core simulation engine
- `BacktestPipeline` - Scheduled backtests
- `BacktestResultStore` - Result persistence

## API Layer

### REST API (`services/api/`)

FastAPI-based REST API for programmatic access.

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/status` | GET | System status |
| `/api/v1/positions` | GET | List positions |
| `/api/v1/trades` | GET | Trade history |
| `/api/v1/strategy/start` | POST | Start strategy |
| `/api/v1/strategy/stop` | POST | Stop strategy |
| `/api/v1/kill-switch` | POST | Emergency stop |
| `/api/v1/users` | CRUD | User management |
| `/api/v1/monitoring/*` | GET | System metrics |

**Authentication:**
- `X-MASP-ADMIN-TOKEN` - Legacy admin token
- `X-MASP-API-KEY` - User API key (RBAC)

### WebSocket (`services/api/websocket/`)

Real-time streaming of positions and trades.

**Channels:**
- `positions` - Position updates
- `trades` - Trade notifications
- `alerts` - System alerts

## Data Flow

### Order Execution Flow

```
Strategy Signal → Position Validator → Risk Check → Order Executor
                                                          │
                  ┌───────────────────────────────────────┘
                  │
                  ▼
           Exchange Adapter → Exchange API → Order Confirmation
                  │
                  ▼
           Position Tracker → Audit Logger → Database
```

### Market Data Flow

```
Exchange API → Adapter → Data Cache → Strategy Engine
                              │
                              ▼
                    Indicator Calculation → Signal Generation
```

## Configuration

### Environment Variables

```bash
# Exchange API Keys
UPBIT_ACCESS_KEY=...
UPBIT_SECRET_KEY=...
BITHUMB_API_KEY=...
BITHUMB_API_SECRET=...
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
EBEST_APP_KEY=...
EBEST_APP_SECRET=...

# Security
MASP_ADMIN_TOKEN=...
API_KEY_HMAC_SECRET=...

# Database
DATABASE_URL=sqlite:///storage/masp.db

# API Server
API_HOST=0.0.0.0
API_PORT=8000
API_SSL_ENABLED=0
```

### Schedule Configuration (`config/schedule_config.json`)

```json
{
  "exchanges": {
    "upbit": {
      "enabled": true,
      "strategy": "KAMA-TSMOM-Gate",
      "symbols": "ALL_KRW",
      "position_size_krw": 10000,
      "schedule": {
        "hour": 6,
        "minute": 0,
        "timezone": "Asia/Seoul"
      }
    }
  }
}
```

## Security Considerations

### API Security
- All endpoints require authentication
- RBAC for fine-grained permission control
- API keys are hashed before storage
- Rate limiting on API endpoints

### Trading Security
- Position limits per symbol and total
- Daily drawdown circuit breaker
- Kill switch for emergency stops
- Order validation before execution

### Data Security
- Audit logging of all operations
- Encrypted API key storage
- HTTPS support for API

## Testing Strategy

### Test Categories
- Unit tests: Individual components
- Integration tests: Adapter interactions
- API tests: Endpoint validation
- Backtest tests: Strategy validation

### Test Coverage
- Total tests: 1683+
- Coverage target: 80%+

### Running Tests
```bash
# All tests
pytest

# Specific module
pytest tests/core/test_rbac.py

# With coverage
pytest --cov=libs --cov-report=html
```

## Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
python -m services.api.main

# Run dashboard
streamlit run services/dashboard/main.py
```

### Production
See `docs/PRODUCTION_DEPLOYMENT_GUIDE.md` for:
- Docker deployment
- Kubernetes deployment
- SSL/TLS configuration
- Health probes

## Monitoring & Alerts

### Health Endpoints
- `/api/v1/health/live` - Liveness probe
- `/api/v1/health/ready` - Readiness probe

### Metrics
- System: CPU, memory, disk usage
- Trading: Order latency, success rate
- Position: P&L, exposure

### Alert Channels
- Console logging
- Callback functions (extensible)

## Future Roadmap

1. **Multi-Strategy Support** - Run multiple strategies concurrently
2. **ML Integration** - Machine learning signal enhancement
3. **Portfolio Optimization** - Cross-asset allocation
4. **Advanced Risk** - VaR, stress testing
5. **Cloud Native** - Kubernetes operators, GitOps
