"""
FastAPI main application (v2).
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.api.config import api_config
from services.api.models.schemas import (
    BaseResponse,
    KillSwitchRequest,
    KillSwitchResponse,
    SystemStatus,
    ExchangeStatus,
    ExchangeStatusResponse,
)
from services.api.routes import (
    strategy,
    positions,
    trades,
    health,
    settings,
    config,
    keys,
    monitoring,
    users,
)
from services.api.websocket.stream import router as ws_router

logger = logging.getLogger(__name__)

_start_time = datetime.now()


# ============================================================================
# Configuration Caching
# ============================================================================
@lru_cache(maxsize=1)
def _load_schedule_config() -> dict:
    """Load schedule config with caching. Call invalidate_config_cache() to refresh."""
    config_path = Path("config/schedule_config.json")
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Failed to load schedule config: %s", e)
    return {}


def invalidate_config_cache():
    """Invalidate the schedule config cache."""
    _load_schedule_config.cache_clear()
    logger.info("[API] Schedule config cache invalidated")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize shared state."""
    from services.api.routes.strategy import StrategyManager

    logger.info("[API] Starting MASP API Server v1.0...")
    logger.info("[API] Host: %s:%s", api_config.host, api_config.port)
    logger.info("[API] Mode: Single Worker (state-safe)")

    # Initialize app state
    app.state.strategy_manager = StrategyManager()
    app.state.schedule_config = _load_schedule_config()

    logger.info("[API] App state initialized")
    yield
    logger.info("[API] Shutting down API Server...")


app = FastAPI(
    title="Multi-Asset Strategy Platform API",
    description="Real-time trading strategy management API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to headers. No body modification for performance."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Timestamp"] = datetime.now().isoformat()
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(
        "[%s] HTTPException: %s - %s", request_id, exc.status_code, exc.detail
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail if exc.status_code < 500 else "Internal server error",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error("[%s] Unhandled exception: %s", request_id, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        },
    )


app.include_router(strategy.router, prefix="/api/v1/strategy", tags=["Strategy"])
app.include_router(positions.router, prefix="/api/v1/positions", tags=["Positions"])
app.include_router(trades.router, prefix="/api/v1/trades", tags=["Trades"])
app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
app.include_router(settings.router, prefix="/api/v1/settings", tags=["Settings"])
app.include_router(config.router)
app.include_router(keys.router)
app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["Monitoring"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(ws_router, prefix="/ws", tags=["WebSocket"])


@app.get("/api/v1/status", response_model=SystemStatus)
async def get_status(request: Request):
    """Get system status (uses app.state for strategy manager with fallback)."""
    from services.api.routes.strategy import strategy_manager as fallback_manager

    uptime = (datetime.now() - _start_time).total_seconds()
    request_id = getattr(request.state, "request_id", "unknown")

    # Use app state if available, fallback for tests
    if hasattr(request.app.state, "strategy_manager"):
        manager = request.app.state.strategy_manager
    else:
        manager = fallback_manager

    return SystemStatus(
        success=True,
        message="MASP API is running",
        version="1.0.0",
        uptime_seconds=uptime,
        exchanges=["upbit", "bithumb", "binance_spot", "binance_futures", "paper"],
        active_strategies=len(manager.active_strategies),
        request_id=request_id,
        timestamp=datetime.now(),
    )


@app.get("/api/v1/exchanges", response_model=ExchangeStatusResponse)
async def get_exchange_status():
    """Get status of all configured exchanges (uses cached config)."""
    exchanges_status = []

    # Default exchange definitions
    exchange_defs = {
        "upbit": {"quote": "KRW", "name": "Upbit"},
        "bithumb": {"quote": "KRW", "name": "Bithumb"},
        "binance_spot": {"quote": "USDT", "name": "Binance Spot"},
        "binance_futures": {"quote": "USDT", "name": "Binance Futures"},
    }

    # Use cached config
    config = _load_schedule_config()
    exchanges_config = config.get("exchanges", {})

    for exchange_name, cfg in exchanges_config.items():
        enabled = cfg.get("enabled", False)
        schedule = cfg.get("schedule", {})
        hour = schedule.get("hour", 0)
        minute = schedule.get("minute", 0)
        timezone = schedule.get("timezone", "Asia/Seoul")
        tz_label = "UTC" if timezone == "UTC" else "KST"

        symbols_cfg = cfg.get("symbols", [])
        if isinstance(symbols_cfg, str):
            if symbols_cfg in ("ALL_KRW", "ALL_USDT", "ALL_USDT_PERP"):
                symbols_count = -1  # Dynamic
            else:
                symbols_count = 1
        else:
            symbols_count = len(symbols_cfg)

        quote_currency = exchange_defs.get(exchange_name, {}).get("quote", "KRW")
        if cfg.get("position_size_usdt"):
            quote_currency = "USDT"

        exchanges_status.append(
            ExchangeStatus(
                exchange=exchange_name,
                enabled=enabled,
                connected=enabled,  # Simplified: assume connected if enabled
                quote_currency=quote_currency,
                schedule=f"{hour:02d}:{minute:02d} {tz_label}",
                symbols_count=symbols_count,
            )
        )

    # Add any missing exchanges from defaults
    configured_exchanges = {ex.exchange for ex in exchanges_status}
    for name, defs in exchange_defs.items():
        if name not in configured_exchanges:
            exchanges_status.append(
                ExchangeStatus(
                    exchange=name,
                    enabled=False,
                    connected=False,
                    quote_currency=defs["quote"],
                )
            )

    return ExchangeStatusResponse(
        success=True,
        message="Exchange status retrieved",
        exchanges=exchanges_status,
    )


@app.post("/api/v1/kill-switch", response_model=KillSwitchResponse)
async def kill_switch(request: Request, body: KillSwitchRequest):
    """Emergency kill switch - stops all strategies."""
    from services.api.routes.strategy import strategy_manager as fallback_manager

    request_id = getattr(request.state, "request_id", "unknown")

    if not body.confirm:
        raise HTTPException(status_code=400, detail="Kill switch requires confirm=true")

    logger.critical("[%s] KILL SWITCH ACTIVATED", request_id)

    # Use app state if available, fallback for tests
    if hasattr(request.app.state, "strategy_manager"):
        manager = request.app.state.strategy_manager
    else:
        manager = fallback_manager

    stopped = manager.stop_all()

    return KillSwitchResponse(
        success=True,
        message="Kill switch activated - all strategies stopped",
        positions_closed=0,
        strategies_stopped=stopped,
    )


def start_server():
    """Start the API server with optional SSL/TLS support.

    SSL is configured via environment variables:
        API_SSL_ENABLED=1
        API_SSL_CERTFILE=/path/to/cert.pem
        API_SSL_KEYFILE=/path/to/key.pem
        API_SSL_KEYFILE_PASSWORD=optional_password
    """
    import uvicorn

    # Build uvicorn configuration
    uvicorn_config = {
        "app": "services.api.main:app",
        "host": api_config.host,
        "port": api_config.port,
        "workers": 1,
        "reload": api_config.debug,
    }

    # Add SSL configuration if enabled
    if api_config.is_https:
        uvicorn_config["ssl_certfile"] = api_config.ssl_certfile
        uvicorn_config["ssl_keyfile"] = api_config.ssl_keyfile
        if api_config.ssl_keyfile_password:
            uvicorn_config["ssl_keyfile_password"] = api_config.ssl_keyfile_password
        logger.info(f"[API] Starting with HTTPS on {api_config.base_url}")
    else:
        logger.info(f"[API] Starting with HTTP on {api_config.base_url}")

    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    start_server()
