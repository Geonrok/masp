"""
FastAPI main application (v2).
"""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.api.config import api_config
from services.api.models.schemas import BaseResponse, KillSwitchRequest, KillSwitchResponse, SystemStatus
from services.api.routes import strategy, positions, trades, health, settings, config, keys
from services.api.websocket.stream import router as ws_router

logger = logging.getLogger(__name__)

_start_time = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[API] Starting MASP API Server v1.0...")
    logger.info("[API] Host: %s:%s", api_config.host, api_config.port)
    logger.info("[API] Mode: Single Worker (state-safe)")
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
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json") and hasattr(response, "body_iterator"):
        try:
            body = b""
            async for chunk in response.body_iterator:
                body += chunk

            data = json.loads(body)
            data["request_id"] = request_id
            data["timestamp"] = datetime.now().isoformat()

            return JSONResponse(
                content=data,
                status_code=response.status_code,
                headers=dict(response.headers),
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("[%s] JSON parse failed: %s", request_id, str(exc))

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
    logger.warning("[%s] HTTPException: %s - %s", request_id, exc.status_code, exc.detail)
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
app.include_router(ws_router, prefix="/ws", tags=["WebSocket"])


@app.get("/api/v1/status", response_model=SystemStatus)
async def get_status():
    from services.api.routes.strategy import strategy_manager

    uptime = (datetime.now() - _start_time).total_seconds()

    return SystemStatus(
        success=True,
        message="MASP API is running",
        version="1.0.0",
        uptime_seconds=uptime,
        exchanges=["upbit", "bithumb", "binance_spot", "binance_futures", "paper"],
        active_strategies=len(strategy_manager.active_strategies),
    )


@app.post("/api/v1/kill-switch", response_model=KillSwitchResponse)
async def kill_switch(request: Request, body: KillSwitchRequest):
    from services.api.routes.strategy import strategy_manager

    request_id = getattr(request.state, "request_id", "unknown")

    if not body.confirm:
        raise HTTPException(status_code=400, detail="Kill switch requires confirm=true")

    logger.critical("[%s] KILL SWITCH ACTIVATED", request_id)

    stopped = strategy_manager.stop_all()

    return KillSwitchResponse(
        success=True,
        message="Kill switch activated - all strategies stopped",
        positions_closed=0,
        strategies_stopped=stopped,
    )


def start_server():
    import uvicorn

    uvicorn.run(
        "services.api.main:app",
        host=api_config.host,
        port=api_config.port,
        workers=1,
        reload=api_config.debug,
    )


if __name__ == "__main__":
    start_server()
