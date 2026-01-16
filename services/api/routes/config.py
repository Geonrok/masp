"""
Config management routes (admin-only).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from libs.core.config_store import ConfigStore, ExchangeConfig
from services.api.middleware.auth import verify_admin_token

router = APIRouter(prefix="/api/v1/config", tags=["config"])

_ALLOWED_EXCHANGES = {"upbit", "bithumb", "binance", "binance_futures"}


def _validate_exchange(name: str) -> None:
    if name not in _ALLOWED_EXCHANGES:
        raise HTTPException(status_code=400, detail=f"Invalid exchange: {name}")


@router.get("/exchanges", dependencies=[Depends(verify_admin_token)])
async def get_exchanges():
    """Get all exchange configs."""
    store = ConfigStore()
    return store.get("exchanges") or {}


@router.get("/exchanges/{name}", dependencies=[Depends(verify_admin_token)])
async def get_exchange(name: str):
    """Get a specific exchange config."""
    _validate_exchange(name)
    store = ConfigStore()
    config = store.get(f"exchanges.{name}")
    if not config:
        raise HTTPException(status_code=404, detail=f"Exchange {name} not found")
    return config


@router.put("/exchanges/{name}", dependencies=[Depends(verify_admin_token)])
async def update_exchange(name: str, config: ExchangeConfig):
    """Update exchange config."""
    _validate_exchange(name)
    store = ConfigStore()
    success = store.set(f"exchanges.{name}", config.model_dump())
    return {"success": success, "exchange": name}


@router.put("/exchanges/{name}/toggle", dependencies=[Depends(verify_admin_token)])
async def toggle_exchange(name: str, enabled: bool):
    """Toggle exchange on/off."""
    _validate_exchange(name)
    store = ConfigStore()
    success = store.set(f"exchanges.{name}.enabled", enabled)
    return {"success": success, "exchange": name, "enabled": enabled}
