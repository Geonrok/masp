"""
API key management routes (admin-only).
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from libs.core.key_manager import KeyManager
from libs.core.startup_validator import (
    check_exchange_ready,
    get_missing_keys,
    validate_api_keys,
)
from services.api.middleware.auth import verify_admin_token

router = APIRouter(prefix="/api/v1/keys", tags=["keys"])


class KeyInput(BaseModel):
    api_key: str
    secret_key: str


@router.get("", dependencies=[Depends(verify_admin_token)])
async def list_keys():
    """List all API keys (masked)."""
    km = KeyManager()
    return km.get_keys()


@router.post("/{exchange}", dependencies=[Depends(verify_admin_token)])
async def store_key(exchange: str, key_input: KeyInput):
    """Store API keys for an exchange."""
    km = KeyManager()
    success = km.store_key(exchange, key_input.api_key, key_input.secret_key)
    return {"success": success, "exchange": exchange}


@router.delete("/{exchange}", dependencies=[Depends(verify_admin_token)])
async def delete_key(exchange: str):
    """Delete API keys for an exchange."""
    km = KeyManager()
    success = km.delete_key(exchange)
    if not success:
        raise HTTPException(status_code=404, detail=f"Key for {exchange} not found")
    return {"success": True, "exchange": exchange}


class KeyValidationResponse(BaseModel):
    """Response model for key validation."""

    is_valid: bool
    exchanges_validated: int
    exchanges_valid: int
    exchanges_invalid: int
    errors: List[str]
    warnings: List[str]
    exchange_details: dict


class ExchangeReadyResponse(BaseModel):
    """Response for exchange readiness check."""

    exchange: str
    ready: bool
    missing_keys: List[str]


@router.get("/validate", dependencies=[Depends(verify_admin_token)])
async def validate_keys() -> KeyValidationResponse:
    """
    Validate API keys for all enabled exchanges.

    Checks that all required API keys are configured and valid
    for exchanges enabled in schedule_config.json.
    """
    result = validate_api_keys()

    exchange_details = {}
    for ex_result in result.exchange_results:
        exchange_details[ex_result.exchange] = {
            "enabled": ex_result.enabled,
            "is_valid": ex_result.is_valid,
            "errors": ex_result.errors,
            "warnings": ex_result.warnings,
            "keys_checked": [kr.key_name for kr in ex_result.key_results],
        }

    return KeyValidationResponse(
        is_valid=result.is_valid,
        exchanges_validated=result.exchanges_validated,
        exchanges_valid=result.exchanges_valid,
        exchanges_invalid=result.exchanges_invalid,
        errors=result.errors,
        warnings=result.warnings,
        exchange_details=exchange_details,
    )


@router.get("/validate/{exchange}", dependencies=[Depends(verify_admin_token)])
async def validate_exchange_keys(exchange: str) -> ExchangeReadyResponse:
    """
    Check if a specific exchange has valid API keys configured.

    Args:
        exchange: Exchange name (upbit, bithumb, binance_spot, etc.)
    """
    ready = check_exchange_ready(exchange)
    missing = get_missing_keys(exchange) if not ready else []

    return ExchangeReadyResponse(
        exchange=exchange,
        ready=ready,
        missing_keys=missing,
    )
