"""
API key management routes (admin-only).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from libs.core.key_manager import KeyManager
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
