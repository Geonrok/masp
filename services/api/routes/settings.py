"""
Settings API routes for API key management.
Secure storage with HMAC hashing + FileLock.
Phase 3B-v3: Blocking I/O fix + Atomic replace.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException
from filelock import FileLock
from pydantic import BaseModel

from services.api.models.schemas import (
    APIKeyRequest,
    APIKeyResponse,
    APIKeyListResponse,
    BaseResponse,
    ExchangeType,
)

router = APIRouter()

API_KEYS_FILE = Path(os.getenv("API_KEYS_FILE", "logs/api_keys.json"))
API_KEYS_LOCK = API_KEYS_FILE.with_suffix(".lock")
HMAC_SECRET = os.getenv("API_KEY_HMAC_SECRET", "change-this-in-production")

if HMAC_SECRET == "change-this-in-production":
    warnings.warn(
        "⚠️  API_KEY_HMAC_SECRET is using default value. "
        "Set API_KEY_HMAC_SECRET environment variable for production!",
        UserWarning,
    )

logger = logging.getLogger(__name__)


def _hash_key(key: str, salt: str) -> str:
    """HMAC-SHA256 hash."""
    return hmac.new(
        HMAC_SECRET.encode(),
        f"{salt}:{key}".encode(),
        hashlib.sha256,
    ).hexdigest()


def _mask_key(key: str) -> str:
    """Mask API key (first3 + *** + last3)."""
    if len(key) <= 6:
        return "***"
    return f"{key[:3]}***{key[-3:]}"


def _load_keys() -> Dict:
    """Load keys from file (with lock)."""
    with FileLock(API_KEYS_LOCK, timeout=5):
        if not API_KEYS_FILE.exists():
            return {}
        try:
            with API_KEYS_FILE.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}


def _save_keys(data: Dict) -> None:
    """Save keys to file (atomic replace with lock)."""
    API_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(API_KEYS_LOCK, timeout=5):
        fd, tmp_path = tempfile.mkstemp(
            dir=API_KEYS_FILE.parent,
            prefix=".api_keys_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, default=str)
            os.replace(tmp_path, API_KEYS_FILE)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise


@router.post("/api-keys", response_model=APIKeyResponse)
def save_api_key(request: APIKeyRequest):
    """Save API key (hash + persistence)."""
    keys = _load_keys()
    salt = secrets.token_hex(16)

    keys[request.exchange.value] = {
        "api_key_hash": _hash_key(request.api_key, salt),
        "secret_key_hash": _hash_key(request.secret_key, salt),
        "salt": salt,
        "masked_api_key": _mask_key(request.api_key),
        "created_at": datetime.now().isoformat(),
    }

    _save_keys(keys)

    return APIKeyResponse(
        success=True,
        message=f"API key saved for {request.exchange.value}",
        exchange=request.exchange,
        masked_key=_mask_key(request.api_key),
        created_at=datetime.now(),
    )


@router.get("/api-keys", response_model=APIKeyListResponse)
def list_api_keys():
    """Return stored API keys (masked only)."""
    keys = _load_keys()
    result = []

    for exchange, data in keys.items():
        try:
            exchange_type = ExchangeType(exchange)
        except ValueError:
            continue

        created_raw = data.get("created_at")
        try:
            created_at = datetime.fromisoformat(created_raw) if created_raw else datetime.now()
        except ValueError:
            created_at = datetime.now()

        result.append(
            APIKeyResponse(
                success=True,
                exchange=exchange_type,
                masked_key=data.get("masked_api_key", "***"),
                created_at=created_at,
            )
        )

    return APIKeyListResponse(
        success=True,
        message=f"Found {len(result)} API keys",
        keys=result,
    )


@router.delete("/api-keys/{exchange}", response_model=BaseResponse)
def delete_api_key(exchange: ExchangeType):
    """Delete API key."""
    keys = _load_keys()

    if exchange.value not in keys:
        raise HTTPException(status_code=404, detail="API key not found")

    del keys[exchange.value]
    _save_keys(keys)

    return BaseResponse(
        success=True,
        message=f"API key deleted for {exchange.value}",
    )


class VerifyKeyRequest(BaseModel):
    """Request body for API key verification."""

    exchange: ExchangeType
    api_key: str
    secret_key: str


@router.post("/exchanges/verify", response_model=BaseResponse)
def verify_api_key(request: VerifyKeyRequest):
    """Verify API key (hash compare). POST prevents query param exposure."""
    keys = _load_keys()

    if request.exchange.value not in keys:
        logger.warning("[VERIFY] Invalid exchange attempt: %s", request.exchange.value)
        raise HTTPException(status_code=401, detail="Verification failed")

    stored = keys[request.exchange.value]
    salt = stored["salt"]

    api_match = hmac.compare_digest(
        _hash_key(request.api_key, salt),
        stored["api_key_hash"],
    )
    secret_match = hmac.compare_digest(
        _hash_key(request.secret_key, salt),
        stored["secret_key_hash"],
    )

    if api_match and secret_match:
        return BaseResponse(success=True, message="API key verified")

    logger.warning("[VERIFY] Invalid credentials for: %s", request.exchange.value)
    raise HTTPException(status_code=401, detail="Verification failed")
