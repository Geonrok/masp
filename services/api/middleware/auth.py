"""
Admin token authentication middleware.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-MASP-ADMIN-TOKEN", auto_error=False)


async def verify_admin_token(token: str = Depends(api_key_header)) -> bool:
    """Verify admin token from header."""
    expected_token = os.getenv("MASP_ADMIN_TOKEN")

    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MASP_ADMIN_TOKEN not configured",
        )
    if expected_token.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MASP_ADMIN_TOKEN cannot be empty",
        )

    if not secrets.compare_digest(token or "", expected_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing admin token",
        )

    return True
