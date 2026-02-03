"""
Authentication and authorization middleware.

Integrates with RBAC system for fine-grained access control.
"""

from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from libs.core.rbac import (
    Permission,
    Role,
    User,
    get_rbac_manager,
)

# API Key header for authentication
api_key_header = APIKeyHeader(name="X-MASP-API-KEY", auto_error=False)

# Legacy admin token header (for backwards compatibility)
admin_token_header = APIKeyHeader(name="X-MASP-ADMIN-TOKEN", auto_error=False)


async def get_current_user(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
    admin_token: Optional[str] = Depends(admin_token_header),
) -> User:
    """
    Get the current authenticated user.

    Supports both new API key authentication and legacy admin token.

    Args:
        request: FastAPI request object
        api_key: API key from X-MASP-API-KEY header
        admin_token: Legacy admin token from X-MASP-ADMIN-TOKEN header

    Returns:
        Authenticated User object

    Raises:
        HTTPException: If authentication fails
    """
    rbac = get_rbac_manager()

    # Try new API key authentication first
    if api_key:
        user = rbac.get_user_by_api_key(api_key)
        if user and user.enabled:
            return user

    # Fall back to legacy admin token
    if admin_token:
        expected_token = os.getenv("MASP_ADMIN_TOKEN")
        if expected_token and secrets.compare_digest(admin_token, expected_token):
            # Return admin user for legacy token
            admin_user = rbac.get_user("admin")
            if admin_user:
                return admin_user

            # Create temporary admin user if not in RBAC
            return User(
                user_id="admin",
                username="admin",
                roles=[Role.SUPER_ADMIN],
            )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication credentials",
        headers={"WWW-Authenticate": "ApiKey"},
    )


async def get_optional_user(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
    admin_token: Optional[str] = Depends(admin_token_header),
) -> Optional[User]:
    """
    Get the current user if authenticated, None otherwise.

    Use this for endpoints that have optional authentication.
    """
    try:
        return await get_current_user(request, api_key, admin_token)
    except HTTPException:
        return None


async def verify_admin_token(token: str = Depends(admin_token_header)) -> bool:
    """
    Verify admin token from header.

    This is the legacy authentication method. Prefer using get_current_user
    with RBAC for new endpoints.
    """
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


def require_permission(*permissions: Permission):
    """
    Dependency factory that requires specific permissions.

    Usage:
        @router.get("/data")
        async def get_data(
            _: bool = Depends(require_permission(Permission.READ_POSITIONS))
        ):
            ...
    """

    async def check_permissions(
        current_user: User = Depends(get_current_user),
    ) -> User:
        if not current_user.has_all_permissions(list(permissions)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {[p.value for p in permissions]}",
            )
        return current_user

    return check_permissions


def require_any_permission(*permissions: Permission):
    """
    Dependency factory that requires any of the specified permissions.

    Usage:
        @router.get("/data")
        async def get_data(
            _: bool = Depends(require_any_permission(
                Permission.READ_POSITIONS,
                Permission.READ_TRADES
            ))
        ):
            ...
    """

    async def check_permissions(
        current_user: User = Depends(get_current_user),
    ) -> User:
        if not current_user.has_any_permission(list(permissions)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {[p.value for p in permissions]}",
            )
        return current_user

    return check_permissions


def require_role(*roles: Role):
    """
    Dependency factory that requires specific roles.

    Usage:
        @router.post("/admin")
        async def admin_action(
            _: bool = Depends(require_role(Role.ADMIN))
        ):
            ...
    """

    async def check_role(
        current_user: User = Depends(get_current_user),
    ) -> User:
        if not any(current_user.has_role(role) for role in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {[r.value for r in roles]}",
            )
        return current_user

    return check_role


# Re-export RBAC components for convenience
__all__ = [
    "get_current_user",
    "get_optional_user",
    "verify_admin_token",
    "require_permission",
    "require_any_permission",
    "require_role",
    "Permission",
    "Role",
    "User",
]
