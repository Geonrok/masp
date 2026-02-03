"""
User management API routes.

Provides endpoints for managing users, roles, and permissions.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from libs.core.rbac import (
    Permission,
    Role,
    User,
    get_rbac_manager,
)
from services.api.middleware.auth import (
    get_current_user,
    require_permission,
)

router = APIRouter()


# Request/Response Models


class CreateUserRequest(BaseModel):
    """Request to create a new user."""

    username: str = Field(..., min_length=3, max_length=50)
    roles: List[str] = Field(default=["viewer"])


class CreateUserResponse(BaseModel):
    """Response for user creation."""

    user_id: str
    username: str
    api_key: str
    roles: List[str]


class UserResponse(BaseModel):
    """User details response."""

    user_id: str
    username: str
    roles: List[str]
    permissions: List[str]
    enabled: bool
    created_at: str
    last_login: Optional[str] = None


class UserListResponse(BaseModel):
    """List of users response."""

    users: List[UserResponse]
    total: int


class UpdateUserRequest(BaseModel):
    """Request to update user details."""

    username: Optional[str] = Field(None, min_length=3, max_length=50)
    roles: Optional[List[str]] = None
    enabled: Optional[bool] = None


class UpdateRoleRequest(BaseModel):
    """Request to update user role."""

    role: str


class UpdatePermissionRequest(BaseModel):
    """Request to update user permission."""

    permission: str
    action: str = Field(..., pattern="^(add|remove|deny)$")


class RegenerateApiKeyResponse(BaseModel):
    """Response for API key regeneration."""

    user_id: str
    api_key: str


class CurrentUserResponse(BaseModel):
    """Current user details."""

    user_id: str
    username: str
    roles: List[str]
    permissions: List[str]


# Endpoints


@router.get("/me", response_model=CurrentUserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
) -> CurrentUserResponse:
    """Get current authenticated user information."""
    return CurrentUserResponse(
        user_id=current_user.user_id,
        username=current_user.username,
        roles=[r.value for r in current_user.roles],
        permissions=[p.value for p in current_user.get_permissions()],
    )


@router.get("", response_model=UserListResponse)
async def list_users(
    current_user: User = Depends(require_permission(Permission.ADMIN_USERS)),
) -> UserListResponse:
    """List all users. Requires ADMIN_USERS permission."""
    rbac = get_rbac_manager()
    users = rbac.list_users()

    user_responses = [
        UserResponse(
            user_id=u.user_id,
            username=u.username,
            roles=[r.value for r in u.roles],
            permissions=[p.value for p in u.get_permissions()],
            enabled=u.enabled,
            created_at=u.created_at.isoformat(),
            last_login=u.last_login.isoformat() if u.last_login else None,
        )
        for u in users
    ]

    return UserListResponse(users=user_responses, total=len(users))


@router.post("", response_model=CreateUserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: CreateUserRequest,
    current_user: User = Depends(require_permission(Permission.ADMIN_USERS)),
) -> CreateUserResponse:
    """Create a new user. Requires ADMIN_USERS permission."""
    rbac = get_rbac_manager()

    # Validate roles
    try:
        roles = [Role(r) for r in request.roles]
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {e}",
        )

    # Check if current user can assign these roles
    # Only super admins can create other super admins
    if Role.SUPER_ADMIN in roles and Role.SUPER_ADMIN not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only super admins can create super admin users",
        )

    try:
        user, api_key = rbac.create_user(
            username=request.username,
            roles=roles,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return CreateUserResponse(
        user_id=user.user_id,
        username=user.username,
        api_key=api_key,
        roles=[r.value for r in user.roles],
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: User = Depends(require_permission(Permission.ADMIN_USERS)),
) -> UserResponse:
    """Get user details. Requires ADMIN_USERS permission."""
    rbac = get_rbac_manager()
    user = rbac.get_user(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}",
        )

    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        roles=[r.value for r in user.roles],
        permissions=[p.value for p in user.get_permissions()],
        enabled=user.enabled,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None,
    )


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    request: UpdateUserRequest,
    current_user: User = Depends(require_permission(Permission.ADMIN_USERS)),
) -> UserResponse:
    """Update user details. Requires ADMIN_USERS permission."""
    rbac = get_rbac_manager()

    # Parse roles if provided
    roles = None
    if request.roles is not None:
        try:
            roles = [Role(r) for r in request.roles]
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {e}",
            )

        # Check permissions for super admin assignment
        if (
            roles
            and Role.SUPER_ADMIN in roles
            and Role.SUPER_ADMIN not in current_user.roles
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only super admins can assign super admin role",
            )

    user = rbac.update_user(
        user_id=user_id,
        username=request.username,
        roles=roles,
        enabled=request.enabled,
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}",
        )

    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        roles=[r.value for r in user.roles],
        permissions=[p.value for p in user.get_permissions()],
        enabled=user.enabled,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None,
    )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    current_user: User = Depends(require_permission(Permission.ADMIN_USERS)),
):
    """Delete a user. Requires ADMIN_USERS permission."""
    rbac = get_rbac_manager()

    # Prevent self-deletion
    if user_id == current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    try:
        success = rbac.delete_user(user_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}",
        )


@router.post("/{user_id}/roles", response_model=UserResponse)
async def add_user_role(
    user_id: str,
    request: UpdateRoleRequest,
    current_user: User = Depends(require_permission(Permission.ADMIN_USERS)),
) -> UserResponse:
    """Add a role to a user. Requires ADMIN_USERS permission."""
    rbac = get_rbac_manager()

    try:
        role = Role(request.role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {request.role}",
        )

    # Check permissions for super admin assignment
    if role == Role.SUPER_ADMIN and Role.SUPER_ADMIN not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only super admins can assign super admin role",
        )

    success = rbac.add_role(user_id, role)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}",
        )

    user = rbac.get_user(user_id)
    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        roles=[r.value for r in user.roles],
        permissions=[p.value for p in user.get_permissions()],
        enabled=user.enabled,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None,
    )


@router.delete("/{user_id}/roles/{role}", response_model=UserResponse)
async def remove_user_role(
    user_id: str,
    role: str,
    current_user: User = Depends(require_permission(Permission.ADMIN_USERS)),
) -> UserResponse:
    """Remove a role from a user. Requires ADMIN_USERS permission."""
    rbac = get_rbac_manager()

    try:
        role_enum = Role(role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {role}",
        )

    success = rbac.remove_role(user_id, role_enum)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}",
        )

    user = rbac.get_user(user_id)
    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        roles=[r.value for r in user.roles],
        permissions=[p.value for p in user.get_permissions()],
        enabled=user.enabled,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None,
    )


@router.post("/{user_id}/permissions", response_model=UserResponse)
async def update_user_permission(
    user_id: str,
    request: UpdatePermissionRequest,
    current_user: User = Depends(require_permission(Permission.ADMIN_USERS)),
) -> UserResponse:
    """
    Update a user's permission.

    Actions:
    - add: Add an additional permission
    - remove: Remove an additional permission
    - deny: Explicitly deny a permission

    Requires ADMIN_USERS permission.
    """
    rbac = get_rbac_manager()

    try:
        permission = Permission(request.permission)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid permission: {request.permission}",
        )

    if request.action == "add":
        success = rbac.add_permission(user_id, permission)
    elif request.action == "remove":
        success = rbac.remove_permission(user_id, permission)
    elif request.action == "deny":
        success = rbac.deny_permission(user_id, permission)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action: {request.action}",
        )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}",
        )

    user = rbac.get_user(user_id)
    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        roles=[r.value for r in user.roles],
        permissions=[p.value for p in user.get_permissions()],
        enabled=user.enabled,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None,
    )


@router.post("/{user_id}/regenerate-api-key", response_model=RegenerateApiKeyResponse)
async def regenerate_user_api_key(
    user_id: str,
    current_user: User = Depends(require_permission(Permission.ADMIN_USERS)),
) -> RegenerateApiKeyResponse:
    """Regenerate API key for a user. Requires ADMIN_USERS permission."""
    rbac = get_rbac_manager()

    api_key = rbac.regenerate_api_key(user_id)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}",
        )

    return RegenerateApiKeyResponse(
        user_id=user_id,
        api_key=api_key,
    )


@router.get("/roles/available", response_model=List[str])
async def list_available_roles(
    current_user: User = Depends(get_current_user),
) -> List[str]:
    """List all available roles."""
    return [r.value for r in Role]


@router.get("/permissions/available", response_model=List[str])
async def list_available_permissions(
    current_user: User = Depends(get_current_user),
) -> List[str]:
    """List all available permissions."""
    return [p.value for p in Permission]
