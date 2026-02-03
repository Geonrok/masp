"""
Role-Based Access Control (RBAC) System.

Provides fine-grained permission control for the MASP platform.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Permission(Enum):
    """Available permissions in the system."""

    # Read permissions
    READ_POSITIONS = "positions:read"
    READ_TRADES = "trades:read"
    READ_BALANCE = "balance:read"
    READ_CONFIG = "config:read"
    READ_STRATEGIES = "strategies:read"
    READ_MONITORING = "monitoring:read"
    READ_AUDIT_LOGS = "audit:read"
    READ_BACKTEST = "backtest:read"

    # Write permissions
    WRITE_CONFIG = "config:write"
    WRITE_STRATEGIES = "strategies:write"
    WRITE_API_KEYS = "apikeys:write"

    # Execute permissions
    EXECUTE_TRADES = "trades:execute"
    EXECUTE_BACKTEST = "backtest:execute"
    EXECUTE_KILL_SWITCH = "killswitch:execute"

    # Admin permissions
    ADMIN_USERS = "users:admin"
    ADMIN_SYSTEM = "system:admin"
    ADMIN_AUDIT = "audit:admin"


class Role(Enum):
    """Predefined roles with associated permissions."""

    VIEWER = "viewer"
    ANALYST = "analyst"
    TRADER = "trader"
    MANAGER = "manager"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.READ_POSITIONS,
        Permission.READ_TRADES,
        Permission.READ_BALANCE,
        Permission.READ_STRATEGIES,
        Permission.READ_MONITORING,
    },
    Role.ANALYST: {
        Permission.READ_POSITIONS,
        Permission.READ_TRADES,
        Permission.READ_BALANCE,
        Permission.READ_CONFIG,
        Permission.READ_STRATEGIES,
        Permission.READ_MONITORING,
        Permission.READ_BACKTEST,
        Permission.EXECUTE_BACKTEST,
    },
    Role.TRADER: {
        Permission.READ_POSITIONS,
        Permission.READ_TRADES,
        Permission.READ_BALANCE,
        Permission.READ_CONFIG,
        Permission.READ_STRATEGIES,
        Permission.READ_MONITORING,
        Permission.READ_BACKTEST,
        Permission.EXECUTE_TRADES,
        Permission.EXECUTE_BACKTEST,
    },
    Role.MANAGER: {
        Permission.READ_POSITIONS,
        Permission.READ_TRADES,
        Permission.READ_BALANCE,
        Permission.READ_CONFIG,
        Permission.READ_STRATEGIES,
        Permission.READ_MONITORING,
        Permission.READ_BACKTEST,
        Permission.READ_AUDIT_LOGS,
        Permission.WRITE_CONFIG,
        Permission.WRITE_STRATEGIES,
        Permission.EXECUTE_TRADES,
        Permission.EXECUTE_BACKTEST,
        Permission.EXECUTE_KILL_SWITCH,
    },
    Role.ADMIN: {
        Permission.READ_POSITIONS,
        Permission.READ_TRADES,
        Permission.READ_BALANCE,
        Permission.READ_CONFIG,
        Permission.READ_STRATEGIES,
        Permission.READ_MONITORING,
        Permission.READ_BACKTEST,
        Permission.READ_AUDIT_LOGS,
        Permission.WRITE_CONFIG,
        Permission.WRITE_STRATEGIES,
        Permission.WRITE_API_KEYS,
        Permission.EXECUTE_TRADES,
        Permission.EXECUTE_BACKTEST,
        Permission.EXECUTE_KILL_SWITCH,
        Permission.ADMIN_USERS,
    },
    Role.SUPER_ADMIN: set(Permission),  # All permissions
}


@dataclass
class User:
    """User with roles and permissions."""

    user_id: str
    username: str
    roles: List[Role] = field(default_factory=list)
    additional_permissions: Set[Permission] = field(default_factory=set)
    denied_permissions: Set[Permission] = field(default_factory=set)
    api_key_hash: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_permissions(self) -> Set[Permission]:
        """Get all effective permissions for this user."""
        permissions: Set[Permission] = set()

        # Add permissions from roles
        for role in self.roles:
            permissions.update(ROLE_PERMISSIONS.get(role, set()))

        # Add additional permissions
        permissions.update(self.additional_permissions)

        # Remove denied permissions
        permissions -= self.denied_permissions

        return permissions

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        if not self.enabled:
            return False
        return permission in self.get_permissions()

    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        user_perms = self.get_permissions()
        return any(p in user_perms for p in permissions)

    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if user has all specified permissions."""
        user_perms = self.get_permissions()
        return all(p in user_perms for p in permissions)

    def has_role(self, role: Role) -> bool:
        """Check if user has a specific role."""
        return role in self.roles

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "roles": [r.value for r in self.roles],
            "additional_permissions": [p.value for p in self.additional_permissions],
            "denied_permissions": [p.value for p in self.denied_permissions],
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            roles=[Role(r) for r in data.get("roles", [])],
            additional_permissions={
                Permission(p) for p in data.get("additional_permissions", [])
            },
            denied_permissions={
                Permission(p) for p in data.get("denied_permissions", [])
            },
            api_key_hash=data.get("api_key_hash", ""),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now(timezone.utc)
            ),
            last_login=(
                datetime.fromisoformat(data["last_login"])
                if data.get("last_login")
                else None
            ),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Session:
    """User session."""

    session_id: str
    user_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    ip_address: str = ""
    user_agent: str = ""

    def is_expired(self) -> bool:
        """Check if session is expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class RBACManager:
    """
    Role-Based Access Control Manager.

    Manages users, roles, and permissions for the MASP platform.
    """

    _instance: Optional["RBACManager"] = None

    def __init__(
        self,
        storage_path: Optional[str] = None,
        session_timeout_hours: int = 24,
    ):
        """
        Initialize RBAC manager.

        Args:
            storage_path: Path to user storage file
            session_timeout_hours: Session timeout in hours
        """
        self._storage_path = Path(
            storage_path or os.getenv("MASP_RBAC_STORAGE", "storage/rbac_users.json")
        )
        self._session_timeout_hours = session_timeout_hours
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Session] = {}
        self._api_key_to_user: Dict[str, str] = {}

        # Load users from storage
        self._load_users()

        # Create default admin if no users exist
        if not self._users:
            self._create_default_admin()

        RBACManager._instance = self

    @classmethod
    def get_instance(cls) -> "RBACManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def _load_users(self) -> None:
        """Load users from storage."""
        if not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for user_data in data.get("users", []):
                user = User.from_dict(user_data)
                self._users[user.user_id] = user
                if user.api_key_hash:
                    self._api_key_to_user[user.api_key_hash] = user.user_id

            logger.info("Loaded %d users from storage", len(self._users))
        except Exception as e:
            logger.error("Failed to load users: %s", e)

    def _save_users(self) -> None:
        """Save users to storage."""
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {"users": [u.to_dict() for u in self._users.values()]}

            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug("Saved %d users to storage", len(self._users))
        except Exception as e:
            logger.error("Failed to save users: %s", e)

    def _create_default_admin(self) -> None:
        """Create default admin user."""
        admin_token = os.getenv("MASP_ADMIN_TOKEN")
        if not admin_token:
            logger.warning("MASP_ADMIN_TOKEN not set, skipping default admin creation")
            return

        api_key_hash = self._hash_api_key(admin_token)

        admin = User(
            user_id="admin",
            username="admin",
            roles=[Role.SUPER_ADMIN],
            api_key_hash=api_key_hash,
        )
        self._users[admin.user_id] = admin
        self._api_key_to_user[api_key_hash] = admin.user_id

        self._save_users()
        logger.info("Created default admin user")

    def _hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def _generate_api_key(self) -> str:
        """Generate a new API key."""
        return secrets.token_urlsafe(32)

    def _generate_session_id(self) -> str:
        """Generate a new session ID."""
        return secrets.token_urlsafe(32)

    # User Management

    def create_user(
        self,
        username: str,
        roles: Optional[List[Role]] = None,
        user_id: Optional[str] = None,
    ) -> tuple[User, str]:
        """
        Create a new user.

        Args:
            username: Username
            roles: List of roles to assign
            user_id: Optional custom user ID

        Returns:
            Tuple of (User, api_key)
        """
        user_id = user_id or secrets.token_hex(8)

        if user_id in self._users:
            raise ValueError(f"User {user_id} already exists")

        api_key = self._generate_api_key()
        api_key_hash = self._hash_api_key(api_key)

        user = User(
            user_id=user_id,
            username=username,
            roles=roles or [Role.VIEWER],
            api_key_hash=api_key_hash,
        )

        self._users[user_id] = user
        self._api_key_to_user[api_key_hash] = user_id

        self._save_users()
        logger.info("Created user: %s", username)

        return user, api_key

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        api_key_hash = self._hash_api_key(api_key)
        user_id = self._api_key_to_user.get(api_key_hash)
        if user_id:
            return self._users.get(user_id)
        return None

    def list_users(self) -> List[User]:
        """List all users."""
        return list(self._users.values())

    def update_user(
        self,
        user_id: str,
        username: Optional[str] = None,
        roles: Optional[List[Role]] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[User]:
        """Update user details."""
        user = self._users.get(user_id)
        if not user:
            return None

        if username is not None:
            user.username = username
        if roles is not None:
            user.roles = roles
        if enabled is not None:
            user.enabled = enabled

        self._save_users()
        logger.info("Updated user: %s", user_id)

        return user

    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        user = self._users.get(user_id)
        if not user:
            return False

        # Don't allow deleting the last admin
        if Role.SUPER_ADMIN in user.roles:
            admin_count = sum(
                1 for u in self._users.values() if Role.SUPER_ADMIN in u.roles
            )
            if admin_count <= 1:
                raise ValueError("Cannot delete the last super admin")

        # Remove from API key mapping
        if user.api_key_hash in self._api_key_to_user:
            del self._api_key_to_user[user.api_key_hash]

        del self._users[user_id]
        self._save_users()
        logger.info("Deleted user: %s", user_id)

        return True

    def regenerate_api_key(self, user_id: str) -> Optional[str]:
        """Regenerate API key for a user."""
        user = self._users.get(user_id)
        if not user:
            return None

        # Remove old API key mapping
        if user.api_key_hash in self._api_key_to_user:
            del self._api_key_to_user[user.api_key_hash]

        # Generate new API key
        api_key = self._generate_api_key()
        api_key_hash = self._hash_api_key(api_key)

        user.api_key_hash = api_key_hash
        self._api_key_to_user[api_key_hash] = user_id

        self._save_users()
        logger.info("Regenerated API key for user: %s", user_id)

        return api_key

    # Role Management

    def add_role(self, user_id: str, role: Role) -> bool:
        """Add a role to a user."""
        user = self._users.get(user_id)
        if not user:
            return False

        if role not in user.roles:
            user.roles.append(role)
            self._save_users()
            logger.info("Added role %s to user %s", role.value, user_id)

        return True

    def remove_role(self, user_id: str, role: Role) -> bool:
        """Remove a role from a user."""
        user = self._users.get(user_id)
        if not user:
            return False

        if role in user.roles:
            user.roles.remove(role)
            self._save_users()
            logger.info("Removed role %s from user %s", role.value, user_id)

        return True

    # Permission Management

    def add_permission(self, user_id: str, permission: Permission) -> bool:
        """Add an additional permission to a user."""
        user = self._users.get(user_id)
        if not user:
            return False

        user.additional_permissions.add(permission)
        user.denied_permissions.discard(permission)
        self._save_users()

        return True

    def remove_permission(self, user_id: str, permission: Permission) -> bool:
        """Remove an additional permission from a user."""
        user = self._users.get(user_id)
        if not user:
            return False

        user.additional_permissions.discard(permission)
        self._save_users()

        return True

    def deny_permission(self, user_id: str, permission: Permission) -> bool:
        """Explicitly deny a permission to a user."""
        user = self._users.get(user_id)
        if not user:
            return False

        user.denied_permissions.add(permission)
        user.additional_permissions.discard(permission)
        self._save_users()

        return True

    # Session Management

    def create_session(
        self,
        user: User,
        ip_address: str = "",
        user_agent: str = "",
    ) -> Session:
        """Create a new session for a user."""
        session_id = self._generate_session_id()

        from datetime import timedelta

        expires_at = datetime.now(timezone.utc) + timedelta(
            hours=self._session_timeout_hours
        )

        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        self._sessions[session_id] = session

        # Update user last login
        user.last_login = datetime.now(timezone.utc)
        self._save_users()

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        session = self._sessions.get(session_id)
        if session and session.is_expired():
            del self._sessions[session_id]
            return None
        return session

    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all active sessions for a user."""
        return [s for s in self._sessions.values() if s.user_id == user_id]

    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user."""
        to_remove = [
            s.session_id for s in self._sessions.values() if s.user_id == user_id
        ]
        for session_id in to_remove:
            del self._sessions[session_id]
        return len(to_remove)

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        expired = [s.session_id for s in self._sessions.values() if s.is_expired()]
        for session_id in expired:
            del self._sessions[session_id]
        return len(expired)

    # Permission Checking

    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if a user has a specific permission."""
        user = self._users.get(user_id)
        if not user:
            return False
        return user.has_permission(permission)

    def check_any_permission(self, user_id: str, permissions: List[Permission]) -> bool:
        """Check if a user has any of the specified permissions."""
        user = self._users.get(user_id)
        if not user:
            return False
        return user.has_any_permission(permissions)

    def check_all_permissions(
        self, user_id: str, permissions: List[Permission]
    ) -> bool:
        """Check if a user has all specified permissions."""
        user = self._users.get(user_id)
        if not user:
            return False
        return user.has_all_permissions(permissions)

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all effective permissions for a user."""
        user = self._users.get(user_id)
        if not user:
            return set()
        return user.get_permissions()


# FastAPI Integration


def get_rbac_manager() -> RBACManager:
    """Get RBAC manager instance."""
    return RBACManager.get_instance()


def require_permission(*permissions: Permission):
    """
    Decorator to require specific permissions for an endpoint.

    Usage:
        @require_permission(Permission.READ_POSITIONS)
        async def get_positions(user: User = Depends(get_current_user)):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Get user from kwargs (injected by FastAPI)
            user = kwargs.get("current_user")
            if not user:
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            # Check permissions
            if not user.has_all_permissions(list(permissions)):
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required permissions: {[p.value for p in permissions]}",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_any_permission(*permissions: Permission):
    """
    Decorator to require any of the specified permissions.

    Usage:
        @require_any_permission(Permission.READ_POSITIONS, Permission.READ_TRADES)
        async def get_data(user: User = Depends(get_current_user)):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            user = kwargs.get("current_user")
            if not user:
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if not user.has_any_permission(list(permissions)):
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requires one of: {[p.value for p in permissions]}",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_role(*roles: Role):
    """
    Decorator to require specific roles.

    Usage:
        @require_role(Role.ADMIN)
        async def admin_action(user: User = Depends(get_current_user)):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            user = kwargs.get("current_user")
            if not user:
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if not any(user.has_role(role) for role in roles):
                from fastapi import HTTPException, status

                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requires one of roles: {[r.value for r in roles]}",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator
