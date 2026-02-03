"""
Tests for RBAC (Role-Based Access Control) System.
"""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from libs.core.rbac import (
    ROLE_PERMISSIONS,
    Permission,
    RBACManager,
    Role,
    Session,
    User,
    get_rbac_manager,
    require_any_permission,
    require_permission,
    require_role,
)


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def rbac_manager(temp_storage_dir):
    """Create test RBAC manager."""
    RBACManager.reset()
    with patch.dict("os.environ", {"MASP_ADMIN_TOKEN": "test_admin_token"}):
        manager = RBACManager(
            storage_path=f"{temp_storage_dir}/rbac_users.json",
            session_timeout_hours=1,
        )
        yield manager
    RBACManager.reset()


class TestPermission:
    """Tests for Permission enum."""

    def test_read_permissions(self):
        """Test read permissions exist."""
        assert Permission.READ_POSITIONS.value == "positions:read"
        assert Permission.READ_TRADES.value == "trades:read"
        assert Permission.READ_BALANCE.value == "balance:read"
        assert Permission.READ_CONFIG.value == "config:read"
        assert Permission.READ_STRATEGIES.value == "strategies:read"

    def test_write_permissions(self):
        """Test write permissions exist."""
        assert Permission.WRITE_CONFIG.value == "config:write"
        assert Permission.WRITE_STRATEGIES.value == "strategies:write"
        assert Permission.WRITE_API_KEYS.value == "apikeys:write"

    def test_execute_permissions(self):
        """Test execute permissions exist."""
        assert Permission.EXECUTE_TRADES.value == "trades:execute"
        assert Permission.EXECUTE_BACKTEST.value == "backtest:execute"
        assert Permission.EXECUTE_KILL_SWITCH.value == "killswitch:execute"

    def test_admin_permissions(self):
        """Test admin permissions exist."""
        assert Permission.ADMIN_USERS.value == "users:admin"
        assert Permission.ADMIN_SYSTEM.value == "system:admin"


class TestRole:
    """Tests for Role enum."""

    def test_role_values(self):
        """Test role values."""
        assert Role.VIEWER.value == "viewer"
        assert Role.ANALYST.value == "analyst"
        assert Role.TRADER.value == "trader"
        assert Role.MANAGER.value == "manager"
        assert Role.ADMIN.value == "admin"
        assert Role.SUPER_ADMIN.value == "super_admin"


class TestRolePermissions:
    """Tests for role to permission mappings."""

    def test_viewer_permissions(self):
        """Test viewer has read-only permissions."""
        perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.READ_POSITIONS in perms
        assert Permission.READ_TRADES in perms
        assert Permission.READ_BALANCE in perms
        assert Permission.WRITE_CONFIG not in perms
        assert Permission.EXECUTE_TRADES not in perms

    def test_analyst_permissions(self):
        """Test analyst permissions include backtest."""
        perms = ROLE_PERMISSIONS[Role.ANALYST]
        assert Permission.READ_BACKTEST in perms
        assert Permission.EXECUTE_BACKTEST in perms
        assert Permission.EXECUTE_TRADES not in perms

    def test_trader_permissions(self):
        """Test trader can execute trades."""
        perms = ROLE_PERMISSIONS[Role.TRADER]
        assert Permission.EXECUTE_TRADES in perms
        assert Permission.EXECUTE_BACKTEST in perms
        assert Permission.WRITE_CONFIG not in perms

    def test_manager_permissions(self):
        """Test manager has management permissions."""
        perms = ROLE_PERMISSIONS[Role.MANAGER]
        assert Permission.WRITE_CONFIG in perms
        assert Permission.WRITE_STRATEGIES in perms
        assert Permission.EXECUTE_KILL_SWITCH in perms
        assert Permission.READ_AUDIT_LOGS in perms

    def test_admin_permissions(self):
        """Test admin has user management."""
        perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.ADMIN_USERS in perms
        assert Permission.WRITE_API_KEYS in perms

    def test_super_admin_has_all_permissions(self):
        """Test super admin has all permissions."""
        perms = ROLE_PERMISSIONS[Role.SUPER_ADMIN]
        assert perms == set(Permission)


class TestUser:
    """Tests for User dataclass."""

    def test_user_creation(self):
        """Test creating a user."""
        user = User(
            user_id="test_user",
            username="Test User",
            roles=[Role.VIEWER],
        )
        assert user.user_id == "test_user"
        assert user.username == "Test User"
        assert Role.VIEWER in user.roles
        assert user.enabled

    def test_get_permissions_from_role(self):
        """Test getting permissions from roles."""
        user = User(
            user_id="test",
            username="Test",
            roles=[Role.VIEWER],
        )
        perms = user.get_permissions()
        assert Permission.READ_POSITIONS in perms
        assert Permission.WRITE_CONFIG not in perms

    def test_get_permissions_multiple_roles(self):
        """Test permissions from multiple roles are combined."""
        user = User(
            user_id="test",
            username="Test",
            roles=[Role.VIEWER, Role.ANALYST],
        )
        perms = user.get_permissions()
        assert Permission.READ_POSITIONS in perms
        assert Permission.EXECUTE_BACKTEST in perms

    def test_additional_permissions(self):
        """Test additional permissions are added."""
        user = User(
            user_id="test",
            username="Test",
            roles=[Role.VIEWER],
            additional_permissions={Permission.WRITE_CONFIG},
        )
        perms = user.get_permissions()
        assert Permission.READ_POSITIONS in perms
        assert Permission.WRITE_CONFIG in perms

    def test_denied_permissions(self):
        """Test denied permissions are removed."""
        user = User(
            user_id="test",
            username="Test",
            roles=[Role.TRADER],
            denied_permissions={Permission.EXECUTE_TRADES},
        )
        perms = user.get_permissions()
        assert Permission.EXECUTE_TRADES not in perms
        assert Permission.READ_POSITIONS in perms

    def test_has_permission(self):
        """Test has_permission method."""
        user = User(
            user_id="test",
            username="Test",
            roles=[Role.VIEWER],
        )
        assert user.has_permission(Permission.READ_POSITIONS)
        assert not user.has_permission(Permission.WRITE_CONFIG)

    def test_disabled_user_has_no_permissions(self):
        """Test disabled user has no permissions."""
        user = User(
            user_id="test",
            username="Test",
            roles=[Role.SUPER_ADMIN],
            enabled=False,
        )
        assert not user.has_permission(Permission.READ_POSITIONS)

    def test_has_any_permission(self):
        """Test has_any_permission method."""
        user = User(
            user_id="test",
            username="Test",
            roles=[Role.VIEWER],
        )
        assert user.has_any_permission(
            [Permission.READ_POSITIONS, Permission.WRITE_CONFIG]
        )
        assert not user.has_any_permission(
            [Permission.WRITE_CONFIG, Permission.ADMIN_USERS]
        )

    def test_has_all_permissions(self):
        """Test has_all_permissions method."""
        user = User(
            user_id="test",
            username="Test",
            roles=[Role.VIEWER],
        )
        assert user.has_all_permissions(
            [Permission.READ_POSITIONS, Permission.READ_TRADES]
        )
        assert not user.has_all_permissions(
            [Permission.READ_POSITIONS, Permission.WRITE_CONFIG]
        )

    def test_has_role(self):
        """Test has_role method."""
        user = User(
            user_id="test",
            username="Test",
            roles=[Role.VIEWER, Role.ANALYST],
        )
        assert user.has_role(Role.VIEWER)
        assert user.has_role(Role.ANALYST)
        assert not user.has_role(Role.ADMIN)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        user = User(
            user_id="test",
            username="Test User",
            roles=[Role.VIEWER],
        )
        d = user.to_dict()
        assert d["user_id"] == "test"
        assert d["username"] == "Test User"
        assert "viewer" in d["roles"]
        assert d["enabled"] is True

    def test_from_dict(self):
        """Test creating user from dictionary."""
        data = {
            "user_id": "test",
            "username": "Test User",
            "roles": ["viewer", "analyst"],
            "additional_permissions": ["config:write"],
            "denied_permissions": [],
            "api_key_hash": "",
            "created_at": "2024-01-01T00:00:00+00:00",
            "last_login": None,
            "enabled": True,
            "metadata": {},
        }
        user = User.from_dict(data)
        assert user.user_id == "test"
        assert user.username == "Test User"
        assert Role.VIEWER in user.roles
        assert Permission.WRITE_CONFIG in user.additional_permissions


class TestSession:
    """Tests for Session dataclass."""

    def test_session_creation(self):
        """Test creating a session."""
        session = Session(
            session_id="sess123",
            user_id="user123",
            ip_address="192.168.1.1",
        )
        assert session.session_id == "sess123"
        assert session.user_id == "user123"

    def test_session_not_expired(self):
        """Test session not expired."""
        session = Session(
            session_id="sess123",
            user_id="user123",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert not session.is_expired()

    def test_session_expired(self):
        """Test session expired."""
        session = Session(
            session_id="sess123",
            user_id="user123",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert session.is_expired()

    def test_session_no_expiry(self):
        """Test session with no expiry."""
        session = Session(
            session_id="sess123",
            user_id="user123",
            expires_at=None,
        )
        assert not session.is_expired()


class TestRBACManager:
    """Tests for RBACManager class."""

    def test_init(self, rbac_manager):
        """Test initialization."""
        assert rbac_manager is not None

    def test_default_admin_created(self, rbac_manager):
        """Test default admin is created."""
        admin = rbac_manager.get_user("admin")
        assert admin is not None
        assert Role.SUPER_ADMIN in admin.roles

    def test_singleton(self, rbac_manager):
        """Test singleton pattern."""
        manager1 = RBACManager.get_instance()
        manager2 = RBACManager.get_instance()
        assert manager1 is manager2

    def test_create_user(self, rbac_manager):
        """Test creating a user."""
        user, api_key = rbac_manager.create_user(
            username="New User",
            roles=[Role.VIEWER],
        )
        assert user.username == "New User"
        assert Role.VIEWER in user.roles
        assert api_key is not None
        assert len(api_key) > 20

    def test_create_user_duplicate_id(self, rbac_manager):
        """Test creating duplicate user fails."""
        rbac_manager.create_user(username="User1", user_id="test_id")
        with pytest.raises(ValueError):
            rbac_manager.create_user(username="User2", user_id="test_id")

    def test_get_user(self, rbac_manager):
        """Test getting a user."""
        user, _ = rbac_manager.create_user(username="Test")
        retrieved = rbac_manager.get_user(user.user_id)
        assert retrieved is not None
        assert retrieved.username == "Test"

    def test_get_user_nonexistent(self, rbac_manager):
        """Test getting nonexistent user."""
        user = rbac_manager.get_user("nonexistent")
        assert user is None

    def test_get_user_by_api_key(self, rbac_manager):
        """Test getting user by API key."""
        user, api_key = rbac_manager.create_user(username="Test")
        retrieved = rbac_manager.get_user_by_api_key(api_key)
        assert retrieved is not None
        assert retrieved.user_id == user.user_id

    def test_list_users(self, rbac_manager):
        """Test listing users."""
        rbac_manager.create_user(username="User1")
        rbac_manager.create_user(username="User2")
        users = rbac_manager.list_users()
        # Admin + 2 new users
        assert len(users) >= 2

    def test_update_user(self, rbac_manager):
        """Test updating user."""
        user, _ = rbac_manager.create_user(username="Original")
        updated = rbac_manager.update_user(
            user_id=user.user_id,
            username="Updated",
            enabled=False,
        )
        assert updated.username == "Updated"
        assert not updated.enabled

    def test_delete_user(self, rbac_manager):
        """Test deleting user."""
        user, _ = rbac_manager.create_user(username="ToDelete")
        success = rbac_manager.delete_user(user.user_id)
        assert success
        assert rbac_manager.get_user(user.user_id) is None

    def test_delete_last_super_admin_fails(self, rbac_manager):
        """Test cannot delete last super admin."""
        with pytest.raises(ValueError):
            rbac_manager.delete_user("admin")

    def test_regenerate_api_key(self, rbac_manager):
        """Test regenerating API key."""
        user, old_api_key = rbac_manager.create_user(username="Test")
        new_api_key = rbac_manager.regenerate_api_key(user.user_id)
        assert new_api_key is not None
        assert new_api_key != old_api_key
        # Old key should not work
        assert rbac_manager.get_user_by_api_key(old_api_key) is None
        # New key should work
        assert rbac_manager.get_user_by_api_key(new_api_key) is not None


class TestRBACManagerRoles:
    """Tests for RBAC role management."""

    def test_add_role(self, rbac_manager):
        """Test adding a role."""
        user, _ = rbac_manager.create_user(username="Test", roles=[Role.VIEWER])
        success = rbac_manager.add_role(user.user_id, Role.ANALYST)
        assert success
        updated = rbac_manager.get_user(user.user_id)
        assert Role.ANALYST in updated.roles

    def test_add_role_duplicate(self, rbac_manager):
        """Test adding duplicate role."""
        user, _ = rbac_manager.create_user(username="Test", roles=[Role.VIEWER])
        success = rbac_manager.add_role(user.user_id, Role.VIEWER)
        assert success
        updated = rbac_manager.get_user(user.user_id)
        assert updated.roles.count(Role.VIEWER) == 1

    def test_remove_role(self, rbac_manager):
        """Test removing a role."""
        user, _ = rbac_manager.create_user(
            username="Test",
            roles=[Role.VIEWER, Role.ANALYST],
        )
        success = rbac_manager.remove_role(user.user_id, Role.ANALYST)
        assert success
        updated = rbac_manager.get_user(user.user_id)
        assert Role.ANALYST not in updated.roles


class TestRBACManagerPermissions:
    """Tests for RBAC permission management."""

    def test_add_permission(self, rbac_manager):
        """Test adding a permission."""
        user, _ = rbac_manager.create_user(username="Test", roles=[Role.VIEWER])
        success = rbac_manager.add_permission(user.user_id, Permission.WRITE_CONFIG)
        assert success
        updated = rbac_manager.get_user(user.user_id)
        assert updated.has_permission(Permission.WRITE_CONFIG)

    def test_remove_permission(self, rbac_manager):
        """Test removing a permission."""
        user, _ = rbac_manager.create_user(
            username="Test",
            roles=[Role.VIEWER],
        )
        rbac_manager.add_permission(user.user_id, Permission.WRITE_CONFIG)
        success = rbac_manager.remove_permission(user.user_id, Permission.WRITE_CONFIG)
        assert success
        updated = rbac_manager.get_user(user.user_id)
        assert not updated.has_permission(Permission.WRITE_CONFIG)

    def test_deny_permission(self, rbac_manager):
        """Test denying a permission."""
        user, _ = rbac_manager.create_user(username="Test", roles=[Role.TRADER])
        success = rbac_manager.deny_permission(user.user_id, Permission.EXECUTE_TRADES)
        assert success
        updated = rbac_manager.get_user(user.user_id)
        assert not updated.has_permission(Permission.EXECUTE_TRADES)

    def test_check_permission(self, rbac_manager):
        """Test check_permission method."""
        user, _ = rbac_manager.create_user(username="Test", roles=[Role.VIEWER])
        assert rbac_manager.check_permission(user.user_id, Permission.READ_POSITIONS)
        assert not rbac_manager.check_permission(user.user_id, Permission.WRITE_CONFIG)

    def test_check_any_permission(self, rbac_manager):
        """Test check_any_permission method."""
        user, _ = rbac_manager.create_user(username="Test", roles=[Role.VIEWER])
        assert rbac_manager.check_any_permission(
            user.user_id,
            [Permission.READ_POSITIONS, Permission.WRITE_CONFIG],
        )

    def test_check_all_permissions(self, rbac_manager):
        """Test check_all_permissions method."""
        user, _ = rbac_manager.create_user(username="Test", roles=[Role.VIEWER])
        assert rbac_manager.check_all_permissions(
            user.user_id,
            [Permission.READ_POSITIONS, Permission.READ_TRADES],
        )
        assert not rbac_manager.check_all_permissions(
            user.user_id,
            [Permission.READ_POSITIONS, Permission.WRITE_CONFIG],
        )


class TestRBACManagerSessions:
    """Tests for RBAC session management."""

    def test_create_session(self, rbac_manager):
        """Test creating a session."""
        user, _ = rbac_manager.create_user(username="Test")
        session = rbac_manager.create_session(
            user=user,
            ip_address="192.168.1.1",
            user_agent="Test Agent",
        )
        assert session is not None
        assert session.user_id == user.user_id
        assert session.ip_address == "192.168.1.1"

    def test_get_session(self, rbac_manager):
        """Test getting a session."""
        user, _ = rbac_manager.create_user(username="Test")
        session = rbac_manager.create_session(user=user)
        retrieved = rbac_manager.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    def test_get_expired_session(self, rbac_manager):
        """Test getting expired session returns None."""
        user, _ = rbac_manager.create_user(username="Test")
        session = rbac_manager.create_session(user=user)
        # Force expire
        session.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        rbac_manager._sessions[session.session_id] = session
        retrieved = rbac_manager.get_session(session.session_id)
        assert retrieved is None

    def test_invalidate_session(self, rbac_manager):
        """Test invalidating a session."""
        user, _ = rbac_manager.create_user(username="Test")
        session = rbac_manager.create_session(user=user)
        success = rbac_manager.invalidate_session(session.session_id)
        assert success
        assert rbac_manager.get_session(session.session_id) is None

    def test_get_user_sessions(self, rbac_manager):
        """Test getting user sessions."""
        user, _ = rbac_manager.create_user(username="Test")
        rbac_manager.create_session(user=user)
        rbac_manager.create_session(user=user)
        sessions = rbac_manager.get_user_sessions(user.user_id)
        assert len(sessions) == 2

    def test_invalidate_user_sessions(self, rbac_manager):
        """Test invalidating all user sessions."""
        user, _ = rbac_manager.create_user(username="Test")
        rbac_manager.create_session(user=user)
        rbac_manager.create_session(user=user)
        count = rbac_manager.invalidate_user_sessions(user.user_id)
        assert count == 2
        assert len(rbac_manager.get_user_sessions(user.user_id)) == 0

    def test_cleanup_expired_sessions(self, rbac_manager):
        """Test cleaning up expired sessions."""
        user, _ = rbac_manager.create_user(username="Test")
        session = rbac_manager.create_session(user=user)
        # Force expire
        session.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        rbac_manager._sessions[session.session_id] = session

        # Create another valid session
        rbac_manager.create_session(user=user)

        count = rbac_manager.cleanup_expired_sessions()
        assert count == 1


class TestRBACPersistence:
    """Tests for RBAC persistence."""

    def test_save_and_load_users(self, temp_storage_dir):
        """Test users are persisted."""
        storage_path = f"{temp_storage_dir}/rbac_users.json"

        # Create manager and add user
        with patch.dict("os.environ", {"MASP_ADMIN_TOKEN": "test_token"}):
            manager1 = RBACManager(storage_path=storage_path)
            user, _ = manager1.create_user(
                username="Persistent User",
                roles=[Role.TRADER],
            )
            user_id = user.user_id

        RBACManager.reset()

        # Create new manager and verify user exists
        with patch.dict("os.environ", {"MASP_ADMIN_TOKEN": "test_token"}):
            manager2 = RBACManager(storage_path=storage_path)
            loaded_user = manager2.get_user(user_id)
            assert loaded_user is not None
            assert loaded_user.username == "Persistent User"
            assert Role.TRADER in loaded_user.roles

        RBACManager.reset()


class TestGetRBACManager:
    """Tests for get_rbac_manager function."""

    def test_returns_singleton(self, rbac_manager):
        """Test get_rbac_manager returns singleton."""
        manager = get_rbac_manager()
        assert manager is rbac_manager
