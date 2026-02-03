"""
Tests for User management API routes.
"""

import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from libs.core.rbac import (
    Permission,
    RBACManager,
    Role,
    User,
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


@pytest.fixture
def test_client(rbac_manager):
    """Create test client with mocked RBAC."""
    from services.api.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def admin_headers():
    """Headers with admin token."""
    return {"X-MASP-ADMIN-TOKEN": "test_admin_token"}


@pytest.fixture
def viewer_user(rbac_manager):
    """Create a viewer user."""
    user, api_key = rbac_manager.create_user(
        username="Viewer User",
        roles=[Role.VIEWER],
    )
    return user, api_key


class TestGetCurrentUser:
    """Tests for /users/me endpoint."""

    def test_get_current_user_with_admin_token(self, test_client, admin_headers):
        """Test getting current user with admin token."""
        response = test_client.get("/api/v1/users/me", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "admin"
        assert "super_admin" in data["roles"]

    def test_get_current_user_with_api_key(self, test_client, viewer_user):
        """Test getting current user with API key."""
        user, api_key = viewer_user
        headers = {"X-MASP-API-KEY": api_key}
        response = test_client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == user.user_id
        assert "viewer" in data["roles"]

    def test_get_current_user_unauthorized(self, test_client):
        """Test getting current user without auth."""
        response = test_client.get("/api/v1/users/me")
        assert response.status_code == 401


class TestListUsers:
    """Tests for GET /users endpoint."""

    def test_list_users_as_admin(self, test_client, admin_headers, rbac_manager):
        """Test listing users as admin."""
        rbac_manager.create_user(username="User1")
        rbac_manager.create_user(username="User2")

        response = test_client.get("/api/v1/users", headers=admin_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 3  # admin + 2 users

    def test_list_users_as_viewer_forbidden(self, test_client, viewer_user):
        """Test listing users as viewer is forbidden."""
        _, api_key = viewer_user
        headers = {"X-MASP-API-KEY": api_key}
        response = test_client.get("/api/v1/users", headers=headers)
        assert response.status_code == 403


class TestCreateUser:
    """Tests for POST /users endpoint."""

    def test_create_user_as_admin(self, test_client, admin_headers):
        """Test creating user as admin."""
        response = test_client.post(
            "/api/v1/users",
            headers=admin_headers,
            json={
                "username": "New User",
                "roles": ["viewer", "analyst"],
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["username"] == "New User"
        assert "viewer" in data["roles"]
        assert "analyst" in data["roles"]
        assert "api_key" in data

    def test_create_user_invalid_role(self, test_client, admin_headers):
        """Test creating user with invalid role."""
        response = test_client.post(
            "/api/v1/users",
            headers=admin_headers,
            json={
                "username": "New User",
                "roles": ["invalid_role"],
            },
        )
        assert response.status_code == 400

    def test_create_super_admin_as_admin(self, test_client, rbac_manager):
        """Test creating super admin requires super admin."""
        # Create a regular admin
        user, api_key = rbac_manager.create_user(
            username="Regular Admin",
            roles=[Role.ADMIN],
        )
        headers = {"X-MASP-API-KEY": api_key}

        response = test_client.post(
            "/api/v1/users",
            headers=headers,
            json={
                "username": "New Super Admin",
                "roles": ["super_admin"],
            },
        )
        assert response.status_code == 403


class TestGetUser:
    """Tests for GET /users/{user_id} endpoint."""

    def test_get_user_as_admin(self, test_client, admin_headers, rbac_manager):
        """Test getting user details as admin."""
        user, _ = rbac_manager.create_user(username="Target User")

        response = test_client.get(
            f"/api/v1/users/{user.user_id}",
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "Target User"

    def test_get_user_not_found(self, test_client, admin_headers):
        """Test getting nonexistent user."""
        response = test_client.get(
            "/api/v1/users/nonexistent",
            headers=admin_headers,
        )
        assert response.status_code == 404


class TestUpdateUser:
    """Tests for PATCH /users/{user_id} endpoint."""

    def test_update_user_username(self, test_client, admin_headers, rbac_manager):
        """Test updating user username."""
        user, _ = rbac_manager.create_user(username="Original")

        response = test_client.patch(
            f"/api/v1/users/{user.user_id}",
            headers=admin_headers,
            json={"username": "Updated"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "Updated"

    def test_update_user_roles(self, test_client, admin_headers, rbac_manager):
        """Test updating user roles."""
        user, _ = rbac_manager.create_user(username="Test", roles=[Role.VIEWER])

        response = test_client.patch(
            f"/api/v1/users/{user.user_id}",
            headers=admin_headers,
            json={"roles": ["trader"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "trader" in data["roles"]

    def test_disable_user(self, test_client, admin_headers, rbac_manager):
        """Test disabling a user."""
        user, _ = rbac_manager.create_user(username="Test")

        response = test_client.patch(
            f"/api/v1/users/{user.user_id}",
            headers=admin_headers,
            json={"enabled": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False


class TestDeleteUser:
    """Tests for DELETE /users/{user_id} endpoint."""

    def test_delete_user(self, test_client, admin_headers, rbac_manager):
        """Test deleting a user."""
        user, _ = rbac_manager.create_user(username="ToDelete")

        response = test_client.delete(
            f"/api/v1/users/{user.user_id}",
            headers=admin_headers,
        )
        assert response.status_code == 204

    def test_delete_self_forbidden(self, test_client, admin_headers):
        """Test cannot delete own account."""
        response = test_client.delete(
            "/api/v1/users/admin",
            headers=admin_headers,
        )
        assert response.status_code == 400


class TestRoleManagement:
    """Tests for role management endpoints."""

    def test_add_role(self, test_client, admin_headers, rbac_manager):
        """Test adding a role to user."""
        user, _ = rbac_manager.create_user(username="Test", roles=[Role.VIEWER])

        response = test_client.post(
            f"/api/v1/users/{user.user_id}/roles",
            headers=admin_headers,
            json={"role": "analyst"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "analyst" in data["roles"]

    def test_remove_role(self, test_client, admin_headers, rbac_manager):
        """Test removing a role from user."""
        user, _ = rbac_manager.create_user(
            username="Test",
            roles=[Role.VIEWER, Role.ANALYST],
        )

        response = test_client.delete(
            f"/api/v1/users/{user.user_id}/roles/analyst",
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "analyst" not in data["roles"]


class TestPermissionManagement:
    """Tests for permission management endpoints."""

    def test_add_permission(self, test_client, admin_headers, rbac_manager):
        """Test adding a permission to user."""
        user, _ = rbac_manager.create_user(username="Test", roles=[Role.VIEWER])

        response = test_client.post(
            f"/api/v1/users/{user.user_id}/permissions",
            headers=admin_headers,
            json={"permission": "config:write", "action": "add"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "config:write" in data["permissions"]

    def test_deny_permission(self, test_client, admin_headers, rbac_manager):
        """Test denying a permission."""
        user, _ = rbac_manager.create_user(username="Test", roles=[Role.TRADER])

        response = test_client.post(
            f"/api/v1/users/{user.user_id}/permissions",
            headers=admin_headers,
            json={"permission": "trades:execute", "action": "deny"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "trades:execute" not in data["permissions"]


class TestRegenerateApiKey:
    """Tests for API key regeneration."""

    def test_regenerate_api_key(self, test_client, admin_headers, rbac_manager):
        """Test regenerating API key."""
        user, old_api_key = rbac_manager.create_user(username="Test")

        response = test_client.post(
            f"/api/v1/users/{user.user_id}/regenerate-api-key",
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "api_key" in data
        assert data["api_key"] != old_api_key


class TestListRolesAndPermissions:
    """Tests for listing available roles and permissions."""

    def test_list_available_roles(self, test_client, admin_headers):
        """Test listing available roles."""
        response = test_client.get(
            "/api/v1/users/roles/available",
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "viewer" in data
        assert "admin" in data
        assert "super_admin" in data

    def test_list_available_permissions(self, test_client, admin_headers):
        """Test listing available permissions."""
        response = test_client.get(
            "/api/v1/users/permissions/available",
            headers=admin_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "positions:read" in data
        assert "config:write" in data
        assert "users:admin" in data
