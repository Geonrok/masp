"""Config/keys API client for the dashboard."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv(override=False)
logger = logging.getLogger(__name__)


class ConfigApiClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("MASP_API_BASE_URL", "http://localhost:8000")
        self.token = os.getenv("MASP_ADMIN_TOKEN", "")
        self.headers = {"X-MASP-ADMIN-TOKEN": self.token}

        if not self.token:
            logger.warning("[ConfigApiClient] MASP_ADMIN_TOKEN not set")

    def get_exchange_config(self, exchange: str) -> Optional[Dict[str, Any]]:
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/config/exchanges/{exchange}",
                headers=self.headers,
                timeout=5,
            )
            return resp.json() if resp.ok else None
        except Exception as exc:
            logger.error("[ConfigApiClient] get_exchange_config failed: %s", exc)
            return None

    def update_exchange_config(self, exchange: str, updates: Dict[str, Any]) -> bool:
        """Fetch current config, deep-merge updates, and PUT the full config."""
        try:
            existing = self.get_exchange_config(exchange)
            if not isinstance(existing, dict):
                logger.error("[ConfigApiClient] Failed to get existing config")
                return False

            def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
                for key, value in source.items():
                    if (
                        isinstance(value, dict)
                        and key in target
                        and isinstance(target[key], dict)
                    ):
                        deep_merge(target[key], value)
                    else:
                        target[key] = value

            merged = dict(existing)
            deep_merge(merged, updates)

            resp = requests.put(
                f"{self.base_url}/api/v1/config/exchanges/{exchange}",
                headers=self.headers,
                json=merged,
                timeout=5,
            )
            return resp.ok
        except Exception as exc:
            logger.error("[ConfigApiClient] update_exchange_config failed: %s", exc)
            return False

    def toggle_exchange(self, exchange: str, enabled: bool) -> bool:
        try:
            resp = requests.put(
                f"{self.base_url}/api/v1/config/exchanges/{exchange}/toggle",
                headers=self.headers,
                params={"enabled": enabled},
                timeout=5,
            )
            return resp.ok
        except Exception as exc:
            logger.error("[ConfigApiClient] toggle_exchange failed: %s", exc)
            return False

    def get_all_keys(self) -> Dict[str, Any]:
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/keys",
                headers=self.headers,
                timeout=5,
            )
            if not resp.ok:
                return {}
            data = resp.json()
            if isinstance(data, dict):
                keys = data.get("keys")
                return keys if isinstance(keys, dict) else data
            return {}
        except Exception as exc:
            logger.error("[ConfigApiClient] get_all_keys failed: %s", exc)
            return {}
