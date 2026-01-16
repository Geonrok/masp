"""
Encrypted API key manager using Fernet.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from cryptography.fernet import Fernet, InvalidToken
from filelock import FileLock

logger = logging.getLogger(__name__)


class KeyManager:
    """
    API key encryption manager.
    - Fernet symmetric encryption
    - master key policy for prod/dev
    - masked returns only
    """

    ALLOWED_EXCHANGES = {"upbit", "bithumb", "binance", "binance_futures"}

    def __init__(self, storage_path: str = "storage/encrypted_keys.json"):
        self._path = Path(storage_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(f"{storage_path}.lock")
        self._fernet = self._init_fernet()
        self._ensure_file()

    def _init_fernet(self) -> Fernet:
        """Initialize master key."""
        master_key = os.getenv("MASP_MASTER_KEY")

        if master_key:
            logger.info("[KeyManager] Using master key from environment")
            return Fernet(master_key.encode())

        if os.getenv("MASP_ALLOW_AUTOGEN_MASTER_KEY") == "1":
            master_key = self._load_or_generate_master_key()
            return Fernet(master_key.encode())

        raise RuntimeError(
            "[KeyManager] MASP_MASTER_KEY not set. "
            "For development, set MASP_ALLOW_AUTOGEN_MASTER_KEY=1"
        )

    def _load_or_generate_master_key(self) -> str:
        """Load or generate master key (dev mode only)."""
        key_path = Path("storage/master.key")

        if key_path.exists():
            logger.info("[KeyManager] Loading existing master key")
            return key_path.read_text().strip()

        logger.warning("[KeyManager] Generating new master key (dev mode)")
        new_key = Fernet.generate_key().decode()

        key_path.write_text(new_key)
        try:
            key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass

        return new_key

    def _validate_exchange(self, exchange: str) -> None:
        if exchange not in self.ALLOWED_EXCHANGES:
            raise ValueError(
                f"Invalid exchange: {exchange}. Allowed: {self.ALLOWED_EXCHANGES}"
            )

    def _ensure_file(self) -> None:
        if self._path.exists() and self._path.stat().st_size > 0:
            return
        data = {"schema_version": 1, "keys": {}}
        self._save_keys(data)

    def _load_keys(self) -> Dict[str, dict]:
        with self._lock:
            try:
                raw = self._path.read_text(encoding="utf-8")
                data = json.loads(raw)
                if isinstance(data, dict):
                    data.setdefault("keys", {})
                    data.setdefault("schema_version", 1)
                    return data
            except Exception as exc:
                logger.warning("[KeyManager] Load failed: %s", exc)
        return {"schema_version": 1, "keys": {}}

    def _save_keys(self, data: Dict[str, dict]) -> None:
        with self._lock:
            temp_path = self._path.with_suffix(".tmp")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                temp_path.replace(self._path)
            except Exception as exc:
                logger.error("[KeyManager] Save failed: %s", exc)
                if temp_path.exists():
                    temp_path.unlink()
                raise

    def store_key(self, exchange: str, api_key: str, secret_key: str) -> bool:
        """Encrypt and store API keys."""
        self._validate_exchange(exchange)

        try:
            data = self._load_keys()
            data["keys"][exchange] = {
                "api_key": self._fernet.encrypt(api_key.encode()).decode(),
                "secret_key": self._fernet.encrypt(secret_key.encode()).decode(),
                "updated_at": datetime.now().isoformat(),
            }
            self._save_keys(data)

            masked = api_key[:8] + "..." if len(api_key) > 8 else "***"
            logger.info(
                "[KeyManager] Key stored: exchange=%s, key=%s",
                exchange,
                masked,
            )
            return True
        except Exception as exc:
            logger.error("[KeyManager] Store failed: %s", exc)
            return False

    def get_keys(self) -> Dict[str, dict]:
        """Get all keys with masking."""
        data = self._load_keys()
        result: Dict[str, dict] = {}

        for exchange, key_data in data.get("keys", {}).items():
            try:
                decrypted_api = self._fernet.decrypt(
                    key_data["api_key"].encode()
                ).decode()
                result[exchange] = {
                    "api_key": decrypted_api[:8] + "..."
                    if len(decrypted_api) > 8
                    else "***",
                    "has_secret": bool(key_data.get("secret_key")),
                    "updated_at": key_data.get("updated_at", ""),
                }
            except InvalidToken:
                result[exchange] = {
                    "api_key": "[DECRYPT_ERROR]",
                    "has_secret": False,
                }

        return result

    def get_raw_key(self, exchange: str) -> Optional[dict]:
        """Get decrypted key (internal use only)."""
        self._validate_exchange(exchange)
        data = self._load_keys()
        key_data = data.get("keys", {}).get(exchange)

        if not key_data:
            return None

        try:
            return {
                "api_key": self._fernet.decrypt(
                    key_data["api_key"].encode()
                ).decode(),
                "secret_key": self._fernet.decrypt(
                    key_data["secret_key"].encode()
                ).decode(),
            }
        except InvalidToken:
            return None

    def delete_key(self, exchange: str) -> bool:
        """Delete a key."""
        self._validate_exchange(exchange)
        data = self._load_keys()

        if exchange in data.get("keys", {}):
            del data["keys"][exchange]
            self._save_keys(data)
            logger.info("[KeyManager] Key deleted: %s", exchange)
            return True
        return False

    def get_key_with_env_fallback(self, exchange: str) -> Optional[dict]:
        """
        Fallback priority: encrypted store > .env.
        """
        raw = self.get_raw_key(exchange)
        if raw:
            return raw

        env_prefix = exchange.upper()
        api_key = os.getenv(f"{env_prefix}_API_KEY") or os.getenv(
            f"{env_prefix}_ACCESS_KEY"
        )
        secret_key = os.getenv(f"{env_prefix}_SECRET_KEY")

        if api_key and secret_key:
            logger.info("[KeyManager] Using .env fallback for %s", exchange)
            return {"api_key": api_key, "secret_key": secret_key}

        return None
