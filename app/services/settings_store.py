# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.utils import get_app_directory, logger

try:
    # Werkzeug is a Flask dependency.
    from werkzeug.security import check_password_hash, generate_password_hash
except Exception:  # pragma: no cover
    check_password_hash = None  # type: ignore
    generate_password_hash = None  # type: ignore


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SettingsPaths:
    config_dir: str
    settings_json: str


class SettingsStore:
    """Small JSON settings loader/saver with thread safety.

    Note: `config/settings.json` is runtime state and should not be committed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paths = self._compute_paths()

    @staticmethod
    def _compute_paths() -> SettingsPaths:
        config_dir = os.path.join(get_app_directory(), "config")
        return SettingsPaths(config_dir=config_dir, settings_json=os.path.join(config_dir, "settings.json"))

    @property
    def paths(self) -> SettingsPaths:
        return self._paths

    def ensure_exists(self) -> None:
        """Create a minimal settings file if missing."""
        with self._lock:
            os.makedirs(self._paths.config_dir, exist_ok=True)
            if os.path.exists(self._paths.settings_json):
                return
            default_settings: Dict[str, Any] = {
                "folder": "",
                "offline_mode": False,
                "local_model_path": "",
                "admin_password_hash": "",
                "server_port": 8080,
                "embed_backend": "torch",
                "embed_normalize": True,
            }
            try:
                with open(self._paths.settings_json, "w", encoding="utf-8") as f:
                    json.dump(default_settings, f, ensure_ascii=False, indent=2)
                logger.info(f"기본 설정 파일 생성: {self._paths.settings_json}")
            except Exception as e:
                logger.warning(f"기본 설정 파일 생성 실패: {e}")

    def load(self) -> Dict[str, Any]:
        with self._lock:
            try:
                if not os.path.exists(self._paths.settings_json):
                    return {}
                with open(self._paths.settings_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning(f"설정 파일 로드 실패: {e}")
            return {}

    def save(self, settings: Dict[str, Any]) -> bool:
        with self._lock:
            try:
                os.makedirs(self._paths.config_dir, exist_ok=True)
                with open(self._paths.settings_json, "w", encoding="utf-8") as f:
                    json.dump(settings, f, ensure_ascii=False, indent=2)
                return True
            except Exception as e:
                logger.warning(f"설정 파일 저장 실패: {e}")
                return False


_settings_store_singleton: Optional[SettingsStore] = None


def get_settings_store() -> SettingsStore:
    global _settings_store_singleton
    if _settings_store_singleton is None:
        _settings_store_singleton = SettingsStore()
    return _settings_store_singleton


def get_admin_password_hash() -> str:
    # Highest priority: explicit env var to avoid storing secrets on disk.
    env_hash = os.environ.get("ADMIN_PASSWORD_HASH", "").strip()
    if env_hash:
        return env_hash

    # Convenience: allow plain password via env var.
    env_pw = os.environ.get("ADMIN_PASSWORD", "")
    if env_pw:
        if generate_password_hash is not None:
            return generate_password_hash(env_pw)
        return hashlib.sha256(env_pw.encode("utf-8")).hexdigest()

    store = get_settings_store()
    settings = store.load()
    return str(settings.get("admin_password_hash", "") or "").strip()


def verify_admin_password(password: str) -> bool:
    if password is None:
        return False
    password = str(password)

    stored_hash = get_admin_password_hash()

    # Back-compat: if no password set, allow default "admin".
    if not stored_hash:
        return password == "admin"

    # Legacy format: raw sha256 hex.
    if _SHA256_HEX_RE.match(stored_hash):
        candidate = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(candidate, stored_hash)

    # Werkzeug format (pbkdf2:sha256:...)
    if check_password_hash is not None:
        try:
            return bool(check_password_hash(stored_hash, password))
        except Exception:
            return False

    return False


def set_admin_password(password: str) -> bool:
    if not password or not isinstance(password, str):
        return False

    store = get_settings_store()
    settings = store.load()

    if generate_password_hash is not None:
        settings["admin_password_hash"] = generate_password_hash(password)
    else:
        settings["admin_password_hash"] = hashlib.sha256(password.encode("utf-8")).hexdigest()

    return store.save(settings)

