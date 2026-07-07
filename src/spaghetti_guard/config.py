"""Layered configuration: config.yaml on disk, env vars override, secrets env-only.

Brief §5.1. Secrets (BAMBU_ACCESS_CODE) must never appear in yaml or logs.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

ENV_PREFIX_IP = "BAMBU_IP"
ENV_PREFIX_SERIAL = "BAMBU_SERIAL"
ENV_PREFIX_ACCESS_CODE = "BAMBU_ACCESS_CODE"
# Notify targets can carry secrets (telegram bot token), so they are settable
# from the environment / secrets.local.txt. The var matching the configured
# backend wins; others are ignored.
ENV_NOTIFY_TARGETS = {"ntfy": "NTFY_TOPIC_URL", "telegram": "TELEGRAM_TARGET"}

YAML_FORBIDDEN_SECRET_KEYS = ("access_code", "lan_access_code")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PrinterConfig(_Frozen):
    ip: str
    serial: str
    access_code: SecretStr

    @field_validator("ip")
    @classmethod
    def _ip_must_be_valid(cls, v: str) -> str:
        ipaddress.ip_address(v)
        return v

    @field_validator("serial")
    @classmethod
    def _serial_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("printer.serial must be non-empty")
        return v


class CameraConfig(_Frozen):
    backend: Literal["raw", "lib"] = "raw"
    port: int = 6000
    timeout_s: float = 15.0
    max_reconnect_attempts: int = 5

    @field_validator("port")
    @classmethod
    def _port_in_range(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("camera.port out of range")
        return v


class DetectorConfig(_Frozen):
    model_path: Path
    conf_threshold: float = 0.55
    consecutive_hits: int = 6
    # Default matches config.yaml's documented posture: only the two
    # operationally severe classes fire the guard.
    failure_classes: list[str] = Field(
        default_factory=lambda: ["spaghetti", "detachment"]
    )

    @field_validator("conf_threshold")
    @classmethod
    def _threshold_range(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("detector.conf_threshold must be in (0, 1]")
        return v

    @field_validator("consecutive_hits")
    @classmethod
    def _hits_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("detector.consecutive_hits must be >= 1")
        return v


class ActionConfig(_Frozen):
    mode: Literal["stop", "pause", "ask"] = "pause"
    dry_run: bool = False
    cooldown_s: float = 30.0
    ask_timeout_s: float = 30.0
    ask_timeout_action: Literal["stop", "pause"] = "stop"


class NotifyConfig(_Frozen):
    backend: Literal["none", "ntfy", "telegram", "homeassistant"] = "none"
    target: str = ""


class SnapshotConfig(_Frozen):
    dir: Path = Path("./failure_snapshots")
    # Oldest trigger-*.jpg beyond this count are pruned; 0 disables pruning.
    max_files: int = 500


class LogConfig(_Frozen):
    level: str = "INFO"


class AppConfig(_Frozen):
    printer: PrinterConfig
    camera: CameraConfig
    detector: DetectorConfig
    action: ActionConfig
    notify: NotifyConfig
    snapshots: SnapshotConfig
    log: LogConfig


def _yaml_has_secret(data: dict) -> str | None:
    """Return the first forbidden key found in any nesting level, else None."""
    if not isinstance(data, dict):
        return None
    for k, v in data.items():
        if k in YAML_FORBIDDEN_SECRET_KEYS:
            return k
        if isinstance(v, dict):
            found = _yaml_has_secret(v)
            if found:
                return found
    return None


def _apply_env_overrides(data: dict, env: dict) -> dict:
    """Overlay env vars onto the yaml data tree.

    Whitelist of env vars, since pydantic-settings env autodiscovery would
    require giving the access code free reign over the schema.
    """
    data = dict(data)
    printer = dict(data.get("printer") or {})
    if ENV_PREFIX_IP in env:
        printer["ip"] = env[ENV_PREFIX_IP]
    if ENV_PREFIX_SERIAL in env:
        printer["serial"] = env[ENV_PREFIX_SERIAL]
    if ENV_PREFIX_ACCESS_CODE not in env:
        raise ValueError(
            f"{ENV_PREFIX_ACCESS_CODE} must be set in environment "
            f"(brief §5.1: secrets are env-only, never yaml)."
        )
    printer["access_code"] = env[ENV_PREFIX_ACCESS_CODE]
    data["printer"] = printer

    notify = dict(data.get("notify") or {})
    env_var = ENV_NOTIFY_TARGETS.get(notify.get("backend", ""))
    if env_var and env.get(env_var):
        notify["target"] = env[env_var]
        data["notify"] = notify
    return data


def load_config(
    yaml_path: str | os.PathLike[str] | None = None,
    env: dict | None = None,
    *,
    check_model_path: bool = True,
) -> AppConfig:
    """Load AppConfig from yaml + env.

    Args:
        yaml_path: Path to config.yaml. None uses CONFIG_YAML env or ./config.yaml.
        env: Environment dict (defaults to os.environ). Injectable for tests.
        check_model_path: If True, require detector.model_path to exist.
            Set False during tests and via CLI --no-model-check.
    """
    env = dict(env if env is not None else os.environ)

    if yaml_path is None:
        yaml_path = env.get("CONFIG_YAML", "config.yaml")
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"config yaml not found: {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config yaml root must be a mapping, got {type(raw).__name__}")

    leaked = _yaml_has_secret(raw)
    if leaked is not None:
        raise ValueError(
            f"forbidden secret key '{leaked}' found in {yaml_path}; "
            f"secrets must come from environment only."
        )

    merged = _apply_env_overrides(raw, env)
    cfg = AppConfig(**merged)

    if check_model_path and not cfg.detector.model_path.exists():
        raise FileNotFoundError(
            f"detector.model_path does not exist: {cfg.detector.model_path}. "
            f"Pass check_model_path=False (or --no-model-check) to bypass."
        )

    return cfg
