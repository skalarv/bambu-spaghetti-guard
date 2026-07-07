"""Brief §6.4: env overrides yaml; access_code env-only; invalid values rejected."""

from __future__ import annotations

from pathlib import Path

import pytest

from spaghetti_guard.config import AppConfig, load_config


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(body, encoding="utf-8")
    return p


_BASE_YAML = """
printer:
  ip: 192.168.1.50
  serial: ABC123SERIAL
camera:
  backend: raw
  port: 6000
  timeout_s: 15
  max_reconnect_attempts: 5
detector:
  model_path: nonexistent.pt
  conf_threshold: 0.6
  consecutive_hits: 4
  failure_classes: [spaghetti, blob]
action:
  mode: pause
  dry_run: false
  cooldown_s: 30
notify:
  backend: none
  target: ""
snapshots:
  dir: ./snaps
log:
  level: INFO
"""


def _env(**extra) -> dict:
    base = {"BAMBU_ACCESS_CODE": "test-access-code"}
    base.update(extra)
    return base


def test_loads_minimum(tmp_path):
    cfg = load_config(_write_yaml(tmp_path, _BASE_YAML), env=_env(), check_model_path=False)
    assert isinstance(cfg, AppConfig)
    assert cfg.printer.ip == "192.168.1.50"
    assert cfg.printer.serial == "ABC123SERIAL"
    assert cfg.printer.access_code.get_secret_value() == "test-access-code"
    assert cfg.detector.consecutive_hits == 4


def test_env_overrides_yaml(tmp_path):
    cfg = load_config(
        _write_yaml(tmp_path, _BASE_YAML),
        env=_env(BAMBU_IP="10.0.0.7", BAMBU_SERIAL="OVERRIDE_SERIAL"),
        check_model_path=False,
    )
    assert cfg.printer.ip == "10.0.0.7"
    assert cfg.printer.serial == "OVERRIDE_SERIAL"


def test_ntfy_target_env_override(tmp_path):
    """NTFY_TOPIC_URL from secrets.local.txt / env must reach notify.target —
    otherwise the notify target has to live in config.yaml."""
    body = _BASE_YAML.replace('backend: none\n  target: ""', 'backend: ntfy\n  target: ""')
    cfg = load_config(
        _write_yaml(tmp_path, body),
        env=_env(NTFY_TOPIC_URL="https://ntfy.sh/my-topic"),
        check_model_path=False,
    )
    assert cfg.notify.target == "https://ntfy.sh/my-topic"


def test_telegram_target_env_override(tmp_path):
    """TELEGRAM_TARGET carries the bot token (a secret) — it must be able to
    come from the environment, never forced into config.yaml."""
    body = _BASE_YAML.replace('backend: none\n  target: ""', 'backend: telegram\n  target: ""')
    cfg = load_config(
        _write_yaml(tmp_path, body),
        env=_env(TELEGRAM_TARGET="123456:AAHtok:987654"),
        check_model_path=False,
    )
    assert cfg.notify.target == "123456:AAHtok:987654"


def test_notify_env_override_ignored_for_other_backend(tmp_path):
    """An NTFY url in the env must not leak into a telegram/none target."""
    cfg = load_config(
        _write_yaml(tmp_path, _BASE_YAML),  # backend: none
        env=_env(NTFY_TOPIC_URL="https://ntfy.sh/my-topic"),
        check_model_path=False,
    )
    assert cfg.notify.target == ""


def test_access_code_required_from_env(tmp_path):
    with pytest.raises(ValueError, match="BAMBU_ACCESS_CODE"):
        load_config(_write_yaml(tmp_path, _BASE_YAML), env={}, check_model_path=False)


def test_yaml_with_access_code_rejected(tmp_path):
    bad = _BASE_YAML.replace(
        "serial: ABC123SERIAL",
        "serial: ABC123SERIAL\n  access_code: should-not-be-here",
    )
    with pytest.raises(ValueError, match="forbidden secret key"):
        load_config(_write_yaml(tmp_path, bad), env=_env(), check_model_path=False)


def test_invalid_ip_rejected(tmp_path):
    bad = _BASE_YAML.replace("192.168.1.50", "not-an-ip")
    with pytest.raises(Exception):
        load_config(_write_yaml(tmp_path, bad), env=_env(), check_model_path=False)


def test_empty_serial_rejected(tmp_path):
    bad = _BASE_YAML.replace("serial: ABC123SERIAL", 'serial: ""')
    with pytest.raises(Exception):
        load_config(_write_yaml(tmp_path, bad), env=_env(), check_model_path=False)


def test_out_of_range_threshold_rejected(tmp_path):
    bad = _BASE_YAML.replace("conf_threshold: 0.6", "conf_threshold: 1.5")
    with pytest.raises(Exception):
        load_config(_write_yaml(tmp_path, bad), env=_env(), check_model_path=False)


def test_zero_consecutive_hits_rejected(tmp_path):
    bad = _BASE_YAML.replace("consecutive_hits: 4", "consecutive_hits: 0")
    with pytest.raises(Exception):
        load_config(_write_yaml(tmp_path, bad), env=_env(), check_model_path=False)


def test_bad_action_mode_rejected(tmp_path):
    bad = _BASE_YAML.replace("mode: pause", "mode: explode")
    with pytest.raises(Exception):
        load_config(_write_yaml(tmp_path, bad), env=_env(), check_model_path=False)


def test_model_path_check_bypassable(tmp_path):
    """check_model_path=False lets us load even when weights are absent (test/CI path)."""
    cfg = load_config(_write_yaml(tmp_path, _BASE_YAML), env=_env(), check_model_path=False)
    assert not cfg.detector.model_path.exists()


def test_model_path_check_enforced_by_default(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(_write_yaml(tmp_path, _BASE_YAML), env=_env(), check_model_path=True)


def test_model_path_check_passes_when_file_exists(tmp_path):
    model = tmp_path / "fake.pt"
    model.write_bytes(b"\x00")
    yaml_body = _BASE_YAML.replace("nonexistent.pt", str(model).replace("\\", "/"))
    cfg = load_config(_write_yaml(tmp_path, yaml_body), env=_env(), check_model_path=True)
    assert cfg.detector.model_path == model


def test_access_code_not_in_repr(tmp_path):
    """SecretStr must mask the value in any default stringification."""
    cfg = load_config(_write_yaml(tmp_path, _BASE_YAML), env=_env(), check_model_path=False)
    s = repr(cfg)
    assert "test-access-code" not in s
