"""CLI secrets loader + live-verify wiring (without actually opening sockets)."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from spaghetti_guard import cli
from spaghetti_guard.cli import (
    SECRETS_OPTIONAL_KEYS,
    SECRETS_REQUIRED_KEYS,
    SecretsFileError,
    apply_secrets_to_env,
    load_secrets_file,
    parse_secrets_file,
)


_RUN_CONFIG_BODY = """\
printer:
  ip: 192.168.1.50
  serial: YAMLSERIAL
camera:
  backend: raw
  port: 6000
  timeout_s: 15
  max_reconnect_attempts: 0
detector:
  model_path: weights.pt
  conf_threshold: 0.5
  consecutive_hits: 2
  failure_classes: [spaghetti]
action:
  mode: pause
  dry_run: true
  cooldown_s: 30
  ask_timeout_s: 30
  ask_timeout_action: stop
notify:
  backend: none
  target: ''
snapshots:
  dir: ./snaps
log:
  level: INFO
"""


def _mock_run_env(monkeypatch):
    """Fake camera + control so _cmd_run never touches the network; the
    control factory records its kwargs."""
    from unittest.mock import MagicMock

    fake_cam = MagicMock()
    fake_cam.frames.return_value = iter([])

    class _State:
        def snapshot(self_inner):
            return ("IDLE", 0, 0.0)

    fake_control = MagicMock()
    fake_control.state = _State()
    fake_control.connect.return_value = True

    control_kwargs = {}

    def control_factory(**kw):
        control_kwargs.update(kw)
        return fake_control

    monkeypatch.setattr("spaghetti_guard.camera.RawSocketBackend", lambda **kw: fake_cam)
    monkeypatch.setattr("spaghetti_guard.control.PrinterControl", control_factory)
    return control_kwargs


def _run_namespace(cfg_yaml, secrets):
    return argparse.Namespace(
        config=cfg_yaml,
        dry_run=True,
        action=None,
        no_model_check=True,
        viewer=False,
        mock_detector=True,
        secrets=secrets,
    )


def test_cmd_run_loads_secrets_file_when_env_missing(monkeypatch, tmp_path):
    """`run` must work from a secrets.local.txt alone — INSTALL.md's flow —
    not just from pre-exported environment variables."""
    for k in SECRETS_REQUIRED_KEYS + SECRETS_OPTIONAL_KEYS:
        monkeypatch.delenv(k, raising=False)
    control_kwargs = _mock_run_env(monkeypatch)
    secrets = tmp_path / "secrets.local.txt"
    secrets.write_text(
        "BAMBU_IP=192.168.9.9\nBAMBU_SERIAL=FILESERIAL\nBAMBU_ACCESS_CODE=filetok\n",
        encoding="utf-8",
    )
    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text(_RUN_CONFIG_BODY, encoding="utf-8")

    rc = cli._cmd_run(_run_namespace(cfg_yaml, secrets))
    assert rc == 3  # ran to reconnect-exhaustion — not a config/secrets error
    assert control_kwargs["access_code"] == "filetok"
    assert control_kwargs["serial"] == "FILESERIAL"


def test_cmd_run_env_wins_over_secrets_file(monkeypatch, tmp_path):
    """A service manager's explicit environment beats the file."""
    for k in SECRETS_REQUIRED_KEYS + SECRETS_OPTIONAL_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BAMBU_ACCESS_CODE", "envtok")
    control_kwargs = _mock_run_env(monkeypatch)
    secrets = tmp_path / "secrets.local.txt"
    secrets.write_text(
        "BAMBU_IP=192.168.9.9\nBAMBU_SERIAL=FILESERIAL\nBAMBU_ACCESS_CODE=filetok\n",
        encoding="utf-8",
    )
    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text(_RUN_CONFIG_BODY, encoding="utf-8")

    rc = cli._cmd_run(_run_namespace(cfg_yaml, secrets))
    assert rc == 3
    assert control_kwargs["access_code"] == "envtok"  # env beats file
    assert control_kwargs["serial"] == "FILESERIAL"  # file fills the gap


def test_cmd_run_missing_secrets_file_is_fine_with_env(monkeypatch, tmp_path):
    """No secrets file at all is a supported deployment (env-only)."""
    for k in SECRETS_REQUIRED_KEYS + SECRETS_OPTIONAL_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BAMBU_ACCESS_CODE", "envtok")
    control_kwargs = _mock_run_env(monkeypatch)
    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text(_RUN_CONFIG_BODY, encoding="utf-8")

    rc = cli._cmd_run(_run_namespace(cfg_yaml, tmp_path / "nonexistent.txt"))
    assert rc == 3
    assert control_kwargs["access_code"] == "envtok"
    assert control_kwargs["serial"] == "YAMLSERIAL"  # falls back to yaml


# ---- parse_secrets_file --------------------------------------------------


def test_parse_basic():
    text = "BAMBU_IP=192.168.1.50\nBAMBU_SERIAL=ABC\nBAMBU_ACCESS_CODE=secret\n"
    out = parse_secrets_file(text)
    assert out == {"BAMBU_IP": "192.168.1.50", "BAMBU_SERIAL": "ABC", "BAMBU_ACCESS_CODE": "secret"}


def test_parse_skips_blank_and_comment_lines():
    text = "\n# comment\nBAMBU_IP=10.0.0.1\n\n   # indented comment\nBAMBU_SERIAL=S\nBAMBU_ACCESS_CODE=C\n"
    out = parse_secrets_file(text)
    assert out["BAMBU_IP"] == "10.0.0.1"


def test_parse_trims_whitespace():
    text = "  BAMBU_IP =  10.0.0.1  \nBAMBU_SERIAL=S\nBAMBU_ACCESS_CODE=C"
    out = parse_secrets_file(text)
    assert out["BAMBU_IP"] == "10.0.0.1"


def test_parse_missing_equals_rejected():
    with pytest.raises(SecretsFileError, match="line 1"):
        parse_secrets_file("BAMBU_IP-no-equals\n")


def test_parse_empty_key_rejected():
    with pytest.raises(SecretsFileError, match="empty key"):
        parse_secrets_file("=value\n")


# ---- load_secrets_file ---------------------------------------------------


def _write_full_secrets(tmp_path: Path) -> Path:
    p = tmp_path / "secrets.local.txt"
    p.write_text(
        "BAMBU_IP=192.168.1.50\nBAMBU_SERIAL=ABC123\nBAMBU_ACCESS_CODE=token\n",
        encoding="utf-8",
    )
    return p


def test_load_secrets_file_ok(tmp_path):
    p = _write_full_secrets(tmp_path)
    s = load_secrets_file(p)
    for k in SECRETS_REQUIRED_KEYS:
        assert s[k]


def test_load_secrets_file_missing_path(tmp_path):
    p = tmp_path / "nonexistent.txt"
    with pytest.raises(SecretsFileError, match="not found"):
        load_secrets_file(p)


def test_load_secrets_file_missing_required(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("BAMBU_IP=10.0.0.1\n", encoding="utf-8")
    with pytest.raises(SecretsFileError, match="missing required keys"):
        load_secrets_file(p)


def test_load_secrets_file_empty_required(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text(
        "BAMBU_IP=192.168.1.50\nBAMBU_SERIAL=\nBAMBU_ACCESS_CODE=tok\n",
        encoding="utf-8",
    )
    with pytest.raises(SecretsFileError, match="BAMBU_SERIAL"):
        load_secrets_file(p)


def test_load_secrets_file_with_optional(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text(
        "BAMBU_IP=10.0.0.1\nBAMBU_SERIAL=S\nBAMBU_ACCESS_CODE=C\n"
        "NTFY_TOPIC_URL=https://ntfy.sh/abc\n",
        encoding="utf-8",
    )
    s = load_secrets_file(p)
    assert s["NTFY_TOPIC_URL"] == "https://ntfy.sh/abc"


# ---- apply_secrets_to_env ----------------------------------------------


def test_apply_secrets_to_env_only_promotes_known_keys():
    target = {}
    apply_secrets_to_env({"BAMBU_IP": "1.2.3.4", "EXTRA_GARBAGE": "x"}, env=target)
    assert target == {"BAMBU_IP": "1.2.3.4"}


def test_apply_secrets_to_env_skips_blank_values():
    target = {}
    apply_secrets_to_env({"BAMBU_IP": "1.2.3.4", "NTFY_TOPIC_URL": ""}, env=target)
    assert "NTFY_TOPIC_URL" not in target


def test_apply_secrets_promotes_optional():
    target = {}
    apply_secrets_to_env(
        {"BAMBU_IP": "1.2.3.4", "NTFY_TOPIC_URL": "https://ntfy.sh/x"}, env=target
    )
    assert target["NTFY_TOPIC_URL"] == "https://ntfy.sh/x"


def test_required_and_optional_keys_disjoint():
    assert set(SECRETS_REQUIRED_KEYS).isdisjoint(SECRETS_OPTIONAL_KEYS)


# ---- live-verify happy path (mocked) -----------------------------------


def test_live_verify_missing_secrets_returns_5(tmp_path):
    args = argparse.Namespace(
        config=tmp_path / "config.yaml",
        secrets=tmp_path / "does-not-exist.txt",
        headless=True,
    )
    rc = cli._cmd_live_verify(args)
    assert rc == 5


def test_live_verify_invokes_run_with_dry_run(tmp_path, monkeypatch):
    p = _write_full_secrets(tmp_path)
    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text("placeholder", encoding="utf-8")

    captured = {}

    def fake_run(args):
        captured["dry_run"] = args.dry_run
        captured["mock_detector"] = args.mock_detector
        captured["viewer"] = args.viewer
        return 0

    monkeypatch.setattr(cli, "_cmd_run", fake_run)
    args = argparse.Namespace(config=cfg_yaml, secrets=p, headless=True)
    rc = cli._cmd_live_verify(args)
    assert rc == 0
    assert captured["dry_run"] is True
    assert captured["mock_detector"] is True
    assert captured["viewer"] is False  # --headless


# ---- parser smoke ------------------------------------------------------


def test_parser_run_accepts_ask_action():
    p = cli._build_parser()
    args = p.parse_args(["run", "--action", "ask", "--viewer", "--no-model-check"])
    assert args.cmd == "run"
    assert args.action == "ask"
    assert args.viewer is True


def test_parser_live_verify_subcommand():
    p = cli._build_parser()
    args = p.parse_args(["live-verify"])
    assert args.cmd == "live-verify"
    assert args.headless is False


# ---- main() routing ----------------------------------------------------


def test_main_dispatches_to_subcommand(monkeypatch):
    called = {}

    def fake(args):
        called["yes"] = True
        return 0

    monkeypatch.setattr(cli, "_cmd_verify", fake)
    rc = cli.main(["verify"])
    assert rc == 0
    assert called["yes"]


def test_main_log_level_parses():
    rc = cli.main(["--log-level", "DEBUG", "verify"]) if False else None
    # Just exercise the parser path; we don't actually run verify here.
    args = cli._build_parser().parse_args(["--log-level", "DEBUG", "verify"])
    assert args.log_level == "DEBUG"


# ---- _build_detector mock path ----------------------------------------


def test_build_detector_mock_returns_never_fires(tmp_path):
    """Mock detector must never report a hit, regardless of the JPEG bytes."""
    from spaghetti_guard.config import load_config

    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text(
        "printer:\n  ip: 192.168.1.50\n  serial: ABC\n"
        "camera:\n  backend: raw\n  port: 6000\n  timeout_s: 15\n  max_reconnect_attempts: 5\n"
        "detector:\n  model_path: x.pt\n  conf_threshold: 0.5\n  consecutive_hits: 3\n"
        "  failure_classes: [spaghetti]\n"
        "action:\n  mode: pause\n  dry_run: false\n  cooldown_s: 30\n"
        "  ask_timeout_s: 30\n  ask_timeout_action: stop\n"
        "notify:\n  backend: none\n  target: ''\n"
        "snapshots:\n  dir: ./snaps\n"
        "log:\n  level: INFO\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_yaml, env={"BAMBU_ACCESS_CODE": "tok"}, check_model_path=False)
    det = cli._build_detector(cfg, mock_detector=True)
    r = det.is_failure_frame(b"\xff\xd8\x00\xff\xd9")  # arbitrary
    assert r.hit is False


# ---- _build_viewer ----------------------------------------------------


def test_build_viewer_off():
    assert cli._build_viewer(False) is None
