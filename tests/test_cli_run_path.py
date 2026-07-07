"""Cover _cmd_run, _build_detector live path, _build_viewer with mocks.

Sockets and Tk are mocked — these tests never open a real connection or
window.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from spaghetti_guard import cli


_CONFIG_BODY = """\
printer:
  ip: 192.168.1.50
  serial: ABC
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


def _make_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_CONFIG_BODY, encoding="utf-8")
    return cfg


# ---- _build_viewer enabled ---------------------------------------------


def test_build_viewer_enabled_returns_tk_viewer():
    v = cli._build_viewer(True)
    assert v is not None
    # We did NOT start it -- no Tk window opened.
    v.stop()


# ---- _build_detector live path (mocked) -------------------------------


def test_build_detector_live_path_handles_empty_results(monkeypatch, tmp_path):
    from spaghetti_guard.config import load_config

    cfg_yaml = _make_config(tmp_path)
    cfg = load_config(cfg_yaml, env={"BAMBU_ACCESS_CODE": "tok"}, check_model_path=False)

    fake_model = MagicMock()
    monkeypatch.setattr("spaghetti_guard.detector.load_yolo_model", lambda p: fake_model)
    monkeypatch.setattr("spaghetti_guard.detector.decode_jpeg", lambda j: b"decoded")

    det = cli._build_detector(cfg, mock_detector=False)
    # Predict returns an Ultralytics-shaped Results list -- empty here
    fake_model.predict.return_value = []
    r = det.is_failure_frame(b"jpeg")
    assert r.hit is False


def test_build_detector_live_path_flattens_boxes(monkeypatch, tmp_path):
    """Make sure the adapter picks names + conf out of a Results-like object."""
    from spaghetti_guard.config import load_config

    cfg_yaml = _make_config(tmp_path)
    cfg = load_config(cfg_yaml, env={"BAMBU_ACCESS_CODE": "tok"}, check_model_path=False)

    class _Boxes:
        cls = [0]
        conf = [0.9]

        def __len__(self):
            return 1

    class _Result:
        names = {0: "spaghetti"}
        boxes = _Boxes()

    fake_model = MagicMock()
    fake_model.predict.return_value = [_Result()]
    monkeypatch.setattr("spaghetti_guard.detector.load_yolo_model", lambda p: fake_model)
    monkeypatch.setattr("spaghetti_guard.detector.decode_jpeg", lambda j: b"decoded")

    det = cli._build_detector(cfg, mock_detector=False)
    r = det.is_failure_frame(b"jpeg")
    assert r.hit is True
    assert r.best_class == "spaghetti"
    assert r.conf == 0.9


# ---- _cmd_run smoke (mocked socket layer) -----------------------------


def test_cmd_run_with_dry_run_and_no_viewer(monkeypatch, tmp_path):
    monkeypatch.setenv("BAMBU_ACCESS_CODE", "tok")
    cfg_yaml = _make_config(tmp_path)

    # Stub camera and control so we never touch the network.
    fake_cam = MagicMock()
    fake_cam.frames.return_value = iter([])  # zero frames -> guard.run returns immediately

    class _State:
        def snapshot(self_inner):
            return ("IDLE", 0, 0.0)

    fake_control = MagicMock()
    fake_control.state = _State()
    fake_control.connect.return_value = True

    monkeypatch.setattr("spaghetti_guard.camera.RawSocketBackend", lambda **kw: fake_cam)
    monkeypatch.setattr("spaghetti_guard.control.PrinterControl", lambda **kw: fake_control)

    args = argparse.Namespace(
        config=cfg_yaml,
        dry_run=True,
        action=None,
        no_model_check=True,
        viewer=False,
        mock_detector=True,  # bypass YOLO
        secrets=Path("nonexistent-secrets-for-tests.txt"),
    )
    rc = cli._cmd_run(args)
    # Empty stream + max_reconnect_attempts=0 → reconnect exhausted → 3.
    assert rc == 3
    fake_cam.connect.assert_called_once()
    fake_cam.close.assert_called_once()
    fake_control.close.assert_called_once()


def test_cmd_run_mqtt_connect_fail_returns_4(monkeypatch, tmp_path):
    monkeypatch.setenv("BAMBU_ACCESS_CODE", "tok")
    cfg_yaml = _make_config(tmp_path)

    fake_cam = MagicMock()
    fake_cam.frames.return_value = iter([])

    class _State:
        def snapshot(self_inner):
            return ("IDLE", 0, 0.0)

    fake_control = MagicMock()
    fake_control.state = _State()
    fake_control.connect.return_value = False  # the failure under test

    monkeypatch.setattr("spaghetti_guard.camera.RawSocketBackend", lambda **kw: fake_cam)
    monkeypatch.setattr("spaghetti_guard.control.PrinterControl", lambda **kw: fake_control)

    args = argparse.Namespace(
        config=cfg_yaml,
        dry_run=True,
        action=None,
        no_model_check=True,
        viewer=False,
        mock_detector=True,
        secrets=Path("nonexistent-secrets-for-tests.txt"),
    )
    rc = cli._cmd_run(args)
    assert rc == 4


def test_cmd_run_wires_viewer_button_hooks(monkeypatch, tmp_path):
    monkeypatch.setenv("BAMBU_ACCESS_CODE", "tok")
    cfg_yaml = _make_config(tmp_path)
    fake_cam = MagicMock()
    fake_cam.frames.return_value = iter([])

    class _State:
        def snapshot(self_inner):
            return ("IDLE", 0, 0.0)

    fake_control = MagicMock()
    fake_control.state = _State()
    fake_control.connect.return_value = True

    fake_viewer = MagicMock()
    fake_viewer.on_pause_clicked = None
    fake_viewer.on_stop_clicked = None

    monkeypatch.setattr("spaghetti_guard.camera.RawSocketBackend", lambda **kw: fake_cam)
    monkeypatch.setattr("spaghetti_guard.control.PrinterControl", lambda **kw: fake_control)
    monkeypatch.setattr(cli, "_build_viewer", lambda enabled: fake_viewer if enabled else None)

    args = argparse.Namespace(
        config=cfg_yaml,
        dry_run=True,
        action=None,
        no_model_check=True,
        viewer=True,
        mock_detector=True,
        secrets=Path("nonexistent-secrets-for-tests.txt"),
    )
    rc = cli._cmd_run(args)
    assert rc == 3  # empty stream + zero reconnect budget → exhausted
    fake_viewer.start.assert_called_once()
    fake_viewer.stop.assert_called_once()
    # Button hooks should have been replaced with lambdas (callable, non-None)
    assert callable(fake_viewer.on_pause_clicked)
    assert callable(fake_viewer.on_stop_clicked)


# ---- CLI subcommand routing through main() ----------------------------


def test_main_replay_routes(monkeypatch):
    called = {}

    def fake(args):
        called["replay"] = True
        return 0

    monkeypatch.setattr(cli, "_cmd_replay", fake)
    rc = cli.main(["replay", "some/clip"])
    assert rc == 0
    assert called.get("replay")


def test_main_train_routes(monkeypatch):
    called = {}

    def fake(args):
        called["train"] = True
        return 0

    monkeypatch.setattr(cli, "_cmd_train", fake)
    rc = cli.main(["train", "--data", "x"])
    assert rc == 0


def test_main_validate_routes(monkeypatch):
    called = {}

    def fake(args):
        called["validate"] = True
        return 0

    monkeypatch.setattr(cli, "_cmd_validate", fake)
    rc = cli.main(["validate", "--weights", "x", "--data", "y"])
    assert rc == 0
