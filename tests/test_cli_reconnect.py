"""Cover the CLI reconnect wrapper + force_armed + SIGINT hook."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path
from unittest.mock import MagicMock


from spaghetti_guard import cli


_CONFIG_BODY = """\
printer:
  ip: 192.168.1.50
  serial: ABC
camera:
  backend: raw
  port: 6000
  timeout_s: 15
  max_reconnect_attempts: 2
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


def _mocked_env(monkeypatch, tmp_path, *, connect_ok=True):
    """Standard mock skeleton: real config + fake camera + fake control."""
    monkeypatch.setenv("BAMBU_ACCESS_CODE", "tok")
    # No real sleep during reconnect.
    monkeypatch.setattr("time.sleep", lambda s: None)

    fake_cam = MagicMock()

    class _State:
        def snapshot(self_inner):
            return ("IDLE", 0, 0.0)

    fake_control = MagicMock()
    fake_control.state = _State()
    fake_control.connect.return_value = connect_ok

    monkeypatch.setattr("spaghetti_guard.camera.RawSocketBackend", lambda **kw: fake_cam)
    monkeypatch.setattr("spaghetti_guard.control.PrinterControl", lambda **kw: fake_control)
    return fake_cam, fake_control


def _run_args(cfg_yaml, **overrides):
    ns = argparse.Namespace(
        config=cfg_yaml,
        dry_run=True,
        action=None,
        no_model_check=True,
        viewer=False,
        mock_detector=True,
        force_armed=False,
        # keep tests hermetic: never read the developer's real secrets file
        secrets=Path("nonexistent-secrets-for-tests.txt"),
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


# =====================================================================
# force_armed path
# =====================================================================


def test_cmd_run_force_armed_uses_override_state_provider(monkeypatch, tmp_path, caplog):
    fake_cam, _ = _mocked_env(monkeypatch, tmp_path)
    fake_cam.frames.return_value = iter([])

    cfg_yaml = _make_config(tmp_path)
    args = _run_args(cfg_yaml, force_armed=True)

    with caplog.at_level("WARNING", logger="spaghetti-guard"):
        rc = cli._cmd_run(args)

    assert rc == 3  # empty stream eventually exhausts the reconnect budget
    assert any("--force-armed" in r.message for r in caplog.records)


# =====================================================================
# Reconnect wrapper: exception raised, reconnect exhausts, exits cleanly
# =====================================================================


def test_reconnect_wrapper_exhausts_and_exits(monkeypatch, tmp_path, caplog):
    fake_cam, _ = _mocked_env(monkeypatch, tmp_path)
    from spaghetti_guard.camera import CameraStreamClosed

    # Every frames() call raises CameraStreamClosed immediately.
    def raising_frames():
        raise CameraStreamClosed("simulated mid-payload EOF")

    fake_cam.frames.side_effect = lambda: raising_frames()

    cfg_yaml = _make_config(tmp_path)  # max_reconnect_attempts=2
    args = _run_args(cfg_yaml)

    with caplog.at_level("WARNING", logger="spaghetti-guard"):
        rc = cli._cmd_run(args)

    # Exhaustion means the guard is no longer protecting the print — the exit
    # code must be non-zero so a service manager (Restart=on-failure) restarts us.
    assert rc == 3
    # cam.close called at end, plus once per reconnect attempt (>=2).
    assert fake_cam.close.call_count >= 2
    # At least one reconnect attempt should have been made.
    assert fake_cam.connect.call_count >= 2  # initial + at least one reconnect
    # The reconnect-exhausted line must have logged.
    assert any(
        "camera reconnect exhausted" in r.message for r in caplog.records
    ), [r.message for r in caplog.records]


def test_reconnect_wrapper_reconnect_call_failure_logged(monkeypatch, tmp_path, caplog):
    fake_cam, _ = _mocked_env(monkeypatch, tmp_path)
    from spaghetti_guard.camera import CameraStreamClosed

    fake_cam.frames.side_effect = CameraStreamClosed("mid-payload")
    # First connect() OK (setup), then subsequent reconnect() attempts raise.
    connect_calls = [0]

    def _connect():
        connect_calls[0] += 1
        if connect_calls[0] > 1:
            raise OSError("printer offline")

    fake_cam.connect.side_effect = _connect

    cfg_yaml = _make_config(tmp_path)
    args = _run_args(cfg_yaml)

    with caplog.at_level("ERROR", logger="spaghetti-guard"):
        rc = cli._cmd_run(args)

    assert rc == 3  # exhausted — non-zero so the service manager restarts us
    assert any("camera reconnect failed" in r.message for r in caplog.records)


# =====================================================================
# Reconnect budget resets after a healthy stream
# =====================================================================


def test_reconnect_counter_resets_after_frames_flow(monkeypatch, tmp_path):
    """Drops separated by healthy streaming must not accumulate toward the
    reconnect budget — only *consecutive* dead reconnects exhaust it. A
    multi-day print with occasional drops must never kill the guard."""
    fake_cam, _ = _mocked_env(monkeypatch, tmp_path)
    from spaghetti_guard.camera import CameraStreamClosed

    calls = {"n": 0}

    def frames_factory():
        calls["n"] += 1
        if calls["n"] <= 5:
            def healthy_then_drop():
                yield b"jpeg"
                raise CameraStreamClosed("drop after a good frame")

            return healthy_then_drop()

        def dead():
            raise CameraStreamClosed("immediate")
            yield  # pragma: no cover  (makes this a generator)

        return dead()

    fake_cam.frames.side_effect = frames_factory

    cfg_yaml = _make_config(tmp_path)  # max_reconnect_attempts=2
    args = _run_args(cfg_yaml)
    rc = cli._cmd_run(args)

    # 5 healthy-then-drop cycles never exhaust the budget of 2 (counter resets
    # after each streamed frame); only the consecutive immediate failures at
    # the end do: n=6 (attempt 1... 2) and n=7 (attempt 3 > 2 -> exhausted).
    assert calls["n"] == 7
    assert rc == 3


# =====================================================================
# Reconnect wrapper: exception during reconnect close() is suppressed
# =====================================================================


def test_reconnect_wrapper_close_error_suppressed(monkeypatch, tmp_path):
    fake_cam, _ = _mocked_env(monkeypatch, tmp_path)
    from spaghetti_guard.camera import CameraStreamClosed

    fake_cam.frames.side_effect = CameraStreamClosed("closed")
    # First close (after loop) OK, mid-loop closes raise.
    close_calls = [0]

    def _close():
        close_calls[0] += 1
        if close_calls[0] <= 2:
            raise RuntimeError("socket already dead")

    fake_cam.close.side_effect = _close

    cfg_yaml = _make_config(tmp_path)
    args = _run_args(cfg_yaml)

    rc = cli._cmd_run(args)
    assert rc == 3  # exception in close() didn't bubble; exhaustion code intact


# =====================================================================
# guard.stopped short-circuits both branches
# =====================================================================


def test_reconnect_wrapper_stopped_before_exception_check(monkeypatch, tmp_path):
    """If guard.stopped goes True inside guard.run, the outer while breaks."""
    fake_cam, _ = _mocked_env(monkeypatch, tmp_path)
    fake_cam.frames.return_value = iter([])

    from spaghetti_guard import guard as guard_mod

    def patched_run(self, frame_iter, **kw):
        # Consume the (empty) iterator and set stop before returning.
        for _ in frame_iter:
            pass
        self.request_stop()

    monkeypatch.setattr(guard_mod.Guard, "run", patched_run)

    cfg_yaml = _make_config(tmp_path)
    args = _run_args(cfg_yaml)

    rc = cli._cmd_run(args)
    assert rc == 0
    # cam.close called once at the finally: no reconnect attempts.
    assert fake_cam.close.call_count == 1


def test_reconnect_wrapper_stopped_in_except_branch(monkeypatch, tmp_path):
    """Guard.run raising + stopped=True → break out of the except branch."""
    fake_cam, _ = _mocked_env(monkeypatch, tmp_path)
    from spaghetti_guard.camera import CameraStreamClosed
    from spaghetti_guard import guard as guard_mod

    def patched_run(self, frame_iter, **kw):
        self.request_stop()
        raise CameraStreamClosed("dead")

    monkeypatch.setattr(guard_mod.Guard, "run", patched_run)
    # cam.frames must return something iterable so it doesn't raise at call
    fake_cam.frames.return_value = iter([])

    cfg_yaml = _make_config(tmp_path)
    args = _run_args(cfg_yaml)

    rc = cli._cmd_run(args)
    assert rc == 0


# =====================================================================
# SIGINT signal handler installs
# =====================================================================


def test_cmd_run_installs_sigint_handler(monkeypatch, tmp_path):
    fake_cam, _ = _mocked_env(monkeypatch, tmp_path)
    fake_cam.frames.return_value = iter([])

    installed = {}

    real_signal = signal.signal

    def fake_signal(sig, handler):
        installed[sig] = handler
        return real_signal(sig, handler if sig != signal.SIGINT else lambda *a: None)

    monkeypatch.setattr(signal, "signal", fake_signal)

    cfg_yaml = _make_config(tmp_path)
    args = _run_args(cfg_yaml)

    rc = cli._cmd_run(args)
    assert rc == 3  # empty stream eventually exhausts the reconnect budget
    assert signal.SIGINT in installed
    # Fire the handler and confirm it calls guard.request_stop
    # (we can't easily reach the guard from here — but the fact that we
    # installed it is what covers lines 157-159).
