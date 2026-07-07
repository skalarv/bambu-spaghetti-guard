"""Final-mile coverage targets — context manager, error suppression paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from spaghetti_guard import camera
from spaghetti_guard.camera import (
    CameraError,
    CameraStreamClosed,
    FRAME_HEADER_LEN,
    iter_frames_from_stream,
    parse_frame_header,
)


# =====================================================================
# CameraBackend context manager + parse_frame_header bad length
# =====================================================================


class _DummyBackend(camera.CameraBackend):
    def __init__(self):
        self.connected = False
        self.closed = False

    def connect(self) -> None:
        self.connected = True

    def frames(self):
        return iter([])

    def close(self) -> None:
        self.closed = True


def test_camera_backend_context_manager():
    """`with backend:` should connect on enter, close on exit."""
    b = _DummyBackend()
    with b as ctx:
        assert ctx is b
        assert b.connected is True
    assert b.closed is True


def test_parse_frame_header_wrong_size_raises():
    with pytest.raises(ValueError):
        parse_frame_header(b"\x00" * (FRAME_HEADER_LEN - 1))


def test_iter_frames_drop_malformed_false_raises_on_bad_header():
    """With drop_malformed=False, bad headers must surface as ValueError."""
    import struct
    import io

    # Length too large to be plausible
    blob = struct.pack("<I", 99_999_999) + b"\x00" * 12 + b"junk"
    buf = io.BytesIO(blob)

    def read_exact(n):
        data = buf.read(n)
        if len(data) < n:
            raise EOFError
        return data

    with pytest.raises(ValueError):
        list(iter_frames_from_stream(read_exact, drop_malformed=False))


def test_iter_frames_eof_mid_payload_raises_stream_closed():
    import struct
    import io

    blob = struct.pack("<I", 100) + b"\x00" * 12  # header, no payload
    buf = io.BytesIO(blob)

    def read_exact(n):
        data = buf.read(n)
        if len(data) < n:
            raise EOFError
        return data

    with pytest.raises(CameraStreamClosed):
        list(iter_frames_from_stream(read_exact))


# =====================================================================
# RawSocketBackend.close — when shutdown raises OSError
# =====================================================================


def test_raw_close_handles_shutdown_oserror(monkeypatch):
    b = camera.RawSocketBackend(host="127.0.0.1", access_code="x")
    fake_sock = MagicMock()
    fake_sock.shutdown.side_effect = OSError("not connected")
    b._sock = fake_sock
    b.close()  # must not raise
    fake_sock.close.assert_called_once()
    assert b._sock is None


# =====================================================================
# Guard — notifier raise during fire is suppressed
# =====================================================================


def test_guard_notifier_exception_suppressed(tmp_path):
    from spaghetti_guard.detector import FailureDetector
    from spaghetti_guard.guard import Guard, GuardState
    from spaghetti_guard.notifier import Notifier

    class BoomNotifier(Notifier):
        def send(self, title, message, image_path=None):
            raise RuntimeError("ntfy down")

    class _Y:
        def predict(self, *a, **kw):
            return [type("B", (), {"cls_name": "spaghetti", "conf": 0.9})()]

    det = FailureDetector(_Y(), failure_classes=("spaghetti",), conf_threshold=0.5, decoder=lambda j: j)
    ctrl = MagicMock()
    g = Guard(
        detector=det,
        control=ctrl,
        notifier=BoomNotifier(),
        gcode_state_provider=lambda: "RUNNING",
        debounce_window=1,
        snapshot_dir=tmp_path,
    )
    g.feed_frame(b"j")
    # Notifier exception must not block the action.
    assert ctrl.pause.call_count + ctrl.stop.call_count == 1
    assert g.state == GuardState.COOLDOWN


def test_guard_control_exception_keeps_triggered(tmp_path):
    """If the control call raises, the guard should stay in TRIGGERED so the
    operator sees the alert (instead of silently moving on)."""
    from spaghetti_guard.detector import FailureDetector
    from spaghetti_guard.guard import Guard, GuardState
    from spaghetti_guard.notifier import NoopNotifier

    class _Y:
        def predict(self, *a, **kw):
            return [type("B", (), {"cls_name": "spaghetti", "conf": 0.9})()]

    det = FailureDetector(_Y(), failure_classes=("spaghetti",), conf_threshold=0.5, decoder=lambda j: j)
    ctrl = MagicMock()
    ctrl.pause.side_effect = RuntimeError("MQTT broker rejected")
    g = Guard(
        detector=det,
        control=ctrl,
        notifier=NoopNotifier(),
        gcode_state_provider=lambda: "RUNNING",
        action_mode="pause",
        debounce_window=1,
        snapshot_dir=tmp_path,
    )
    g.feed_frame(b"j")
    assert g.state == GuardState.TRIGGERED  # never transitioned to COOLDOWN
    ctrl.pause.assert_called_once()


def test_guard_cooldown_skips_detection(tmp_path):
    """While in COOLDOWN with `now < cooldown_until`, feed_frame must not call
    the detector."""
    from spaghetti_guard.detector import FailureDetector
    from spaghetti_guard.guard import Guard, GuardState
    from spaghetti_guard.notifier import NoopNotifier

    yolo = MagicMock()
    yolo.predict.return_value = []
    det = FailureDetector(yolo, failure_classes=("x",), conf_threshold=0.5, decoder=lambda j: j)

    class _Clock:
        def __init__(self):
            self.t = 1000.0

        def __call__(self):
            return self.t

    clock = _Clock()
    g = Guard(
        detector=det,
        control=MagicMock(),
        notifier=NoopNotifier(),
        gcode_state_provider=lambda: "RUNNING",
        snapshot_dir=tmp_path,
        now=clock,
    )
    # Force into cooldown
    g._state = GuardState.COOLDOWN
    g._cooldown_until = clock.t + 30
    g.feed_frame(b"j")
    assert yolo.predict.call_count == 0
    assert g.state == GuardState.COOLDOWN


# =====================================================================
# Snapshot write failure is suppressed
# =====================================================================


def test_snapshot_write_failure_suppressed(tmp_path, monkeypatch):
    from spaghetti_guard.detector import FailureDetector, FrameResult
    from spaghetti_guard.guard import Guard
    from spaghetti_guard.notifier import NoopNotifier

    class _Y:
        def predict(self, *a, **kw):
            return [type("B", (), {"cls_name": "x", "conf": 0.9})()]

    det = FailureDetector(_Y(), failure_classes=("x",), conf_threshold=0.5, decoder=lambda j: j)
    g = Guard(
        detector=det,
        control=MagicMock(),
        notifier=NoopNotifier(),
        gcode_state_provider=lambda: "RUNNING",
        debounce_window=1,
        snapshot_dir=tmp_path,
    )
    # Patch Path.write_bytes to raise
    original = Path.write_bytes
    monkeypatch.setattr(
        Path, "write_bytes", lambda self, data: (_ for _ in ()).throw(OSError("disk full"))
    )
    # Must not crash
    g.feed_frame(b"j")
    monkeypatch.setattr(Path, "write_bytes", original)


# =====================================================================
# Viewer update with no jpeg payload
# =====================================================================


def test_viewer_update_with_none_jpeg(tmp_path):
    """When the printer state isn't RUNNING the guard pushes `jpeg=None` to the
    viewer; it should still go through without error."""
    from spaghetti_guard.detector import FailureDetector
    from spaghetti_guard.guard import Guard, GuardState
    from spaghetti_guard.notifier import NoopNotifier

    calls = []

    class V:
        def update(self, **kwargs):
            calls.append(kwargs)

        def start(self):
            pass

        def stop(self):
            pass

    class _Y:
        def predict(self, *a, **kw):
            return []

    det = FailureDetector(_Y(), failure_classes=("x",), conf_threshold=0.5, decoder=lambda j: j)
    g = Guard(
        detector=det,
        control=MagicMock(),
        notifier=NoopNotifier(),
        gcode_state_provider=lambda: "IDLE",
        snapshot_dir=tmp_path,
        viewer=V(),
    )
    g.feed_frame(b"j")
    assert len(calls) == 1
    assert calls[0]["last_result"] is None
