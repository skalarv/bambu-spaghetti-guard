"""Tests targeting otherwise-uncovered branches to push coverage above 95%."""

from __future__ import annotations

import argparse
import sys
from unittest.mock import MagicMock

import pytest

from spaghetti_guard import camera, cli, control, detector, viewer


# =====================================================================
# control.py callbacks and reconnect
# =====================================================================


class _FakePahoMin:
    def __init__(self, cid):
        self.client_id = cid
        self.on_connect = None
        self.on_disconnect = None
        self.on_message = None
        self.subscribed = []
        self.published = []
        self.reconnect_calls = 0
        self.reconnect_delay_args = None

    def username_pw_set(self, u, p):
        pass

    def tls_set(self, **kw):
        pass

    def tls_insecure_set(self, b):
        pass

    def connect(self, *a, **kw):
        pass

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def subscribe(self, t, qos=0):
        self.subscribed.append((t, qos))

    def publish(self, *a, **kw):
        m = MagicMock()
        m.rc = 0
        return m

    def reconnect(self):
        self.reconnect_calls += 1

    def reconnect_delay_set(self, min_delay=1, max_delay=120):
        self.reconnect_delay_args = (min_delay, max_delay)


def _build_control(**kw):
    fake = _FakePahoMin("cli-id")
    pc = control.PrinterControl(
        host="127.0.0.1",
        serial="S",
        access_code="tok",
        client_factory=lambda cid: fake,
        **kw,
    )
    return pc, fake


def test_on_connect_success_subscribes():
    pc, fake = _build_control()
    pc._on_connect(fake, None, {}, 0, None)
    assert ("device/S/report", 0) in fake.subscribed
    assert pc.is_connected is True


def test_on_connect_failure_no_subscribe():
    pc, fake = _build_control()
    pc._on_connect(fake, None, {}, 5, None)  # non-zero reason -> failure
    assert fake.subscribed == []
    assert pc.is_connected is False


def test_on_disconnect_clears_connected_without_spawning_threads():
    """Reconnection is paho's job (loop_start auto-reconnects with the delays
    from reconnect_delay_set). A second hand-rolled reconnect thread racing
    paho's own can wedge the client — _on_disconnect must stay passive."""
    import threading

    pc, fake = _build_control()
    pc._connected.set()
    before = threading.active_count()
    pc._on_disconnect(fake, None, {}, 1, None)
    assert threading.active_count() == before  # no thread spawned
    assert fake.reconnect_calls == 0  # no direct reconnect either
    assert pc.is_connected is False


def test_reconnect_delay_configured_from_backoff_max():
    """The backoff ceiling must be handed to paho's built-in reconnect."""
    pc, fake = _build_control(reconnect_backoff_max_s=45.0)
    assert fake.reconnect_delay_args is not None
    min_delay, max_delay = fake.reconnect_delay_args
    assert min_delay >= 1
    assert max_delay == 45


def test_use_tls_false_skips_tls_setup():
    fake = _FakePahoMin("c")
    called = []

    def boom(**kw):
        called.append(True)

    fake.tls_set = boom
    control.PrinterControl(
        host="127.0.0.1",
        serial="S",
        access_code="tok",
        use_tls=False,
        client_factory=lambda cid: fake,
    )
    assert called == []  # tls_set NOT invoked


def test_close_handles_loop_stop_exception(monkeypatch):
    pc, fake = _build_control()

    def boom():
        raise RuntimeError("nope")

    monkeypatch.setattr(fake, "loop_stop", boom)
    pc.close()  # must not raise


# =====================================================================
# camera.py edge cases
# =====================================================================


def test_raw_backend_frames_before_connect_raises():
    backend = camera.RawSocketBackend(host="127.0.0.1", access_code="tok")
    with pytest.raises(camera.CameraError):
        list(backend.frames())


def test_raw_backend_close_before_connect_is_noop():
    backend = camera.RawSocketBackend(host="127.0.0.1", access_code="tok")
    backend.close()  # must not raise


def test_lib_backend_methods_raise():
    """The stub's abstract methods (never reached in practice) raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        # Calling __init__ raises; cover the path
        camera.LibBackend(any_kwarg=1)


# =====================================================================
# detector.py
# =====================================================================


def test_decode_jpeg_invalid_raises(monkeypatch):
    class FakeCV2:
        IMREAD_COLOR = 1

        @staticmethod
        def imdecode(arr, flag):
            return None  # simulate "not a jpeg"

    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    with pytest.raises(ValueError):
        detector.decode_jpeg(b"not-a-jpeg")


def test_decode_jpeg_success(monkeypatch):
    import numpy as np

    class FakeCV2:
        IMREAD_COLOR = 1

        @staticmethod
        def imdecode(arr, flag):
            return np.zeros((10, 10, 3), dtype=np.uint8)

    monkeypatch.setitem(sys.modules, "cv2", FakeCV2)
    img = detector.decode_jpeg(b"fake")
    assert img.shape == (10, 10, 3)


def test_flatten_boxes_empty_predictions():
    assert detector._flatten_boxes([]) == []


def test_flatten_boxes_unknown_shape_raises():
    # No cls_name / boxes attribute -> must raise, never silently return [].
    class _X:
        pass

    with pytest.raises(ValueError):
        detector._flatten_boxes([_X()])


def test_load_yolo_model_lazy(monkeypatch):
    """load_yolo_model is the only place ultralytics is imported live."""

    class FakeYOLO:
        def __init__(self, path):
            self.path = path

    fake_module = MagicMock()
    fake_module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    m = detector.load_yolo_model("x.pt")
    assert isinstance(m, FakeYOLO)
    assert m.path == "x.pt"


# =====================================================================
# cli.py — _cmd_replay / _cmd_train / _cmd_validate / _cmd_verify
# =====================================================================


def test_cmd_replay_forwards_args(monkeypatch, tmp_path):
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("verification.replay_harness.main", fake_main)
    args = argparse.Namespace(
        clip=tmp_path / "c", model="marker", window=4, conf=0.5, json_out=None
    )
    rc = cli._cmd_replay(args)
    assert rc == 0
    assert "--window" in seen["argv"] and "4" in seen["argv"]


def test_cmd_replay_with_json_out(monkeypatch, tmp_path):
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("verification.replay_harness.main", fake_main)
    args = argparse.Namespace(
        clip=tmp_path / "c",
        model="marker",
        window=4,
        conf=0.5,
        json_out=tmp_path / "out.json",
    )
    cli._cmd_replay(args)
    assert "--json-out" in seen["argv"]


def test_cmd_train_forwards(monkeypatch):
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("training.train.main", fake_main)
    args = argparse.Namespace(data="d.yaml", epochs=10)
    cli._cmd_train(args)
    assert "--epochs" in seen["argv"] and "10" in seen["argv"]


def test_cmd_validate_forwards(monkeypatch):
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("training.validate.main", fake_main)
    args = argparse.Namespace(weights="w.pt", data="d.yaml")
    cli._cmd_validate(args)
    assert "--weights" in seen["argv"]


def test_cmd_verify_uses_subprocess(monkeypatch):
    """_cmd_verify shells out to pytest. Patch subprocess.call to confirm."""
    seen = {}

    def fake_call(cmd):
        seen["cmd"] = cmd
        return 0

    monkeypatch.setattr("subprocess.call", fake_call)
    rc = cli._cmd_verify(argparse.Namespace())
    assert rc == 0
    assert "pytest" in seen["cmd"]


# =====================================================================
# viewer.py — remaining branches
# =====================================================================


def test_status_text_cooldown_no_streak_shown():
    s = viewer._status_text(
        viewer.GuardState.COOLDOWN, streak=2, window=6, last_result=None, last_trigger_ts=None
    )
    # COOLDOWN should not include streak counter (only ARMED/ALERTING do)
    assert "2/6" not in s


def test_status_text_zero_conf_omitted():
    from spaghetti_guard.detector import FrameResult

    r = FrameResult(hit=False, conf=0.0, best_class=None)
    s = viewer._status_text(
        viewer.GuardState.ARMED, streak=0, window=6, last_result=r, last_trigger_ts=None
    )
    assert "0.00" not in s  # no confidence text when conf == 0


# =====================================================================
# guard.py — sigint / camera-loss not-armed branches
# =====================================================================


def test_guard_tick_when_idle_is_noop(tmp_path):
    from spaghetti_guard.detector import FailureDetector
    from spaghetti_guard.guard import Guard
    from spaghetti_guard.notifier import NoopNotifier

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
    )
    g.tick()  # no-op while IDLE


def test_guard_request_stop_sets_event(tmp_path):
    from spaghetti_guard.detector import FailureDetector
    from spaghetti_guard.guard import Guard
    from spaghetti_guard.notifier import NoopNotifier

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
    )
    assert g.stopped is False
    g.request_stop()
    assert g.stopped is True


def test_guard_run_loop_exits_on_stop_event(tmp_path):
    """run() should bail out if request_stop() is set before the iterator yields."""
    from spaghetti_guard.detector import FailureDetector
    from spaghetti_guard.guard import Guard
    from spaghetti_guard.notifier import NoopNotifier

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
    )
    g.request_stop()
    g.run(iter([b"f1", b"f2", b"f3"]))  # exits without processing


# =====================================================================
# Frame iter helpers — clean EOF on empty buffer
# =====================================================================


def test_iter_frames_resilient_to_bad_payload_then_good():
    """A bad frame in the middle of a good stream must be skipped, not crash."""
    from spaghetti_guard.camera import iter_frames_from_stream
    import struct
    import io

    good_jpeg = b"\xff\xd8payload\xff\xd9"
    bad_jpeg = b"\x00\x00not-a-jpeg-at-all-but-the-right-length-12345"
    blob = (
        struct.pack("<I", len(bad_jpeg)) + b"\x00" * 12 + bad_jpeg
        + struct.pack("<I", len(good_jpeg)) + b"\x00" * 12 + good_jpeg
    )
    buf = io.BytesIO(blob)

    def read_exact(n):
        data = buf.read(n)
        if len(data) < n:
            raise EOFError
        return data

    frames = list(iter_frames_from_stream(read_exact))
    assert frames == [good_jpeg]
