"""Brief §6.5 — full guard loop against MockPrinter.

These tests boot the real camera + control modules against the in-process mock
printer (TLS camera socket + plain-MQTT amqtt broker). The detector is a
deliberately tiny fake that reads the marker byte injected by
`mock_printer.make_jpeg` so we don't need YOLO weights / cv2 to exercise the
loop end-to-end.
"""

from __future__ import annotations

import asyncio
import ssl
import threading
import time
from pathlib import Path

import pytest

from spaghetti_guard.camera import RawSocketBackend
from spaghetti_guard.control import PrinterControl
from spaghetti_guard.detector import FailureDetector
from spaghetti_guard.guard import Guard, GuardState
from spaghetti_guard.notifier import NoopNotifier
from verification.mock_printer import (
    MARKER_CLEAN,
    MARKER_FAILURE,
    MockConfig,
    MockPrinter,
    extract_marker,
)


# ---------------------------------------------------------------------------
# Marker-based fake detector adapter
# ---------------------------------------------------------------------------


class _MarkerBox:
    __slots__ = ("cls_name", "conf")

    def __init__(self, cls_name: str, conf: float) -> None:
        self.cls_name = cls_name
        self.conf = conf


class MarkerYolo:
    """A 'YOLO' that returns one box if the JPEG carries the failure marker.

    Used only in the integration test: it lets us drive ground-truth failure
    onsets via mock_printer.inject_failure_at without involving cv2 or
    ultralytics.
    """

    def predict(self, image, **kwargs):
        # image is the raw JPEG bytes (decoder=identity below)
        marker = extract_marker(image)
        if marker == MARKER_FAILURE:
            return [_MarkerBox("spaghetti", 0.95)]
        return []


def _identity_decoder(jpeg):
    return jpeg


# ---------------------------------------------------------------------------
# Helper: run the guard in a background thread
# ---------------------------------------------------------------------------


def _start_guard_thread(guard: Guard, frame_iter):
    t = threading.Thread(target=guard.run, args=(frame_iter,), daemon=True)
    t.start()
    return t


def _ssl_client_context_insecure() -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def _wait_for(predicate, timeout_s=10.0, interval_s=0.05):
    """Poll until predicate() is truthy or timeout. Raises AssertionError on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval_s)
    raise AssertionError(f"predicate {predicate} did not become true within {timeout_s}s")


# ---------------------------------------------------------------------------
# Shared setup builder
# ---------------------------------------------------------------------------


def _build_guard_stack(mock: MockPrinter, tmp_path: Path, *, debounce_window=3):
    cam = RawSocketBackend(
        host="127.0.0.1",
        port=mock.camera_port,
        username=mock.cfg.username,
        access_code=mock.cfg.access_code,
        ssl_context=_ssl_client_context_insecure(),
        recv_timeout_s=5.0,
    )
    control = PrinterControl(
        host="127.0.0.1",
        serial=mock.cfg.serial,
        access_code=mock.cfg.access_code,
        port=mock.mqtt_port,
        use_tls=False,
    )
    detector = FailureDetector(
        MarkerYolo(),
        failure_classes=("spaghetti",),
        conf_threshold=0.5,
        decoder=_identity_decoder,
    )
    guard = Guard(
        detector=detector,
        control=control,
        notifier=NoopNotifier(),
        gcode_state_provider=lambda: control.state.snapshot()[0],
        action_mode="stop",
        debounce_window=debounce_window,
        cooldown_s=60,
        camera_timeout_s=10,
        snapshot_dir=tmp_path / "snaps",
    )
    return cam, control, guard


async def _connect_off_loop(*funcs) -> None:
    """Run blocking connect calls in a worker thread so the asyncio mock can serve them."""
    loop = asyncio.get_event_loop()
    for fn in funcs:
        await loop.run_in_executor(None, fn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spaghetti_clip_fires_stop_after_window(tmp_path):
    """Scenario A: failure injected mid-stream -> single stop after exactly
    `consecutive_hits` qualifying frames."""
    cfg = MockConfig(fps=20.0)  # fast frames so the test finishes quickly
    async with MockPrinter(cfg) as mock:
        cam, control, guard = _build_guard_stack(mock, tmp_path, debounce_window=3)
        try:
            await _connect_off_loop(
                cam.connect,
                lambda: control.connect(wait_s=5.0) or None,
            )
            await mock.broker.set_gcode_state("RUNNING", layer_num=10)
            thread = _start_guard_thread(guard, cam.frames())

            # Wait for the guard to actually arm against the printer state.
            await _wait_for(lambda: guard.state == GuardState.ARMED, timeout_s=5)

            # Stream a few clean frames first, then inject failure.
            await asyncio.sleep(0.4)
            assert len(mock.broker.recorder) == 0  # no command yet
            assert mock.camera is not None
            mock.camera.inject_failure_at(mock.camera.frames_streamed + 1)

            # Wait for a single request command to land in the recorder.
            await _wait_for(lambda: len(mock.broker.recorder) >= 1, timeout_s=10)

            recorded = mock.broker.recorder.snapshot()
            assert len(recorded) == 1, f"expected exactly 1 command, got {len(recorded)}"
            cmd = recorded[0]
            assert cmd.topic == f"device/{mock.cfg.serial}/request"
            assert cmd.payload == {"print": {"command": "stop", "sequence_id": "0"}}
            assert cmd.qos == 1
            assert guard.state == GuardState.COOLDOWN
        finally:
            guard.request_stop()
            cam.close()
            control.close()
            # give the thread a beat to wind down on its own
            if 'thread' in locals():
                thread.join(timeout=2)


@pytest.mark.asyncio
async def test_clean_clip_silent(tmp_path):
    """Scenario B: stream clean frames for many cycles -> no command ever sent."""
    cfg = MockConfig(fps=20.0)
    async with MockPrinter(cfg) as mock:
        cam, control, guard = _build_guard_stack(mock, tmp_path, debounce_window=3)
        try:
            await _connect_off_loop(
                cam.connect,
                lambda: control.connect(wait_s=5.0) or None,
            )
            await mock.broker.set_gcode_state("RUNNING", layer_num=10)
            thread = _start_guard_thread(guard, cam.frames())
            await _wait_for(lambda: guard.state == GuardState.ARMED, timeout_s=5)
            # Stream long enough that we'd have triggered if anything were hitting.
            await _wait_for(
                lambda: mock.camera is not None and mock.camera.frames_streamed >= 30,
                timeout_s=10,
            )
            assert len(mock.broker.recorder) == 0
            assert guard.state in (GuardState.ARMED, GuardState.ALERTING)
        finally:
            guard.request_stop()
            cam.close()
            control.close()
            if 'thread' in locals():
                thread.join(timeout=2)


@pytest.mark.asyncio
async def test_disarms_on_finish_before_fire(tmp_path):
    """Scenario C: failure stream begins, then state -> FINISH before window
    fills. Guard must disarm and never publish a command."""
    cfg = MockConfig(fps=20.0)
    async with MockPrinter(cfg) as mock:
        # Large debounce window so FINISH lands before the trigger.
        cam, control, guard = _build_guard_stack(mock, tmp_path, debounce_window=20)
        try:
            await _connect_off_loop(
                cam.connect,
                lambda: control.connect(wait_s=5.0) or None,
            )
            await mock.broker.set_gcode_state("RUNNING", layer_num=10)
            thread = _start_guard_thread(guard, cam.frames())
            await _wait_for(lambda: guard.state == GuardState.ARMED, timeout_s=5)

            # Inject failure immediately so frames start hitting.
            assert mock.camera is not None
            mock.camera.inject_failure_at(0)

            # Wait for the guard to start alerting (some hits have accumulated).
            await _wait_for(lambda: guard.debounce_streak >= 3, timeout_s=5)
            assert guard.state == GuardState.ALERTING
            assert len(mock.broker.recorder) == 0

            # Now switch state to FINISH — guard must disarm without firing.
            await mock.broker.set_gcode_state("FINISH")
            await _wait_for(lambda: guard.state == GuardState.IDLE, timeout_s=5)

            # Let a few more frames flow to be sure no late stop appears.
            await asyncio.sleep(0.5)
            assert len(mock.broker.recorder) == 0
        finally:
            guard.request_stop()
            cam.close()
            control.close()
            if 'thread' in locals():
                thread.join(timeout=2)
