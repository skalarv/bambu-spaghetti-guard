"""Brief §3.1 + §6.4 — state-machine + watchdog correctness."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from spaghetti_guard.detector import FailureDetector
from spaghetti_guard.guard import Guard, GuardState
from spaghetti_guard.notifier import NoopNotifier


# ---- fakes --------------------------------------------------------------


@dataclass
class FakeBox:
    cls_name: str
    conf: float


class FakeYolo:
    def __init__(self):
        self._next_boxes: list[FakeBox] = []

    def set_next(self, boxes: list[FakeBox]) -> None:
        self._next_boxes = boxes

    def predict(self, image, **kwargs):
        return list(self._next_boxes)


class FakeControl:
    def __init__(self):
        self.stop_calls = 0
        self.pause_calls = 0

    def stop(self):
        self.stop_calls += 1

    def pause(self):
        self.pause_calls += 1


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class StateProvider:
    """Holds a printer state value the test can flip on demand."""

    def __init__(self, state: str = "IDLE"):
        self.state = state

    def __call__(self) -> str:
        return self.state


def _identity_decoder(jpeg):
    return jpeg


# ---- fixtures -----------------------------------------------------------


@pytest.fixture
def env(tmp_path):
    yolo = FakeYolo()
    detector = FailureDetector(
        yolo,
        failure_classes=("spaghetti",),
        conf_threshold=0.5,
        decoder=_identity_decoder,
    )
    control = FakeControl()
    notifier = NoopNotifier()
    clock = FakeClock()
    provider = StateProvider("IDLE")

    def make_guard(**overrides) -> Guard:
        kw = dict(
            detector=detector,
            control=control,
            notifier=notifier,
            gcode_state_provider=provider,
            action_mode="stop",
            debounce_window=3,
            cooldown_s=30,
            camera_timeout_s=15,
            snapshot_dir=tmp_path / "snaps",
            now=clock,
        )
        kw.update(overrides)
        return Guard(**kw)

    return {
        "yolo": yolo,
        "detector": detector,
        "control": control,
        "notifier": notifier,
        "clock": clock,
        "provider": provider,
        "make_guard": make_guard,
        "tmp_path": tmp_path,
    }


# ---- detection only while RUNNING ---------------------------------------


def test_detection_skipped_when_idle(env):
    g = env["make_guard"]()
    env["yolo"].set_next([FakeBox("spaghetti", 0.99)])
    # printer is IDLE — feeding frames should never fire
    for _ in range(10):
        r = g.feed_frame(b"jpeg")
        assert not r.fired
    assert env["control"].stop_calls == 0
    assert g.state == GuardState.IDLE


def test_arms_on_running_then_fires_after_window(env):
    g = env["make_guard"](debounce_window=3)
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.9)])
    r1 = g.feed_frame(b"j")
    r2 = g.feed_frame(b"j")
    assert not r1.fired and not r2.fired
    assert g.state == GuardState.ALERTING
    r3 = g.feed_frame(b"j")
    assert r3.fired
    assert env["control"].stop_calls == 1
    assert g.state == GuardState.COOLDOWN


def test_single_miss_resets_alert(env):
    g = env["make_guard"](debounce_window=3)
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.9)])
    g.feed_frame(b"j")
    g.feed_frame(b"j")
    env["yolo"].set_next([])  # clean frame
    r = g.feed_frame(b"j")
    assert not r.fired
    assert g.state == GuardState.ARMED
    assert g.debounce_streak == 0


# ---- disarm transitions -------------------------------------------------


@pytest.mark.parametrize("end_state", ["FINISH", "FAILED", "IDLE"])
def test_disarms_on_end_state(env, end_state):
    g = env["make_guard"]()
    env["provider"].state = "RUNNING"
    g.feed_frame(b"j")  # arms
    env["provider"].state = end_state
    g.feed_frame(b"j")
    assert g.state == GuardState.IDLE
    assert env["control"].stop_calls == 0


def test_finish_mid_alert_aborts_without_firing(env):
    g = env["make_guard"](debounce_window=4)
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.9)])
    g.feed_frame(b"j")
    g.feed_frame(b"j")
    g.feed_frame(b"j")  # 3 of 4 — still ALERTING
    assert g.state == GuardState.ALERTING
    env["provider"].state = "FINISH"
    g.feed_frame(b"j")
    assert g.state == GuardState.IDLE
    assert env["control"].stop_calls == 0


# ---- camera loss policy -------------------------------------------------


def test_camera_loss_notifies_never_stops(env):
    g = env["make_guard"](camera_timeout_s=5)
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([])
    g.feed_frame(b"j")  # marks last_frame_ts
    assert g.state == GuardState.ARMED
    # simulate 10s of camera silence
    env["clock"].advance(10)
    g.tick()
    # guard should still be armed; no stop sent
    assert g.state == GuardState.ARMED
    assert env["control"].stop_calls == 0
    assert env["control"].pause_calls == 0


def test_camera_loss_alert_only_once_per_outage(env, monkeypatch):
    """Watchdog must not spam notifications on every tick."""
    g = env["make_guard"](camera_timeout_s=5)
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([])
    g.feed_frame(b"j")
    calls = []
    monkeypatch.setattr(env["notifier"], "send", lambda *a, **kw: calls.append(a) or True)
    env["clock"].advance(10)
    g.tick()
    g.tick()
    g.tick()
    assert len(calls) == 1  # single alert per outage


# ---- printer-report staleness -------------------------------------------
# A guard that trusts a stale gcode_state is silently unprotected (stale
# IDLE never arms; stale RUNNING trusts a print that may have ended).


def test_stale_printer_report_notifies_once(env):
    notifier = CaptureNotifier()
    age = {"v": 1.0}
    g = env["make_guard"](
        notifier=notifier, state_age_provider=lambda: age["v"], mqtt_timeout_s=30
    )
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([])
    g.feed_frame(b"j")
    age["v"] = 45.0
    g.tick()
    g.tick()
    g.tick()
    stale_alerts = [t for t, _ in notifier.sent if "report" in t.lower()]
    assert len(stale_alerts) == 1  # once per outage, no spam
    # staleness never touches the printer
    assert env["control"].stop_calls == 0
    assert env["control"].pause_calls == 0


def test_stale_report_realerts_after_recovery(env):
    notifier = CaptureNotifier()
    age = {"v": 45.0}
    g = env["make_guard"](
        notifier=notifier, state_age_provider=lambda: age["v"], mqtt_timeout_s=30
    )
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([])
    g.feed_frame(b"j")
    g.tick()  # first outage
    age["v"] = 1.0
    g.tick()  # recovered
    age["v"] = 60.0
    g.tick()  # second outage
    stale_alerts = [t for t, _ in notifier.sent if "report" in t.lower()]
    assert len(stale_alerts) == 2


def test_stale_report_checked_even_when_disarmed(env):
    """Stale IDLE is the nastiest case: the guard never arms and nobody
    notices — the operator must be told."""
    notifier = CaptureNotifier()
    g = env["make_guard"](
        notifier=notifier, state_age_provider=lambda: 120.0, mqtt_timeout_s=30
    )
    env["provider"].state = "IDLE"
    g.tick()
    stale_alerts = [t for t, _ in notifier.sent if "report" in t.lower()]
    assert len(stale_alerts) == 1


def test_no_report_yet_is_not_stale(env):
    """Before the first report arrives (provider returns None) there is no
    baseline — don't alert at startup."""
    notifier = CaptureNotifier()
    g = env["make_guard"](
        notifier=notifier, state_age_provider=lambda: None, mqtt_timeout_s=30
    )
    g.tick()
    assert notifier.sent == []


def test_watchdog_fires_while_frame_iterator_is_blocked(env):
    """A stalled camera blocks run() inside the frame iterator (recv never
    returns). The camera-silence alert must fire anyway — the watchdog cannot
    depend on frames arriving to run."""
    import threading
    import time as real_time

    notifier = CaptureNotifier()
    g = env["make_guard"](camera_timeout_s=5, notifier=notifier)
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([])
    g.feed_frame(b"j")  # ARMED + last_frame_ts recorded
    assert g.state == GuardState.ARMED

    release = threading.Event()

    def stalled_stream():
        release.wait(5.0)  # simulates a recv() that never returns
        if False:
            yield b""  # pragma: no cover  (makes this a generator)

    env["clock"].advance(10)  # camera silent well past the 5s timeout

    t = threading.Thread(
        target=g.run,
        args=(stalled_stream(),),
        kwargs={"tick_interval_s": 0.01},
        daemon=True,
    )
    t.start()
    deadline = real_time.time() + 2.0
    while real_time.time() < deadline and not notifier.sent:
        real_time.sleep(0.01)
    g.request_stop()
    release.set()
    t.join(timeout=2.0)

    assert notifier.sent, "camera-silence alert never fired while the iterator was blocked"
    title, _ = notifier.sent[0]
    assert "camera" in title.lower()


# ---- cooldown prevents back-to-back ------------------------------------


def test_cooldown_blocks_second_trigger(env):
    g = env["make_guard"](debounce_window=2, cooldown_s=30)
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.95)])
    g.feed_frame(b"j")
    g.feed_frame(b"j")  # fires
    assert env["control"].stop_calls == 1
    assert g.state == GuardState.COOLDOWN
    # advance only 5s — still cooling
    env["clock"].advance(5)
    g.feed_frame(b"j")
    g.feed_frame(b"j")
    assert env["control"].stop_calls == 1  # no second fire


def test_cooldown_expiry_rearms(env):
    g = env["make_guard"](debounce_window=2, cooldown_s=10)
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.95)])
    g.feed_frame(b"j")
    g.feed_frame(b"j")  # fires
    env["clock"].advance(11)  # cooldown elapsed
    g.feed_frame(b"j")
    assert g.state == GuardState.ALERTING
    g.feed_frame(b"j")
    assert env["control"].stop_calls == 2  # fired again


# ---- pause-vs-stop action mode -----------------------------------------


def test_pause_mode_calls_pause_not_stop(env):
    g = env["make_guard"](action_mode="pause", debounce_window=1)
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.9)])
    g.feed_frame(b"j")
    assert env["control"].pause_calls == 1
    assert env["control"].stop_calls == 0


# ---- fire ordering + action-failure handling ----------------------------
# The control action is the safety-critical step: it must run before the
# (blocking, HTTP) notification, and a failed action must alert the operator
# and retry — but never storm at frame rate.


class CaptureNotifier:
    def __init__(self, order: list | None = None):
        self.sent: list[tuple[str, str]] = []
        self._order = order

    def send(self, title, message, image_path=None):
        self.sent.append((title, message))
        if self._order is not None:
            self._order.append("notify")
        return True


class FailingControl:
    def __init__(self, fail_times: int | None = None):
        """fail_times=None → always fail; N → fail the first N calls."""
        self.calls = 0
        self._fail_times = fail_times

    def stop(self):
        self.calls += 1
        if self._fail_times is None or self.calls <= self._fail_times:
            raise RuntimeError("broker unreachable")

    def pause(self):
        self.stop()


def test_fire_control_action_precedes_notification(env):
    order = []

    class OrderControl:
        def stop(self):
            order.append("control")

        def pause(self):
            order.append("control")

    g = env["make_guard"](
        control=OrderControl(), notifier=CaptureNotifier(order), debounce_window=1
    )
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.9)])
    r = g.feed_frame(b"j")
    assert r.fired
    assert order == ["control", "notify"]


def test_control_failure_sends_failure_alert_and_keeps_triggered(env):
    notifier = CaptureNotifier()
    control = FailingControl()
    g = env["make_guard"](control=control, notifier=notifier, debounce_window=1)
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.9)])
    g.feed_frame(b"j")
    assert g.state == GuardState.TRIGGERED
    assert control.calls == 1
    assert len(notifier.sent) == 1
    title, message = notifier.sent[0]
    # The single alert must be the action-failure alert, not a plain
    # "detected, action taken" message that would read as success.
    assert "fail" in title.lower()


def test_control_failure_retries_are_rate_limited(env):
    control = FailingControl()
    notifier = CaptureNotifier()
    g = env["make_guard"](
        control=control, notifier=notifier, debounce_window=1, action_retry_s=10
    )
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.9)])
    g.feed_frame(b"j")  # first attempt fails
    for _ in range(5):  # frame storm within the same second
        g.feed_frame(b"j")
    assert control.calls == 1  # no per-frame hammering
    assert len(notifier.sent) == 1  # no notification spam either
    env["clock"].advance(11)
    g.feed_frame(b"j")
    assert control.calls == 2  # retried after the backoff window


def test_control_retry_success_enters_cooldown(env):
    control = FailingControl(fail_times=1)
    notifier = CaptureNotifier()
    g = env["make_guard"](
        control=control, notifier=notifier, debounce_window=1, action_retry_s=5
    )
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.9)])
    g.feed_frame(b"j")
    assert g.state == GuardState.TRIGGERED
    env["clock"].advance(6)
    g.feed_frame(b"j")
    assert control.calls == 2
    assert g.state == GuardState.COOLDOWN
    # one snapshot per incident, not one per retry
    snaps = list((env["tmp_path"] / "snaps").iterdir())
    assert len(snaps) == 1


# ---- snapshot persisted on trigger -------------------------------------


def test_snapshot_written_on_trigger(env):
    g = env["make_guard"](debounce_window=1)
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.9)])
    g.feed_frame(b"jpeg-bytes")
    snaps = list((env["tmp_path"] / "snaps").iterdir())
    assert len(snaps) == 1
    assert snaps[0].read_bytes() == b"jpeg-bytes"
    assert "spaghetti" in snaps[0].name


# ---- invalid construction ---------------------------------------------


def test_invalid_action_mode_rejected(env):
    with pytest.raises(ValueError):
        env["make_guard"](action_mode="explode")
