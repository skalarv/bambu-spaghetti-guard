"""Guard ask-mode wiring + viewer.update flow."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from spaghetti_guard.detector import FailureDetector, FrameResult
from spaghetti_guard.guard import Guard, GuardState
from spaghetti_guard.notifier import NoopNotifier
from spaghetti_guard.viewer import ConfirmDecision


# ---- fakes --------------------------------------------------------------


@dataclass
class FakeBox:
    cls_name: str
    conf: float


class FakeYolo:
    def __init__(self):
        self._next: list[FakeBox] = []

    def set_next(self, boxes):
        self._next = boxes

    def predict(self, image, **kwargs):
        return list(self._next)


class FakeControl:
    def __init__(self):
        self.stop_calls = 0
        self.pause_calls = 0

    def stop(self):
        self.stop_calls += 1

    def pause(self):
        self.pause_calls += 1


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class StateProvider:
    def __init__(self, state="IDLE"):
        self.state = state

    def __call__(self):
        return self.state


class RecordingViewer:
    def __init__(self, *, decision=ConfirmDecision.STOP, raise_on_confirm=False):
        self.updates = []
        self.confirm_calls = []
        self._decision = decision
        self._raise = raise_on_confirm
        self.start_called = False
        self.stop_called = False

    def start(self):
        self.start_called = True

    def stop(self):
        self.stop_called = True

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def request_confirm_stop(self, *, timeout_s):
        self.confirm_calls.append(timeout_s)
        if self._raise:
            raise RuntimeError("viewer is sad")
        return self._decision


# ---- fixtures -----------------------------------------------------------


@pytest.fixture
def env(tmp_path):
    yolo = FakeYolo()
    detector = FailureDetector(
        yolo,
        failure_classes=("spaghetti",),
        conf_threshold=0.5,
        decoder=lambda j: j,
    )
    control = FakeControl()
    clock = FakeClock()
    provider = StateProvider("IDLE")

    def make_guard(*, viewer=None, action_mode="ask", **overrides):
        kw = dict(
            detector=detector,
            control=control,
            notifier=NoopNotifier(),
            gcode_state_provider=provider,
            action_mode=action_mode,
            debounce_window=2,
            cooldown_s=30,
            camera_timeout_s=15,
            snapshot_dir=tmp_path / "snaps",
            now=clock,
            viewer=viewer,
        )
        kw.update(overrides)
        return Guard(**kw)

    return {
        "yolo": yolo,
        "control": control,
        "clock": clock,
        "provider": provider,
        "make_guard": make_guard,
    }


# ---- viewer.update is invoked per frame -------------------------------


def test_viewer_update_called_each_frame(env):
    v = RecordingViewer()
    g = env["make_guard"](viewer=v, action_mode="pause")
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([])
    for _ in range(3):
        g.feed_frame(b"jpeg")
    assert len(v.updates) == 3
    # Every update has the keys the viewer needs
    for u in v.updates:
        assert {"jpeg", "state", "streak", "window", "last_result", "last_trigger_ts"} <= set(u)


def test_viewer_update_called_when_state_not_running(env):
    """Even before the print starts, the viewer should get frames so the user
    can confirm the camera is working."""
    v = RecordingViewer()
    g = env["make_guard"](viewer=v, action_mode="pause")
    env["provider"].state = "IDLE"
    g.feed_frame(b"jpeg")
    assert len(v.updates) == 1
    assert v.updates[0]["last_result"] is None  # detection skipped


def test_viewer_update_failure_suppressed(env):
    class BoomViewer(RecordingViewer):
        def update(self, **kwargs):
            raise RuntimeError("UI broke")

    v = BoomViewer()
    g = env["make_guard"](viewer=v, action_mode="pause")
    env["provider"].state = "RUNNING"
    # Must not raise
    g.feed_frame(b"jpeg")


# ---- ask mode routes through viewer -----------------------------------


def test_ask_mode_stop_decision_routes_to_stop(env):
    v = RecordingViewer(decision=ConfirmDecision.STOP)
    g = env["make_guard"](viewer=v, action_mode="ask", ask_timeout_s=5)
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.95)])
    g.feed_frame(b"j")
    g.feed_frame(b"j")  # window=2, fires
    assert env["control"].stop_calls == 1
    assert env["control"].pause_calls == 0
    assert v.confirm_calls == [5]


def test_ask_mode_cancel_decision_skips_action(env):
    v = RecordingViewer(decision=ConfirmDecision.CANCEL)
    g = env["make_guard"](viewer=v, action_mode="ask")
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.95)])
    g.feed_frame(b"j")
    g.feed_frame(b"j")
    assert env["control"].stop_calls == 0
    assert env["control"].pause_calls == 0
    # Guard still transitions through TRIGGERED -> COOLDOWN even on skip
    assert g.state == GuardState.COOLDOWN


def test_ask_mode_timeout_falls_back_to_stop_by_default(env):
    v = RecordingViewer(decision=ConfirmDecision.TIMEOUT)
    g = env["make_guard"](viewer=v, action_mode="ask")
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.95)])
    g.feed_frame(b"j")
    g.feed_frame(b"j")
    assert env["control"].stop_calls == 1


def test_ask_mode_timeout_can_fall_back_to_pause(env):
    v = RecordingViewer(decision=ConfirmDecision.TIMEOUT)
    g = env["make_guard"](viewer=v, action_mode="ask", ask_timeout_action="pause")
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.95)])
    g.feed_frame(b"j")
    g.feed_frame(b"j")
    assert env["control"].pause_calls == 1
    assert env["control"].stop_calls == 0


def test_ask_mode_no_viewer_falls_back(env):
    g = env["make_guard"](viewer=None, action_mode="ask", ask_timeout_action="pause")
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.95)])
    g.feed_frame(b"j")
    g.feed_frame(b"j")
    assert env["control"].pause_calls == 1


def test_ask_mode_viewer_raises_falls_back(env):
    v = RecordingViewer(raise_on_confirm=True)
    g = env["make_guard"](viewer=v, action_mode="ask")
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.95)])
    g.feed_frame(b"j")
    g.feed_frame(b"j")
    assert env["control"].stop_calls == 1


# ---- ask-mode ordering + retry -----------------------------------------


def test_ask_mode_notifies_before_prompting(env):
    """The phone alert must be on its way while the ask dialog waits for the
    operator — not after a 30s prompt timeout."""
    order = []

    class OrderNotifier(NoopNotifier):
        def send(self, title, message, image_path=None):
            order.append("notify")
            return True

    class OrderViewer(RecordingViewer):
        def request_confirm_stop(self, *, timeout_s):
            order.append("prompt")
            return super().request_confirm_stop(timeout_s=timeout_s)

    v = OrderViewer(decision=ConfirmDecision.STOP)
    g = env["make_guard"](viewer=v, notifier=OrderNotifier())
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.95)])
    g.feed_frame(b"j")
    g.feed_frame(b"j")
    assert order[:2] == ["notify", "prompt"]


def test_ask_mode_failed_action_retry_does_not_reprompt(env):
    """When the operator already answered and only the publish failed, the
    retry must reuse the decision instead of asking again."""

    class FailOnceControl:
        def __init__(self):
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise RuntimeError("broker unreachable")

        def pause(self):
            raise AssertionError("operator chose stop")

    v = RecordingViewer(decision=ConfirmDecision.STOP)
    control = FailOnceControl()
    g = env["make_guard"](viewer=v, control=control, action_retry_s=5)
    env["provider"].state = "RUNNING"
    env["yolo"].set_next([FakeBox("spaghetti", 0.95)])
    g.feed_frame(b"j")
    g.feed_frame(b"j")  # fires; stop fails
    assert control.stop_calls == 1
    env["clock"].advance(6)
    g.feed_frame(b"j")  # retry
    assert control.stop_calls == 2
    assert len(v.confirm_calls) == 1  # prompted exactly once
    assert g.state == GuardState.COOLDOWN


# ---- invalid construction ---------------------------------------------


def test_invalid_ask_timeout_action_rejected(env):
    with pytest.raises(ValueError):
        env["make_guard"](action_mode="ask", ask_timeout_action="bogus")
