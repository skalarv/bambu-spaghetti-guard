"""ViewerLogic — display projection + confirm lifecycle."""

from __future__ import annotations

import pytest

from spaghetti_guard.detector import FrameResult
from spaghetti_guard.guard import GuardState
from spaghetti_guard.viewer import (
    ConfirmDecision,
    HeadlessViewer,
    ViewerLogic,
    _border_for,
    _header_text,
    _status_text,
)


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ---- border colour per state -------------------------------------------


def test_border_for_each_state():
    for state in GuardState:
        c = _border_for(state)
        assert c.startswith("#") and len(c) == 7


def test_border_distinct_alert_vs_armed():
    assert _border_for(GuardState.ARMED) != _border_for(GuardState.ALERTING)
    assert _border_for(GuardState.TRIGGERED) != _border_for(GuardState.ARMED)


# ---- status text -------------------------------------------------------


def test_status_text_idle_minimal():
    s = _status_text(GuardState.IDLE, streak=0, window=6, last_result=None, last_trigger_ts=None)
    assert "IDLE" in s


def test_status_text_armed_shows_streak():
    s = _status_text(GuardState.ARMED, streak=2, window=6, last_result=None, last_trigger_ts=None)
    assert "ARMED" in s
    assert "2/6" in s


def test_status_text_with_result():
    r = FrameResult(hit=True, conf=0.81, best_class="spaghetti")
    s = _status_text(GuardState.ALERTING, streak=3, window=6, last_result=r, last_trigger_ts=None)
    assert "0.81" in s
    assert "spaghetti" in s


def test_status_text_with_trigger_age():
    import time as _t

    now = _t.time()
    s = _status_text(
        GuardState.COOLDOWN, streak=0, window=6, last_result=None, last_trigger_ts=now - 5
    )
    assert "last trigger" in s


# ---- header text -------------------------------------------------------


def test_header_pending_confirm_overrides_state():
    assert "confirm" in _header_text(GuardState.ALERTING, pending_confirm=True).lower()


def test_header_each_state_has_text():
    for state in GuardState:
        h = _header_text(state, pending_confirm=False)
        assert h and isinstance(h, str)


# ---- ViewerLogic.current_display --------------------------------------


def test_current_display_idle():
    logic = ViewerLogic()
    d = logic.current_display(
        state=GuardState.IDLE, streak=0, window=6, last_result=None, last_trigger_ts=None
    )
    assert d.show_confirm is False
    assert d.confirm_seconds_left is None
    assert "IDLE" in d.status_text


def test_current_display_alerting():
    logic = ViewerLogic()
    r = FrameResult(hit=True, conf=0.7, best_class="spaghetti")
    d = logic.current_display(
        state=GuardState.ALERTING, streak=3, window=6, last_result=r, last_trigger_ts=None
    )
    assert "ALERTING" in d.status_text
    assert d.border_color == _border_for(GuardState.ALERTING)


# ---- Confirm lifecycle ------------------------------------------------


def test_confirm_request_then_stop():
    clock = FakeClock()
    logic = ViewerLogic(now=clock)
    assert logic.poll_confirm() is None  # nothing pending
    logic.request_confirm(timeout_s=10)
    assert logic.has_pending_confirm()
    assert logic.poll_confirm() is None  # still pending, before deadline
    assert logic.submit_decision(ConfirmDecision.STOP) is True
    assert logic.poll_confirm() is ConfirmDecision.STOP
    # Second poll clears it
    assert logic.poll_confirm() is None


def test_confirm_request_then_cancel():
    logic = ViewerLogic()
    logic.request_confirm(timeout_s=10)
    assert logic.submit_decision(ConfirmDecision.CANCEL) is True
    assert logic.poll_confirm() is ConfirmDecision.CANCEL


def test_confirm_timeout_promoted():
    clock = FakeClock()
    logic = ViewerLogic(now=clock)
    logic.request_confirm(timeout_s=5)
    clock.advance(6)
    assert logic.poll_confirm() is ConfirmDecision.TIMEOUT
    assert logic.has_pending_confirm() is False


def test_submit_decision_without_request_is_noop():
    logic = ViewerLogic()
    assert logic.submit_decision(ConfirmDecision.STOP) is False


def test_double_submit_keeps_first():
    logic = ViewerLogic()
    logic.request_confirm(timeout_s=10)
    logic.submit_decision(ConfirmDecision.CANCEL)
    # A second decision should be ignored.
    assert logic.submit_decision(ConfirmDecision.STOP) is False
    assert logic.poll_confirm() is ConfirmDecision.CANCEL


def test_confirm_seconds_left():
    clock = FakeClock()
    logic = ViewerLogic(now=clock)
    assert logic.confirm_seconds_left() is None
    logic.request_confirm(timeout_s=10)
    assert logic.confirm_seconds_left() == pytest.approx(10.0)
    clock.advance(7)
    assert logic.confirm_seconds_left() == pytest.approx(3.0)
    clock.advance(5)
    assert logic.confirm_seconds_left() == pytest.approx(0.0)


def test_display_show_confirm_flag():
    logic = ViewerLogic()
    logic.request_confirm(timeout_s=10)
    d = logic.current_display(
        state=GuardState.TRIGGERED, streak=6, window=6, last_result=None, last_trigger_ts=None
    )
    assert d.show_confirm is True
    assert d.confirm_seconds_left is not None


# ---- HeadlessViewer ---------------------------------------------------


def test_headless_viewer_no_window_defaults_stop():
    v = HeadlessViewer()
    v.start()
    v.update(jpeg=None, state=GuardState.ARMED, streak=0, window=6, last_result=None, last_trigger_ts=None)
    assert v.request_confirm_stop(timeout_s=1) is ConfirmDecision.STOP
    v.stop()
