"""TkViewer queue + confirm-stop polling — without spinning a Tk mainloop.

The mainloop itself (`_run_ui`) is excluded from coverage; it can only be
exercised against a real display. Everything *around* the mainloop is
testable from a regular thread.
"""

from __future__ import annotations

import threading
import time

from spaghetti_guard.detector import FrameResult
from spaghetti_guard.guard import GuardState
from spaghetti_guard.viewer import (
    ConfirmDecision,
    TkViewer,
    ViewerLogic,
)


# ---- queue behavior -----------------------------------------------------


def _payload(jpeg=b"jpg"):
    return dict(
        jpeg=jpeg,
        state=GuardState.ARMED,
        streak=1,
        window=6,
        last_result=None,
        last_trigger_ts=None,
    )


def test_update_pushes_packet():
    v = TkViewer()
    v.update(**_payload())
    assert v._latest.qsize() == 1


def test_update_drops_old_packet_when_queue_full():
    """The queue has maxlen=1. A second update before the UI thread drains
    must overwrite, not block — the live behavior we want."""
    v = TkViewer()
    v.update(**_payload(jpeg=b"first"))
    v.update(**_payload(jpeg=b"second"))
    pkt = v._latest.get_nowait()
    assert pkt.jpeg == b"second"
    assert v._latest.empty()


# ---- request_confirm_stop polling ---------------------------------------


def test_request_confirm_stop_returns_user_decision():
    v = TkViewer(logic=ViewerLogic())
    result = {}

    def reply_after_short_delay():
        time.sleep(0.1)
        v.logic.submit_decision(ConfirmDecision.STOP)

    t = threading.Thread(target=reply_after_short_delay)
    t.start()
    result["d"] = v.request_confirm_stop(timeout_s=2.0)
    t.join()
    assert result["d"] is ConfirmDecision.STOP


def test_request_confirm_stop_cancel_decision():
    v = TkViewer(logic=ViewerLogic())

    def cancel():
        time.sleep(0.05)
        v.logic.submit_decision(ConfirmDecision.CANCEL)

    threading.Thread(target=cancel, daemon=True).start()
    assert v.request_confirm_stop(timeout_s=2.0) is ConfirmDecision.CANCEL


def test_request_confirm_stop_times_out():
    v = TkViewer(logic=ViewerLogic())
    # No submitter → polling loop should hit the deadline.
    decision = v.request_confirm_stop(timeout_s=0.2)
    assert decision is ConfirmDecision.TIMEOUT


# ---- click hooks -------------------------------------------------------


def test_button_hooks_default_noop():
    v = TkViewer()
    # default lambdas should accept zero args
    v.on_pause_clicked()
    v.on_stop_clicked()


def test_button_hooks_replaceable():
    v = TkViewer()
    calls = []
    v.on_pause_clicked = lambda: calls.append("pause")
    v.on_stop_clicked = lambda: calls.append("stop")
    v.on_pause_clicked()
    v.on_stop_clicked()
    assert calls == ["pause", "stop"]


# ---- start / stop without UI ------------------------------------------


def test_stop_without_start_is_noop():
    v = TkViewer()
    v.stop()  # must not raise


# ---- ViewerLogic.current_display includes detection details -----------


def test_current_display_propagates_result():
    logic = ViewerLogic()
    r = FrameResult(hit=True, conf=0.92, best_class="blob")
    d = logic.current_display(
        state=GuardState.ALERTING, streak=4, window=6, last_result=r, last_trigger_ts=None
    )
    assert "0.92" in d.status_text
    assert "blob" in d.status_text
