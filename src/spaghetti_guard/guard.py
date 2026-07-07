"""Guard orchestrator + state machine (brief §3.1, §5.6).

The state machine in this module is the safety-critical core. Two invariants
matter most:

1. **Detection runs only while `gcode_state == RUNNING`.** Disarming on
   FINISH/FAILED/IDLE eliminates the largest source of false positives
   (idle camera frames during bed clear, filament load, etc.).
2. **Camera loss never sends a stop.** A missing frame is not a print
   failure. We notify and try to reconnect; we never publish `stop` on the
   absence of evidence.

The state machine itself is synchronous: every step is a pure function of
the previous state plus the current frame outcome. The run loop is a thin
wrapper that pulls frames, calls into the state machine, and minds the
watchdog. Tests drive `feed_frame` and `tick` directly without spinning a
thread.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .detector import Debouncer, FailureDetector, FrameResult
from .notifier import Notifier

if TYPE_CHECKING:
    # Import-time only: keeps tkinter out of the runtime import graph.
    from .viewer import ViewerLike

logger = logging.getLogger(__name__)


class GuardState(enum.Enum):
    IDLE = "idle"
    ARMED = "armed"
    ALERTING = "alerting"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"


# Printer report states that should disarm the guard (brief §3.1).
DISARM_STATES = frozenset({"FINISH", "FAILED", "IDLE", "PAUSE"})
RUN_STATE = "RUNNING"


class ControlLike(Protocol):
    def stop(self) -> object: ...
    def pause(self) -> object: ...


@dataclass(frozen=True)
class FireResult:
    fired: bool
    state: GuardState
    debounce_streak: int
    frame_result: FrameResult | None


class Guard:
    """Wires a detector + control + notifier into the brief's state machine.

    The printer state is read on every step via the injected
    `gcode_state_provider` callable (typically `lambda: control.state.snapshot()[0]`).
    """

    def __init__(
        self,
        *,
        detector: FailureDetector,
        control: ControlLike,
        notifier: Notifier,
        gcode_state_provider: Callable[[], str],
        action_mode: str = "pause",
        debounce_window: int = 6,
        cooldown_s: float = 30.0,
        camera_timeout_s: float = 15.0,
        snapshot_dir: Path | str = "./failure_snapshots",
        now: Callable[[], float] = time.time,
        viewer: ViewerLike | None = None,
        ask_timeout_s: float = 30.0,
        ask_timeout_action: str = "stop",
        action_retry_s: float = 10.0,
        state_age_provider: Callable[[], float | None] | None = None,
        mqtt_timeout_s: float = 30.0,
        snapshot_max_files: int | None = 500,
    ) -> None:
        if action_mode not in ("stop", "pause", "ask"):
            raise ValueError(f"action_mode must be stop|pause|ask, got {action_mode}")
        if ask_timeout_action not in ("stop", "pause"):
            raise ValueError(
                f"ask_timeout_action must be stop|pause, got {ask_timeout_action}"
            )
        self._detector = detector
        self._control = control
        self._notifier = notifier
        self._gcode_state_provider = gcode_state_provider
        self._action_mode = action_mode
        self._debouncer = Debouncer(debounce_window)
        self._debounce_window = debounce_window
        self._cooldown_s = cooldown_s
        self._camera_timeout_s = camera_timeout_s
        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._now = now
        self._viewer = viewer
        self._ask_timeout_s = ask_timeout_s
        self._ask_timeout_action = ask_timeout_action
        self._action_retry_s = action_retry_s
        # Seconds since the last printer report, or None before the first
        # report. A stale gcode_state means the guard is deciding blind.
        self._state_age_provider = state_age_provider
        self._mqtt_timeout_s = mqtt_timeout_s
        self._report_stale = False
        self._snapshot_max_files = snapshot_max_files

        self._state = GuardState.IDLE
        # Per-incident bookkeeping for the action-failed retry path.
        self._action_retry_at: float = 0.0
        self._pending_action: str | None = None
        self._incident_snapshot: Path | None = None
        self._last_frame_ts: float | None = None
        self._last_frame_result: FrameResult | None = None
        self._last_trigger_ts: float | None = None
        self._cooldown_until: float = 0.0
        self._camera_lost = False
        self._stop_event = threading.Event()

    # ---------------------------------------------------------------
    # Inspection
    # ---------------------------------------------------------------
    @property
    def state(self) -> GuardState:
        return self._state

    @property
    def debounce_streak(self) -> int:
        return self._debouncer.streak()

    # ---------------------------------------------------------------
    # State machine
    # ---------------------------------------------------------------
    def _check_printer_state(self) -> str:
        gs = self._gcode_state_provider()
        if gs in DISARM_STATES:
            self._disarm(reason=f"printer state={gs}")
        elif gs == RUN_STATE and self._state == GuardState.IDLE:
            self._arm()
        return gs

    def _arm(self) -> None:
        if self._state != GuardState.ARMED:
            logger.info("guard ARMED")
        self._state = GuardState.ARMED
        self._debouncer.reset()
        self._camera_lost = False

    def _disarm(self, *, reason: str) -> None:
        if self._state != GuardState.IDLE:
            logger.info("guard disarming (%s)", reason)
        self._state = GuardState.IDLE
        self._debouncer.reset()
        self._last_frame_ts = None
        self._cooldown_until = 0.0
        self._camera_lost = False
        self._action_retry_at = 0.0
        self._pending_action = None
        self._incident_snapshot = None

    def feed_frame(self, jpeg: bytes) -> FireResult:
        """Process one camera frame. Returns FireResult describing the outcome."""
        gs = self._check_printer_state()
        if gs != RUN_STATE:
            self._push_to_viewer(jpeg, last_result=None)
            return FireResult(False, self._state, self._debouncer.streak(), None)

        now = self._now()
        self._last_frame_ts = now
        self._camera_lost = False

        # COOLDOWN bookkeeping: skip detection until cooldown expires.
        if self._state == GuardState.COOLDOWN:
            if now < self._cooldown_until:
                self._push_to_viewer(jpeg, last_result=self._last_frame_result)
                return FireResult(False, self._state, self._debouncer.streak(), None)
            logger.info("guard cooldown elapsed; re-arming")
            self._state = GuardState.ARMED

        if self._state == GuardState.IDLE:
            # Defensive: arm if we got here without a state callback
            self._arm()

        result = self._detector.is_failure_frame(jpeg)
        self._last_frame_result = result
        self._debouncer.update(result.hit)

        if result.hit and self._state == GuardState.ARMED:
            self._state = GuardState.ALERTING

        if self._debouncer.confirmed():
            self._fire(jpeg, result, now)
            self._push_to_viewer(jpeg, last_result=result)
            return FireResult(True, self._state, self._debouncer.streak(), result)

        # On any miss while ALERTING, drop back to ARMED (debouncer already reset).
        if not result.hit and self._state == GuardState.ALERTING:
            self._state = GuardState.ARMED

        self._push_to_viewer(jpeg, last_result=result)
        return FireResult(False, self._state, self._debouncer.streak(), result)

    def _push_to_viewer(self, jpeg: bytes | None, last_result: FrameResult | None) -> None:
        if self._viewer is None:
            return
        try:
            self._viewer.update(
                jpeg=jpeg,
                state=self._state,
                streak=self._debouncer.streak(),
                window=self._debounce_window,
                last_result=last_result,
                last_trigger_ts=self._last_trigger_ts,
            )
        except Exception:
            logger.exception("viewer.update raised (suppressed)")

    def _notify(self, title: str, message: str, image_path: Path | None = None) -> None:
        try:
            self._notifier.send(title, message, image_path=image_path)
        except Exception:
            logger.exception("notifier raised — should have been suppressed internally")

    def _fire(self, jpeg: bytes, result: FrameResult, now: float) -> None:
        # The control action is the safety-critical step: it runs before the
        # (blocking, HTTP) notification. When the action failed, we stay
        # TRIGGERED and retry — but at most once per action_retry_s, never at
        # frame rate.
        retrying = self._state == GuardState.TRIGGERED
        if retrying and now < self._action_retry_at:
            return
        self._state = GuardState.TRIGGERED
        self._last_trigger_ts = now
        msg = (
            f"Print-failure detected on class={result.best_class} conf={result.conf:.2f}. "
            f"Action: {self._action_mode}."
        )
        if not retrying:
            logger.warning(msg)
            self._incident_snapshot = self._save_snapshot(jpeg, result, now)
        snapshot_path = self._incident_snapshot

        if retrying and self._pending_action is not None:
            # The operator (or the mode) already decided; only the publish
            # failed. Reuse the decision — never re-prompt in ask mode.
            effective_action = self._pending_action
        else:
            if self._action_mode == "ask":
                # Ask blocks up to ask_timeout_s; get the alert on its way first.
                self._notify("Bambu spaghetti detected", msg, image_path=snapshot_path)
            effective_action = self._resolve_action()

        try:
            if effective_action == "stop":
                self._control.stop()
            elif effective_action == "pause":
                self._control.pause()
            else:
                # "skip" — operator cancelled in ask mode; don't touch printer.
                logger.info("operator cancelled the action; staying out of printer's way")
        except Exception:
            logger.exception("control action failed; staying in TRIGGERED to flag operator")
            self._pending_action = effective_action
            self._action_retry_at = now + self._action_retry_s
            if not retrying:
                self._notify(
                    "Bambu action FAILED",
                    f"{msg} The {effective_action} command did NOT reach the printer "
                    f"(will retry every {self._action_retry_s:.0f}s) — check the printer NOW.",
                    image_path=snapshot_path,
                )
            return

        if self._action_mode != "ask":
            self._notify("Bambu spaghetti detected", msg, image_path=snapshot_path)
        self._pending_action = None
        self._incident_snapshot = None
        self._state = GuardState.COOLDOWN
        self._cooldown_until = now + self._cooldown_s
        # Reset the debounce buffer so the next ARMED window starts fresh — without
        # this, the still-full buffer would re-confirm on the very first hit after
        # cooldown and fire again.
        self._debouncer.reset()

    def _resolve_action(self) -> str:
        """For stop/pause modes return the same string. For ask mode prompt the
        viewer; on timeout fall back to `ask_timeout_action`. Returns 'stop',
        'pause', or 'skip' (operator cancelled)."""
        if self._action_mode != "ask":
            return self._action_mode
        if self._viewer is None:
            logger.warning("ask mode without a viewer — defaulting to %s", self._ask_timeout_action)
            return self._ask_timeout_action
        try:
            decision = self._viewer.request_confirm_stop(timeout_s=self._ask_timeout_s)
        except Exception:
            logger.exception("viewer.request_confirm_stop raised; defaulting to %s", self._ask_timeout_action)
            return self._ask_timeout_action
        # The viewer returns a ConfirmDecision enum (str-valued).
        d = getattr(decision, "value", decision)
        if d == "stop":
            return "stop"
        if d == "cancel":
            logger.info("operator cancelled the action via ask mode")
            return "skip"
        # timeout
        logger.warning("ask mode timed out after %.1fs; defaulting to %s", self._ask_timeout_s, self._ask_timeout_action)
        return self._ask_timeout_action

    def _save_snapshot(self, jpeg: bytes, result: FrameResult, now: float) -> Path:
        ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now))
        name = f"trigger-{ts}-{result.best_class or 'unknown'}-{result.conf:.2f}.jpg"
        path = self._snapshot_dir / name
        try:
            path.write_bytes(jpeg)
        except OSError:
            logger.exception("could not write snapshot %s", path)
        self._prune_snapshots()
        return path

    def _prune_snapshots(self) -> None:
        """Keep at most snapshot_max_files trigger images (oldest deleted).

        Timestamped names sort chronologically, so a plain sort suffices."""
        if not self._snapshot_max_files:
            return
        try:
            snaps = sorted(self._snapshot_dir.glob("trigger-*.jpg"))
            for old in snaps[: -self._snapshot_max_files]:
                old.unlink()
        except OSError:
            logger.exception("snapshot pruning failed (non-fatal)")

    # ---------------------------------------------------------------
    # Watchdog
    # ---------------------------------------------------------------
    def tick(self) -> None:
        """Run liveness checks. Caller invokes periodically (e.g. once/second).

        On camera timeout while armed: notify (once), but DO NOT send stop.
        Reconnect is the caller's responsibility — guard only flags loss.
        Printer-report staleness is checked in every state: a stale IDLE
        means the guard will never arm, and nobody would notice.
        """
        self._check_report_staleness()
        if self._state not in (GuardState.ARMED, GuardState.ALERTING):
            return
        if self._last_frame_ts is None:
            return
        elapsed = self._now() - self._last_frame_ts
        if elapsed >= self._camera_timeout_s and not self._camera_lost:
            self._camera_lost = True
            logger.warning("camera silent for %.1fs while armed", elapsed)
            try:
                self._notifier.send(
                    "Bambu camera silent",
                    f"No frame in {elapsed:.0f}s while ARMED — guard cannot detect.",
                )
            except Exception:
                logger.exception("notifier raised on camera-loss alert")

    def _check_report_staleness(self) -> None:
        if self._state_age_provider is None:
            return
        age = self._state_age_provider()
        if age is None:  # no report yet — no baseline to be stale against
            return
        if age >= self._mqtt_timeout_s:
            if not self._report_stale:
                self._report_stale = True
                logger.warning("printer report silent for %.1fs", age)
                self._notify(
                    "Bambu printer reports silent",
                    f"No MQTT report in {age:.0f}s — the guard cannot see the "
                    f"print state and may be silently unprotected.",
                )
        else:
            self._report_stale = False

    # ---------------------------------------------------------------
    # Run loop
    # ---------------------------------------------------------------
    def request_stop(self) -> None:
        self._stop_event.set()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(
        self,
        frame_iter: Iterator[bytes],
        *,
        tick_interval_s: float = 1.0,
    ) -> None:
        """Drive the loop until `request_stop()` is called or the iterator ends.

        `frame_iter` may block indefinitely inside a socket recv, so the
        watchdog runs on its own daemon thread — a stalled camera must not
        starve the very check that reports the camera as stalled.
        For real use the caller wires `frame_iter = camera.frames()`.
        """
        stop_watchdog = threading.Event()

        def _watchdog_loop() -> None:
            while not stop_watchdog.wait(tick_interval_s):
                try:
                    self.tick()
                except Exception:
                    logger.exception("watchdog tick raised; continuing")

        watchdog = threading.Thread(
            target=_watchdog_loop, name="guard-watchdog", daemon=True
        )
        watchdog.start()
        try:
            for jpeg in frame_iter:
                if self._stop_event.is_set():
                    return
                try:
                    self.feed_frame(jpeg)
                except Exception:
                    logger.exception("feed_frame raised; continuing loop")
        finally:
            stop_watchdog.set()
