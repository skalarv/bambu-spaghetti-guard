"""Live camera viewer with status overlay + interactive confirm.

Two pieces, separated for testability:

* `ViewerLogic` — pure-Python policy. Maps guard state to colours and status
  text and manages the lifecycle of an "ask mode" confirmation request.
  Imports no GUI deps; fully unit-tested.

* `TkViewer` — Tkinter window shell. Runs on a background thread; receives
  JPEG bytes via a thread-safe queue, decodes with Pillow, draws the latest
  frame plus the overlay produced by `ViewerLogic`. Buttons publish
  decisions back into the logic object.

Design choice: Tkinter + Pillow instead of cv2.imshow because Tkinter is
stdlib and Pillow is already a test dependency. The live-only deps
(`ultralytics`, `torch`, `cv2`) stay out of the viewer path.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .detector import FrameResult
from .guard import GuardState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------


class ConfirmDecision(StrEnum):
    STOP = "stop"
    CANCEL = "cancel"
    TIMEOUT = "timeout"


@dataclass
class ViewerDisplay:
    """A snapshot of what the UI should render right now."""

    border_color: str  # hex like "#3a7d3a"
    status_text: str  # one-liner: "ARMED  streak 3/6  conf 0.72"
    header_text: str  # big top-of-window text
    show_confirm: bool  # show the Stop/Cancel modal
    confirm_seconds_left: float | None  # countdown for the modal


@dataclass
class _PendingConfirm:
    deadline: float
    decision: ConfirmDecision | None = None


# Colour scheme picked to be legible on a workshop monitor:
_BORDER_BY_STATE: dict[GuardState, str] = {
    GuardState.IDLE: "#3c4150",
    GuardState.ARMED: "#2f7a3b",
    GuardState.ALERTING: "#d59f1a",
    GuardState.TRIGGERED: "#b8332b",
    GuardState.COOLDOWN: "#6b6f7d",
}


def _border_for(state: GuardState) -> str:
    return _BORDER_BY_STATE.get(state, "#444444")


def _status_text(
    state: GuardState,
    streak: int,
    window: int,
    last_result: FrameResult | None,
    last_trigger_ts: float | None,
) -> str:
    parts = [state.name]
    if state in (GuardState.ARMED, GuardState.ALERTING):
        parts.append(f"streak {streak}/{window}")
    if last_result is not None and last_result.conf > 0:
        parts.append(f"conf {last_result.conf:.2f}")
        if last_result.best_class:
            parts.append(last_result.best_class)
    if last_trigger_ts is not None:
        elapsed = max(0, int(time.time() - last_trigger_ts))
        parts.append(f"last trigger {elapsed}s ago")
    return "  |  ".join(parts)


def _header_text(state: GuardState, pending_confirm: bool) -> str:
    if pending_confirm:
        return "SPAGHETTI DETECTED — confirm action"
    if state == GuardState.TRIGGERED:
        return "SPAGHETTI DETECTED — action sent"
    if state == GuardState.ALERTING:
        return "Possible spaghetti — accumulating evidence"
    if state == GuardState.COOLDOWN:
        return "Cooldown — guard idle for a moment"
    if state == GuardState.ARMED:
        return "Watching"
    return "Idle (waiting for print to start)"


class ViewerLogic:
    """Renders state -> display + drives the ask-mode confirm lifecycle.

    Thread-safe: `request_confirm`, `submit_decision`, `poll_confirm`,
    `current_display` may be called from any thread.
    """

    def __init__(self, *, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._lock = threading.Lock()
        self._pending: _PendingConfirm | None = None

    # ---- confirmation -------------------------------------------------
    def request_confirm(self, *, timeout_s: float) -> None:
        with self._lock:
            self._pending = _PendingConfirm(deadline=self._now() + timeout_s)

    def submit_decision(self, decision: ConfirmDecision) -> bool:
        """Record a user choice. Returns True if a request was pending."""
        with self._lock:
            if self._pending is None or self._pending.decision is not None:
                return False
            self._pending.decision = decision
            return True

    def has_pending_confirm(self) -> bool:
        with self._lock:
            return self._pending is not None and self._pending.decision is None

    def poll_confirm(self) -> ConfirmDecision | None:
        """Return a resolved decision once, then clear. Promotes a deadline
        miss to TIMEOUT."""
        with self._lock:
            if self._pending is None:
                return None
            if self._pending.decision is not None:
                d = self._pending.decision
                self._pending = None
                return d
            if self._now() >= self._pending.deadline:
                self._pending = None
                return ConfirmDecision.TIMEOUT
            return None

    def confirm_seconds_left(self) -> float | None:
        with self._lock:
            if self._pending is None or self._pending.decision is not None:
                return None
            return max(0.0, self._pending.deadline - self._now())

    # ---- display projection ------------------------------------------
    def current_display(
        self,
        *,
        state: GuardState,
        streak: int,
        window: int,
        last_result: FrameResult | None,
        last_trigger_ts: float | None,
    ) -> ViewerDisplay:
        show_confirm = self.has_pending_confirm()
        return ViewerDisplay(
            border_color=_border_for(state),
            status_text=_status_text(state, streak, window, last_result, last_trigger_ts),
            header_text=_header_text(state, show_confirm),
            show_confirm=show_confirm,
            confirm_seconds_left=self.confirm_seconds_left(),
        )


# ---------------------------------------------------------------------------
# Viewer protocol that the Guard depends on
# ---------------------------------------------------------------------------


class ViewerLike(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def update(
        self,
        *,
        jpeg: bytes | None,
        state: GuardState,
        streak: int,
        window: int,
        last_result: FrameResult | None,
        last_trigger_ts: float | None,
    ) -> None: ...
    def request_confirm_stop(self, *, timeout_s: float) -> ConfirmDecision: ...


# ---------------------------------------------------------------------------
# Headless viewer (used when --viewer is off but ask mode is on; also for tests)
# ---------------------------------------------------------------------------


class HeadlessViewer:
    """No window. ask-mode falls back to auto-stop on every request.

    Useful for service deployments where there's no operator at the screen.
    """

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def update(self, **_kwargs) -> None:
        pass

    def request_confirm_stop(self, *, timeout_s: float) -> ConfirmDecision:
        logger.warning("ask mode without a viewer — defaulting to STOP for safety")
        return ConfirmDecision.STOP


# ---------------------------------------------------------------------------
# Tk viewer
# ---------------------------------------------------------------------------


@dataclass
class _FramePacket:
    jpeg: bytes | None
    state: GuardState
    streak: int
    window: int
    last_result: FrameResult | None
    last_trigger_ts: float | None


class TkViewer:
    """Tk window driven by a background thread.

    Frame updates are pushed onto a single-element queue (newest wins) so the
    guard never blocks on UI throughput. The mainloop polls every ~50 ms.

    Tk is constructed inside the worker thread so all Tk objects belong to
    that thread (Tk dislikes cross-thread access).
    """

    def __init__(
        self,
        *,
        title: str = "Bambu Spaghetti Guard",
        canvas_size: tuple[int, int] = (640, 480),
        poll_ms: int = 50,
        logic: ViewerLogic | None = None,
    ) -> None:
        self._title = title
        self._canvas_size = canvas_size
        self._poll_ms = poll_ms
        self.logic = logic or ViewerLogic()
        self._latest: queue.Queue[_FramePacket] = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready = threading.Event()
        # External "button clicked" hooks: filled by the live wiring
        # (`PauseRequested` / `StopRequested` go into a callback the guard sets).
        self.on_pause_clicked: Callable[[], None] = lambda: None
        self.on_stop_clicked: Callable[[], None] = lambda: None

    # ---- lifecycle ----------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_ui, daemon=True, name="tk-viewer")
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # ---- ViewerLike API ----------------------------------------------
    def update(
        self,
        *,
        jpeg: bytes | None,
        state: GuardState,
        streak: int,
        window: int,
        last_result: FrameResult | None,
        last_trigger_ts: float | None,
    ) -> None:
        packet = _FramePacket(
            jpeg=jpeg,
            state=state,
            streak=streak,
            window=window,
            last_result=last_result,
            last_trigger_ts=last_trigger_ts,
        )
        # Drop the previous packet if the UI hasn't picked it up — we want the
        # newest frame on screen, not a backlog.
        try:
            self._latest.put_nowait(packet)
        except queue.Full:
            try:
                self._latest.get_nowait()
            except queue.Empty:
                pass
            try:
                self._latest.put_nowait(packet)
            except queue.Full:
                pass

    def request_confirm_stop(self, *, timeout_s: float) -> ConfirmDecision:
        self.logic.request_confirm(timeout_s=timeout_s)
        # Poll for a decision; the UI thread fills it in via button clicks.
        deadline = time.monotonic() + timeout_s + 1.0  # +1s grace for poll loop
        while time.monotonic() < deadline:
            decision = self.logic.poll_confirm()
            if decision is not None:
                return decision
            time.sleep(0.05)
        return ConfirmDecision.TIMEOUT

    # ---- internals ----------------------------------------------------
    def _run_ui(self) -> None:  # pragma: no cover -- exercised manually by `--viewer`
        import tkinter as tk
        from PIL import Image, ImageTk
        import io

        root = tk.Tk()
        root.title(self._title)
        root.configure(bg="#1c1f27")
        try:
            root.geometry(f"{self._canvas_size[0] + 60}x{self._canvas_size[1] + 160}")
        except Exception:
            pass

        header_var = tk.StringVar(value="Bambu Spaghetti Guard")
        status_var = tk.StringVar(value="(no frames yet)")
        confirm_var = tk.StringVar(value="")

        border = tk.Frame(root, bg="#3c4150", padx=4, pady=4)
        border.pack(padx=12, pady=12)

        header = tk.Label(
            root,
            textvariable=header_var,
            fg="#ffffff",
            bg="#1c1f27",
            font=("Segoe UI", 14, "bold"),
        )
        header.pack(pady=(0, 4))

        canvas = tk.Canvas(
            border,
            width=self._canvas_size[0],
            height=self._canvas_size[1],
            bg="#000000",
            highlightthickness=0,
        )
        canvas.pack()
        canvas_img_id: list[int] = []
        canvas_photo: list = []  # keep ref to PhotoImage so it isn't GC'd

        status_bar = tk.Label(
            root,
            textvariable=status_var,
            fg="#cfd6e4",
            bg="#1c1f27",
            font=("Consolas", 11),
        )
        status_bar.pack(pady=(8, 0))

        # Action row
        btn_row = tk.Frame(root, bg="#1c1f27")
        btn_row.pack(pady=8)
        pause_btn = tk.Button(
            btn_row,
            text="Pause",
            width=10,
            command=lambda: self.on_pause_clicked(),
        )
        stop_btn = tk.Button(
            btn_row,
            text="Stop",
            width=10,
            command=lambda: self.on_stop_clicked(),
        )
        pause_btn.pack(side="left", padx=4)
        stop_btn.pack(side="left", padx=4)

        # Confirm row (hidden unless ask mode triggers)
        confirm_row = tk.Frame(root, bg="#1c1f27")
        confirm_label = tk.Label(
            confirm_row,
            textvariable=confirm_var,
            fg="#ffd166",
            bg="#1c1f27",
            font=("Segoe UI", 11, "bold"),
        )
        confirm_label.pack()
        confirm_btns = tk.Frame(confirm_row, bg="#1c1f27")
        confirm_btns.pack(pady=4)
        tk.Button(
            confirm_btns,
            text="Confirm STOP",
            width=14,
            bg="#b8332b",
            fg="#ffffff",
            command=lambda: self.logic.submit_decision(ConfirmDecision.STOP),
        ).pack(side="left", padx=4)
        tk.Button(
            confirm_btns,
            text="Cancel (keep printing)",
            width=20,
            command=lambda: self.logic.submit_decision(ConfirmDecision.CANCEL),
        ).pack(side="left", padx=4)

        def on_close():
            self._stop_event.set()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)

        self._ready.set()

        def tick():
            if self._stop_event.is_set():
                root.destroy()
                return
            # Drain queue: keep newest
            packet: _FramePacket | None = None
            while True:
                try:
                    packet = self._latest.get_nowait()
                except queue.Empty:
                    break
            if packet is not None:
                disp = self.logic.current_display(
                    state=packet.state,
                    streak=packet.streak,
                    window=packet.window,
                    last_result=packet.last_result,
                    last_trigger_ts=packet.last_trigger_ts,
                )
                header_var.set(disp.header_text)
                status_var.set(disp.status_text)
                border.configure(bg=disp.border_color)
                if disp.show_confirm:
                    sl = disp.confirm_seconds_left
                    confirm_var.set(
                        f"Stop the print? (auto-stop in {int(sl)}s)" if sl is not None else "Stop the print?"
                    )
                    if not confirm_row.winfo_ismapped():
                        confirm_row.pack(pady=8)
                else:
                    if confirm_row.winfo_ismapped():
                        confirm_row.pack_forget()
                if packet.jpeg is not None:
                    try:
                        img = Image.open(io.BytesIO(packet.jpeg)).convert("RGB")
                        img.thumbnail(self._canvas_size)
                        photo = ImageTk.PhotoImage(img)
                        if canvas_img_id:
                            canvas.itemconfig(canvas_img_id[0], image=photo)
                        else:
                            cid = canvas.create_image(
                                self._canvas_size[0] // 2,
                                self._canvas_size[1] // 2,
                                image=photo,
                                anchor="center",
                            )
                            canvas_img_id.append(cid)
                        canvas_photo.clear()
                        canvas_photo.append(photo)
                    except Exception:
                        logger.exception("frame decode failed")
            else:
                # Even without new frames, keep the countdown updating
                sl = self.logic.confirm_seconds_left()
                if sl is not None:
                    confirm_var.set(f"Stop the print? (auto-stop in {int(sl)}s)")

            root.after(self._poll_ms, tick)

        root.after(self._poll_ms, tick)
        try:
            root.mainloop()
        except Exception:
            logger.exception("Tk mainloop crashed")
        self._stop_event.set()
