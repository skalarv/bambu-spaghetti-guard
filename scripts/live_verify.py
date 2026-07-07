"""Live verification against a real Bambu P1S.

What it does (in order):

1. Loads `secrets.local.txt`.
2. Probes the network — is the printer pingable / port-reachable?
3. Opens the camera socket and waits for at least N frames. Reports actual
   fps, frame size, and the marker (if any) so we know the wire format is
   right for your firmware.
4. Connects MQTT and listens for a `report` message; reports the gcode_state
   it sees.
5. Optionally opens the viewer window so you can watch live frames + the
   guard's state machine (dry-run; no stop is ever published).

Exit codes:
  0 — all checks passed
  1 — secrets missing / malformed
  2 — network unreachable
  3 — camera handshake or stream failed
  4 — MQTT connect / report failed
  5 — viewer crashed

Never publishes a control command. Safe to run mid-print.
"""

from __future__ import annotations

import argparse
import logging
import socket
import ssl
import sys
import threading
import time
from pathlib import Path

# Make this script importable from the repo root without an editable install.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spaghetti_guard.camera import RawSocketBackend  # noqa: E402
from spaghetti_guard.cli import (  # noqa: E402
    SecretsFileError,
    apply_secrets_to_env,
    load_secrets_file,
)
from spaghetti_guard.control import PrinterControl  # noqa: E402
from spaghetti_guard.detector import FailureDetector, FrameResult  # noqa: E402
from spaghetti_guard.guard import Guard, GuardState  # noqa: E402
from spaghetti_guard.notifier import NoopNotifier  # noqa: E402
from verification.mock_printer import extract_marker  # noqa: E402

logger = logging.getLogger("live-verify")


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def probe_network(host: str, ports: tuple[int, ...] = (6000, 8883), timeout_s: float = 5.0) -> dict:
    """TCP-connect to the given ports and report what worked. No data sent."""
    out = {}
    for port in ports:
        t0 = time.monotonic()
        try:
            with socket.create_connection((host, port), timeout=timeout_s) as s:
                out[port] = {"open": True, "rtt_ms": int((time.monotonic() - t0) * 1000)}
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            out[port] = {"open": False, "error": repr(e)}
    return out


def probe_camera(
    host: str, port: int, access_code: str, n_frames: int = 5, recv_timeout_s: float = 30.0
) -> dict:
    """Connect to the camera channel, collect a few frames, report stats."""
    backend = RawSocketBackend(
        host=host, port=port, access_code=access_code, recv_timeout_s=recv_timeout_s
    )
    backend.connect()
    try:
        sizes: list[int] = []
        markers: list[int | None] = []
        timestamps: list[float] = []
        for jpeg in backend.frames():
            timestamps.append(time.monotonic())
            sizes.append(len(jpeg))
            markers.append(extract_marker(jpeg))
            if len(sizes) >= n_frames:
                break
        elapsed = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0
        fps = (len(timestamps) - 1) / elapsed if elapsed > 0 else 0
        return {
            "frames": len(sizes),
            "avg_bytes": int(sum(sizes) / len(sizes)) if sizes else 0,
            "min_bytes": min(sizes) if sizes else 0,
            "max_bytes": max(sizes) if sizes else 0,
            "fps_observed": round(fps, 2),
            "markers_seen": [m for m in markers if m is not None],
        }
    finally:
        backend.close()


def probe_mqtt(
    host: str, serial: str, access_code: str, wait_s: float = 20.0
) -> dict:
    """Connect MQTT (TLS, port 8883) and wait for at least one report."""
    control = PrinterControl(host=host, serial=serial, access_code=access_code)
    if not control.connect(wait_s=10.0):
        return {"connected": False, "error": "MQTT CONNACK never arrived"}

    deadline = time.monotonic() + wait_s
    got_report = False
    state, layer, ts = "?", -1, 0.0
    while time.monotonic() < deadline:
        state, layer, ts = control.state.snapshot()
        if ts > 0:
            got_report = True
            break
        time.sleep(0.2)
    control.close()
    return {
        "connected": True,
        "report_received": got_report,
        "gcode_state": state if got_report else None,
        "layer_num": layer if got_report else None,
    }


# ---------------------------------------------------------------------------
# Viewer dry-run (optional, longer-running)
# ---------------------------------------------------------------------------


def viewer_dry_run(host: str, serial: str, access_code: str, duration_s: float) -> None:
    """Stream live frames into the viewer window for `duration_s` seconds.

    Uses a never-fires detector; sole purpose is to confirm the camera + viewer
    work end-to-end on your hardware.
    """
    from spaghetti_guard.viewer import TkViewer

    class _NoFireYolo:
        def predict(self, image, **kwargs):
            return []

    detector = FailureDetector(
        _NoFireYolo(),
        failure_classes=("spaghetti",),
        conf_threshold=0.5,
        decoder=lambda j: j,
    )
    control = PrinterControl(host=host, serial=serial, access_code=access_code)
    if not control.connect(wait_s=10.0):
        logger.error("MQTT connect failed during viewer dry-run")
        return
    cam = RawSocketBackend(host=host, port=6000, access_code=access_code, recv_timeout_s=30.0)
    cam.connect()
    viewer = TkViewer()
    viewer.on_pause_clicked = lambda: logger.info("[dry-run] pause requested by operator (not sent)")
    viewer.on_stop_clicked = lambda: logger.info("[dry-run] stop requested by operator (not sent)")
    viewer.start()

    guard = Guard(
        detector=detector,
        control=_DryRunControl(),  # never publishes
        notifier=NoopNotifier(),
        gcode_state_provider=lambda: control.state.snapshot()[0],
        action_mode="pause",
        debounce_window=6,
        camera_timeout_s=30.0,
        snapshot_dir="./failure_snapshots",
        viewer=viewer,
    )

    stop_at = time.monotonic() + duration_s

    def watchdog():
        while time.monotonic() < stop_at:
            time.sleep(0.5)
        guard.request_stop()

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        guard.run(cam.frames())
    finally:
        viewer.stop()
        cam.close()
        control.close()


class _DryRunControl:
    """Sink that logs commands but never publishes."""

    def stop(self) -> None:
        logger.warning("[dry-run] would publish STOP")

    def pause(self) -> None:
        logger.warning("[dry-run] would publish PAUSE")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secrets", type=Path, default=ROOT / "secrets.local.txt")
    parser.add_argument(
        "--viewer-seconds",
        type=int,
        default=0,
        help="open the viewer window and stream live frames for this many seconds (0 = skip)",
    )
    parser.add_argument(
        "--camera-frames",
        type=int,
        default=5,
        help="frames to collect from the camera channel in the probe stage",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # --- secrets
    try:
        secrets = load_secrets_file(args.secrets)
    except SecretsFileError as e:
        logger.error("%s", e)
        return 1
    apply_secrets_to_env(secrets)
    ip = secrets["BAMBU_IP"]
    serial = secrets["BAMBU_SERIAL"]
    access = secrets["BAMBU_ACCESS_CODE"]
    logger.info("secrets loaded for %s (serial=%s)", ip, serial[:6] + "..." if len(serial) > 6 else serial)

    # --- network
    logger.info("probing TCP ports 6000 (camera) and 8883 (MQTT)...")
    net = probe_network(ip)
    for port, info in net.items():
        if info["open"]:
            logger.info("  port %d open (rtt %dms)", port, info["rtt_ms"])
        else:
            logger.error("  port %d closed: %s", port, info["error"])
    if not all(info["open"] for info in net.values()):
        logger.error("network unreachable — fix routing / firewall / LAN-mode first")
        return 2

    # --- camera
    logger.info("probing camera channel (collecting %d frames)...", args.camera_frames)
    try:
        cam_stats = probe_camera(ip, 6000, access, n_frames=args.camera_frames)
        logger.info(
            "  ok: %d frames, avg %d bytes, observed %s fps",
            cam_stats["frames"],
            cam_stats["avg_bytes"],
            cam_stats["fps_observed"],
        )
    except Exception as e:
        logger.exception("camera probe failed: %s", e)
        return 3

    # --- MQTT
    logger.info("probing MQTT and waiting up to 20s for a report message...")
    try:
        mqtt_stats = probe_mqtt(ip, serial, access)
    except Exception as e:
        logger.exception("MQTT probe failed: %s", e)
        return 4
    if not mqtt_stats["connected"]:
        logger.error("  connect failed: %s", mqtt_stats.get("error"))
        return 4
    if not mqtt_stats["report_received"]:
        logger.warning(
            "  connected but no report yet — printer is reachable but quiet. "
            "If LAN Mode is disabled the broker will not deliver reports."
        )
    else:
        logger.info(
            "  ok: gcode_state=%s, layer_num=%s",
            mqtt_stats["gcode_state"],
            mqtt_stats["layer_num"],
        )

    # --- viewer dry-run (optional)
    if args.viewer_seconds > 0:
        logger.info("opening viewer for %d seconds (dry-run; no stop will be published)...", args.viewer_seconds)
        try:
            viewer_dry_run(ip, serial, access, duration_s=args.viewer_seconds)
        except Exception as e:
            logger.exception("viewer dry-run crashed: %s", e)
            return 5
        logger.info("viewer closed")

    logger.info("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
