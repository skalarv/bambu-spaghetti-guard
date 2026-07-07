"""CLI entry point (brief §5.7).

Modes:
  spaghetti-guard run        # live: real camera + real MQTT
  spaghetti-guard run --dry-run
  spaghetti-guard run --action stop|pause
  spaghetti-guard verify     # spin mock printer + run integration scenarios
  spaghetti-guard replay <clip>
  spaghetti-guard train ...
  spaghetti-guard validate ...

Live modes pull weights / camera / MQTT — they will error cleanly if the
required deps aren't installed (see `docs/INSTALL.md`).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

from .config import load_config


SECRETS_REQUIRED_KEYS = ("BAMBU_IP", "BAMBU_SERIAL", "BAMBU_ACCESS_CODE")
SECRETS_OPTIONAL_KEYS = ("NTFY_TOPIC_URL", "TELEGRAM_TARGET")


class SecretsFileError(ValueError):
    """Raised when secrets.local.txt is missing, malformed, or incomplete."""


def parse_secrets_file(text: str) -> dict[str, str]:
    """Parse a KEY=VALUE / comment / blank-line config (.env-style).

    Public so tests can drive it without touching the filesystem.
    """
    out: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SecretsFileError(f"line {lineno}: expected KEY=VALUE, got {line!r}")
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            raise SecretsFileError(f"line {lineno}: empty key")
        out[key] = value
    return out


def load_secrets_file(path: Path) -> dict[str, str]:
    if not path.exists():
        raise SecretsFileError(
            f"{path} not found. Copy secrets.local.txt.template to {path.name} and fill it in."
        )
    parsed = parse_secrets_file(path.read_text(encoding="utf-8"))
    missing = [k for k in SECRETS_REQUIRED_KEYS if not parsed.get(k)]
    if missing:
        raise SecretsFileError(
            f"{path} missing required keys (or values are empty): {', '.join(missing)}"
        )
    return parsed


def apply_secrets_to_env(secrets: dict[str, str], env: dict[str, str] | None = None) -> None:
    """Push the secrets into the given env dict (defaults to os.environ)."""
    target = env if env is not None else os.environ
    for k in SECRETS_REQUIRED_KEYS + SECRETS_OPTIONAL_KEYS:
        if k in secrets and secrets[k]:
            target[k] = secrets[k]

logger = logging.getLogger("spaghetti-guard")


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    # Secrets can come from secrets.local.txt (the documented flow) or from
    # the environment (service managers). Explicit env vars always win; the
    # file only fills the gaps. os.environ is never mutated.
    env = dict(os.environ)
    secrets_path: Path = getattr(args, "secrets", None) or Path("secrets.local.txt")
    if secrets_path.exists():
        try:
            parsed = parse_secrets_file(secrets_path.read_text(encoding="utf-8"))
        except SecretsFileError as e:
            logger.error("%s", e)
            return 5
        for key in SECRETS_REQUIRED_KEYS + SECRETS_OPTIONAL_KEYS:
            if parsed.get(key):
                env.setdefault(key, parsed[key])
        logger.info("loaded secrets from %s", secrets_path)

    cfg = load_config(
        yaml_path=args.config,
        env=env,
        check_model_path=not args.no_model_check,
    )

    # Lazy imports so `--help` / `verify` work without ultralytics installed.
    from .camera import RawSocketBackend
    from .control import PrinterControl
    from .guard import Guard
    from .notifier import build_notifier

    action_mode = args.action or cfg.action.mode
    dry_run = args.dry_run or cfg.action.dry_run

    detector = _build_detector(cfg, mock_detector=args.mock_detector)
    viewer = _build_viewer(args.viewer)

    cam = RawSocketBackend(
        host=cfg.printer.ip,
        port=cfg.camera.port,
        username="bblp",
        access_code=cfg.printer.access_code.get_secret_value(),
        recv_timeout_s=cfg.camera.timeout_s + 5.0,
    )
    control = PrinterControl(
        host=cfg.printer.ip,
        serial=cfg.printer.serial,
        access_code=cfg.printer.access_code.get_secret_value(),
        dry_run=dry_run,
    )

    notifier = build_notifier(cfg.notify.backend, cfg.notify.target)

    if getattr(args, "force_armed", False):
        logger.warning(
            "--force-armed: bypassing gcode_state gate. Guard will run detection "
            "regardless of the printer's actual print state. DEMO ONLY."
        )
        state_provider = lambda: "RUNNING"
        state_age_provider = None
    else:
        state_provider = lambda: control.state.snapshot()[0]

        def state_age_provider() -> float | None:
            import time as _time

            ts = control.state.snapshot()[2]
            return None if ts <= 0 else _time.time() - ts

    guard = Guard(
        detector=detector,
        control=control,
        notifier=notifier,
        gcode_state_provider=state_provider,
        action_mode=action_mode,
        debounce_window=cfg.detector.consecutive_hits,
        cooldown_s=cfg.action.cooldown_s,
        camera_timeout_s=cfg.camera.timeout_s,
        snapshot_dir=cfg.snapshots.dir,
        viewer=viewer,
        ask_timeout_s=cfg.action.ask_timeout_s,
        ask_timeout_action=cfg.action.ask_timeout_action,
        state_age_provider=state_age_provider,
    )

    # Wire UI buttons -> control.
    if viewer is not None and hasattr(viewer, "on_pause_clicked"):
        viewer.on_pause_clicked = lambda: control.pause()
        viewer.on_stop_clicked = lambda: control.stop()

    if not control.connect(wait_s=10.0):
        logger.error("MQTT connect failed; check LAN mode / credentials.")
        return 4
    cam.connect()
    if viewer is not None:
        viewer.start()
    logger.info("guard live (action=%s, dry_run=%s, viewer=%s)", action_mode, dry_run, viewer is not None)

    def _on_sigint(_signum, _frame):
        logger.info("SIGINT — shutting down")
        guard.request_stop()

    signal.signal(signal.SIGINT, _on_sigint)

    import time
    from .camera import CameraError, CameraStreamClosed

    max_attempts = cfg.camera.max_reconnect_attempts
    exit_code = 0

    def _tracking_frames(src, flag):
        for frame in src:
            flag["got_frame"] = True
            yield frame

    try:
        attempt = 0
        while not guard.stopped:
            reason = None
            flag = {"got_frame": False}
            try:
                guard.run(_tracking_frames(cam.frames(), flag))
                if guard.stopped:
                    break
                # Clean iterator exit isn't done — P1S may close the channel
                # between frames (brief §5.2). Fall through and reconnect.
                reason = "stream ended between frames"
            except (CameraStreamClosed, CameraError, EOFError, OSError) as e:
                if guard.stopped:
                    break
                reason = e.__class__.__name__

            if flag["got_frame"]:
                # The stream was healthy since the last reconnect — only
                # *consecutive* dead reconnects count toward the budget.
                attempt = 0
            attempt += 1
            if attempt > max_attempts:
                logger.error(
                    "camera reconnect exhausted after %d attempts; exiting run loop",
                    max_attempts,
                )
                # Non-zero so a service manager (Restart=on-failure) brings the
                # guard back instead of leaving the print unprotected.
                exit_code = 3
                break
            backoff = min(60.0, 2 ** min(attempt, 6))
            logger.warning(
                "camera stream lost (%s); reconnect attempt %d/%d in %.1fs",
                reason, attempt, max_attempts, backoff,
            )
            try:
                cam.close()
            except Exception:
                pass
            time.sleep(backoff)
            try:
                cam.connect()
                logger.info("camera reconnected")
            except Exception as ce:
                logger.error("camera reconnect failed: %s", ce)
    finally:
        if viewer is not None:
            viewer.stop()
        cam.close()
        control.close()
    return exit_code


def _build_detector(cfg, *, mock_detector: bool):
    """Build the live detector. If `mock_detector` is set, use a deterministic
    'never-fires' detector — keeps the live verification safe when we don't want
    to send real stops.
    """
    from .detector import FailureDetector

    if mock_detector:
        class _NoFireYolo:
            def predict(self, image, **kwargs):
                return []

        return FailureDetector(
            _NoFireYolo(),
            failure_classes=cfg.detector.failure_classes,
            conf_threshold=cfg.detector.conf_threshold,
            decoder=lambda j: j,
        )

    from .detector import decode_jpeg, load_yolo_model

    # FailureDetector understands Ultralytics Results natively (and raises on
    # anything it can't read) — no adapter layer needed.
    model = load_yolo_model(cfg.detector.model_path)
    return FailureDetector(
        model,
        failure_classes=cfg.detector.failure_classes,
        conf_threshold=cfg.detector.conf_threshold,
        decoder=decode_jpeg,
    )


def _build_viewer(enabled: bool):
    if not enabled:
        return None
    from .viewer import TkViewer

    return TkViewer()


def _cmd_live_verify(args: argparse.Namespace) -> int:
    """Load secrets.local.txt, dry-run the guard against the real printer,
    open the viewer. Verifies camera handshake + MQTT report parsing
    end-to-end without ever publishing a stop or pause.
    """
    secrets_path: Path = args.secrets
    try:
        secrets = load_secrets_file(secrets_path)
    except SecretsFileError as e:
        logger.error("%s", e)
        return 5
    apply_secrets_to_env(secrets)
    logger.info("loaded secrets from %s", secrets_path)

    # Synthesise an argparse namespace for _cmd_run with dry_run + viewer + mock-detector
    forwarded = argparse.Namespace(
        config=args.config,
        dry_run=True,
        action=None,
        no_model_check=True,
        viewer=not args.headless,
        mock_detector=True,
        force_armed=False,
        secrets=args.secrets,
    )
    return _cmd_run(forwarded)


def _cmd_verify(args: argparse.Namespace) -> int:
    """Spin the mock printer and run pytest's integration loop."""
    import subprocess

    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_integration_loop.py"]
    logger.info("$ %s", " ".join(cmd))
    return subprocess.call(cmd)


def _import_repo_module(name: str):
    """Import a repo-checkout-only module (verification/, training/).

    These aren't shipped in the wheel — the installed console script can only
    reach them when run from the repository root, so fall back to cwd.
    """
    import importlib

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        cwd = str(Path.cwd())
        if cwd not in sys.path and (Path.cwd() / name.split(".")[0]).is_dir():
            sys.path.insert(0, cwd)
            return importlib.import_module(name)
        raise ModuleNotFoundError(
            f"{name} is only available from a repository checkout; "
            f"run this subcommand from the repo root."
        ) from e


def _cmd_replay(args: argparse.Namespace) -> int:
    replay_main = _import_repo_module("verification.replay_harness").main

    fwd = [str(args.clip), "--window", str(args.window), "--conf", str(args.conf)]
    if args.model:
        fwd += ["--model", args.model]
    if args.json_out:
        fwd += ["--json-out", str(args.json_out)]
    return replay_main(fwd)


def _cmd_train(args: argparse.Namespace) -> int:
    train_main = _import_repo_module("training.train").main

    fwd = ["--data", str(args.data), "--epochs", str(args.epochs)]
    return train_main(fwd)


def _cmd_validate(args: argparse.Namespace) -> int:
    validate_main = _import_repo_module("training.validate").main

    fwd = ["--weights", str(args.weights), "--data", str(args.data)]
    return validate_main(fwd)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spaghetti-guard", description=__doc__)
    p.add_argument("--log-level", default="INFO", help="DEBUG / INFO / WARNING / ERROR")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the guard against the real printer")
    run.add_argument("--config", type=Path, default=Path("config.yaml"))
    run.add_argument(
        "--secrets",
        type=Path,
        default=Path("secrets.local.txt"),
        help="optional KEY=VALUE secrets file; explicit env vars take precedence",
    )
    run.add_argument("--dry-run", action="store_true", help="never publish; log payloads")
    run.add_argument("--action", choices=("stop", "pause", "ask"), help="override action mode")
    run.add_argument(
        "--no-model-check",
        action="store_true",
        help="skip model_path existence check (testing only)",
    )
    run.add_argument("--viewer", action="store_true", help="open the live camera window")
    run.add_argument(
        "--mock-detector",
        action="store_true",
        help="use a deterministic never-fires detector (for live verification)",
    )
    run.add_argument(
        "--force-armed",
        action="store_true",
        help="bypass the gcode_state==RUNNING gate. DEMO ONLY — treats every frame as if a print is active.",
    )
    run.set_defaults(func=_cmd_run)

    live = sub.add_parser(
        "live-verify",
        help="dry-run against the real printer with the viewer; safe end-to-end check",
    )
    live.add_argument("--config", type=Path, default=Path("config.yaml"))
    live.add_argument(
        "--secrets",
        type=Path,
        default=Path("secrets.local.txt"),
        help="KEY=VALUE file with BAMBU_IP/SERIAL/ACCESS_CODE",
    )
    live.add_argument("--headless", action="store_true", help="no viewer (CI only)")
    live.set_defaults(func=_cmd_live_verify)

    verify = sub.add_parser("verify", help="run integration tests against the mock printer")
    verify.set_defaults(func=_cmd_verify)

    replay = sub.add_parser("replay", help="replay a clip and report would-fire frames")
    replay.add_argument("clip", type=Path)
    replay.add_argument("--model", default="marker")
    replay.add_argument("--window", type=int, default=6)
    replay.add_argument("--conf", type=float, default=0.55)
    replay.add_argument("--json-out", type=Path)
    replay.set_defaults(func=_cmd_replay)

    train = sub.add_parser("train", help="fine-tune YOLO on the prepared dataset")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=80)
    train.set_defaults(func=_cmd_train)

    val = sub.add_parser("validate", help="validate weights + emit summary")
    val.add_argument("--weights", type=Path, required=True)
    val.add_argument("--data", type=Path, required=True)
    val.set_defaults(func=_cmd_validate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
