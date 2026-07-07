# Bambu P1S Spaghetti Guard: Claude Code Implementation Brief

**Document type:** Implementation brief and handoff for Claude Code
**Target hardware:** Bambu Lab P1S (stock chamber camera)
**Primary dev machine:** Windows desktop (i9-14900K, RTX 5070 12 GB, 96 GB DDR5)
**Training machine:** Lenovo Legion Pro 7 (RTX 5090 24 GB, 96 GB RAM)
**Optional edge target:** Orange Pi 3B (RK3566, RKNN) for always-on inference
**Language:** Python 3.11
**Status:** Greenfield build. A working single-file scaffold exists (`bambu_spaghetti_guard.py`) and should be treated as a reference, not the final structure.

---

## 1. Mission

Build a self-hosted system that watches the P1S stock camera during an active print, runs a YOLO failure-detection model on each frame, and stops or pauses the print over MQTT when a spaghetti or detachment failure is confirmed across several consecutive frames. The system must be fully testable offline against a mock printer before it is ever pointed at the real machine.

Two deliverables of equal priority:

1. The guard application itself.
2. A verification environment that proves the guard behaves correctly without risking a live print.

Everything must run locally. No cloud dependency, no Chinese-origin models (use Ultralytics YOLO weights, Meta/Microsoft/Google-lineage backbones only, or self-trained weights).

---

## 2. Key facts and constraints (do not re-derive these)

**Camera (P1/A1 series):**
- Proprietary stream on TCP port **6000** (X1/H2D use 322, not relevant here), TLS wrapped.
- Frame rate is roughly **1 fps**. This is adequate: spaghetti develops over minutes. Do not chase higher frame rates.
- Auth: username `bblp`, password is the LAN access code. Frames are length-prefixed JPEG (16-byte header, first 4 bytes little-endian payload length, then a JPEG that starts `FF D8` and ends `FF D9`).
- The exact handshake byte layout is community-reverse-engineered and firmware-sensitive. Treat the camera module as the one place to either (a) verify against the OpenBambuAPI repo (`Doridian/OpenBambuAPI`) or (b) delegate to the `bambulabs_api` PyPI package. Make this swappable behind an interface.

**Control (MQTT):**
- Port **8883**, TLS, username `bblp`, password is the LAN access code.
- Request topic: `device/{SERIAL}/request`. Report topic: `device/{SERIAL}/report`.
- Stop: `{"print": {"command": "stop", "sequence_id": "0"}}`. Pause and resume use the same shape with `"pause"` / `"resume"`. Send stop and pause at **QoS 1**.
- This path is stable and well documented. It is the most reliable part of the system.

**Firmware authorization:**
- Post-2024-05 firmware locks local MQTT, camera, and FTP behind **LAN Mode / Developer Mode**. Without it, MQTT publishes fail with "MQTT command verification failed".
- Operator must enable LAN Mode (or Developer Mode), set a static IP, and record the access code and serial from the touchscreen. This is an install-step, not a code concern, but the install guide must cover it.

**Print state awareness:**
- Subscribe to `device/{SERIAL}/report` and parse `print.gcode_state` (`RUNNING`, `PAUSE`, `FINISH`, `FAILED`, `IDLE`) and `print.layer_num`.
- Only run detection while `gcode_state == RUNNING`. This single guard eliminates most false positives (no detection during idle, loading, or bed clear).

---

## 3. Architecture

```
                 device/{serial}/report (MQTT 8883)
                          |  gcode_state, layer_num
                          v
   port 6000  +----------------------+   stop/pause   port 8883
   JPEG ~1fps |       GUARD          |--------------->  printer
  ----------> |   (state machine)    |   MQTT request
   camera     +----------------------+
                  |            |
                  v            v
              detector     notifier
            (YOLO+debounce) (ntfy/TG/HA)
```

Six modules, each independently testable:

| Module | File | Responsibility |
|---|---|---|
| Config | `config.py` | Load + validate config from env and `config.yaml`. Secrets only from env. |
| Camera | `camera.py` | Connect port 6000, yield JPEG frames. Interface with two backends. |
| Control | `control.py` | MQTT connect 8883, publish stop/pause, subscribe to report topic, expose print state. |
| Detector | `detector.py` | Decode JPEG, run YOLO, apply per-frame threshold, rolling N-of-N debounce. |
| Notifier | `notifier.py` | Push alerts (ntfy / Telegram / Home Assistant webhook). Pluggable, all optional. |
| Guard | `guard.py` | Orchestrator. State machine. Watchdog. Snapshot logging. |

### 3.1 Guard state machine

```
IDLE ──(gcode_state RUNNING)──> ARMED
ARMED ──(frame hit)──> ALERTING (accumulating consecutive hits)
ALERTING ──(miss)──> ARMED            (debounce resets)
ALERTING ──(N consecutive hits)──> TRIGGERED (send action, notify)
TRIGGERED ──(action sent)──> COOLDOWN
any ──(gcode_state FINISH/FAILED/IDLE)──> IDLE   (disarm)
```

The guard must **only** act in ARMED/ALERTING. It must disarm cleanly at print end so it never sends a stop to an idle printer or a freshly started next job.

### 3.2 Watchdog

- If no camera frame arrives for `camera_timeout_s` while ARMED, attempt reconnect (exponential backoff). After `max_reconnect_attempts`, **notify only**. Camera loss is not a print failure, so never send stop on camera loss.
- If MQTT disconnects, reconnect with backoff. Queue nothing; only the current report state matters.

### 3.3 Safety posture

- Default action is configurable: `pause` while tuning, `stop` once trusted. Document the trade. Pause lets the operator inspect; stop saves filament and reduces fire risk on a confirmed failure.
- Debounce (`consecutive_hits`, default 6) is the primary false-positive defense. A single bad frame must never trigger.
- Always log every triggering frame to disk with timestamp and confidence, for retraining.
- Provide a hard `--dry-run` mode that runs the full live pipeline but only logs the action payload instead of publishing it.

---

## 4. Repository layout

```
bambu-spaghetti-guard/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── config.yaml
├── tasks.ps1                    # Windows task runner (setup/test/run-dry/run-live/train)
├── Makefile                     # same targets for Linux/Orange Pi
├── src/spaghetti_guard/
│   ├── __init__.py
│   ├── config.py
│   ├── camera.py                # CameraBackend interface + RawSocketBackend + LibBackend
│   ├── detector.py
│   ├── control.py
│   ├── notifier.py
│   ├── guard.py
│   └── cli.py                   # entrypoint, modes, arg parsing
├── models/
│   └── README.md                # weight placement + provenance
├── training/
│   ├── datasets.md
│   ├── prepare_dataset.py
│   ├── train.py
│   └── validate.py
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_camera_parse.py
│   ├── test_detector_debounce.py
│   ├── test_control_payload.py
│   ├── test_guard_state.py
│   └── test_integration_loop.py
├── verification/
│   ├── mock_printer.py          # fake camera server (6000) + fake MQTT broker (8883)
│   ├── replay_harness.py        # run detector over recorded clips, report would-fire timeline
│   ├── metrics.py               # precision/recall, detection latency, FP-per-print-hour
│   └── fixtures/                # sample good + spaghetti JPEG sequences
├── deploy/
│   ├── windows_service.md       # NSSM or Task Scheduler
│   ├── systemd/spaghetti-guard.service
│   └── orange_pi_rknn.md        # optional RKNN edge inference
└── docs/
    ├── ARCHITECTURE.md
    ├── PROTOCOL.md              # camera + MQTT protocol reference
    ├── INSTALL.md
    ├── RUNBOOK.md
    └── SAFETY.md
```

---

## 5. Module specifications

### 5.1 `config.py`
- Pydantic (v2) settings model. Sources: environment variables override `config.yaml`.
- Secrets (`access_code`) come **only** from env (`BAMBU_ACCESS_CODE`), never from yaml, never logged.
- Validate: IP format, non-empty serial, thresholds in range, `action in {stop, pause}`, model path exists (unless `--no-model-check`).
- Expose a frozen config object to all modules.

Config fields:
```
printer.ip, printer.serial            # env: BAMBU_IP, BAMBU_SERIAL
printer.access_code                   # env only: BAMBU_ACCESS_CODE
camera.backend                        # "raw" | "lib"
camera.timeout_s = 15
camera.max_reconnect_attempts = 5
detector.model_path
detector.conf_threshold = 0.55
detector.consecutive_hits = 6
detector.failure_classes = [spaghetti, detachment, blob, failure]
action.mode = "pause"                 # "stop" | "pause"
action.dry_run = false
notify.backend = "none"               # "none"|"ntfy"|"telegram"|"homeassistant"
notify.target                         # url/token/topic as relevant
snapshots.dir = "./failure_snapshots"
log.level = "INFO"
```

### 5.2 `camera.py`
- Define `CameraBackend` ABC with `connect()`, `frames() -> Iterator[bytes]`, `close()`.
- `RawSocketBackend`: TLS socket to port 6000, send auth packet, parse length-prefixed JPEG, handle partial reads (`recv_exact`), drop malformed frames, raise on closed stream.
- `LibBackend`: thin wrapper around `bambulabs_api` camera frame source. Selected when handshake reliability is a concern.
- Document the handshake byte layout in `docs/PROTOCOL.md` and add a TODO to verify against OpenBambuAPI for the operator's firmware.

### 5.3 `control.py`
- paho-mqtt **2.x** (`CallbackAPIVersion.VERSION2`). TLS with `cert_reqs=CERT_NONE`, `tls_insecure_set(True)`.
- Connect, `loop_start()`, subscribe `device/{serial}/report`.
- Parse incoming report JSON, maintain a thread-safe `PrinterState` (gcode_state, layer_num, last_update_ts).
- `stop()` / `pause()` / `resume()` build the exact payload and publish at QoS 1. In dry-run, log payload and return without publishing.
- Reconnect with backoff on disconnect.

### 5.4 `detector.py`
- Lazy-import `ultralytics` and `cv2` so the module imports without GPU deps (tests can stub).
- `decode(jpeg) -> ndarray`, `is_failure_frame(jpeg) -> (hit, conf, img)` checks any box whose class is in `failure_classes` and `conf >= threshold`.
- `Debouncer`: maxlen deque of last N booleans. `confirmed()` true only when full and all-true. Resets on any miss.
- Keep model load behind a `load()` call so tests can inject a fake model.

### 5.5 `notifier.py`
- `Notifier` ABC: `send(title, message, image_path=None)`.
- Implementations: `NoopNotifier`, `NtfyNotifier`, `TelegramNotifier`, `HomeAssistantNotifier`.
- Never let a notify failure crash the guard. Wrap and log.

### 5.6 `guard.py`
- Wire camera + detector + control + notifier per the state machine in section 3.1.
- Only run detection while printer state is RUNNING.
- On TRIGGERED: send action, save annotated snapshot, notify, enter COOLDOWN.
- Watchdog thread or async task for camera/MQTT liveness.
- Clean shutdown on SIGINT.

### 5.7 `cli.py`
- Modes: `run` (live), `--dry-run`, `--action {stop,pause}` override, `verify` (run integration test loop against mock), `replay <clip>`, `train`, `validate`.
- `--dry-run` and `--action pause` are the safe-rollout path.

---

## 6. Verification environment (build this first, before any live test)

The point of this section: prove correctness with zero risk to a real print. **Claude Code must complete and pass this environment before the guard is considered done.**

### 6.1 `mock_printer.py`
A self-contained fake printer:

- **Fake camera server**: TLS TCP server on a configurable port. Accepts the auth packet (validates `bblp` + a test access code), then streams length-prefixed JPEG frames from a directory or video file at a configurable fps (default 1). Supports injecting a "spaghetti sequence" at a chosen frame index.
- **Fake MQTT broker**: use `amqtt` (pure-Python broker) or spin a local mosquitto in a subprocess. Accept TLS with the test creds, accept subscriptions, publish synthetic `report` messages (drive `gcode_state` transitions: IDLE -> RUNNING -> ... -> FINISH), and **record every message received on `device/{serial}/request`** into a `CommandRecorder` for assertions.

This lets the entire guard run end-to-end offline, exactly as it would against hardware.

### 6.2 `replay_harness.py`
- Input: a folder or video of frames (good print and/or known failure with ground-truth onset frame).
- Runs the detector over the clip, prints a per-frame timeline (frame index, conf, hit), and reports the frame at which the debounce **would have fired** under current settings.
- Primary tool for tuning `conf_threshold` and `consecutive_hits` against real chamber footage.

### 6.3 `metrics.py`
- Given labeled clips, compute:
  - **Detection latency**: frames and seconds from true failure onset to fire.
  - **Frame-level precision / recall** over the failure class.
  - **False-positive rate per print-hour** measured on clean footage (the number that actually matters operationally).
  - Threshold sweep (ROC-style) to pick an operating point.

### 6.4 Unit tests (pytest)
- `test_config.py`: env overrides yaml; access_code never sourced from yaml; invalid values rejected.
- `test_camera_parse.py`: crafted length-prefixed buffer yields correct frames; handles split reads across recv boundaries; drops malformed (missing JPEG markers).
- `test_detector_debounce.py`: N-of-N fires exactly at N, not N-1; a single miss inside the window resets; never fires below threshold.
- `test_control_payload.py`: stop/pause produce the exact JSON, correct topic, QoS 1 (mock the paho client, assert `publish` call args); dry-run does not publish.
- `test_guard_state.py`: detection ignored unless RUNNING; disarms on FINISH/FAILED/IDLE; camera loss notifies but never stops.

### 6.5 Integration test
- `test_integration_loop.py`: start `mock_printer`, run the guard against it.
  - Clip with an injected spaghetti sequence: assert a **stop** command is recorded, and that it appears only after exactly `consecutive_hits` qualifying frames.
  - All-good clip: assert **no** command is ever recorded.
  - Report driven to FINISH mid-watch: assert guard disarms and ignores subsequent frames.

### 6.6 Hardware-in-the-loop (operator-run, not automated)
- `--dry-run` against the real printer during a normal print: validates the real camera handshake and live detection on the actual chamber, logging the action payload without sending it.
- This step is run **by the operator**, never by Claude Code.

---

## 7. Detection model pipeline (`training/`)

### 7.1 Data
- Source labeled 3D-print-failure / spaghetti datasets (Roboflow Universe has several; pick permissively licensed, non-Chinese-origin where the license is clear).
- Reuse Obico's open failure-detection dataset/model as a baseline reference if license permits.
- Classes: `spaghetti`, `detachment`, `blob` (warping/blob), optionally `stringing` (low priority, often benign).
- `datasets.md` documents each source, license, and class mapping.

### 7.2 Train (`train.py`)
- Base: `yolo11n.pt` (or `yolo11s` if precision needs it). Fine-tune on the RTX 5090.
- Output weights to `models/`, plus a `validate.py` run producing confusion matrix, PR curves, and the FP-per-print-hour estimate on a held-out clean set.
- Target: high precision over recall. A missed failure costs filament; a false stop costs a whole good print. Tune the operating point accordingly.

### 7.3 Retraining loop
- The guard logs all triggering frames. Periodically label them and fold back into training. The stock P1S camera angle/lighting is specific enough that a model trained only on other printers will over-trigger until adapted to this chamber. Document this in `RUNBOOK.md`.

### 7.4 Optional edge (`deploy/orange_pi_rknn.md`)
- Export to RKNN for the Orange Pi 3B if always-on inference without the workstation is wanted (mirrors the SENTINEL-B1 RKNN path). Keep this optional and behind the `camera.backend`/inference abstraction.

---

## 8. Installation guide (also goes in `docs/INSTALL.md`)

### Step 1: Prepare the printer
1. On the P1S touchscreen, enable **LAN Mode** (or Developer Mode).
2. Set a **static IP** for the printer in the router.
3. Record the **LAN access code** and **serial number** from the touchscreen.

### Step 2: Set up the environment (Windows dev machine)
```powershell
git clone <repo> bambu-spaghetti-guard
cd bambu-spaghetti-guard
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Install the CUDA build of torch matching the RTX driver:
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Step 3: Configure
```powershell
copy .env.example .env
# Edit .env:
#   BAMBU_IP=192.168.1.50
#   BAMBU_SERIAL=01P00A...
#   BAMBU_ACCESS_CODE=xxxxxxxx
```
Adjust `config.yaml` for thresholds, action mode, and notifier.

### Step 4: Get a model
Either drop trained weights into `models/` (see `models/README.md`), or run the training pipeline (section 7).

### Step 5: Verify offline (mandatory before live)
```powershell
.\tasks.ps1 test          # unit + integration against the mock printer
.\tasks.ps1 replay <clip> # tune thresholds on recorded footage
```
All tests must pass.

### Step 6: Dry-run against the real printer
```powershell
.\tasks.ps1 run-dry       # real camera + detection, action logged not sent
```
Run through a real print (including a deliberately sabotaged one if you want to see a real trigger). Confirm the camera handshake works on your firmware and detection fires sensibly.

### Step 7: Arm conservatively, then promote
```powershell
.\tasks.ps1 run-live      # starts with action.mode = pause
```
Watch a few prints. Once confident, set `action.mode = stop` and optionally install as a service.

### Step 8 (optional): Run as a service
- Windows: NSSM or Task Scheduler (`deploy/windows_service.md`).
- Linux / Orange Pi: `deploy/systemd/spaghetti-guard.service`.

---

## 9. Documentation deliverables

Claude Code must generate, alongside the code:
- `README.md`: one-paragraph what/why, quickstart, link to docs.
- `docs/ARCHITECTURE.md`: the diagram, state machine, module responsibilities.
- `docs/PROTOCOL.md`: camera handshake (with the verify-against-OpenBambuAPI note), MQTT topics, command payloads, report fields used.
- `docs/INSTALL.md`: section 8 expanded.
- `docs/RUNBOOK.md`: day-to-day operation, threshold tuning, snapshot review, retraining loop, common failures.
- `docs/SAFETY.md`: debounce rationale, pause-vs-stop, camera-loss policy, dry-run, kill switch, fire-risk note.

---

## 10. Definition of done (acceptance checklist)

- [ ] All unit tests green.
- [ ] Integration test: stop fired on spaghetti clip after exactly `consecutive_hits` frames; no command on a clean clip; disarms on FINISH.
- [ ] `replay_harness` and `metrics` produce a report on the provided fixtures.
- [ ] Dry-run mode proven to never publish (asserted in tests).
- [ ] Detection only runs while `gcode_state == RUNNING`.
- [ ] Camera loss notifies and never sends a stop.
- [ ] Secrets sourced only from env; `.env` git-ignored; `.env.example` has placeholders only.
- [ ] paho-mqtt 2.x API used correctly; dependencies pinned.
- [ ] Reconnect/backoff implemented for camera and MQTT.
- [ ] Training pipeline runs and emits a validation report.
- [ ] All docs present and accurate.
- [ ] `tasks.ps1` and `Makefile` expose: setup, test, lint, replay, run-dry, run-live, train, validate.

---

## 11. Instructions to Claude Code

**Build order** (test each module before moving on):
1. `config.py` + `test_config.py`
2. `control.py` + `test_control_payload.py` (testable against the mock broker alone)
3. `verification/mock_printer.py` (needed by later tests)
4. `camera.py` + `test_camera_parse.py`
5. `detector.py` + `test_detector_debounce.py`
6. `notifier.py`
7. `guard.py` + `test_guard_state.py`
8. `test_integration_loop.py` against the mock printer
9. `verification/replay_harness.py` + `metrics.py`
10. `training/` pipeline
11. `docs/`, `deploy/`, task runners

**Rules:**
- Run the test suite after each module. Do not advance on red.
- Do all loop testing against `mock_printer`. **Never open a socket to a real printer IP.** The only live steps (`run-dry`, `run-live`) are run by the operator, not by you.
- Pin every dependency. Note the paho-mqtt 2.x callback API explicitly.
- Generate `.env.example` with placeholders. Never write a real access code anywhere.
- Keep the camera handshake behind the `CameraBackend` interface so the `lib` backend can replace it if the raw handshake misbehaves on this firmware.
- Prefer high precision over recall in the model and thresholds; explain the trade in `SAFETY.md`.
- Write clear commit messages per module so the build history is reviewable.

When the acceptance checklist in section 10 is fully satisfied, summarize what was built, list any handshake or dependency assumptions that need operator verification, and hand back the three operator-run steps (printer LAN-mode prep, dry-run, conservative arming).
