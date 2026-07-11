# Bambu P1S Spaghetti Guard

Self-hosted print-failure detector for the Bambu Lab P1S. Watches the stock
chamber camera at ~1 fps, runs a YOLO model on each frame, and pauses or stops
the print over MQTT when a spaghetti / detachment failure is confirmed across
several consecutive frames.

See `bambu_spaghetti_guard_brief.md` for the full implementation brief.

## Current status (as of 2026-07-10)

- **Model**: `models/yolo11n-spaghetti.pt` — yolo26n fine-tuned on 24,098 CC-BY
  images from Roboflow Universe (spaghetti-3d + 3d-printing-flaws + syLucauc).
  60 epochs on RTX 5070. Final val: **P=0.86, R=0.68, mAP50=0.72, mAP50-95=0.42**.
- **Test suite**: 260 tests green.
- **Live-verify**: passes end-to-end against the operator's P1S
  (camera port 6000 + MQTT port 8883, LAN Mode enabled; printer IP lives in
  `secrets.local.txt`).
- **Dry-run through a real ~2.4h print (2026-07-10)**: 0 false triggers;
  detection armed/disarmed correctly on printer state. Observed 7 brief
  camera-silent dropouts (~15s each) — logged, no action taken (correct).
- **Notifications**: `notify.backend: ntfy` is live; topic in
  `secrets.local.txt` (`NTFY_TOPIC_URL`). The notifier verifies HTTPS against
  the **OS trust store**, so alerts survive this LAN's TLS-inspection proxy.
  Subscribe to the topic in the ntfy phone app to receive pushes.
- **Desktop launcher**: `Spaghetti Guard - LIVE.bat` on the operator's desktop
  runs `spaghetti-guard run --viewer` (real publish).
- **Guard runtime**: camera-reconnect wrapper caps at
  `camera.max_reconnect_attempts` *consecutive* dead reconnects (default 5;
  the budget resets after any healthy streaming), then exits with code 3 so
  a service manager restarts the guard.

## Quickstart (offline / test only)

```powershell
.\tasks.ps1 setup      # venv + editable install with test/lint tooling
# Live-only deps (skip if you only want to run tests):
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .[live]
.\tasks.ps1 test       # 260 tests
.\tasks.ps1 lint       # ruff + mypy
.\tasks.ps1 coverage   # coverage gate (>= 90% on src/)
```

## Live operation

1. Configure printer secrets in `secrets.local.txt` (gitignored — see
   `secrets.local.txt.template`). Required: `BAMBU_IP`, `BAMBU_SERIAL`,
   `BAMBU_ACCESS_CODE`; optional `NTFY_TOPIC_URL` / `TELEGRAM_TARGET` for
   notifications. Both `run` and `live-verify` load this file automatically
   (explicit env vars win). LAN Mode must be enabled on the P1S touchscreen.
   For push alerts, set `notify.backend: ntfy` in `config.yaml` and subscribe
   to your `NTFY_TOPIC_URL` topic in the ntfy phone app (see `docs/INSTALL.md`).
2. Probe the printer without publishing:
   ```powershell
   spaghetti-guard live-verify        # or: python scripts\live_verify.py
   ```
3. Dry-run against a real print (real detection, no publish):
   ```powershell
   .\tasks.ps1 run-dry
   ```
4. When trusted, run live (drops `--dry-run`):
   ```powershell
   .\tasks.ps1 run-live
   ```

## Repository layout

```
src/spaghetti_guard/        # runtime (camera, control, detector, guard, cli, viewer, notifier)
tests/                      # pytest suite; coverage gated at >= 90% on src/
verification/               # mock_printer + replay + metrics harness
training/
  data/{spaghetti-3d, 3d-printing-flaws, sylucauc-3dpfd, merged}/
  merge_datasets.py         # unified 7-class taxonomy builder
  train.py                  # yolo26n / yolo11n / yolo11s fine-tuner
  train_from_snapshots.py   # retraining loop (brief §7.3)
models/yolo11n-spaghetti.pt # active detector weights
scripts/live_verify.py      # end-to-end probe against the real P1S
docs/{ARCHITECTURE,INSTALL,PROTOCOL,RUNBOOK,SAFETY,RETRAINING}.md
```

## Retraining loop

Every trigger writes a `trigger-*.jpg` under `failure_snapshots/`. Review those
with an annotator (`labelImg`, Roboflow, etc.), then:

```powershell
python training\train_from_snapshots.py --review         # see what's unlabelled
python training\train_from_snapshots.py --promote        # fold into training set
python training\merge_datasets.py                        # rebuild merged/
python training\train.py --data training\data\merged\data.yaml --base models\yolo11n-spaghetti.pt --name chamber-YYYYMMDD
```

Full guide: `docs/RETRAINING.md`.

## Safety posture

- Default action is `pause` (safer while tuning). Flip to `stop` in
  `config.yaml` once trusted.
- Debounce (`consecutive_hits=6`) means a single bad frame never fires.
- The pause/stop publish is **verified** (broker ack within a timeout); an
  unconfirmed command keeps the guard TRIGGERED, alerts the operator, and
  retries every ~10 s instead of pretending it acted.
- The control action fires **before** the (blocking, HTTP) notification.
- Camera loss **notifies but never publishes stop** (brief §3.3); the
  watchdog runs on its own thread so a stalled socket can't starve it.
- A stale printer report (> 30 s without MQTT data) raises an operator alert
  — a guard trusting stale state is silently unprotected.
- MQTT loss reconnects with exponential backoff (paho built-in; actions are
  never queued).
- `--dry-run` runs the full pipeline including snapshotting but logs the MQTT
  payload instead of publishing.
