# Bambu P1S Spaghetti Guard

Self-hosted print-failure detector for the Bambu Lab P1S. Watches the stock
chamber camera at ~1 fps, runs a YOLO model on each frame, and pauses or stops
the print over MQTT when a spaghetti / detachment failure is confirmed across
several consecutive frames.

See `bambu_spaghetti_guard_brief.md` for the full implementation brief.

## Current status (as of 2026-07-07)

- **Model**: `models/yolo11n-spaghetti.pt` — yolo26n fine-tuned on 24,098 CC-BY
  images from Roboflow Universe (spaghetti-3d + 3d-printing-flaws + syLucauc).
  60 epochs on RTX 5070. Final val: **P=0.86, R=0.68, mAP50=0.72, mAP50-95=0.42**.
- **Test suite**: 260 tests green.
- **Live-verify**: passes end-to-end against the operator's P1S
  (camera port 6000 + MQTT port 8883, LAN Mode enabled; printer IP lives in
  `secrets.local.txt`).
- **Guard runtime**: camera-reconnect wrapper caps at
  `camera.max_reconnect_attempts` *consecutive* dead reconnects (default 5;
  the budget resets after any healthy streaming), then exits with code 3 so
  a service manager restarts the guard.

## Quickstart (offline / test only)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Live-only deps (skip if you only want to run tests):
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics opencv-python
.\tasks.ps1 test
```

## Live operation

1. Configure printer secrets in `secrets.local.txt` (gitignored — see
   `secrets.local.txt.template`). Required: `BAMBU_IP`, `BAMBU_SERIAL`,
   `BAMBU_ACCESS_CODE`. LAN Mode must be enabled on the P1S touchscreen.
2. Probe the printer without publishing:
   ```powershell
   python scripts\live_verify.py
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
tests/                      # 229 pytest tests, 95.58% coverage
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
- Camera loss **notifies but never publishes stop** (brief §3.3).
- MQTT loss triggers exponential-backoff reconnect (never queues actions).
- `--dry-run` runs the full pipeline including snapshotting but logs the MQTT
  payload instead of publishing.
