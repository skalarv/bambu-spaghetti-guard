# Install

## Step 1 — Prepare the printer

1. Open the P1S touchscreen → **Settings → General → LAN Mode** (or
   **Developer Mode** on older firmware) and turn it on.
2. In your router, give the P1S a **static IP** (DHCP reservation by MAC).
3. From the printer, note the **LAN access code** and the **serial number**
   (Settings → Device).

The guard will refuse to publish if LAN Mode is off, and it will refuse to
connect if the printer's IP isn't reachable.

## Step 2 — Set up the workstation (Windows)

```powershell
git clone <repo> bambu-spaghetti-guard
cd bambu-spaghetti-guard
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

For the live detector you also need CUDA-PyTorch and Ultralytics. Pin to
`cu124` to match the RTX driver:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics opencv-python
```

These three are intentionally **not** in `requirements.txt`. The unit and
integration suite stubs them so test installs stay fast and disk-light.

## Step 3 — Configure

```powershell
copy secrets.local.txt.template secrets.local.txt
```

Edit `secrets.local.txt`:

```
BAMBU_IP=192.168.1.50
BAMBU_SERIAL=01P00A...
BAMBU_ACCESS_CODE=<from-touchscreen>
```

`secrets.local.txt` is git-ignored. Both `run` and `live-verify` load it
automatically; explicit environment variables take precedence (useful for
service managers). The access code is never read from `config.yaml` —
putting it there fails validation. The optional `NTFY_TOPIC_URL` /
`TELEGRAM_TARGET` keys in the secrets file feed `notify.target` when the
matching `notify.backend` is selected in `config.yaml`.

Then tune `config.yaml`: pick `action.mode` (start with `pause`),
`detector.conf_threshold` and `detector.consecutive_hits`, and your
`notify.backend` if you want push alerts.

## Step 4 — Get a model

Two paths:

* **Drop in weights** at `models/yolo11n-spaghetti.pt`. Document where they
  came from in `models/README.md`.
* **Train your own** — see `training/datasets.md` and run the pipeline in
  section 7 of the brief (`prepare_dataset.py` → `train.py` → `validate.py`).

## Step 5 — Verify offline (mandatory before going live)

```powershell
.\tasks.ps1 test
.\tasks.ps1 replay <path-to-clip>
```

`tasks.ps1 test` runs the unit + integration suite against the mock printer.
**Every test must pass** before pointing the guard at the real machine.

`tasks.ps1 replay` tunes thresholds on recorded footage and tells you which
frame the debouncer would fire on under the current settings.

## Step 6 — Dry-run against the real printer

```powershell
.\tasks.ps1 run-dry
```

This runs the full live pipeline — real camera, real detector, real MQTT
report parsing — but logs the action payload instead of publishing it. Run
through a normal print, then through a deliberately sabotaged one (an
already-detached print is the easiest reproducer). Confirm:

* The camera handshake works on your firmware.
* The detector trips on real failures and stays silent on normal prints.
* The action payload that would have been published matches the expected
  shape from `docs/PROTOCOL.md`.

## Step 7 — Arm conservatively, then promote

```powershell
.\tasks.ps1 run-live      # starts with action.mode=pause
```

Watch a few prints. When you trust the detection, set `action.mode = stop`
in `config.yaml` and restart. Stop saves filament; pause lets you inspect.
Make the change explicit; never default-rebrand a pause guard into a stop
guard without telling the operator.

## Step 8 (optional) — Run as a service

Linux / Orange Pi: `deploy/systemd/spaghetti-guard.service`.

Windows: see `deploy/windows_service.md` for NSSM / Task Scheduler recipes.

Orange Pi 3B (RKNN edge inference): `deploy/orange_pi_rknn.md`.
